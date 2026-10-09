"""Reconstruct saved native token means/weights, never model logits or Adam."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ORIGINAL_MANIFEST_SHA = 'af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
ORIGINAL_TRAINER_SHA = '74b9f70e6940b03a61035224d8d22605832798967346761167cd5b9284ae25fe'
ORIGINAL_SCORE_SHA = '75828103d939872e9d3cc3d35880db50fa3cf0615852372fa0197e4dcb589f0e'
VARIANTS = {'G_POS': ('answer_gap', 1), 'G_NEG': ('answer_gap', -1),
            'H_POS': ('student_entropy', 1), 'H_NEG': ('student_entropy', -1)}
ATOL = 2e-5
RTOL = 2e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def checked_file(ref):
    path = Path(ref['path']).resolve(); before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Bound audit input changed during SHA read')
    require(digest.hexdigest() == ref['sha256'], 'Bound audit input SHA differs')
    require('bytes' not in ref or ref['bytes'] == after.st_size, 'Bound audit input size differs')
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=after.st_size,
                device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns)


def read_bytes(ref):
    actual = checked_file(ref); payload = Path(actual['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == actual['sha256'], 'Input changed after SHA read')
    return payload, actual


def checked_json(ref):
    payload, actual = read_bytes(ref)
    return json.loads(payload), actual


def jsonl(ref):
    payload, actual = read_bytes(ref)
    require(not payload or payload.endswith(b'\n'), 'Committed JSONL has an incomplete trailing line')
    return [json.loads(line) for line in payload.splitlines() if line.strip()], actual


def same_ref(left, right):
    return left['path'] == right['path'] and left['sha256'] == right['sha256']


def original_functions(config, manifest_sha):
    """Hash the entire original helper, compile only exact stdlib functions."""
    require(manifest_sha == config['helper_source_manifest']['sha256'] == ORIGINAL_MANIFEST_SHA,
            'Exact original00fec71 helper manifest required')
    manifest, proof = checked_json(config['helper_source_manifest'])
    root = Path(config['paths']['helper_root']).resolve(); source = {}
    for name, entry in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe original source inventory')
        payload, actual = read_bytes(dict(path=str(root / relative), sha256=entry['sha256'], bytes=entry['bytes']))
        source[name] = payload
    require({name for name in manifest['files'] if name.endswith('.py')}
            == {str(path.relative_to(root)) for path in root.rglob('*.py')}, 'Original Python source inventory differs')
    require(manifest['files']['trainer.py']['sha256'] == ORIGINAL_TRAINER_SHA
            and manifest['files']['sibling_scores.py']['sha256'] == ORIGINAL_SCORE_SHA,
            'Original committed/score code identity differs')
    namespace = dict(math=math)
    wanted = {'trainer.py': {'verify_committed_receipts'},
              'sibling_scores.py': {'weights', 'compare_selected_training_metrics'},
              'checkpoint.py': {'science_contract'}}
    constants = {'WEIGHTS_VERSION', 'AMPLITUDE', 'NUMERICAL_FLOOR'}
    for name, functions in wanted.items():
        tree = ast.parse(source[name]); selected = []
        found = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
                selected.append(node); found.add(node.name)
            elif name == 'sibling_scores.py' and isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in constants for target in node.targets):
                ast.literal_eval(node.value); selected.append(node)
        require(found == functions, 'Exact original readonly function inventory differs')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / name), 'exec'), namespace)
    require('torch' not in sys.modules, 'Readonly source extraction imported Torch')
    return namespace, dict(manifest=proof, source_files=len(source),
                          complete_byte_sha_readback=True, selected_original_functions=sorted(
                              function for group in wanted.values() for function in group))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, label, errors):
    require(finite(actual) and finite(expected) and math.isclose(actual, expected, abs_tol=ATOL, rel_tol=RTOL),
            'Saved native scalar reconstruction differs: ' + label)
    errors[label] = max(errors.get(label, 0.), abs(actual - expected))


def check_normalization(score, functions):
    require(score['schema'] == 'opsd-current-preupdate-five-sibling-scores-v1' and score['accepted'] is True,
            'Signal score schema/acceptance differs')
    signal, direction = VARIANTS[score['variant']]
    require(score['signal'] == signal and score['scoring_scope'] == 'no_grad_current_preupdate_hf_replay_not_generation_scores'
            and score['original_training_objective'] is False and score['selected_D_only_training'] is True,
            'Signal score changed mathematical/execution scope')
    siblings = score['siblings']
    require(len(siblings) == 5 and [row['sibling'] for row in siblings] == list(range(5)),
            'Five unique original signal sibling slots required')
    skipped = [row.get('skipped_reason') for row in siblings]
    reason = None
    if any(skipped):
        require(skipped == ['sibling_encoding_failure'] * 5 and score['neutral_details'],
                'Whole-group encoding fallback lacks five explicit NA siblings')
        require(all(row['answer_gap'] is None and row['student_entropy'] is None for row in siblings),
                'Encoding fallback fabricated signal scores')
        reason = 'sibling_encoding_failure'
    else:
        for row in siblings:
            require(finite(row['student_entropy']), 'Native entropy signal unavailable without encoding fallback')
            policy_indices, answer_indices = row['valid_policy_token_indices'], row['answer_token_indices']
            require(all(type(index) is int and index >= 0 for index in policy_indices + answer_indices)
                    and policy_indices == sorted(set(policy_indices))
                    and answer_indices == sorted(set(answer_indices))
                    and set(answer_indices) <= set(policy_indices),
                    'Answer indices must be sorted unique and within valid policy positions')
            if row['answer_gap'] is None:
                require(row['answer_valid_tokens'] == 0 and row['answer_token_indices'] == []
                        and row['gap_missing_reason'] == 'no_valid_final_answer_payload_tokens',
                        'Unavailable answer gap lacks explicit zero-answer-token reason')
            else:
                require(finite(row['answer_gap']) and row['answer_valid_tokens'] > 0
                        and len(row['answer_token_indices']) == row['answer_valid_tokens'],
                        'Available answer gap has invalid token/finite inventory')
    normalization = functions['weights']([row[signal] for row in siblings], direction, neutral_reason=reason)
    require(normalization == score['normalization'], 'Recomputed five-sibling weights/normalization differ')
    selected = score['selected_sibling']
    require(type(selected) is int and 0 <= selected < 5 and score['selected_score'] == siblings[selected],
            'Selected D score changed sibling')
    require(score['selected_pair_weight'] == normalization['weights'][selected], 'Selected D scalar weight differs')
    restore = score['state_restore']
    require(restore['state_restored'] is True and restore['rng_consumed'] is False
            and restore['rng_before_sha256'] == restore['restored_rng_sha256'],
            'Native scorer lacks unchanged/restored production RNG proof')
    body = {key: value for key, value in score.items() if key != 'sha256'}
    require(score['sha256'] == canonical(body), 'Signal receipt inner identity differs')
    return normalization, bool(reason)


def check_pair(header, rows, score, metrics, update, batch, config, encoded_sha, source_pair, functions):
    errors = {}; normalization, fallback = check_normalization(score, functions)
    pair_index = metrics['pair_index']; rank = header['rank']; actual = update['global_actual_pairs']
    before = update['pre_update_adapter_sha256']; identity = score['preupdate_identity']
    require(header['schema'] == 'opsd-online-training-token-header-v1'
            and header['run_id'] == update['run_id'] == config['run_id']
            and header['outer_index'] == update['outer_index']
            and header['optimizer_update'] == update['optimizer_updates']
            and header['pair_index'] == pair_index and header['retry'] == update['retries']
            and header['encoded_sha256'] == encoded_sha,
            'Actual forward header update/selected encoding differs')
    require(header['contract'] == update['contract'] == functions['science_contract'](config)
            and header['frozen_teacher_identity'] == config['source_model_manifest']['sha256']
            and header['component_clip'] == config['scientific']['component_clip'] == 10
            and header['loss_scope'] == update['loss_scope'] == 'actual_training_forward_before_optimizer',
            'Actual forward teacher/science/scope differs')
    require(header['student_pre_update_adapter_sha256'] == batch['preupdate_adapter_sha256'] == before
            and identity['adapter_tensor_sha256'] == before
            and identity['run_id'] == config['run_id'] and identity['variant'] == config['variant'] == score['variant']
            and identity['outer_index'] == update['outer_index'] and identity['optimizer_update'] == update['optimizer_updates']
            and identity['rank'] == rank and identity['pair_index'] == pair_index and identity['retry'] == update['retries']
            and identity['encoded_sha256'] == encoded_sha
            and identity['source_model_identity'] == config['source_model_manifest']['sha256']
            and identity['cursor'] == batch['cursor'] and identity['order_sha256'] == batch['order_sha256']
            and same_ref(identity['checkpoint'], batch['checkpoint']),
            'Scorer/header actual preupdate model/cursor/rank differs')
    require(header['sampling_model_identity']['source_model_identity'] == config['source_model_manifest']['sha256']
            and header['outer_old_adapter'] == update['outer_old_adapter'], 'Sampling/source/original-old identity differs')
    original = header['identity']; mining = header['mining']
    require(original['question_id'] == source_pair['question_id'] == score['question_id']
            and original['pair_row_sha256'] == score['selected_pair_sha256'] == canonical(source_pair)
            and mining == source_pair['mining'] and mining['student_sibling'] == score['selected_sibling']
            and original['student_reward'] == source_pair['student_reward']
            and original['reference_reward'] == source_pair['reference_reward'] == 1,
            'Selected original D/reference pair/source SHA differs')
    require(original['student_response_sha256'] == hashlib.sha256(source_pair['student'].encode()).hexdigest()
            and original['reference_response_sha256'] == score['reference_response_sha256']
            == hashlib.sha256(source_pair['reference'].encode()).hexdigest(), 'Original B/D text identity differs')
    require([row['original_row_sha256'] for row in score['siblings']] == mining['sibling_row_sha256']
            and score['selected_score']['encoded_sha256'] == encoded_sha,
            'Sibling/selected-D encoding provenance differs')
    require(header['pair_weight'] == metrics['pair_weight'] == metrics['signal_scalar']['weight']
            == score['selected_pair_weight'], 'Actual selected D weight changed')
    require(metrics['signal_scalar']['question_id'] == original['question_id']
            and metrics['signal_scalar']['sibling'] == mining['student_sibling']
            and metrics['signal_scalar']['strict_reward'] == original['student_reward']
            and metrics['signal_scalar']['success_fraction'] == sum(mining['rewards_by_sibling']) / 5,
            'Actual selected signal metadata differs')
    ids, mask, offsets, regions = (header[name] for name in ('token_ids', 'policy_mask', 'token_offsets', 'token_regions'))
    require(len(ids) == len(mask) == len(offsets) == len(regions) and ids
            and all(type(token) is int and token >= 0 for token in ids)
            and all(type(value) in (int, bool) and value in (0, 1) for value in mask), 'Header token/mask alignment differs')
    token_rows = [row for row in rows if row.get('policy_mask') is True]
    positions = [index for index, enabled in enumerate(mask) if enabled]
    require([row['token_index'] for row in token_rows] == positions and len(positions) == metrics['valid_tokens'] > 0,
            'Actual effective mask/unique token inventory differs')
    for row in rows:
        if row.get('policy_mask') is not True:
            require(row.get('full_vocab_dump') in ('complete', 'cap_skip'), 'Unexpected uncommitted token-log row')
    for row in token_rows:
        index = row['token_index']
        require(row['token_id'] == ids[index] and row['token_offsets'] == offsets[index]
                and row['region'] == regions[index] and row['valid_trajectory_tokens'] == len(positions)
                and row['global_actual_pairs'] == actual and row['pair_weight'] == header['pair_weight'],
                'Actual token identity/mask/global denominator differs')
        require(row['target_kind'] == 'teacher' and row['direction'] == 'student_to_fixed_target_reverse'
                and row['loss_scope'] == 'actual_training_forward_before_optimizer'
                and row['teacher_forced_fixed_student_prefix'] is True and row['actual_generation_scores'] is False
                and row['outer_old_identity']['manifest'] == header['outer_old_adapter'],
                'Saved token changed target/forward/outer-old scope')
        for name in ('pre_clip_full_vocabulary_kl', 'post_clip_full_vocabulary_loss', 'student_to_teacher_raw_kl',
                     'student_to_outer_old_raw_kl', 'student_to_mixture_raw_kl', 'student_observed_token_logp',
                     'teacher_observed_token_logp', 'student_entropy', 'teacher_entropy',
                     'trajectory_contribution', 'global_batch_contribution'):
            require(finite(row[name]), 'Actual native token scalar is unavailable or nonfinite: ' + name)
        close(row['trajectory_contribution'], row['post_clip_full_vocabulary_loss'] / len(positions), 'token_mean_contribution', errors)
        close(row['global_batch_contribution'], header['pair_weight'] * row['post_clip_full_vocabulary_loss']
              / len(positions) / actual, 'actual_pair_global_contribution', errors)
        close(row['pre_clip_full_vocabulary_kl'], row['student_to_teacher_raw_kl'], 'pure_teacher_raw_kl', errors)
        close(row['student_to_mixture_raw_kl'], row['student_to_teacher_raw_kl'], 'pure_teacher_target_kl', errors)
        require(type(row['clipped_vocabulary_count']) is int and type(row['vocabulary_size']) is int
                and 0 <= row['clipped_vocabulary_count'] <= row['vocabulary_size']
                and row['vocabulary_size'] > row['token_id'], 'Native vocabulary/clip inventory differs')
        close(row['clipped_vocabulary_fraction'], row['clipped_vocabulary_count'] / row['vocabulary_size'], 'clip_fraction', errors)
        for name in ('clipped_student_probability_mass', 'clipped_teacher_probability_mass', 'clipped_mixture_probability_mass'):
            require(finite(row[name]) and -ATOL <= row[name] <= 1 + ATOL, 'Native clipped probability mass differs')
    means = dict(raw_loss=math.fsum(row['pre_clip_full_vocabulary_kl'] for row in token_rows) / len(positions),
                 post_loss=math.fsum(row['post_clip_full_vocabulary_loss'] for row in token_rows) / len(positions),
                 student_to_teacher_raw_kl=math.fsum(row['student_to_teacher_raw_kl'] for row in token_rows) / len(positions),
                 student_to_outer_old_raw_kl=math.fsum(row['student_to_outer_old_raw_kl'] for row in token_rows) / len(positions),
                 student_to_mixture_raw_kl=math.fsum(row['student_to_mixture_raw_kl'] for row in token_rows) / len(positions))
    for name, expected in means.items(): close(metrics[name], expected, name, errors)
    close(metrics['weighted_post_loss'], header['pair_weight'] * means['post_loss'], 'weighted_trajectory_mean', errors)
    require(metrics['target_kind'] == 'teacher' and metrics['direction'] == 'reverse'
            and metrics['clipped_vocabulary_count'] == sum(row['clipped_vocabulary_count'] for row in token_rows),
            'Actual trajectory target/clip count differs')
    comparison = functions['compare_selected_training_metrics'](score, token_rows)
    require(comparison == metrics['score_training_match'], 'Saved comparison differs from independently repeated original comparison')
    require(comparison['accepted'] is True and comparison['checked'] is not fallback,
            'Signal/training replay fallback was misrepresented as a logits comparison')
    if not fallback:
        for name, difference in comparison['differences'].items():
            if difference is not None: errors['scorer_' + name] = abs(difference)
    return dict(valid_tokens=len(positions), scorer_checked=not fallback, scorer_fallback_na=fallback,
                gap_missing=score['selected_score']['answer_gap'] is None, weight=header['pair_weight'],
                weighted_post_loss=metrics['weighted_post_loss'], max_absolute_errors=errors,
                top32_positions=sum('student_top32_logp' in row for row in token_rows),
                full_vocab_metadata_complete=sum(row.get('full_vocab_dump') == 'complete' for row in rows),
                full_vocab_metadata_cap_skip=sum(row.get('full_vocab_dump') == 'cap_skip' for row in rows))


def audit(config_ref, checkpoint_ref, expected_updates, expected_outer, helper_sha):
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Token audit must remain stdlib with empty CUDA')
    config, config_proof = checked_json(config_ref); config['_config_sha256'] = config_proof['sha256']
    require(config['schema'] == 'opsd-signal-weight-config-v1' and config['variant'] in VARIANTS,
            'Explicit original four-signal config required')
    functions, source = original_functions(config, helper_sha)
    checkpoint, cp_proof = checked_json(checkpoint_ref)
    require(checkpoint['schema'] == 'opsd-online-checkpoint-v1' and checkpoint['accepted'] is True
            and checkpoint['run_id'] == config['run_id'] and checkpoint['config_sha256'] == config_proof['sha256']
            and checkpoint['optimizer_updates'] == expected_updates and checkpoint['cursor']['outer_index'] == expected_outer
            and checkpoint['source_model_identity'] == config['source_model_manifest']['sha256']
            and checkpoint['runtime_identity'] == config['runtime_manifest']['sha256']
            and checkpoint['contract'] == functions['science_contract'](config), 'Pinned checkpoint config/source/science/counter differs')
    for ref in (config['source_model_manifest'], config['runtime_manifest'], config['common_initial_adapter_manifest']): checked_file(ref)
    outer, _ = checked_json(checkpoint['training_input']['outer_manifest'])
    order, order_proof = checked_json(checkpoint['training_input']['order'])
    encoding, _ = checked_json(checkpoint['training_input']['encoding'])
    pairs, _ = jsonl(checkpoint['training_input']['pairs']); pair_map = {pair['question_id']: pair for pair in pairs}
    require(len(pair_map) == len(pairs) and order['outer_index'] == expected_outer
            and outer['run_id'] == config['run_id'] and outer['config_sha256'] == config_proof['sha256']
            and outer['outer_index'] == expected_outer
            and outer['sampling_model_identity'] == checkpoint['sampling_model_identity']
            and outer['outer_old_adapter'] == checkpoint['outer_old_adapter'], 'Original outer/pair/order provenance differs')
    K = checkpoint['cursor']['usable_pair_count']; cursor = checkpoint['cursor']['next_pair_cursor']
    require(order['encoded_sha256'] == encoding['sha256'] == canonical(encoding['rows'])
            and len(encoding['rows']) == encoding['K'] == K and encoding['P'] == len(pairs)
            and sorted(order['pair_indices']) == list(range(K)) and order_proof['sha256'] == checkpoint['cursor']['order_sha256']
            and encoding['sha256'] == checkpoint['cursor']['encoded_sha256'], 'Original encoding/order inventory differs')
    native_tokens = functions['verify_committed_receipts'](checkpoint['outer_update_receipts'], checkpoint,
                outer['cumulative_optimizer_updates_before'], checked_json, checked_file, order['pair_indices'])
    require(checkpoint['cumulative_pair_exposure'] >= cursor and expected_updates > 0,
            'No committed Adam/token record at the explicit checkpoint')
    summaries = []; files = []; errors = {}; previous_digest = None; accumulated_tokens = 0
    for ref in checkpoint['outer_update_receipts']:
        update, update_ref = checked_json(ref); batch, batch_ref = checked_json(update['score_batch'])
        actual = update['global_actual_pairs']; offset = sum(row['actual_pairs'] for row in summaries)
        require(update['contract'] == functions['science_contract'](config)
                and update['learning_rate'] == config['scientific']['learning_rate'] == 1e-6
                and update['loss_reduction'] == 'trajectory_mean' and update['micro_batch'] == 1
                and update['reference_student_prefix_shared'] is True and update['frozen_teacher'] is True,
                'Committed actual update changed science/target scope')
        require(batch['schema'] == 'opsd-signal-score-batch-v1' and batch['accepted'] is True
                and batch['run_id'] == config['run_id'] and batch['outer_index'] == expected_outer
                and batch['optimizer_update'] == update['optimizer_updates'] and batch['cursor'] == offset
                and batch['global_actual_pairs'] == actual and batch['order_sha256'] == order_proof['sha256']
                and batch['preupdate_adapter_sha256'] == update['pre_update_adapter_sha256'],
                'Durable score batch differs from actual committed update')
        precheckpoint, _ = checked_json(batch['checkpoint'])
        require(precheckpoint['optimizer_updates'] == update['optimizer_updates'] - 1
                and precheckpoint['all_rank_adapter_sha256'] == [update['pre_update_adapter_sha256']] * 4
                and precheckpoint['run_id'] == config['run_id'] and precheckpoint['config_sha256'] == config_proof['sha256'],
                'Scoring did not use immediately preceding actual accepted checkpoint')
        require(previous_digest is None or previous_digest == update['pre_update_adapter_sha256'], 'Committed adapter update chain broken')
        previous_digest = update['post_update_adapter_sha256']; score_refs = {}
        require(len(batch['all_rank_scores']) == 4 and {row['rank'] for row in batch['all_rank_scores']} == set(range(4)),
                'Signal score batch lacks four unique ranks')
        for rank in batch['all_rank_scores']:
            for item in rank['scores']:
                key = (rank['rank'], item['pair_index']); require(key not in score_refs, 'Signal score pair/rank duplicate')
                score_refs[key] = item['score']
        expected_batch = order['pair_indices'][offset:offset + actual]
        require(set(score_refs) == {(rank, index) for rank in range(4) for index in expected_batch[rank::4]},
                'Actual score ranks differ from original four-rank shuffled batch partition')
        pair_results = []
        for rank in update['all_rank_actual_token_metrics']:
            for metrics in rank['pairs']:
                key = (rank['rank'], metrics['pair_index'])
                require(key in score_refs and same_ref(score_refs[key], metrics['signal_score']), 'Actual selected forward unbound to score batch')
                score, sref = checked_json(metrics['signal_score']); records, tref = jsonl(metrics['token_log'])
                require(records and records[0].get('schema') == 'opsd-online-training-token-header-v1', 'Committed token header absent')
                header = records[0]
                require(header['sampling_model_identity'] == outer['sampling_model_identity']
                        == checkpoint['sampling_model_identity'], 'Full recorded sampling model identity differs from original outer/checkpoint')
                require(header['rank'] == rank['rank'] and same_ref(header['signal_score'], sref)
                        and same_ref(header['score_batch'], batch_ref)
                        and header['identity']['question_id'] in pair_map, 'Token header actual rank/score identity differs')
                result = check_pair(header, records[1:], score, metrics, update, batch, config,
                                    encoding['rows'][metrics['pair_index']], pair_map[header['identity']['question_id']], functions)
                pair_results.append(result); files.extend([sref, tref])
                for name, value in result['max_absolute_errors'].items(): errors[name] = max(errors.get(name, 0.), value)
        require(len(pair_results) == len(score_refs) == actual, 'Actual score/forward pair exposure inventory differs')
        count = sum(row['valid_tokens'] for row in pair_results); require(count == update['global_valid_tokens'], 'Actual global token count differs')
        accumulated_tokens += count; files.extend([update_ref, batch_ref])
        summaries.append(dict(optimizer_update=update['optimizer_updates'], actual_pairs=actual, valid_tokens=count,
                scorer_checked=sum(row['scorer_checked'] for row in pair_results), scorer_fallback_na=sum(row['scorer_fallback_na'] for row in pair_results),
                selected_answer_gap_missing=sum(row['gap_missing'] for row in pair_results),
                selected_weight_mean=math.fsum(row['weight'] for row in pair_results) / actual,
                actual_global_weighted_post_loss=math.fsum(row['weighted_post_loss'] for row in pair_results) / actual,
                top32_positions=sum(row['top32_positions'] for row in pair_results),
                full_vocab_metadata_complete=sum(row['full_vocab_metadata_complete'] for row in pair_results),
                full_vocab_metadata_cap_skip=sum(row['full_vocab_metadata_cap_skip'] for row in pair_results)))
    require(accumulated_tokens == native_tokens and previous_digest == checkpoint['all_rank_adapter_sha256'][0]
            and checkpoint['all_rank_adapter_sha256'] == [previous_digest] * 4, 'Pinned last adapter/token count does not close committed chain')
    require('torch' not in sys.modules, 'Token audit unexpectedly imported Torch')
    return dict(schema='opsd-signal-committed-token-audit-v1', accepted=True, checked_unix=time.time(),
                config=config_proof, checkpoint=cp_proof, variant=config['variant'], run_id=config['run_id'],
                optimizer_updates=expected_updates, outer_index=expected_outer, committed_updates_read=len(summaries),
                actual_pair_exposures=sum(row['actual_pairs'] for row in summaries), valid_token_rows=accumulated_tokens,
                updates=summaries, original_source=source, referenced_record_file_count=len(files),
                referenced_record_inventory_sha256=canonical([{'path': ref['path'], 'sha256': ref['sha256'], 'bytes': ref['bytes']} for ref in files]),
                max_absolute_errors=errors, tolerances=dict(absolute=ATOL, relative=RTOL), torch_imported=False,
                scope='Committed saved actual-forward scalar/mask/weight reconstruction only; no new logits, tokenizer, full-vocabulary arrays, Adam/RNG restoration, generation or EM acceptance')


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'checkpoint', 'checkpoint-sha256', 'original-helper-manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-updates', type=int, required=True)
    parser.add_argument('--expected-outer', type=int, required=True)
    args = parser.parse_args()
    try:
        result = audit(dict(path=args.config, sha256=args.config_sha256),
                       dict(path=args.checkpoint, sha256=args.checkpoint_sha256),
                       args.expected_updates, args.expected_outer, args.original_helper_manifest_sha256)
    except Exception as error:
        result = dict(schema='opsd-signal-committed-token-audit-v1', accepted=False,
                      error_type=type(error).__name__, error=str(error), torch_imported='torch' in sys.modules,
                      no_signals_sent=True, no_gpu_or_training_started=True)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
