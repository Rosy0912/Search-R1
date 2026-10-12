"""H100-only CPU raw-action support and existing-forward residual diagnostics.

No forward/model/restore/Adam/Ray. No threshold selection or probability approval.
Raw tensors and texts remain on H100; output contains only bounded scalar metadata.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
RUN = ROOT / 'runs/gigpo-e1-policy-safety-calibration25-job3289-1791251062110516012'
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
BINDING = '7cc3098a35d6f56e809e293e2f26642cc365eac8c49c1ddaedb592e989c79ce2'
INVENTORY = '0cc1015260365ad2ec0b644fe28b04105e16490932eb182d993c4b394fb26079'
PARENT_IDENTITY = 'fbaa431835eddcde7472ffa35fe969970905a2e7e14832c630f9d8c99c0bc6a5'
SCOPE = 'CPU_RAW_ACTION_SUPPORT_AND_EXISTING_FORWARD_RESIDUAL_NOT_ACCEPTANCE'
OUTPUT_NAME = 'CALIBRATION3289-RAW-ACTION-SUPPORT-RESIDUAL-v1.json'
MAX_OUTPUT_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def derive_real_lengths(sampled_ids, chosen_logp, pad, torch):
    """Frozen llm.py pads chosen logprobs with the positive PAD token ID.

    A true sampled PAD has an ordinary <=0 logprob and remains in the action.
    Only a suffix of exact positive sentinels is padding. This derivation is
    specific to the SHA-bound 52 output adapter, never a generic token-ID trim.
    """
    require(pad > 0 and sampled_ids.shape == chosen_logp.shape and sampled_ids.ndim == 2,
            'Invalid positive sentinel or sampled tensor shape')
    require(sampled_ids.dtype == torch.int64, 'Sampled IDs are not int64')
    sentinel = chosen_logp.eq(pad)
    require(bool((torch.isfinite(chosen_logp) & (sentinel | chosen_logp.le(0))).all()),
            'Chosen logprob is nonfinite/positive and not the bound padding sentinel')
    lengths = []
    masks = []
    width = sampled_ids.shape[1]
    for index in range(len(sampled_ids)):
        positions = sentinel[index].nonzero().flatten()
        length = int(positions[0]) if len(positions) else width
        require(bool(sentinel[index, length:].all()) and not bool(sentinel[index, :length].any()),
                'Logprob padding sentinel is not a contiguous suffix')
        require(bool(sampled_ids[index, length:].eq(pad).all()),
                'Sentinel suffix does not match token tensor padding')
        lengths.append(length)
        masks.append([1] * length + [0] * (width - length))
    return lengths, masks


def analyze(spec, parent, torch, tokenizer, processor, candidate):
    turns = json.loads((RUN / 'training-prefix-turns.json').read_text())
    require(len(turns) == 4, 'Actual turn serial coverage changed')
    counters = dict(actions=0, raw_tokens=0, canonical_tokens=0, token_path_changed=0,
                    empty_actions=0, closing_inside_token=0, terminal_eos_excluded=0,
                    genuine_sampled_pad_tokens=0, repeated_probe_state_changed=0,
                    hf_response_token_mismatch=0, hf_position_or_mask_mismatch=0,
                    canonical_replay_mismatch=0, support_rejected=0)
    cases, errors, joints, sources, seen = [], [], [], set(), set()
    global_token = dict(hf_repeat_max_abs=0., hf_vllm_max_abs=0.)
    global_joint = dict(hf_repeat_max_abs=0., hf_vllm_max_abs=0.)
    coverage = []
    for serial, rows in enumerate(turns):
        per_rank = []
        for rank in range(8):
            path = RUN / f'probabilities/train/call-{serial}-rank-{rank}.json'
            record = json.loads(path.read_text())
            require(record['rank'] == rank and record['serial'] == serial and record['phase'] == 'train'
                    and record['source_commit'] == SOURCE, 'Wrong actual raw record')
            raw_path = Path(record['raw']['path'])
            inventory_record = next(row for row in parent['raw_material_inventory'] if row['path'] == str(raw_path))
            require(record['raw'] == {key: inventory_record[key] for key in ('path', 'bytes', 'sha256')},
                    'Raw PT record differs from the actual immutable inventory')
            raw = torch.load(raw_path, map_location='cpu', mmap=True, weights_only=False)
            lengths, masks = derive_real_lengths(raw['sampled_ids'], raw['sampled_log_probs'], tokenizer.pad_token_id, torch)
            require(len(lengths) == record['rows'] and raw['rank'] == rank and raw['serial'] == serial,
                    'Raw PT/JSON row identity differs')
            per_rank.append((raw, record, lengths, masks))
        counts = [len(value[2]) for value in per_rank]
        require(len(set(counts)) == 1 and len(rows) <= sum(counts) and sum(counts) - len(rows) < 8,
                'Unexpected actual GPU padding coverage')
        active_tokens = 0
        for index, row in enumerate(rows):
            rank, local = divmod(index, counts[0])
            raw, record, lengths, masks = per_rank[rank]
            key = (int(row['source']), int(row['step_id']))
            require(key not in seen, 'Duplicate source/action identity')
            seen.add(key); sources.add(key[0]); counters['actions'] += 1
            ids = raw['sampled_ids'][local].tolist()
            full = raw['tensors']
            require(full['responses'][local, :lengths[local]].tolist() == ids[:lengths[local]],
                    'Returned raw response differs from actual engine chosen tokens')
            prompt_width = full['prompts'].shape[1]
            prompt_mask = full['attention_mask'][local, :prompt_width].bool()
            prefix = full['prompts'][local][prompt_mask].tolist()
            require(prefix == row['generation_prefix'] == raw['actual_engine_prompt_token_ids'][local],
                    'Actual worker/engine/recorded canonical prefix differs')
            canonical_ids, canonical_text = processor._postprocess_responses(full['responses'][local:local + 1])
            text = canonical_text[0]
            try:
                action = candidate.retain_sampled_action(tokenizer, ids, lengths[local], masks[local], text)
            except Exception as exc:
                counters['support_rejected'] += 1
                if len(errors) < 20:
                    errors.append(dict(source=key[0], step_id=key[1], serial=serial, rank=rank, local_row=local,
                                       real_sampled_length=lengths[local], error_class=type(exc).__name__, error=str(exc)[:1000]))
                continue
            actor = list(action.actor_ids)
            require(actor == ids[:len(actor)], 'Candidate actor path is not the exact sampled prefix')
            old_canonical = canonical_ids[0][canonical_ids[0].ne(tokenizer.pad_token_id)].tolist()
            # The old observer retains the prefix after the unchanged right-side
            # cap. Full text parsing and that canonical retained prefix stay exact.
            counters['canonical_replay_mismatch'] += old_canonical[:len(row['retained_action_tokens'])] != row['retained_action_tokens']
            counters['raw_tokens'] += len(actor); counters['canonical_tokens'] += len(old_canonical)
            counters['token_path_changed'] += actor != old_canonical
            counters['empty_actions'] += len(actor) == 0
            counters['closing_inside_token'] += action.closing_inside_token
            counters['terminal_eos_excluded'] += action.terminal_eos_excluded
            counters['genuine_sampled_pad_tokens'] += actor.count(tokenizer.pad_token_id)
            counters['repeated_probe_state_changed'] += not (
                record['repeated_probability_probe_rng_unchanged'] and record['repeated_probability_probe_complete_state_unchanged']
                and record['repeated_probe_identity_before'] == record['repeated_probe_identity_after'])
            count = len(actor)
            old = full['old_log_probs'][local, :count].double()
            repeated = raw['actor_repeated_log_probs'][local, :count].double()
            sampled = raw['sampled_log_probs'][local, :count].double()
            require(old.shape == repeated.shape == sampled.shape and bool(torch.isfinite(old).all())
                    and bool(torch.isfinite(repeated).all()) and bool(torch.isfinite(sampled).all()),
                    'Effective raw action probabilities are nonfinite or incomplete')
            counters['hf_response_token_mismatch'] += full['responses'][local, :count].tolist() != actor
            expect_prefix_positions = list(range(len(prefix)))
            expect_action_positions = list(range(len(prefix), len(prefix) + count))
            same_positions = (full['position_ids'][local, :prompt_width][prompt_mask].tolist() == expect_prefix_positions
                              and full['position_ids'][local, prompt_width:prompt_width + count].tolist() == expect_action_positions)
            same_mask = full['attention_mask'][local, prompt_width:prompt_width + count].tolist() == [1] * count
            counters['hf_position_or_mask_mismatch'] += not (same_positions and same_mask)
            repeat_delta, sample_delta = old - repeated, old - sampled
            max_repeat = float(repeat_delta.abs().max()) if count else 0.
            max_sample = float(sample_delta.abs().max()) if count else 0.
            joint_repeat, joint_sample = float(repeat_delta.sum()), float(sample_delta.sum())
            global_token['hf_repeat_max_abs'] = max(global_token['hf_repeat_max_abs'], max_repeat)
            global_token['hf_vllm_max_abs'] = max(global_token['hf_vllm_max_abs'], max_sample)
            global_joint['hf_repeat_max_abs'] = max(global_joint['hf_repeat_max_abs'], abs(joint_repeat))
            global_joint['hf_vllm_max_abs'] = max(global_joint['hf_vllm_max_abs'], abs(joint_sample))
            joints.append(joint_sample); active_tokens += count
            cases.append(dict(source=key[0], step_id=key[1], serial=serial, rank=rank, local_row=local,
                              sampled_length=lengths[local], raw_actor_length=count, canonical_length=len(old_canonical),
                              token_path_changed=actor != old_canonical, selected_closing=action.selected_closing,
                              closing_inside_token=action.closing_inside_token, sampled_ids_sha256=action.sampled_ids_sha256,
                              hf_repeat_token_max=max_repeat, hf_vllm_token_max=max_sample,
                              hf_repeat_joint_delta=joint_repeat, hf_vllm_joint_delta=joint_sample))
        coverage.append(dict(serial=serial, active_rows=len(rows), padded_rank_rows=counts, effective_actor_tokens=active_tokens))
    require(counters['actions'] == 6145 and sources == set(range(2560)), 'Actual complete 6145/2560 coverage changed')
    selected = sorted(cases, key=lambda row: (-row['hf_vllm_token_max'], row['serial'], row['rank'], row['local_row']))[:19]
    bad_length_case = next((row for row in cases if row['source'] == 664 and row['step_id'] == 2), None)
    if bad_length_case is not None and bad_length_case not in selected:
        selected.append(bad_length_case)
    return dict(status='DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE', counters=counters, turn_coverage=coverage,
                unique_source_steps=len(seen), covered_sources=len(sources), token_residuals=global_token,
                complete_action_joint_residuals=global_joint,
                hf_vllm_signed_joint_sum_all_actions=float(sum(joints)), bounded_scalar_cases=selected,
                support_errors=errors, selected_original_length_case=bad_length_case,
                probability_scope='ORIGINAL_PER_TURN_HF_VS_ORIGINAL_ENGINE_ON_CANDIDATE_EFFECTIVE_RAW_PREFIX',
                new_full_action_prefix_scorer_executed=False, old_full_canonical_score_not_reused=True,
                sampler_runner_positions_observed=False,
                engine_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION',
                thresholds_or_tolerances_selected=False, probability_accepted=False)


def execute(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and socket.gethostname().split('.')[0] == 'h100-8',
            'CPU-only H100-local entry required')
    identity_path, output = Path(args.identity), Path(args.output)
    require(identity_path.parent == ROOT / 'logs' and output.parent == ROOT / 'logs'
            and output.name == OUTPUT_NAME and not output.exists(), 'Identity/output path is not isolated')
    require(sha256(identity_path) == args.identity_sha256, 'Wrong support diagnostic identity')
    spec = json.loads(identity_path.read_text())
    require(spec['scope'] == SCOPE and spec['source_commit'] == SOURCE and spec['binding_sha256'] == BINDING
            and spec['actual_run'] == str(RUN) and spec['job_id'] == 3289, 'Wrong support scope/source/run')
    here = ROOT / 'tools/diagnostic-raw-support-3289-v1'
    helper = Path(__file__).resolve()
    require(helper == here / 'diagnose_raw_support_v1.py' and sha256(helper) == spec['helper']['sha256'], 'Wrong support helper')
    old_path = ROOT / 'tools/diagnostic-action-3289-v2/diagnose_actions_v1.py'
    require(sha256(old_path) == spec['parent_helper']['sha256'], 'Old immutable CPU parser/inventory module changed')
    old = load_module(old_path, 'diagnostic_action_parent_3289')
    parent_path, inventory_path = Path(spec['parent_identity']['path']), Path(spec['inventory']['path'])
    require(sha256(parent_path) == PARENT_IDENTITY and sha256(inventory_path) == INVENTORY, 'Wrong actual old input/failure')
    parent, failure = json.loads(parent_path.read_text()), json.loads(inventory_path.read_text())
    old.validate_spec(parent, failure)
    binding_path = ROOT / 'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json'
    require(sha256(binding_path) == BINDING, 'Actual source52 calibration binding changed')
    binding = json.loads(binding_path.read_text())
    llm_record = spec['frozen_llm_output_source']
    require(next(row for row in binding['files'] if row['path'] == llm_record['path'])
            == {key: llm_record[key] for key in ('path', 'bytes', 'sha256')}, 'Sentinel adapter does not derive from actual source binding')
    llm_source = Path(llm_record['path']).read_text()
    selector_record = spec['frozen_vllm_selector_source']
    require(llm_record['path'] == str(ROOT / 'repo-policy-safety-52befe9/verl/third_party/vllm/vllm_v_0_6_3/llm.py'),
            'Wrong actual 0.6.3 output provider')
    require(next(row for row in binding['files'] if row['path'] == selector_record['path'])
            == {key: selector_record[key] for key in ('path', 'bytes', 'sha256')}, 'Wrong selector source binding')
    require('from .vllm_v_0_6_3.llm import LLM' in Path(selector_record['path']).read_text(),
            'Actual selected 0.6.3 LLM provider differs')
    version_record = spec['bound_installed_vllm_metadata']
    require(next(row for row in binding['files'] if row['path'] == version_record['path'])
            == {key: version_record[key] for key in ('path', 'bytes', 'sha256')}, 'Wrong installed vLLM metadata binding')
    require('Name: vllm\n' in Path(version_record['path']).read_text()
            and 'Version: 0.6.3\n' in Path(version_record['path']).read_text(), 'Bound installed vLLM version differs')
    require('logprob.append(logprobs_dict[id].logprob)' in llm_source
            and 'logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)' in llm_source,
            'Bound original chosen-logprob padding adapter differs')
    candidate_path = here / 'action_identity.py'
    require(sha256(candidate_path) == spec['candidate_module']['sha256'], 'Candidate action identity bytes changed')
    inputs = (parent['raw_material_inventory'] + parent['official_model_metadata'] + parent['source_files']
              + parent['control_originals'] + [spec['helper'], spec['parent_helper'], spec['parent_identity'], spec['inventory'],
                                             spec['candidate_module'], llm_record, selector_record, version_record,
                  dict(path=str(binding_path), bytes=binding_path.stat().st_size, sha256=BINDING),
                  dict(path=str(identity_path), bytes=identity_path.stat().st_size, sha256=args.identity_sha256)])
    by_path = {}
    for row in inputs:
        if row['path'] in by_path:
            require(all(by_path[row['path']][key] == row[key] for key in ('path', 'bytes', 'sha256')), 'Conflicting input identity')
        else:
            by_path[row['path']] = row
    before = [old.checked_file(row) for row in by_path.values()]
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    require(torch.__version__.startswith('2.4.') and not torch.cuda.is_initialized(), 'Wrong CPU Torch/version')
    torch.set_num_threads(4)
    candidate = load_module(candidate_path, 'raw_action_identity_candidate_3289')
    parser_path = ROOT / 'repo-policy-safety-52befe9/search_r1/llm_agent/generation.py'
    require(any(row['path'] == str(parser_path) for row in parent['source_files']), 'Frozen parser is not in the inherited source closure')
    processor, parser_ast = old.extract_frozen_postprocess(parser_path, torch)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'), local_files_only=True, trust_remote_code=False)
    require(tokenizer.pad_token_id == 151643 and tokenizer.eos_token_id == 151645, 'Official tokenizer PAD/EOS changed')
    processor.tokenizer = tokenizer
    try:
        result = analyze(spec, parent, torch, tokenizer, processor, candidate)
    except Exception as exc:
        result = dict(status='DIAGNOSTIC_FAILED_NOT_ACCEPTANCE', error_class=type(exc).__name__, error=str(exc)[:2000],
                      thresholds_or_tolerances_selected=False, probability_accepted=False)
    after = [old.checked_file(row) for row in by_path.values()]
    require(before == after and not torch.cuda.is_initialized(), 'Inputs changed or CUDA initialized')
    result.update(scope=SCOPE, job_id=3289, source_commit=SOURCE, binding_sha256=BINDING, actual_run=str(RUN),
                  candidate_module_sha256=spec['candidate_module']['sha256'], originals_before=before,
                  originals_after=after, all_originals_unchanged=True, exact_frozen_postprocess_ast_sha256=parser_ast,
                  cpu_torch_version=torch.__version__, model_loaded=False, forward_calls=0, restore_calls=0,
                  optimizer_writes=0, retained_updates_contributed=0, cuda_initialized_before=False, cuda_initialized_after=False)
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    require(len(payload) <= MAX_OUTPUT_BYTES, 'Bounded scalar aggregate overflow')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(output), bytes=len(payload), sha256=sha256(output), scope=SCOPE)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True)
    parser.add_argument('--identity-sha256', required=True)
    parser.add_argument('--output', required=True)
    execute(parser.parse_args())
