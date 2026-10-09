"""Final explicit 100-outer/4608-slot coverage. Stdlib; never models or signals."""
import argparse
import importlib
import math
import os
from pathlib import Path
import sys
try:
    from .common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, canonical, checked_file,
                         read, rows, same, exclusive, bind_helper, execution_packet)
except ImportError:
    from common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, canonical, checked_file,
                        read, rows, same, exclusive, bind_helper, execution_packet)


def wait_release(exit_proof, count, config, phase):
    require(exit_proof.get('accepted') is True and len(exit_proof['child_exit_evidence']) == count,
            'Actual child wait inventory incomplete')
    identities = exit_proof['child_exit_evidence']
    require(all(r['actual_wait_returncode'] == 0 and r['identity']['uid'] == config['owner']['uid']
                and type(r['identity']['starttick']) is int and r['identity']['run_id'] == config['run_id'] for r in identities)
            and len({(r['identity']['pid'], r['identity']['starttick']) for r in identities}) == count,
            'Actual child waits are unknown/nonzero or reused')
    terminal, _ = read(exit_proof['actual_terminal'])
    require(terminal['child_exit_evidence'] == identities and terminal['run_id'] == config['run_id']
            and terminal['config_sha256'] == config['_config_sha256'] and terminal['phase'] == phase
            and exit_proof['gpu_release']['all_owned_release']['accepted'] is True,
            'Actual terminal wait/release differs')


def bridge(config_ref, sampling_ref, output):
    """Independent analysis input adapter; never overwrite the native receipt."""
    config, cfg = read(config_ref); sampling, sp = read(sampling_ref); body = sampling['train_em']
    terminal, _ = read(sampling['exit_proof']['actual_terminal'])
    require(body.get('accepted') is True and body['variant'] == config['variant']
            and sampling['run_id'] == config['run_id'] and terminal['run_id'] == config['run_id']
            and terminal['config_sha256'] == cfg['sha256'] and terminal['phase'] == 'sampling-running',
            'Native sampling/body/config identity differs')
    require('config' not in body and 'run_id' not in body and Path(output).resolve() != Path(sp['path']).resolve(),
            'Bridge must be a separate copy of the original pure calculate body')
    return exclusive(output, dict(body, config=cfg, run_id=config['run_id'], original_sampling_complete=sp,
                                  input_format_bridge_only=True))


def ledger_identity(ledger, config, cfg):
    """The native final body has no top-level config SHA; its25 rows do."""
    require(ledger['accepted'] is True and ledger['run_id'] == config['run_id']
            and ledger['variant'] == config['variant'] and ledger['outer_iterations'] == 25
            and len(ledger['rows']) == 25
            and [row['outer_index'] for row in ledger['rows']] == list(range(25))
            and all(row['run_id'] == config['run_id'] and row['config_sha256'] == cfg['sha256']
                    and row['variant'] == config['variant'] for row in ledger['rows']),
            'Final native25 TRAIN ledger/row identity differs')


def verify_token_inventory(checkpoint, result):
    """Fresh SHA every record used by a previously accepted token audit."""
    files = []; actual_pairs = tokens = 0
    for ref in checkpoint['outer_update_receipts']:
        update, up = read(ref); batch, bp = read(update['score_batch'])
        count = 0
        for rank in update['all_rank_actual_token_metrics']:
            for metric in rank['pairs']:
                files.extend([checked_file(metric['signal_score']), checked_file(metric['token_log'])])
                count += 1; tokens += metric['valid_tokens']
        require(count == update['global_actual_pairs'], 'Committed actual pair denominator differs')
        actual_pairs += count; files.extend([up, bp])
    require(result['referenced_record_inventory_sha256'] == canonical(files)
            and result['referenced_record_file_count'] == len(files)
            and result['actual_pair_exposures'] == actual_pairs and result['valid_token_rows'] == tokens,
            'Accepted token output record inventory/count has changed')
    return tokens


def token_audit(entry, config, cfg, checkpoint, cp, previous_updates):
    K = checkpoint['cursor']['usable_pair_count']
    if K == 0:
        require(checkpoint['cursor']['next_pair_cursor'] == 0 and not checkpoint['outer_update_receipts']
                and checkpoint['optimizer_updates'] == previous_updates, 'Zero-K invented updates/token loss')
        require(entry.get('token_audit') is None, 'Zero-K must be explicit NA, not an unsupported audit failure')
        return dict(accepted=True, state='K0_METADATA_NA', valid_token_rows=0, actual_pair_exposures=0,
                    scalar_loss=None, token_math_not_applicable=True)
    packet = entry['token_audit']
    if packet.get('mode') == 'recompute':
        require(packet['source_manifest']['sha256'] == packet['published_manifest_sha256'], 'Token audit source pin differs')
        publication, _ = read(packet['publication'])
        require(publication.get('accepted') is True
                and publication.get('source_commit') == publication.get('remote_readback_sha')
                and publication.get('source_manifest_sha256', publication.get('helper_source_manifest_sha256')) == packet['source_manifest']['sha256'],
                'Recomputed token helper source not published')
        helper = bind_helper(packet['helper_root'], packet['source_manifest'], ('audit',))['audit']
        result = helper.audit(cfg, cp, checkpoint['optimizer_updates'], checkpoint['cursor']['outer_index'], ORIGINAL_SHA)
        evidence = dict(mode='recomputed_published_token_audit', source_manifest=checked_file(packet['source_manifest']),
                        publication=checked_file(packet['publication']), compact_output_semantic_sha256=canonical(result))
    else:
        result, evidence = execution_packet(packet)
    require(result['schema'] == 'opsd-signal-committed-token-audit-v1' and result['variant'] == config['variant']
            and result['run_id'] == config['run_id'] and same(result['config'], cfg) and same(result['checkpoint'], cp)
            and result['optimizer_updates'] == checkpoint['optimizer_updates']
            and result['outer_index'] == checkpoint['cursor']['outer_index']
            and result['committed_updates_read'] == math.ceil(K / 32), 'Accepted token audit bound to another checkpoint')
    require(same(result['original_source']['manifest'], config['helper_source_manifest'])
            and result['original_source']['complete_byte_sha_readback'] is True,
            'Token math did not reuse the exact original source')
    tokens = verify_token_inventory(checkpoint, result)
    return dict(accepted=True, state='COMMITTED_TOKEN_RECONSTRUCTION', valid_token_rows=tokens,
                actual_pair_exposures=result['actual_pair_exposures'], evidence=evidence,
                full_vocab_arithmetic_not_verified_here=True)


def outer(entry, config, cfg, native, core, previous, updates, exposure, cursor):
    index = entry['outer_index']; manifest, mp = read(entry['outer_manifest']); sample, sp = read(entry['sampling_complete'])
    require(manifest['outer_index'] == index and manifest['run_id'] == config['run_id']
            and manifest['config_sha256'] == cfg['sha256'] and manifest['runtime_identity'] == config['runtime_manifest']['sha256']
            and manifest['source_model_identity'] == config['source_model_manifest']['sha256']
            and manifest['contract'] == native['checkpoint'].science_contract(config)
            and manifest['common_initial_adapter_manifest'] == config['common_initial_adapter_manifest']
            and manifest['common_cohort_manifest'] == config['common_cohort_manifest']
            and manifest['previous_complete_checkpoint'] == previous
            and manifest['outer_old_adapter'] == (previous or config['common_initial_adapter_manifest'])
            and manifest['cumulative_optimizer_updates_before'] == updates,
            'Cross-outer source/old/checkpoint/Adam chain differs')
    expected_questions, selection = native['sampling'].cohort(config, cursor)
    questions, _ = rows(manifest['questions']); sampled, _ = rows(manifest['rollouts']); pairs, _ = rows(manifest['pairs'])
    require(questions == expected_questions and manifest['train_cursor'] == selection,
            'TRAIN cursor/order differs from common cohort stream')
    recomputed_pairs, stats = native['sampling'].verify_rollouts(sampled, questions, core, config['variant'], index)
    require(pairs == recomputed_pairs and manifest['P'] == len(pairs) and manifest['sampling_stats'] == stats,
            'Original mined B/D/full512x5 strict scoring changed')
    require(sample['schema'] == 'opsd-online-sampling-complete-v1' and sample['accepted'] is True
            and sample['run_id'] == config['run_id'] and sample['train_em']['outer_index'] == index
            and same(sample['rollouts'], manifest['rollouts'])
            and same(sample['pairs'], manifest['pairs']) and sample['sampling_model_identity'] == manifest['sampling_model_identity'],
            'Sampling complete does not bind immutable outer')
    wait_release(sample['exit_proof'], 8, config, 'sampling-running')
    identity = manifest['sampling_model_identity']
    prior_cp = read(previous)[0] if previous else None
    require(identity['checkpoint'] == previous and identity['optimizer_updates'] == updates
            and identity['source_model_identity'] == config['source_model_manifest']['sha256']
            and manifest['explicit_fresh'] is (previous is None)
            and identity['adapter_tensor_sha256'] == (prior_cp['all_rank_adapter_sha256'][0] if previous else config['common_initial_adapter_tensor_sha256']),
            'Sampling did not use the original outer-start raw adapter')
    train = native['training_metrics'].calculate(sampled, questions, config['variant'], index, identity, core)
    require(train == sample['train_em'], 'Recorded preupdate2560 TRAIN EM differs from native strict recalculation')
    complete, _ = native['checkpoint'].check_outer_complete(entry['outer_complete'], config, config['run_id'], index, updates, len(pairs))
    require(same(complete['checkpoint'], entry['end_checkpoint']), 'Outer completion uses another end checkpoint')
    cp, _ = native['checkpoint'].check_checkpoint(entry['end_checkpoint'], config, config['run_id'], complete['optimizer_updates'], index)
    require(same(cp['training_input']['outer_manifest'], mp) and cp['outer_old_adapter'] == manifest['outer_old_adapter']
            and cp['sampling_model_identity'] == identity and complete['cumulative_pair_exposure'] == cp['cumulative_pair_exposure'] == exposure + complete['K'],
            'Outer checkpoint input/cumulative exposure changed')
    wait_release(read(entry['trainer_exit'])[0], 1, config, 'training-running')
    expected_summary = native['training_metrics'].complete_outer_summary(train, complete)
    expected_summary.update(run_id=config['run_id'], config_sha256=cfg['sha256'], raw_checkpoint=complete['checkpoint'], sampling_model_sha256=identity['sha256'])
    summary, _ = read(entry['summary']); require(summary == expected_summary, 'Final outer summary does not reconstruct all committed batches')
    if complete['K'] == 0:
        digest = prior_cp['all_rank_adapter_sha256'][0] if prior_cp else config['common_initial_adapter_tensor_sha256']
        require(cp['all_rank_adapter_sha256'] == [digest] * 4 and complete['optimizer_updates_delta'] == 0
                and summary['global_valid_tokens'] == 0 and summary['actual_outer_weighted_post_loss'] is None
                and summary['weight']['mean'] is None, 'Zero-K fabricated adapter change/exposure/loss')
    token = token_audit(entry, config, cfg, cp, entry['end_checkpoint'], updates)
    require(token['actual_pair_exposures'] == complete['K'] and token['valid_token_rows'] == summary['global_valid_tokens'],
            'Outer token audit/summary exposure differs')
    return dict(outer_index=index, P=complete['P'], K=complete['K'], optimizer_updates=complete['optimizer_updates'],
                delta=complete['optimizer_updates_delta'], cumulative_pair_exposure=cp['cumulative_pair_exposure'],
                train_correct=train['overall']['strict_correct'], train_denominator=2560, token=token,
                checkpoint=entry['end_checkpoint'], completion=entry['outer_complete'],
                sampling_complete=entry['sampling_complete']), summary, selection['after']


def evaluation(parent_ref, child_ref, config, cfg, completed, checkpoint, native, validation):
    parent, pp = read(parent_ref); child, cp = read(child_ref)
    require(parent['schema'] == 'opsd-signal-evaluation-parent-v1' and parent['accepted'] is True
            and parent['run_id'] == config['run_id'] and parent['config_sha256'] == cfg['sha256']
            and parent['completed_outer'] == completed and parent['raw_checkpoint'] == checkpoint
            and same(parent['child'], child_ref) and parent['isolated_child_no_training_rng_consumed'] is True
            and parent['raw_state_unchanged'] is True and parent['parent_torch_imported'] is False,
            'Formal endpoint parent/raw state identity differs')
    wait_release(parent['exit_proof'], 1, config, 'evaluation-running')
    require(child['schema'] == 'opsd-signal-native512-complete-v1' and child['accepted'] is True
            and child['run_id'] == config['run_id'] and same(child['config'], cfg)
            and child['variant'] == config['variant'] and child['study_id'] == config['study_id']
            and child['completed_outer'] == completed and child['primary_slots'] == 512
            and child['actual_child_exit_and_release'] is True and child['no_training'] is True
            and child['checkpoint_metadata_unchanged'] is True, 'Formal endpoint child is partial or foreign')
    exits = child['child_exit_evidence']
    require(len(exits) == 8 and len({(r['identity']['pid'], r['identity']['starttick']) for r in exits}) == 8
            and all(r['actual_wait_returncode'] == 0 and r['identity']['run_id'] == config['run_id']
                    and r['identity']['uid'] == config['owner']['uid'] for r in exits)
            and child['gpu_release']['released'] is True,
            'Formal eight logical worker waits/release incomplete')
    identity = child['input_identity']; require(identity['protocol'] == native['evaluation'].PROTOCOL
            and identity['source_model_manifest']['sha256'] == config['source_model_manifest']['sha256']
            and identity['runtime']['manifest']['sha256'] == config['runtime_manifest']['sha256']
            and same(identity['validation'], config['validation_identity'])
            and identity['common_initial'] == config['common_initial_adapter_manifest'], 'Formal native512 conditions changed')
    if completed:
        pinned, _ = read(checkpoint)
        require(pinned['cursor']['outer_index'] == completed - 1
                and pinned['cursor']['next_pair_cursor'] == pinned['cursor']['usable_pair_count']
                and same(child['raw_checkpoint']['manifest'], checkpoint)
                and child['raw_checkpoint']['kind'] == 'raw_end_of_outer'
                and identity['expected_adapter_tensor_sha256'] == pinned['all_rank_adapter_sha256'][0],
                'Formal endpoint is not raw outer14/24 adapter')
    else:
        require(identity['expected_adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'], 'Common-zero adapter changed')
    records, _ = rows(child['results']); request, _ = read(child['request']); saved, _ = read(child['summary'])
    require(request['input_identity'] == identity and request['completed_outer'] == completed
            and child['raw_checkpoint'] == identity['raw_checkpoint']
            and request['version'] == child['version'] == ('source100' if completed == 0 else config['variant'])
            and request['evaluation_id'] == child['evaluation_id'] and same(request['config'], cfg),
            'Formal request/input identity differs')
    fresh = native['evaluation'].aggregate(records, validation, config, request, verify_traces=True)
    require(fresh == saved, 'Formal unique512/native strict traces/summary changed')
    require(len(child['worker_proofs']) == 16, 'Formal eight worker/result receipt pairs incomplete')
    ranks = set(); saved_slots = set()
    for n in range(0, 16, 2):
        value, _ = read(child['worker_proofs'][n]); rr, rp = rows(child['worker_proofs'][n + 1]); rank = value['rank']
        require(value['schema'] == 'opsd-signal-native512-worker-v1' and value['accepted'] is True
                and value['status'] == 'COMPLETE' and value['evaluation_id'] == child['evaluation_id']
                and value['run_id'] == config['run_id'] and same(value['config'], cfg)
                and same(value['request'], child['request']) and value['primary_rows'] == len(rr) == 64
                and rank in range(8) and rank not in ranks and same(value['results'], rp)
                and value['actual_loaded_adapter']['tensor_sha256'] == identity['expected_adapter_tensor_sha256'],
                'Actual complete logical worker/model/results differ')
        loaded = value['actual_loaded_forward']
        require(loaded['accepted'] is True and loaded['adapter_name'] == child['version']
                and loaded['layers'] == len(loaded['layer_checks']) == 252 and loaded['scaling'] == 2.
                and loaded['rank'] == 16 and loaded['alpha'] == 32 and loaded['dropout'] == 0.
                and loaded['A_B_dtype'] == 'torch.float32', 'Actual selected adapter forward differs')
        require(any(r['identity'] == value['process'] for r in exits), 'Worker proof does not match actually waited process')
        require([r['validation_index'] for r in rr] == list(range(rank, 512, 8)), 'Worker logical64 slice differs')
        require(all(records[r['validation_index']] == r for r in rr), 'Worker/result merge differs')
        ranks.add(rank); saved_slots.update(r['validation_index'] for r in rr)
    require(ranks == set(range(8)) and saved_slots == set(range(512)), 'Logical8 worker coverage incomplete')
    return dict(parent=pp, child=cp, completed_outer=completed, slots=512, hits=fresh['overall']['hits'])


def numeric_final(packets, configs, finals):
    require(set(packets) == set(VARIANTS), 'Four explicit final numeric outputs required')
    evidence = {}
    for variant in VARIANTS:
        value, proof = execution_packet(packets[variant], require_no_torch=False)
        cfg = configs[variant]; checkpoint, _ = read(finals[variant]); ranks = value['ranks']
        require(value['schema'] == 'opsd-ablation-independent-cpu-state-readback-v1'
                and value['variant'] == variant and value['run_id'] == cfg['run_id']
                and value['config']['sha256'] == cfg['_config_sha256']
                and same(value['config'], cfg['_original_config_ref'])
                and same(value['checkpoint']['manifest'], finals[variant])
                and value['outer_index'] == 24 and value['optimizer_updates'] == checkpoint['optimizer_updates']
                and value['all_four_exact_full_adam'] is True and len(ranks) == 4
                and {r['rank'] for r in ranks} == set(range(4))
                and all(r['actual_adam_step'] == checkpoint['optimizer_updates'] and r['full_rng_validated'] is True for r in ranks)
                and [r['optimizer_tensor_sha256'] for r in sorted(ranks, key=lambda x: x['rank'])]
                    == checkpoint['all_rank_optimizer_tensor_sha256']
                and value['adapter_tensors'] == 504 and value['adapter_finite_fp32'] is True
                and value['adapter_tensor_sha256'] == checkpoint['all_rank_adapter_sha256'][0]
                and value['cuda_initialized'] is False and same(value['common_initial'], cfg['common_initial_adapter_manifest']),
                'Native final actual Adam/ranks/finite/digest/config checkpoint readback differs')
        require((value['optimizer_updates'] == 0 and value['changed_tensors_vs_common_initial'] == 0)
                or (value['optimizer_updates'] > 0 and 0 < value['changed_tensors_vs_common_initial'] <= 504),
                'Native final changed adapter relation differs')
        evidence[variant] = proof
    return dict(accepted=True, state='ACTUAL_FOUR_FINAL_NUMERIC_READBACKS', evidence=evidence,
                scope='Parsed native audit_state output and published actual execution; tensors are not deserialized by this assembler')


def backups(packets, configs, expected):
    require(len(packets) == len(expected), 'All protected and complete outer-end backups required')
    seen = set(); evidence = []
    for packet in packets:
        value, proof = execution_packet(packet); marker, copied = read(packet['target_marker_copy'])
        require(value['remote_backup']['accepted'] is True and value['remote_backup']['state'] == 'COMPLETE_FRESH_SHA_READBACK'
                and value['remote_backup']['receipt']['sha256'] == copied['sha256']
                and marker['schema'] == 'opsd-signal-pinned-backup-complete-v1' and marker['accepted'] is True
                and marker['actual_tar_and_transfer_returncodes_zero'] is True and marker['large_outputs_remote_only'] is True,
                'Target marker copy does not match actually verified remote backup')
        inventory = marker['inventory']; variant = inventory['variant']; cpref = inventory['checkpoint']; key = (variant, cpref['path'], cpref['sha256'])
        require(key in expected and key not in seen and inventory['config']['sha256'] == configs[variant]['_config_sha256']
                and inventory['run_id'] == configs[variant]['run_id'] and inventory['study_id'] == configs[variant]['study_id']
                and value['inventory_sha256'] == marker['inventory_sha256'] == canonical(inventory)
                and value['remote_backup']['member_bytes'] == inventory['member_bytes']
                and value['remote_backup']['member_count'] == inventory['member_count']
                and value['inventory_actual_wait']['actual_ssh_wait_returncode'] == 0
                and value['target_probe_actual_wait']['actual_ssh_wait_returncode'] == 0
                and value['local_checkpoint_payload_files_created'] == 0,
                'Backup copied another input, duplicate checkpoint or unknown remote wait')
        cp, _ = read(cpref); refs = [cpref] + [r['file'] for r in cp['ranks']] + [cp['shared_optimizer']] + cp['adapter']['files']
        members = {str(Path(r['path']).relative_to(Path(cpref['path']).parent)): checked_file(r) for r in refs}
        require(set(members) == set(inventory['members']) == set(marker['member_readback'])
                and all(members[n]['sha256'] == inventory['members'][n]['sha256'] == marker['member_readback'][n]['sha256']
                        and members[n]['bytes'] == inventory['members'][n]['bytes'] == marker['member_readback'][n]['bytes'] for n in members),
                'Backup member byte inventory differs from preserved original checkpoint')
        if not value['already_complete']:
            require(value['actual_source_stream']['source_tar_actual_wait_returncode'] == 0
                    and value['actual_source_stream']['before_after_inventory_equal'] is True
                    and value['actual_transport']['source_ssh_actual_wait_returncode'] == 0
                    and value['actual_transport']['target_ssh_actual_wait_returncode'] == 0
                    and value['target_commit_actual_wait']['actual_ssh_wait_returncode'] == 0,
                    'Actual source tar/dual transport/target commit waits differ from0')
        seen.add(key); evidence.append(dict(execution=proof, target_marker_copy=copied))
    require(seen == expected, 'Some explicit protected/outer-end checkpoints are not backed up')
    return dict(accepted=True, state='PINNED_REMOTE_BYTE_BACKUPS', checkpoints=len(seen), evidence=evidence,
                scope='Original full member bytes and recorded remote fresh target verification; no standalone restoration claim')


def vocabulary_scope(spec, configs):
    ledger, proof = read(spec['ledger']); maximum = 24 * (1 << 30)
    roots = {c['logging']['full_vocab_study_root'] for c in configs.values()}
    require(len(roots) == 1 and Path(proof['path']).parent.resolve() == Path(next(iter(roots))).resolve()
            and type(ledger['bytes']) is int and 0 <= ledger['bytes'] <= maximum,
            'Actual shared vocabulary cap ledger changed')
    files = [checked_file(f) for f in ledger['files']]
    require(len({f['path'] for f in files}) == len(files) and sum(f['bytes'] for f in files) <= ledger['bytes']
            and all(any(Path(f['path']).is_relative_to(Path(c['paths']['output_root'])) for c in configs.values()) for f in files),
            'Vocabulary file inventory/cap byte accounting differs')
    # A cap can reserve space for an interrupted write; it must not be called a complete distribution dump.
    completed = sum(r['status'] == 'complete' for r in ledger.get('reservations', []))
    sampled = []
    for packet in spec.get('sampled_arithmetic', []):
        value, evidence = execution_packet(packet, require_no_torch=False)
        require(value.get('sampled_only') is True, 'Vocabulary arithmetic must state its actual sampled scope')
        sampled.append(evidence)
    return dict(accepted=True, state='SHARED_CAP_AND_SAVED_FILE_SCOPE', ledger=proof,
                cap_bytes=maximum, reserved_or_saved_bytes=ledger['bytes'], saved_npy_files=len(files), completed_positions=completed,
                sampled_arithmetic_evidence=sampled, all_positions_or_all_npy_arithmetic_required=False,
                scope='Fresh saved file SHA and study24GiB cap; token complete/cap_skip counts separately audited. No full vocabulary reconstruction from top32')


def source_publications(specs):
    require(specs, 'Explicit actual execution overlay/source publications required')
    proofs = []
    for spec in specs:
        publication, pp = read(spec['publication']); manifest, mp = read(spec['source_manifest']); root = Path(spec['helper_root']).resolve()
        require(publication['accepted'] is True and publication['source_commit'] == publication['remote_readback_sha']
                and publication.get('source_manifest_sha256', publication.get('helper_source_manifest_sha256')) == mp['sha256'],
                'Execution overlay source not published/read back')
        for name, item in manifest['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe published overlay source path')
            checked_file(dict(path=str(root / name), sha256=item['sha256'], bytes=item['bytes']))
        proofs.append(dict(publication=pp, source_manifest=mp))
    return dict(accepted=True, state='FRESH_PUBLISHED_SOURCE_BYTES', evidence=proofs,
                scope='Original training source also checked per group; source-only overlays independently bound to actual terminal/executions')


def analysis_artifact(packet, kind):
    """Bind the actual native CLI stdout to its report artifact, never relabel it."""
    require(packet.get('mode') == 'native_cli_artifact', 'Analysis packet requires explicit native CLI artifact mode')
    cli_packet = dict(packet, output=packet['actual_stdout'])
    cli, execution = execution_packet(cli_packet, require_accepted_output=False)
    if kind == 'train_relationship':
        complete, complete_ref = read(cli)
        require(complete['accepted'] is True and complete['complete_100_outers'] is True,
                'Native TRAIN CLI complete artifact is partial/unaccepted')
        report_ref = complete['outputs']['summary']; request_ref = complete['request']
        evidence = dict(execution=execution, native_complete=complete_ref)
    else:
        require(cli.get('accepted') is True, 'Native analysis CLI did not accept its output artifact')
        report_ref = cli['output']; evidence = dict(execution=execution)
        request_ref = None
    require(same(report_ref, packet['output']), 'Actual native CLI stdout names another report artifact')
    report, proof = read(report_ref); require(report.get('accepted') is True, 'Native analysis report not accepted')
    request_ref = request_ref or report['request']; request, rp = read(request_ref)
    evidence.update(report=proof, native_request=rp)
    return report, request, evidence


def external_analysis(spec, configs, finals, evals, outer_rows):
    require(set(spec) == {'effects', 'train_relationship', 'heldout_relationship'}, 'All predefined independent analysis reports required')
    proofs = {}
    endpoints = {(r['child']['path'], r['child']['sha256']) for r in evals}
    config_refs = {v: configs[v]['_original_config_ref'] for v in VARIANTS}
    outer_index = {(r['variant'], r['outer_index']): r for r in outer_rows}
    for name, packet in spec.items():
        value, request, proof = analysis_artifact(packet, name)
        binding, bp = read(packet['input_binding'])
        require(binding['schema'] == 'opsd-signal-analysis-input-binding-v1' and binding['input_format_bridge_only'] is True
                and same(binding['original_output'], packet['output']) and binding['study_id'] == configs['G_POS']['study_id']
                and binding['configs'] == config_refs
                and binding['final_checkpoints'] == finals
                and {r['sha256'] for r in binding['formal_evaluations']} == {r['child']['sha256'] for r in evals},
                'Independent analysis bridge is not bound to actual configs/endpoints/final checkpoints')
        expected = {'effects': ('opsd-signal-weight-final-effects-v1', 'formal_predictions', 4608),
                    'heldout_relationship': ('opsd-signal-heldout-response-relations-v1', 'formal_EM_rows', 4608),
                    'train_relationship': ('opsd-signal-train-analysis-summary-v1', 'completed_outers', 100)}[name]
        require(value['schema'] == expected[0] and value[expected[1]] == expected[2], 'Independent report has partial/foreign scope')
        if name == 'effects':
            require({(r['path'], r['sha256']) for r in request['evaluation_receipts']} == endpoints
                    and len(request['evaluation_receipts']) == 9
                    and {(r['path'], r['sha256']) for r in request['configs']} == {(r['path'], r['sha256']) for r in config_refs.values()},
                    'Actual effect CLI request does not bind the original nine endpoints/four configs')
            require(all(value['models'][(('source100' if r['completed_outer'] == 0 else read(r['child'])[0]['variant']) + '/' + str(r['completed_outer']))]
                        ['overall']['hits'] == r['hits'] for r in evals), 'Effect report differs from actual native512 hits')
        elif name == 'train_relationship':
            require(value['study_id'] == binding['study_id'] and value['complete_100_outers'] is True
                    and set(value['groups']) == set(VARIANTS) and request['configs'] == config_refs
                    and request.get('require_complete', True) is True
                    and len(request['outers']) == 100
                    and {(e['variant'], e['outer_index']) for e in request['outers']} == set(outer_index),
                    'TRAIN relationship report/request not the original all100 outers')
            for row in request['outers']:
                original = outer_index[row['variant'], row['outer_index']]
                train_bridge, _ = read(row['sampling_summary'])
                require(same(row['training_completion'], original['completion'])
                        and same(train_bridge['original_sampling_complete'], original['sampling_complete']),
                        'Actual TRAIN analysis input bridge/completion belongs to another outer')
        else:
            require(len(request['replay_receipts']) == 9, 'Heldout relationship request incomplete')
            actual = set()
            for ref in request['replay_receipts']:
                replay, _ = read(ref)
                require(replay['accepted'] is True and replay['actual_workers_exit_and_release'] is True,
                        'Heldout analysis request uses incomplete/unreleased replay')
                checked_file(replay['results'])
                actual.add((replay['endpoint']['path'], replay['endpoint']['sha256']))
            require(actual == endpoints, 'Actual heldout relationship CLI request does not use the original nine endpoints')
            require(all(value['models'][(('source100' if r['completed_outer'] == 0 else read(r['child'])[0]['variant']) + '/' + str(r['completed_outer']))]
                        ['all512']['hits'] == r['hits'] for r in evals), 'Heldout relationship report differs from actual native512 hits')
        proofs[name] = dict(execution=proof, input_binding=bp)
    return dict(accepted=True, state='INDEPENDENT_EFFECT_AND_RELATIONSHIP_REPORTS', evidence=proofs,
                scope='Checks exact inputs and report scope/hits; original statistical calculations not repeated here')


def final_requirements(external, configs, finals, expected_backups, evals, outer_rows):
    validators = {'final_numeric_adam': lambda x: numeric_final(x, configs, finals),
                  'protected_and_outer_end_backups': lambda x: backups(x, configs, expected_backups),
                  'published_source': source_publications,
                  'full_vocab_scope': lambda x: vocabulary_scope(x, configs),
                  'effect_and_relationship_analysis': lambda x: external_analysis(x, configs, finals, evals, outer_rows)}
    result = {}
    for name, validator in validators.items():
        result[name] = validator(external[name]) if name in external else dict(accepted=False, state='MISSING_EXTERNAL_EVIDENCE')
    return result


def assemble(request):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Coverage assembly must remain CPU stdlib with empty CUDA')
    require(request['schema'] == 'opsd-signal-final-coverage-request-v1', 'Wrong coverage request')
    study, study_ref = read(request['study_manifest']); terminal, terminal_ref = read(request['study_terminal'])
    require([g['variant'] for g in study['groups']] == list(VARIANTS)
            and [g['variant'] for g in request['groups']] == list(VARIANTS), 'Exactly four ordered distinct arms required')
    require(terminal['schema'] == 'opsd-signal-weight-driver-terminal-v1' and terminal['accepted'] is True
            and same(terminal['study_manifest'], study_ref) and terminal['actual_supervisor_wait_returncode'] == 0
            and terminal['driver_exit_evidence']['source'] == 'same-retained-tmux-pane'
            and terminal['driver_exit_evidence']['pane_dead'] == 1 and terminal['driver_exit_evidence']['pane_dead_status'] == 0
            and terminal['driver_exit_evidence']['no_exit_code_inferred_from_absence'] is True
            and terminal['owned_gpu_release']['accepted'] is True
            and [g['variant'] for g in terminal['groups']] == list(VARIANTS),
            'Study actual supervisor terminal/owned release not independently accepted')
    native_complete, nref = read(request['native_study_completion'])
    require(native_complete['accepted'] is True and native_complete['completed_groups'] == 4
            and native_complete['completed_outers'] == native_complete['full_training_em_rows'] == 100
            and same(native_complete['study_manifest'], study_ref)
            and same(terminal['native_study_completion'], nref)
            and [g['variant'] for g in native_complete['groups']] == list(VARIANTS),
            'Native study terminal has incomplete group/outer coverage')
    configs = {}; all_rows = []; finals = {}; evals = []; seen_updates = set(); expected_backups = set()
    for entry, group in zip(request['groups'], study['groups']):
        require(entry['variant'] == group['variant'] and same(entry['config'], group['config']), 'Explicit group config differs from study')
        config, cfg = read(entry['config']); config['_config_sha256'] = cfg['sha256']; config['_original_config_ref'] = entry['config']
        variant = entry['variant']; configs[variant] = config
        require(config['schema'] == 'opsd-signal-weight-config-v1' and config['variant'] == variant
                and config['study_id'] == study['study_id'] and config['helper_source_manifest']['sha256'] == ORIGINAL_SHA
                and config['scientific']['outer_iterations'] == 25, 'Original four-arm25 contract changed')
        if variant != 'G_POS':
            baseline = configs['G_POS']
            require(all(config[k] == baseline[k] for k in ('source_model_manifest', 'runtime_manifest',
                    'common_initial_adapter_manifest', 'common_cohort_manifest', 'validation_identity')),
                    'Four arms no longer share model/runtime/initial/cohort/validation identity')
        native = bind_helper(config['paths']['helper_root'], config['helper_source_manifest'],
                             ('checkpoint', 'sampling', 'training_metrics', 'scoring', 'evaluation'))
        native['scoring'].verify_scoring(config); core = importlib.import_module('search_e1.core')
        gates = native['evaluation'].gates
        gates.publication_admission(config, gates.scoped(config, gates.bound(config['gates']['publication'])))
        validation, _ = rows(config['validation_identity'])
        require([o['outer_index'] for o in entry['outers']] == list(range(25)), 'Explicit outer chain is missing/duplicated/out of order')
        ledger, lp = read(entry['train_ledger']); ledger_identity(ledger, config, cfg)
        previous = None; updates = exposure = 0; cursor = dict(cycle=0, offset=0); summaries = []; by_index = {}
        for item in entry['outers']:
            row, summary, cursor = outer(item, config, cfg, native, core, previous, updates, exposure, cursor)
            for update in range(updates + 1, row['optimizer_updates'] + 1):
                require((variant, update) not in seen_updates, 'Duplicated true Adam update'); seen_updates.add((variant, update))
            previous = row['checkpoint']; updates = row['optimizer_updates']; exposure = row['cumulative_pair_exposure']
            expected_backups.add((variant, previous['path'], previous['sha256']))
            by_index[item['outer_index']] = previous; all_rows.append(dict(row, variant=variant)); summaries.append(summary)
        require(ledger['rows'] == summaries, 'Final TRAIN ledger differs from100 explicit reconstructed outer rows')
        native_group = next((g for g in native_complete['groups'] if g['variant'] == variant), None)
        captured_group = next(g for g in terminal['groups'] if g['variant'] == variant)
        require(native_group is not None and same(native_group['summary'], lp) and native_group['actual_wait_returncode'] == 0
                and native_group['release']['accepted'] is True and same(captured_group['config'], cfg)
                and same(captured_group['train_ledger'], lp) and same(captured_group['actual_terminal'], native_group['actual_terminal'])
                and captured_group['actual_controller_wait_returncode'] == 0,
                'Group actual controller wait/release missing or capture/native chain changed')
        group_exit, _ = read(native_group['actual_terminal'])
        require(group_exit['actual_wait_returncode'] == 0 and group_exit['controller']['run_id'] == config['run_id'], 'Native group terminal wait differs')
        finals[variant] = previous
        require([p['optimizer_updates'] for p in entry['protected_checkpoints']] == [1, 5], 'Actual protected u1/u5 identities required')
        for pinned in entry['protected_checkpoints']:
            native['checkpoint'].check_checkpoint(pinned['checkpoint'], config, config['run_id'], pinned['optimizer_updates'], pinned['outer_index'])
            expected_backups.add((variant, pinned['checkpoint']['path'], pinned['checkpoint']['sha256']))
        require([e['completed_outer'] for e in entry['evaluations']] == [15, 25], 'Both predefined formal15/25 endpoints required')
        for endpoint in entry['evaluations']:
            t = endpoint['completed_outer']; evals.append(evaluation(endpoint['parent'], endpoint['child'], config, cfg, t, by_index[t - 1], native, validation))
        if variant == 'G_POS':
            zero, _ = read(request['common_zero']); require(zero['schema'] == 'opsd-signal-common-zero-reference-v1'
                    and zero['accepted'] is True and zero['study_id'] == study['study_id'], 'Shared common-zero wrapper invalid')
            native['evaluation'].verify_common_zero(config, request['common_zero'])
            evals.append(evaluation(zero['parent'], zero['child'], config, cfg, 0, None, native, validation))
    requirements = final_requirements(request.get('external_requirements', {}), configs, finals, expected_backups, evals, all_rows)
    final = all(v['accepted'] for v in requirements.values())
    require('torch' not in sys.modules, 'Final coverage assembly imported Torch')
    return dict(schema='opsd-signal-final-coverage-v1', accepted=True, coverage_accepted=True,
                final_accepted=final, state='FINAL_ACCEPTED' if final else 'COVERAGE_ACCEPTED_HOLD_EXTERNAL_EVIDENCE',
                study_id=study['study_id'], study_manifest=study_ref, study_terminal=terminal_ref,
                native_study_completion=nref, complete_groups=4, complete_outers=len(all_rows),
                train_trajectory_slots=len(all_rows) * 2560, formal_native512_slots=sum(e['slots'] for e in evals),
                true_adam_updates=len(seen_updates), total_pair_exposures=sum(r['K'] for r in all_rows),
                valid_training_token_rows=sum(r['token']['valid_token_rows'] for r in all_rows),
                final_checkpoints=finals, outer_rows=all_rows, evaluations=evals, final_requirements=requirements,
                torch_imported=False, no_new_training_generation_or_signals=True,
                scope='Fresh explicit coverage, native strict TRAIN/heldout EM and committed token proof; external numeric/backup/fullvocabulary/effects remain separately scoped')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    target = Path(args.output).resolve()
    require(os.getuid() == 1001 and target.is_relative_to(Path(STUDY_ROOT) / 'analysis'), 'Production output must remain authorized HKUST4 analysis storage')
    try:
        request, proof = read(dict(path=args.request, sha256=args.request_sha256)); value = assemble(request); value['request'] = proof
    except Exception as error:
        value = dict(schema='opsd-signal-final-coverage-v1', accepted=False, final_accepted=False,
                     error_type=type(error).__name__, error=str(error), torch_imported='torch' in sys.modules)
    out = exclusive(target, value)
    print(__import__('json').dumps(dict(accepted=value['accepted'], final_accepted=value['final_accepted'], output=out), sort_keys=True))
    return 0 if value['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
