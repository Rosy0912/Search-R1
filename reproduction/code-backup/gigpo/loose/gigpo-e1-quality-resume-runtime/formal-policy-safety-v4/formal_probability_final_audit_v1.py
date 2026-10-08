"""Reconstruct completed outer identity from original PT bytes; never runs a forward."""
import json
import os
from pathlib import Path

from formal_probability_v1 import require, collective_gate, validate_live_arms, validate_outer
from formal_probability_proofs_v1 import file_record, verify_record, read_json_record, validate_manifest_closure


def audit_probability_outers(run, start, end, source, config_sha, workers, binding):
    # Torch/DataProto are the already bound actual environment. Every file is
    # SHA/bytes checked before deserialization; no CUDA/model/RNG operation.
    import torch
    from verl import DataProto
    from calibration_state_v1 import state_digest
    from formal_probability_worker_v1 import scored_batch_state
    from formal_probability_driver_v1 import reconstruct_actions, reconstruct_reference_records
    run = Path(run)
    require(start == 26 and end == 300 and binding['code_commit'] == source,
            'Wrong selected25 production audit range')
    root = run / 'probability-outer'
    require({p.name for p in root.iterdir() if p.is_dir()} ==
            {f'iter-{step:06d}' for step in range(start, end+1)},
            'Missing/extra actual outer probability evidence')
    accepted = binding['probability_identity']['acceptance']
    summaries = []
    for outer in range(start, end+1):
        directory = root / f'iter-{outer:06d}'
        require(not (directory/'before-adam-rejection.json').exists(), 'Rejected outer cannot be completed')
        path = directory/'before-adam-manifest.json'; manifest_receipt = file_record(path)
        manifest = read_json_record(manifest_receipt, directory)
        require(manifest['outer_step'] == outer and manifest['source_commit'] == source
                and manifest['job_id'] == os.environ['SLURM_JOB_ID'] and manifest['run_id'] == run.name
                and manifest['config_sha256'] == config_sha
                and manifest['production_artifact_sha256'] == binding['production_artifact']['sha256']
                and manifest['probability_acceptance_sha256'] == accepted['sha256']
                and manifest['actual_driver_tokenizer_identity'] == dict(
                    pad_token_id=binding['probability_identity']['pad_token_id'],
                    eos_token_id=binding['probability_identity']['eos_token_id']),
                'Actual outer identity/production/probability acceptance changed')
        validate_manifest_closure(manifest, directory)
        fixed = torch.load(manifest['scored_global_batch']['path'], map_location='cpu', weights_only=False)
        require(state_digest(fixed) == manifest['global_batch_digest'], 'Actual scored full batch changed')
        batch = DataProto.from_dict(tensors=fixed['tensors'], non_tensors=fixed['non_tensors'],
                                    meta_info=fixed['metadata'])
        turns = read_json_record(manifest['observed_turns'], directory)
        reconstructed, raw_files, raw_receipts = reconstruct_actions(batch, turns, directory, binding, source, outer)
        physical_actions = read_json_record(manifest['actions'], directory)
        require(physical_actions['outer_step'] == outer and physical_actions['source_commit'] == source
                and state_digest(physical_actions['records']) == state_digest(reconstructed)
                and raw_files == manifest['raw_files'] and raw_receipts == manifest['raw_receipts'],
                'Actual engine/HF/processed/full-batch reconstruction differs')
        checks = validate_outer(reconstructed, binding['probability_identity']['tolerances'],
                                binding['probability_identity']['pad_token_id'])
        require(checks == manifest['checks'], 'Actual complete action token/joint probability checks differ')
        refs = reconstruct_reference_records(batch, directory, source, outer)
        require(refs == manifest['reference_records'], 'Actual full-prefix reference material differs')
        require(manifest['chunks'] == [dict(rank=rank, rows=len(chunk), batch_digest=state_digest(scored_batch_state(chunk)))
                for rank, chunk in enumerate(batch.chunk(chunks=8))], 'Actual DP row/provenance partition differs')
        arm_path = directory/'armed-workers.json'; arm_receipt = file_record(arm_path)
        armed = read_json_record(arm_receipt, directory)
        require(armed['outer_step'] == outer and armed['manifest_sha256'] == manifest_receipt['sha256']
                and armed['driver_post_sampling_state_digest'] == manifest['driver_post_sampling_state_digest'],
                'Actual all-worker/driver post-sampling boundary differs')
        arms = validate_live_arms(armed['workers'], outer, manifest_receipt['sha256'])
        for arm in arms:
            rank = arm['rank']; physical = read_json_record(arm['arm_receipt'], directory)
            require(physical == {k:v for k,v in arm.items() if k != 'arm_receipt'}, 'Actual worker arm record differs')
            verify_record(arm['pause_state'], directory)
            pause = torch.load(arm['pause_state']['path'], map_location='cpu', weights_only=False)
            require(state_digest(pause) == arm['complete_pause_state_sha256']
                    and pause['identity']['rank'] == rank
                    and (pause['identity']['pid'], pause['identity']['start_ticks']) ==
                        (workers[rank]['pid'], workers[rank]['start_ticks'])
                    and pause['identity']['scheduler_step'] == outer-1,
                    'Actual model/Adam/RNG/PID post-sampling arm state differs')
            complete_path = run/'audit/probability-admission'/f'complete-{outer}-rank-{rank}.json'
            complete = read_json_record(file_record(complete_path), run)
            require(complete['status'] == 'MATCHED_ADMISSION_AND_ORIGINAL_UPDATE_RETURNED'
                    and complete['outer_step'] == outer and complete['rank'] == rank
                    and complete['pid'] == workers[rank]['pid'] and complete['actual_optimizer_steps'] == 10,
                    'Actual first-Adam admission/update completion missing')
            decision = collective_gate(complete['all_ranks'])
            require(complete['decision'] == decision
                    and decision['global_manifest_sha256'] == manifest_receipt['sha256']
                    and decision['global_arm_sha256'] == arm_receipt['sha256'],
                    'Matched last collective did not bind physical complete probability/pause material')
            for proposal in complete['all_ranks']:
                expected = next(r for r in arms if r['rank'] == proposal['rank'])
                require((proposal['pid'], proposal['start_ticks']) == (expected['pid'], expected['start_ticks'])
                        and proposal['local_batch_digest'] == manifest['chunks'][proposal['rank']]['batch_digest'],
                        'Actual distributed worker/chunk identity changed')
        summaries.append(dict(outer_step=outer, manifest=manifest_receipt, arm_receipt=arm_receipt,
                              trajectories=checks['trajectories'], actions=checks['actions'],
                              effective_tokens=checks['effective_tokens']))
    return dict(status='PHYSICAL_RECONSTRUCTION_NOT_INDEPENDENT_ACCEPTANCE',
                retained25=25, new_complete_outers=len(summaries), rank_count=8, outers=summaries,
                sampler_runner_positions_observed=False,
                engine_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION')
