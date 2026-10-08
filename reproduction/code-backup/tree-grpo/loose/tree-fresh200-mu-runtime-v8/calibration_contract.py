"""Fresh200 wrapper contract aligned to the actual fresh source/RPC schema.

No model forwards or optimizer calls. Missing actual evidence refuses; official
configured paths plus asset identity are not a full parameter-byte comparison.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PROTECTED = {**{pid: '1502352181' for pid in (1680782,1680783,1680784,1680785)},
             **{pid: '1502398762' for pid in (1696443,1696444,1696445,1696446)}}
PROFILE = 'tree_fresh200_five_leaf_v1'
MU_PROTOCOL = 'tree_behavior_mu_centered_ppo_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def held_json(path, expected=None):
    with Path(path).open('rb') as stream:
        data = stream.read()
        digest = hashlib.sha256(data).hexdigest()
    require(expected is None or digest == expected, 'JSON bytes differ: '+str(path))
    return json.loads(data), digest


def held_tensor(path, expected):
    """Hash and deserialize from one open file; original asset remains on h100-9."""
    import torch
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for part in iter(lambda: stream.read(8*1024**2), b''): digest.update(part)
        require(digest.hexdigest() == expected, 'Tensor evidence SHA mismatch')
        stream.seek(0)
        value = torch.load(stream, map_location='cpu', weights_only=False)
        after = os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
                == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Tensor evidence changed while reading')
        stream.seek(0); digest = hashlib.sha256()
        for part in iter(lambda: stream.read(8*1024**2), b''): digest.update(part)
        require(digest.hexdigest() == expected, 'Tensor evidence bytes changed while reading')
    return value


def protected_processes():
    for pid, start in PROTECTED.items():
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rindex(')')+2:].split()
        require(fields[19] == start and fields[0] not in ('Z','X'),
                'Protected PID/start changed: '+str(pid))


def forbid_recovery_environment(environ):
    present = [name for name in environ if name.startswith('TREE_RECOVERY_')]
    require(not present, 'Fresh200 forbids inherited recovery environment: '+str(present))


def expected_runtime(run, package, config_hash):
    return dict(code_commit=package['code_commit'], config_sha256=config_hash,
                data_identity=package['common_data_identity'], model_identity=package['model_identity']['sha256'],
                job_id=Path(run).name.rsplit('-',1)[1], run_id=Path(run).name)


def expected_mu_provenance(receipt):
    """Collector metadata retains original ENV names; receipts use short names."""
    return {'SEARCH_R1_BASELINE_COMMIT':receipt['code_commit'],
            'SEARCH_R1_BASELINE_CONFIG_SHA256':receipt['config_sha256'],
            'SEARCH_R1_BASELINE_DATA_IDENTITY':receipt['data_identity'],
            'SEARCH_R1_MODEL_IDENTITY':receipt['model_identity'],
            'SEARCH_R1_BASELINE_RUN_ID':receipt['run_id'],
            'SLURM_JOB_ID':receipt['job_id']}


def validate_fresh_start(run, package, config_hash):
    value, digest = held_json(Path(run)/'output/fresh-initialization.json')
    expected = expected_runtime(run,package,config_hash)
    require(value['status'] == 'FRESH_OFFICIAL_INITIAL_STATE_OBSERVED', 'Missing fresh actual startup evidence')
    require(value['training_protocol'] == PROFILE and value['calibration_profile'] == 'fresh_base_zero_v1'
            and value['initialized_global_steps'] == 0 and value['parent_completed_step'] == 0
            and value['parent_cursor'] == 0 and value['driver_completed_outer'] == 0
            and value['checkpoint_loaded'] is False,
            'Fresh driver restored/advanced a prior training state')
    require(all(value.get(key) == val for key,val in expected.items()), 'Fresh startup identity changed')
    workers = sorted(value['workers'],key=lambda record: record['rank'])
    require(len(workers) == 8, 'Fresh startup lacks eight direct rank receipts')
    for rank, record in enumerate(workers):
        require(record['rank'] == rank and record['world_size'] == 8
                and record['training_protocol'] == PROFILE and record['calibration_profile'] == 'fresh_base_zero_v1'
                and record['checkpoint_recovery_environment_absent'] is True
                and record['optimizer_state_entries'] == 0 and record['optimizer_step_values'] == []
                and record['scheduler_step'] == 0 and record['optimizer_committed'] == 0
                and record['scheduler_advanced'] is False,
                'Fresh rank has inherited Adam/scheduler/checkpoint state')
        require(Path(record['configured_model_path']).resolve() == Path(record['expected_model_path']).resolve()
                == Path(package['official_model_path']).resolve(), 'Fresh rank configured source is not official model')
        require(record['complete_model_parameter_bytes_verified'] is False,
                'Configured-path startup proof must not claim exhaustive weight-byte verification')
        require(all(record.get(key) == val for key,val in expected.items()), 'Fresh rank provenance changed')
    return dict(status='FRESH_START_CONTRACT_VALIDATED', evidence_sha256=digest, workers=8)


def validate_ready_boundary(value,expected):
    """Actual finalize_calibration fields; this does not invent driver counters."""
    require(value['status'] == 'FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES',
            'Wrong corrected fresh calibration status')
    require(value['calibration_profile'] == 'fresh_base_zero_v1'
            and value['initialized_global_steps'] == 0 and value['behavior_mu_corrected_profile'] is True,
            'Fresh corrected profile was not actually executed')
    require(value['parent_completed_outer'] == 0 and value['observed_outer'] == 1
            and value['new_completed_outer'] == 0 and value['optimizer_committed'] == 0
            and value['scheduler_advanced'] is False, 'Fresh calibration received update credit or writes')
    require(value['original_sampler_actor_probability_gate_accepted'] is False,
            'Old sampler/actor gate must remain unaccepted')
    require(all(value.get(key) == val for key,val in expected.items() if key!='run_id'),
            'Fresh calibration provenance changed')
    require(value['sampler_cursor_before_outer'] == 0 and value['sampler_cursor_after_observation'] == 512,
            'Fresh observation question cursor differs from one actual512 batch')


def validate_calibration_rank_state(record,rank,expected):
    require(record['rank'] == rank and record['world_size'] == 8
            and record['calibration_profile'] == 'fresh_base_zero_v1'
            and record['parent_completed_outer'] == 0 and record['observed_outer'] == 1
            and record['optimizer_committed'] == 0 and record['optimizer_state_entries'] == 0
            and record['restored_optimizer_steps'] == [] and record['scheduler_step'] == 0
            and record['scheduler_advanced'] is False,
            'Fresh calibration rank advanced optimizer/scheduler')
    require(record['fresh_post_monitor_state'] == dict(calibration_profile='fresh_base_zero_v1',
        restored_optimizer_steps=[],optimizer_state_entries=0,scheduler_step=0),
        'Fresh monitor changed actual Adam/scheduler state')
    require(all(record.get(key) == val for key,val in expected.items()), 'Calibration rank identity changed')


def validate_calibration_ready(path, run, package, config_hash):
    import torch
    from search_r1.baselines.tree_behavior_mu import validate_behavior_mu_batch
    value, ready_sha = held_json(path)
    expected = expected_runtime(run,package,config_hash)
    validate_ready_boundary(value,expected)
    validate_fresh_start(run,package,config_hash)
    root = Path(run)/'output'
    trajectories = root/'trajectories/outer-000001'
    receipt, _ = held_json(trajectories/'manifest.json')
    require(receipt['status'] == 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED' and receipt['rows'] == 2560,
            'Fresh evidence must contain all five leaves for512 questions')
    full_path = trajectories/'actor-inputs.pt'
    full = held_tensor(full_path,receipt['payloads']['actor-inputs.pt'])
    require(sha(trajectories/'trajectories.jsonl.gz') == receipt['payloads']['trajectories.jsonl.gz'],
            'Full Tree trace SHA differs')
    require(full['rows'] == 2560 and full['identity']['outer_step'] == 1,
            'Actual full actor evidence row/outer count changed')
    # Lossless save_rollouts plain lists are accepted by the pure per-row helper,
    # but reconstruct actual DataProto for the wrapper contract as well.
    import numpy as np
    from verl import DataProto
    metadata = {}
    for key, rows in full['non_tensor_batch'].items():
        require(len(rows)==2560,'Saved non-tensor field row count differs: '+key)
        array=np.empty(2560,dtype=object); array[:]=rows; metadata[key]=array
    data=DataProto.from_dict(tensors=full['tensors'],non_tensors=metadata,meta_info=full['meta_info'])
    mu_receipt = validate_behavior_mu_batch(data,expected_provenance=expected_mu_provenance(expected),required=True)
    require(data.meta_info['tree_behavior_mu_corrected_profile'] is True
            and torch.equal(data.batch['old_log_probs'],data.batch['sampled_old_log_probs']),
            'Actual full PPO denominator is not fixed sampled FP32 mu')
    width = data.batch['responses'].shape[1]
    effective = data.batch['attention_mask'][:,-width:].bool() & data.batch['loss_mask'].bool()
    beta = data.batch['actor_old_log_probs']
    require(beta.is_floating_point() and beta.shape == data.batch['sampled_old_log_probs'].shape
            and bool(torch.isfinite(beta[effective]).all()), 'Independent full beta was not preserved')
    workers = sorted(value['workers'], key=lambda row: row['rank'])
    require(len(workers) == 8, 'Calibration lacks eight rank records')
    common_decision = None
    for rank, record in enumerate(workers):
        validate_calibration_rank_state(record,rank,expected)
        mode=record['actor_calibration_mode']; mode_path=Path(mode['mode_evidence_path'])
        require(mode_path.resolve() == (root/f'probability-calibration/rank-{rank:02d}-mode.json').resolve(),
                'Actual actor mode evidence escaped current run')
        saved_mode,_=held_json(mode_path,mode['mode_evidence_sha256'])
        require(dict(saved_mode,mode_evidence_sha256=mode['mode_evidence_sha256']) == mode
                and mode['requested_monitor_mode'] == 'train' and mode['all_monitor_modules_training'] is True
                and mode['original_module_modes_restored'] is True and mode['body_returned'] is True
                and mode['optimizer_committed'] == 0 and mode['scheduler_advanced'] is False
                and mode['identity'] == dict(rank=rank,world_size=8,**expected),
                'Actual monitor train mode/restoration or identity proof differs')
        require(record['behavior_mu_corrected_profile'] is True
                and record['actual_behavior_mu_row_bindings_validated'] is True
                and record['actual_same_partition_beta_monitor_saved'] is True,
                'Actual corrected probability evidence is incomplete')
        require(record['diagnostic_records'] == [] and record['diagnostic_entries'] == [],
                'Disabled old sampler diagnostic must not be fabricated')
        indices = record['monitor_local_row_indices']
        require(len(indices) == 32 and len(set(indices)) == 32
                and all(type(i) is int and 0<=i<320 for i in indices), 'Wrong fixed monitor rows')
        require(record['monitor_native_forward_calls'] == 32
                and record['additional_initial_actor_repeat_native_forward_calls'] == 16,
                'Same-card2 initial monitor/repeat forward budget changed')
        monitor_path = Path(record['monitor_path'])
        require(monitor_path.resolve() == (root/f'probability-calibration/rank-{rank:02d}-monitor.pt').resolve(),
                'Monitor payload escaped this run')
        monitor = held_tensor(monitor_path,record['monitor_sha256'])
        selected = torch.tensor([rank*320+i for i in indices],dtype=torch.long)
        for key in ('input_ids','responses','attention_mask','position_ids','old_log_probs',
                    'sampled_old_log_probs','actor_old_log_probs','loss_mask','policy_weights','advantages'):
            require(key in monitor and key in data.batch, 'Missing saved monitor field: '+key)
            actual, wanted = monitor[key], data.batch[key].index_select(0,selected)
            valid = effective.index_select(0,selected)
            same = torch.equal(actual[valid],wanted[valid]) if key=='actor_old_log_probs' else torch.equal(actual,wanted)
            require(actual.shape == wanted.shape and actual.dtype == wanted.dtype and same,
                    'Saved monitor no longer maps to full balanced rows: '+key)
        frozen = monitor['actor_same_partition_beta_monitor']
        require(frozen.is_floating_point() and frozen.shape == monitor['old_log_probs'].shape
                and bool(torch.isfinite(frozen[effective.index_select(0,selected)]).all()),
                'Actual frozen beta monitor missing/nonfinite')
        bindings = monitor['behavior_mu_source_row_bindings']
        wanted_bindings = [data.non_tensor_batch['tree_behavior_mu_binding'][int(i)] for i in selected]
        if hasattr(bindings,'tolist'): bindings = bindings.tolist()
        require(bindings == wanted_bindings, 'Saved monitor sampled source binding changed')
        decision = record['decision']
        require(decision['probability_denominator'] == 'actual_behavior_mu'
                and decision['original_sampler_actor_consistency_accepted'] is False
                and decision['initial_mu_kl_baseline_subtracted'] is False
                and decision['initial_actor_repeat_checked'] is True,
                'Corrected observation hid initial mismatch or skipped repeat')
        if common_decision is None: common_decision = decision
        require(decision == common_decision, 'Distributed global observation decision differs')
    audit = root/'tree-rank-audit'
    require(not list(audit.glob('step-*.json')) and not list((audit/'ratio-tail').glob('*')),
            'Training rank update evidence exists in zero-write calibration')
    require(not (root/'checkpoints').exists() or not any((root/'checkpoints').rglob('*')),
            'Zero stage saved new training state')
    require(not (root/'acceptance-step5').exists(), 'Zero stage entered short-training acceptance')
    return dict(status='FRESH_MU_ZERO_OBSERVATION_CONTRACT_VALIDATED_NOT_TRAINING_ACCEPTED',
                ready_sha256=ready_sha, all_leaf_binding=mu_receipt, workers=8,
                decision=common_decision, original_sampler_actor_gate_accepted=False,
                actual_optimizer_writes=0, production_thresholds_accepted=False)
