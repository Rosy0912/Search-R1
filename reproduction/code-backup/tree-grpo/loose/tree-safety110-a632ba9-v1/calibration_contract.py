"""CPU-only identity checks for the healthy110 zero-write calibration stage."""
import hashlib
import json
from pathlib import Path

PARENT_STEP = 110
OBSERVED_STEP = 111
PARENT_ADAM = 2500
PARENT_CURSOR = 57600
PERMUTATION = '3e0101af98996d517a4a344de01199d6e91fd9cb433a88b3e1ad5481d7d9748a'
ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PROTECTED = {**{pid: '1502352181' for pid in (1680782,1680783,1680784,1680785)},
             **{pid: '1502398762' for pid in (1696443,1696444,1696445,1696446)}}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protected_processes():
    for pid, start in PROTECTED.items():
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rindex(')')+2:].split()
        require(fields[19] == start and fields[0] not in ('Z', 'X'),
                f'Protected process identity changed: {pid}')

def validate_parent_manifest(package):
    expected = package['parent_manifest']
    require(sha(expected['path']) == expected['sha256'], 'Parent110 manifest SHA changed')
    value = json.loads(Path(expected['path']).read_bytes())
    require(value['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED', 'Incomplete parent110 status')
    require(value['identity'] == expected['identity'], 'Parent110 identity changed')
    require(value['identity']['completed_step'] == PARENT_STEP, 'Wrong parent step')
    records = sorted(value['worker_states'], key=lambda record: record['identity']['rank'])
    require(len(records) == 8, 'Parent world size changed')
    pinned = package['healthy110_metadata']
    for rank, record in enumerate(records):
        require(record['identity']['rank'] == rank and record['identity']['world_size'] == 8,
                'Parent rank coverage changed')
        require(record == pinned['worker_states'][rank], 'Parent worker metadata changed')
        require(record['scheduler_last_epoch'] == PARENT_STEP, 'Parent scheduler changed')
        require(Path(record['path']).is_relative_to(ROOT/'runs/tree-recovery300-3234'),
                'Parent weights outside this task node directory')
    require(value['driver'] == pinned['driver'], 'Parent driver metadata changed')
    require(value['hf_identity'] == pinned['hf_identity'], 'Parent HF identity changed')
    # Full worker/driver SHA, finite state and cold object digest are verified by
    # load_recovery_training_state/restore_recovery_driver, not replaced here.

def validate_restored110(run, package):
    path = Path(run)/'output/recovery-loaded.json'
    value = json.loads(path.read_bytes())
    require(value['status'] == 'COLD_PARENT_STATE_LOADED_PENDING_EVALUATION', 'Wrong restore status')
    records = sorted(value['workers'], key=lambda row: row['rank'])
    require(len(records) == 8, 'Missing actual cold restore workers')
    for rank, record in enumerate(records):
        require(record['rank'] == rank and record['scheduler'] == PARENT_STEP,
                'Actual restored rank/scheduler mismatch')
        require(record['adam_steps'] == [PARENT_ADAM], 'Actual restored Adam step mismatch')
        require(record['parent_sha256'] == package['healthy110_metadata']['worker_states'][rank]['sha256'],
                'Actual restored worker source mismatch')
        parent_identity = dict(record['parent_identity'])
        require(parent_identity.pop('rank') == rank and parent_identity.pop('world_size') == 8,
                'Actual restored shard layout changed')
        parent_identity.pop('cuda_visible_devices', None)
        require(parent_identity == package['parent_manifest']['identity'], 'Actual restore identity changed')
    migration = value['sampler_migration']
    require(migration['parent_batch_size'] == migration['batch_size'] == 640,
            'Healthy110 batch migration is not approved')
    require(migration['consumed_cursor'] == PARENT_CURSOR and migration['epoch'] == 0,
            'Restored driver consumed cursor/epoch changed')
    require(migration['permutation_sha256'] == PERMUTATION, 'Restored question order changed')
    require(migration['exact_executor_resume'] is False, 'Do not claim executor replay equivalence')
    provenance = value['provenance']
    require(provenance['parent_identity'] == package['parent_manifest']['identity'] and
            provenance['parent_completed_step'] == PARENT_STEP and
            provenance['parent_cursor'] == PARENT_CURSOR and provenance['parent_epoch'] == 0 and
            provenance['new_batch_size'] == 640 and provenance['exact_executor_resume'] is False,
            'Restored driver provenance changed')
    return value

def validate_calibration_ready(path, run, package, config_hash):
    value = json.loads(Path(path).read_bytes())
    contract = package['probability_calibration']
    require(value['status'] == contract['ready_status'], 'Wrong calibration status')
    require(value['code_commit'] == package['code_commit'] and value['config_sha256'] == config_hash,
            'Calibration source/config identity changed')
    require(value['job_id'] == Path(run).name.rsplit('-', 1)[1], 'Calibration job changed')
    require(value['data_identity'] == package['common_data_identity'] and
            value['model_identity'] == package['model_identity']['sha256'], 'Calibration assets/split changed')
    require(value['parent_completed_outer'] == PARENT_STEP and value['observed_outer'] == OBSERVED_STEP,
            'Calibration logical parent/observed outer changed')
    require(value['new_completed_outer'] == 0 and value['optimizer_committed'] == 0 and
            value['scheduler_advanced'] is False, 'Calibration performed training work')
    require(value['sampler_cursor_before_outer'] == PARENT_CURSOR and
            value['sampler_cursor_after_observation'] == PARENT_CURSOR+640,
            'Calibration question cursor changed')
    records = sorted(value['workers'], key=lambda row: row['rank'])
    require(len(records) == 8, 'Calibration rank coverage incomplete')
    expected_identity = dict(job_id=value['job_id'],run_id=Path(run).name,
                             code_commit=package['code_commit'],config_sha256=config_hash,
                             data_identity=package['common_data_identity'],
                             model_identity=package['model_identity']['sha256'])
    common_calls = None
    common_raw_decisions = {}
    common_raw_rows = {}
    common_monitor_decision = None
    for rank, record in enumerate(records):
        require(record['rank'] == rank and record['world_size'] == 8, 'Calibration world layout changed')
        require(record['parent_completed_outer'] == PARENT_STEP and record['observed_outer'] == OBSERVED_STEP,
                'Calibration worker outer differs')
        require(record['optimizer_committed'] == 0 and record['scheduler_advanced'] is False,
                'Calibration worker performed optimizer/scheduler write')
        require(record['restored_optimizer_steps'] == [PARENT_ADAM] and record['scheduler_step'] == PARENT_STEP,
                'Calibration actual Adam/scheduler differs from parent')
        require(all(record.get(key) == expected for key,expected in expected_identity.items()),
                'Calibration worker identity changed')
        if common_monitor_decision is None:
            common_monitor_decision = record['decision']
        require(record['decision'] == common_monitor_decision, 'Distributed monitor decision differs across ranks')
        monitor_path = Path(record['monitor_path'])
        require(monitor_path.resolve().parent == (Path(run)/'output/probability-calibration').resolve() and
                monitor_path.name == f'rank-{rank:02d}-monitor.pt' and
                sha(monitor_path) == record['monitor_sha256'], 'Actual calibration monitor SHA/path changed')
        require(isinstance(record['diagnostic_records'], list) and record['diagnostic_records'],
                'Missing actual policy diagnostic records')
        entries = record['diagnostic_entries']
        require([entry['path'] for entry in entries] == record['diagnostic_records'] and
                len(record['diagnostic_records']) == len(set(record['diagnostic_records'])),
                'Diagnostic entries are duplicated or do not match pinned paths')
        calls = []
        for entry in entries:
            diagnostic_path = entry['path']
            diagnostic_path = Path(diagnostic_path)
            require(diagnostic_path.resolve().parent == (Path(run)/'output/sampling-consistency').resolve(),
                    'Sampler diagnostic escaped current same-node run')
            require(sha(diagnostic_path) == entry['sha256'], 'Actual raw diagnostic JSON SHA changed')
            diagnostic = json.loads(diagnostic_path.read_bytes())
            require(diagnostic['rank'] == rank and diagnostic['world_size'] == 8 and
                    diagnostic['scheduler_step'] == PARENT_STEP and diagnostic['optimizer_committed'] == 0,
                    'Sampler diagnostic actual actor boundary changed')
            require(all(diagnostic.get(key) == expected for key,expected in expected_identity.items()) and
                    diagnostic['enforced'] is False, 'Sampler diagnostic configuration binding changed')
            require(diagnostic['saved_rows'] == diagnostic['local_rows'] and diagnostic['saved_rows'] > 0,
                    'Calibration must preserve every actual raw row for numerical verification')
            call = diagnostic['generation_call']
            require(diagnostic_path.name == f'rank-{rank:02d}-call-{call:05d}.json',
                    'Actual generation call/path differs')
            calls.append(call)
            if call not in common_raw_decisions:
                common_raw_decisions[call] = diagnostic['decision']
                common_raw_rows[call] = diagnostic['local_rows']
            require(diagnostic['decision'] == common_raw_decisions[call],
                    'Distributed raw sampling decision differs across workers')
            require(diagnostic['local_rows'] == common_raw_rows[call],
                    'Actual per-call local sample counts differ across workers')
            snapshot_path = Path(diagnostic['snapshot_path'])
            require(snapshot_path.resolve().parent == diagnostic_path.resolve().parent and snapshot_path.suffix == '.pt',
                    'Sampler snapshot escaped current same-node run')
            require(str(snapshot_path) == entry['snapshot_path'] and
                    diagnostic['snapshot_sha256'] == entry['snapshot_sha256'],
                    'Sampler snapshot differs from the pinned worker entry')
            require(sha(snapshot_path) == diagnostic['snapshot_sha256'], 'Actual sampler snapshot SHA changed')
        if common_calls is None:
            common_calls = calls
        require(calls == common_calls and calls == list(range(1,len(calls)+1)),
                'Actual generation call coverage/order differs across workers')
    audit = Path(run)/'output/tree-rank-audit'
    direct_records = list(audit.glob('step-*.json'))
    ratio_records = list((audit/'ratio-tail').glob('*')) if (audit/'ratio-tail').exists() else []
    require(not direct_records and not ratio_records, 'Direct training optimizer evidence in zero-write calibration')
    events = (audit/'policy-safety').glob('*.json') if (audit/'policy-safety').exists() else []
    for event_path in events:
        event = json.loads(event_path.read_bytes())
        require(event['committed_optimizer_calls'] == 0 and event['scheduler_advanced'] is False,
                'Safety event records an optimizer/scheduler write during calibration')
    checkpoints = Path(run)/'output/checkpoints'
    require(not checkpoints.exists() or not any(checkpoints.rglob('*')),
            'Calibration unexpectedly saved a new training checkpoint')
    return value
