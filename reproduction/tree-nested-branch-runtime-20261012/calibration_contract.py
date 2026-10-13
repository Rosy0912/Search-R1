from production_contract import run_job_id,read_bound
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
    """Hash and deserialize from one open file; original asset remains on h100-10."""
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


_BACKGROUND_LAST_STATUS={}
_BACKGROUND_MISSING_OBSERVED=set(PROTECTED)  # all8 were native ENOENT in bound FAILED3331 proof; future present PIDs are observed anew

def historical_original9_process_observer():
    """Observe original identities; an exited foreign background is not a trainer fault.

    Only ENOENT for this exact original PID/stat is a missing identity observation.
    Permission, other I/O, parsing and unexpected live-identity errors propagate.
    No process is created, signaled, exempted from the foreign guard or relabelled.
    """
    import errno,time
    rows=[]
    for pid,start in PROTECTED.items():
        path=Path(f'/proc/{pid}/stat')
        try:text=path.read_text()
        except FileNotFoundError as error:
            if error.errno!=errno.ENOENT or error.filename!=str(path):raise
            _BACKGROUND_MISSING_OBSERVED.add(pid)
            row=dict(pid=pid,expected_starttime_jiffies=start,status='ORIGINAL_BACKGROUND_PROC_STAT_MISSING',
                actual_starttime_jiffies=None,original_alive=False,identity_known=False,
                error_type='FileNotFoundError',error_errno=errno.ENOENT,stat_path=str(path),
                scope='Original identity no longer observable; exit cause/time are unknown, not a training fault')
        else:
            fields=text[text.rindex(')')+2:].split();actual=fields[19];state=fields[0]
            if actual!=start and pid in _BACKGROUND_MISSING_OBSERVED:
                row=dict(pid=pid,expected_starttime_jiffies=start,status='FOREIGN_PID_REUSED_AFTER_ORIGINAL_MISSING',
                    actual_starttime_jiffies=actual,original_alive=False,identity_known=False,
                    stat_path=str(path),scope='Reused PID is foreign; no original preservation claim or signal')
            else:
                require(actual==start,'Protected live PID identity changed: '+str(pid))
                row=dict(pid=pid,expected_starttime_jiffies=start,
                    status='ORIGINAL_BACKGROUND_EXITED_STATE' if state in ('Z','X') else 'ORIGINAL_BACKGROUND_ALIVE_IDENTITY_PRESERVED',
                    actual_starttime_jiffies=actual,original_alive=state not in ('Z','X'),identity_known=True,
                    stat_path=str(path),process_state=state)
        signature=(row['status'],row['actual_starttime_jiffies'])
        if _BACKGROUND_LAST_STATUS.get(pid)!=signature:
            directory=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
            require(directory.is_dir(),'Own output missing for original-background archive')
            with (directory/'original-background-observations.jsonl').open('a') as stream:
                json.dump(dict(row,observed_epoch=time.time(),job_id=os.environ['SLURM_JOB_ID'],
                    code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),process_signals_sent=False),stream,allow_nan=False)
                stream.write('\n')
            _BACKGROUND_LAST_STATUS[pid]=signature
        rows.append(row)
    return dict(status='ORIGINAL_BACKGROUND_IDENTITIES_OBSERVED',originals=rows,
        original_eight_pid_start_times_preserved=all(row['original_alive'] and row['identity_known'] for row in rows),
        missing_original_pids=[row['pid'] for row in rows if row['status']=='ORIGINAL_BACKGROUND_PROC_STAT_MISSING'],
        current_foreign_processes_untouched=True,process_signals_sent=False)


def protected_processes():
    """Original9 is retired; its PID namespace is not the retained18 namespace."""
    rows=[dict(pid=pid,expected_starttime_jiffies=start,status='RETIRED_ORIGINAL_NODE_IDENTITY_UNOBSERVABLE',
               actual_starttime_jiffies=None,original_alive=False,identity_known=False,original_node='h100-9',
               current_node='h100-10',process_signals_sent=False) for pid,start in PROTECTED.items()]
    return dict(status='ORIGINAL_NODE_RETIRED_CURRENT_FOREIGN_GUARD_REMAINS',originals=rows,
        original_eight_pid_start_times_preserved=None,missing_original_pids=list(PROTECTED),
        current_foreign_processes_untouched=True,process_signals_sent=False)


def final_gpu_process_observation(snapshot,own_relative,background):
    """Classify actual remaining GPU PIDs; only the own allocation may be cleaned."""
    import errno
    alive={row['pid'] for row in background['originals'] if row['original_alive'] and row['identity_known']}
    rows=[]
    for pid in sorted({int(line.split(',')[1].strip()) for line in snapshot['processes']}):
        stat_path=Path(f'/proc/{pid}/stat');cgroup_path=Path(f'/proc/{pid}/cgroup')
        try:
            text=stat_path.read_text();fields=text[text.rindex(')')+2:].split()
            membership=cgroup_path.read_text().splitlines()
        except FileNotFoundError as error:
            if error.errno!=errno.ENOENT or error.filename not in (str(stat_path),str(cgroup_path)):raise
            rows.append(dict(pid=pid,status='GPU_PID_PROC_IDENTITY_MISSING',identity_known=False,own_membership=None))
            continue
        own=any(line.split(':',2)[-1]==own_relative or line.split(':',2)[-1].startswith(own_relative+'/') for line in membership)
        require(not own or fields[0] in ('Z','X'),'Live own GPU process remains after own cleanup: '+str(pid))
        rows.append(dict(pid=pid,process_state=fields[0],starttime_jiffies=fields[19],own_membership=own,
            original_preserved_background=pid in alive,
            status='OWN_EXITED_GPU_CONTEXT_OBSERVED' if own else ('ORIGINAL_BACKGROUND_GPU_PROCESS' if pid in alive else 'CURRENT_FOREIGN_GPU_PROCESS'),
            process_signals_sent=False))
    return dict(status='FINAL_GPU_PROCESS_IDENTITIES_OBSERVED',processes=rows,
        unexpected_current_foreign_pids=[row['pid'] for row in rows if row['status']=='CURRENT_FOREIGN_GPU_PROCESS'],
        missing_gpu_pid_identities=[row['pid'] for row in rows if row['status']=='GPU_PID_PROC_IDENTITY_MISSING'],
        current_foreign_processes_untouched=True,process_signals_sent=False,
        resource_guard_scope='During living training: original mandatory realtime foreignMEM/SM15%; cleanup never signals foreign processes')


def forbid_recovery_environment(environ):
    present = [name for name in environ if name.startswith(('TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_'))]
    require(not present, 'Fresh200 forbids inherited recovery environment: '+str(present))
    from search_r1.baselines.tree_nested_chain_resume import validate_entry
    validate_entry(environ=environ)


def expected_runtime(run, package, config_hash):
    return dict(code_commit=package['code_commit'], config_sha256=config_hash,
                data_identity=package['common_data_identity'], model_identity=package['model_identity']['sha256'],
                job_id=run_job_id(run), run_id=Path(run).name)


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
    native = package.get('nested_native_continuation') is not None
    expected_status = ('COLD_OFFICIAL_INITIAL_STATE_OBSERVED_BEFORE_NESTED_NATIVE1_CONTINUATION'
                       if native else 'FRESH_OFFICIAL_INITIAL_STATE_OBSERVED')
    require(value['status'] == expected_status, 'Missing actual official cold initialization evidence')
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
                and record['checkpoint_recovery_environment_absent'] is (not native)
                and record.get('native_nested_continuation_requested',False) is native
                and record['optimizer_state_entries'] == 0 and record['optimizer_step_values'] == []
                and record['scheduler_step'] == 0 and record['optimizer_committed'] == 0
                and record['scheduler_advanced'] is False,
                'Fresh rank has inherited Adam/scheduler/checkpoint state')
        require(Path(record['configured_model_path']).resolve() == Path(record['expected_model_path']).resolve()
                == Path(package['official_model_path']).resolve(), 'Fresh rank configured source is not official model')
        require(record['complete_model_parameter_bytes_verified'] is False,
                'Configured-path startup proof must not claim exhaustive weight-byte verification')
        require(all(record.get(key) == val for key,val in expected.items()), 'Fresh rank provenance changed')
    return dict(status=('COLD_OFFICIAL_BOOTSTRAP_BEFORE_NESTED_NATIVE_CONTINUATION_VALIDATED'
                        if native else 'FRESH_START_CONTRACT_VALIDATED'),
                evidence_sha256=digest, workers=8, native_continuation_loaded=False)


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


def validate_native_start(run, package, config_hash):
    from search_r1.baselines.tree_nested_chain_resume import (PROTOCOL, PARENT_MANIFEST,
        PARENT_MANIFEST_SHA256, parent_manifest, failed_save_evidence)
    require(package.get('nested_native_continuation') is not None, 'Explicit native package credential required')
    cold = validate_fresh_start(run, package, config_hash)
    value, receipt_sha, _ = read_bound(Path(run)/'output/nested-native1-loaded.json')
    parent = parent_manifest(); provenance = value['provenance']; expected = expected_runtime(run, package, config_hash)
    require(value['status'] == 'COLD_NESTED_NATIVE1_STATE_LOADED' and value['protocol'] == PROTOCOL
            and provenance['protocol'] == PROTOCOL and provenance['same_fresh_nested_chain'] is True
            and provenance['parent_identity'] == parent['identity']
            and provenance['parent_manifest_path'] == str(PARENT_MANIFEST)
            and provenance['parent_manifest_sha256'] == PARENT_MANIFEST_SHA256
            and provenance['parent_completed_step'] == 1 and provenance['parent_cursor'] == 512
            and provenance['parent_epoch'] == 0 and provenance['new_batch_size'] == 512
            and provenance['failed_parent_outer2'] == failed_save_evidence(parent)
            and all(provenance['child_'+key] == expected[key] for key in ('code_commit','config_sha256','job_id','run_id'))
            and provenance['exact_executor_resume'] is False and value['executor_equivalence'] is False,
            'Actual native continuation mother/child/failure binding differs')
    driver = value['driver']
    require(driver['state_path'] == parent['driver']['path'] and driver['state_file_sha256'] == parent['driver']['sha256']
            and driver['actual_file_sha256_verified'] is True and driver['actual_cold_driver_state_equal'] is True
            and driver['completed_outer_updates'] == 1 and driver['next_outer_update'] == 2
            and driver['sampler_cursor'] == 512 and driver['sampler_epoch'] == 0 and driver['training_attempts'] == 1
            and driver['completed_optimizer_calls_per_rank'] == {str(r):20 for r in range(8)},
            'Actual loaded driver effective native1 boundary differs')
    workers = sorted(value['workers'], key=lambda r:r['rank'])
    require(len(workers) == 8 and [r['rank'] for r in workers] == list(range(8)), 'Loaded native rank coverage differs')
    for record in workers:
        mother = next(r for r in parent['worker_states'] if r['identity']['rank'] == record['rank'])
        require(record['status'] == 'COLD_NESTED_NATIVE1_WORKER_LOADED' and record['protocol'] == PROTOCOL
                and record['world_size'] == 8 and record['parent_identity'] == mother['identity']
                and record['actual_cold_native_state_equal'] is True and record['actual_file_sha256_verified'] is True
                and record['serialized_logical_state_sha256'] == record['actual_loaded_logical_state_sha256'] == mother['logical_state_sha256']
                and record['state_file_sha256'] == mother['sha256'] and record['state_file_bytes'] == mother['bytes']
                and record['scheduler_step'] == 1 and record['adam_steps'] == [20]
                and record['actual_total_optimizer_calls'] == 20 and record['child_code_commit'] == expected['code_commit']
                and record['exact_executor_resume'] is False, 'Actual cold-loaded native rank receipt differs')
    return dict(status='ACTUAL_NESTED_NATIVE1_CONTINUATION_START_VALIDATED', receipt_sha256=receipt_sha,
        cold_official_bootstrap=cold, loaded_completed_outer=1, loaded_adam_calls_per_rank=20,
        loaded_scheduler_step=1, loaded_sampler_cursor=512, next_outer=2,
        failed_parent_outer2=provenance['failed_parent_outer2'], exact_executor_resume=False)
