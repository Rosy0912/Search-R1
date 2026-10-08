"""Cross-job restore of accepted step50; preserve training state, rebind execution identity."""
import copy
import hashlib
import json
import os
from pathlib import Path

MANIFEST_SHA = '09b682b1661172760ca5d042372d0cca7aadb929e25214efeb2e484948ec8bc0'
PARENT_RUN = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3340-1791441919057393097')
CHECKPOINT = PARENT_RUN / 'checkpoints/actor/global_step_50'

def require(value, message):
    if not value:
        raise RuntimeError(message)

def manifest():
    raw = (CHECKPOINT / 'training_state/manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'Restore50 manifest changed')
    value = json.loads(raw)
    require(value['logical_step'] == 50 and str(value['job_id']) == '3340', 'Wrong resume checkpoint')
    return value

def binding():
    raw = Path(os.environ['GIGPO_POLICY_NATIVE_BINDING']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'], 'Resume binding SHA changed')
    value = json.loads(raw)
    require(value['resume50']['manifest_sha256'] == MANIFEST_SHA
            and value['resume50']['checkpoint'] == str(CHECKPOINT), 'Unbound restore50')
    return value

def rebind_loaded_worker_metadata(worker, state, path, logical_step, optimizer_steps):
    from search_r1.baselines.fresh200_live import worker_admission_state
    from search_r1.baselines.instruct300 import equal_state
    require(Path(path) == CHECKPOINT and logical_step == 50 and optimizer_steps == 1000, 'Wrong restore endpoint')
    binding(); manifest()
    original = state['runtime_metadata']['fresh200_live_admission']
    current = worker_admission_state(worker)
    require(str(original['payload']['job_id']) == '3340'
            and original['payload']['run_id'] == PARENT_RUN.name
            and original['payload']['policy'] == current['payload']['policy'], 'Restore changed policy or parent')
    old, new = copy.deepcopy(original['worker_config']), copy.deepcopy(current['worker_config'])
    # Only generation cache capacity is changed by the latest human instruction.
    old['rollout']['gpu_memory_utilization'] = new['rollout']['gpu_memory_utilization']
    require(equal_state(old, new), 'Restore changed worker scientific configuration')
    state['runtime_metadata']['fresh200_live_admission'] = copy.deepcopy(current)

def restore_driver(trainer, run):
    import torch
    from omegaconf import OmegaConf
    from search_r1.baselines.recovery import restore_rng, file_sha256
    from search_r1.baselines.fresh200 import validate_attempt_ledger, driver_flow_state
    from search_r1.baselines.instruct300 import equal_state, _live_gigpo_state
    from search_r1.baselines.fresh200_live import driver_admission_state
    from calibration_driver_v1 import json_save
    m = manifest(); binding()
    row = next(row for row in m['files'] if row['path'].endswith('/driver.pt'))
    require(file_sha256(row['path']) == row['sha256'], 'Restore driver SHA differs')
    saved = torch.load(row['path'], map_location='cpu', weights_only=False)
    old, new = copy.deepcopy(saved['config']), OmegaConf.to_container(trainer.config, resolve=True)
    old['trainer']['default_local_dir'] = new['trainer']['default_local_dir']
    old['actor_rollout_ref']['rollout']['gpu_memory_utilization'] = new['actor_rollout_ref']['rollout']['gpu_memory_utilization']
    require(equal_state(old, new), 'Restore changed driver scientific configuration')
    require(saved['logical_step'] == saved['actual_new_updates'] == 50, 'Driver endpoint not50')
    ledger = saved['fresh200_driver_flow']['attempt_ledger']
    validate_attempt_ledger(ledger)
    require(ledger['completed_outer_updates'] == 50 and ledger['actual_adam_total'] == 1000, 'Wrong accepted writes')
    driver_admission_state(trainer)  # Establish this new job's real actor/reference metadata.
    shas = {str(row['pause_identity']['rank']): row['sha256'] for row in m['files'] if 'pause_identity' in row}
    loaded = trainer.actor_rollout_wg.calibration_resume50_full_state(str(CHECKPOINT), shas, 50, 1000)
    require(len(loaded) == 8 and {row['rank'] for row in loaded} == set(range(8))
            and all(row['actual_rank_pt_reload'] and row['actual_complete_runtime_readback'] for row in loaded),
            'Incomplete real eight-rank restore')
    trainer.recovery_sampler.load_state_dict(saved['sampler'])
    trainer.fresh200_attempt_ledger = copy.deepcopy(ledger)
    vars(trainer.kl_ctrl).clear(); vars(trainer.kl_ctrl).update(copy.deepcopy(saved['kl_controller']))
    trainer.baseline_controller.stage.load_state_dict(copy.deepcopy(saved['algorithm_state']['bapo']))
    trainer.global_steps = trainer.actual_outer_updates = 50
    trainer._calibration_training_iterator = iter(trainer.train_dataloader)
    flow = saved['fresh200_driver_flow']
    require(type(trainer._calibration_training_iterator).__module__ + '.' + type(trainer._calibration_training_iterator).__qualname__
            == flow['iterator_class'], 'Different resume iterator type')
    for key, value in flow['iterator_fields'].items():
        setattr(trainer._calibration_training_iterator, key, value)
    trainer.validation_trace_serial = flow['validation_trace_serial']
    restore_rng(saved['rng'])  # Restore after constructor/new iterator have consumed their RNG.
    require(equal_state(trainer.recovery_sampler.state_dict(), saved['sampler'])
            and equal_state(driver_flow_state(trainer), flow)
            and equal_state(_live_gigpo_state(trainer, 50), saved['algorithm_state']), 'Actual driver readback differs')
    trainer.resume50_parent = dict(run=str(PARENT_RUN), manifest_sha256=MANIFEST_SHA, retained_outer=50,
        source_commit=m['code_commit'], parent_binding_sha256='5f65aa458f3607f546608f38368da48321874f33b526fcc15e3764271d48199c')
    # Preserve parent provenance verbatim, filtering strictly to the accepted prefix.
    # Audit reads the parent identity for inherited records and current identity thereafter.
    for rank in range(8):
        for rel in (f'audit/{rank}.jsonl', f'audit/policy-safety/rank-{rank}.jsonl',
                    f'audit/numeric/micro-{rank}.jsonl'):
            source, target = PARENT_RUN / rel, run / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            accepted = []
            for line in source.read_bytes().splitlines(keepends=True):
                entry = json.loads(line)
                if entry.get('attempt_id', 10**9) <= ledger['attempts']:
                    accepted.append(line if line.endswith(b'\n') else line + b'\n')
            with target.open('xb') as stream:
                stream.write(b''.join(accepted))
    json_save(run / 'actual-cross-job-resume50.json', dict(status='ACTUAL_COMPLETE_STATE_RESTORED',
        same_original_process=False, current_job_id=os.environ['SLURM_JOB_ID'],
        current_run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'], parent=trainer.resume50_parent,
        workers=loaded, actual_adam_per_rank=1000, scheduler_step=50,
        data_cursor=trainer.recovery_sampler.cursor, driver_flow_readback_equal=True,
        discarded_unsaved_outer=[51,52,53,54,55,56,57,58,59,60], next_outer=51))
