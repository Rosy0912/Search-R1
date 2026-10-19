"""No torch, scheduler mutation, or implicit retry in the continuation contract."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = '/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1/official-branpo'
PARENTS = {
    '3b': ('139135', '3b-v6', '1d48f3158340470acd6765a8c83f8afc3ababf1b76945787e85fda3e45794899'),
    '7b': ('139136', '7b-v5', 'd352190d3928266f628019d065ae7111e2277e143075802546a39c010d3f58fe'),
}
PARENT_SOURCE = '79790add4d605b7fbb7a4aacaceb8ef7a6910079'
RESUME_LABEL, FINAL_LABEL, START_LABEL = 240, 250, 100
CHANGES = {'trainer.total_steps', 'trainer.resume_mode', 'trainer.resume_from_path',
           'trainer.default_local_dir', 'trainer.experiment_name',
           'trainer.val_before_train', 'trainer.del_local_ckpt_after_load',
           'trainer.episode_log_dir'}  # derived logging path follows experiment_name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def job_name(size):
    assert size in PARENTS
    return 'zhangjunming-branpo-' + size + '-150-v1'


def overrides(parent, size):
    assert size == parent['size'] and parent['adapter_commit'] == PARENT_SOURCE
    values = dict(item.split('=', 1) for item in parent['overrides'])
    assert values['trainer.total_steps'] == '300'
    assert values['trainer.start_step'] == '100'
    assert values['trainer.resume_mode'] == 'disable'
    root = ROOT + '/runs/' + size + '-150-v1'
    values.update({'trainer.total_steps': '250',
                   'trainer.resume_mode': 'resume_path',
                   'trainer.resume_from_path': parent['root'] + '/checkpoints/global_step_240',
                   'trainer.default_local_dir': root + '/checkpoints',
                   'trainer.experiment_name': job_name(size),
                   'trainer.val_before_train': 'False',
                   'trainer.del_local_ckpt_after_load': 'False'})
    return [k+'='+v for k, v in values.items()]


def validate_override_delta(parent_values, new_values):
    old = dict(x.split('=', 1) for x in parent_values)
    new = dict(x.split('=', 1) for x in new_values)
    assert all(old.get(k) == new.get(k) for k in (set(old)|set(new))-CHANGES), 'Scientific recipe changed'
    assert new['trainer.total_steps'] == '250' and new['trainer.start_step'] == '100'
    assert new['trainer.resume_mode'] == 'resume_path'
    assert new['trainer.resume_from_path'].endswith('/checkpoints/global_step_240')
    assert new['trainer.val_before_train'] == 'False'
    assert new['trainer.del_local_ckpt_after_load'] == 'False'
    assert FINAL_LABEL-START_LABEL == 150 and FINAL_LABEL-RESUME_LABEL == 10
    return True


def patched_sources(task_source, branch_source):
    """Only new imports and observational hooks; original loop statements preserved."""
    old = 'from rllm.trainer.verl.agent_ppo_branch_trainer import AgentPPOTrainer\n'
    assert task_source.count(old) == 1
    task_source = task_source.replace(old, 'from reproduction.branpo150.trainer import AgentPPOTrainer\n')
    old_worker = '            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker\n'
    assert task_source.count(old_worker) == 1
    task_source = task_source.replace(old_worker,
        '            from verl.workers.fsdp_workers import ActorRolloutRefWorker\n'
        '            from reproduction.branpo150.worker import AsyncActorRolloutRefWorker\n')
    # No modification of the actual official branch training loop.
    return task_source, branch_source


def validate_plan(plan, parent):
    size = plan['size']; jid, directory, parent_sha = PARENTS[size]
    assert plan['parent_job_id'] == jid and plan['parent_plan_sha256'] == parent_sha
    assert parent['root'] == ROOT + '/runs/' + directory
    assert parent['adapter_commit'] == PARENT_SOURCE
    assert plan['root'] == ROOT + '/runs/' + size + '-150-v1'
    assert plan['job'] == job_name(size)
    assert plan['resume_label'] == 240 and plan['final_label'] == 250
    assert plan['actual_updates_target'] == 150 and plan['new_updates'] == 10
    assert plan['allocation'] == dict(gpus=8, cpus=64, memory_gib=600)
    assert plan['no_auto_retry'] is True and plan['del_local_ckpt_after_load'] is False
    assert plan['overrides'] == overrides(parent, size)
    validate_override_delta(parent['overrides'], plan['overrides'])
    assert plan['no_claim_of_bitwise_uninterrupted_rollout_trajectory'] is True
    return True
