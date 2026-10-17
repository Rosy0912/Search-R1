"""Explicit full-state PJ -> Lumia continuation; storage/host changes only."""
import copy
from pathlib import Path

BASE = Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
MODEL = '/home/jmzhang/Workspace/Search-E1-20261001/models/Qwen2.5-3B-Instruct'
PARENT_SOURCE = '3a13349a6ab6373582758925e2405fafc4445545'
PROFILES = {
    'budget-5': dict(step=85, job='zhangjunming-ttt-b5-s53-v7',
        config='be8bfd13c2ea9e077d055be2b230cefcac365b2ddcb9d7a25a60da442f05b372',
        run_id='02ccdba4db9f4e68aacbb7edfa3042ab', node='node9', host='gpu-ada6000-2',
        stage='pj-budget5-complete85-backup-20261017T1300'),
    'generation-2': dict(step=137, job='zhangjunming-ttt-d2-s108-v3',
        config='c52da22b7dfa12584871bcc6e275209a65ceda9c0432ffab813c43d4eccdf001',
        run_id='4968fc6b0e0c4817a80641f03e95317f', node='node10', host='gpu-l40s-1',
        stage='pj-depth2-complete137-backup-20261017T1350'),
}


def project_config(parent, group):
    """Enumerate every allowed physical-location delta, never a science delta."""
    p = PROFILES[group]
    cfg = copy.deepcopy(parent)
    data = BASE / 'common-data'
    cfg['baselines']['local_runtime_host'] = p['host']
    for key, name in [('train_files', 'train.parquet'), ('val_files', 'validation.parquet'),
                      ('final_val_files', 'final.parquet')]:
        cfg['data'][key] = str(data / name)
    cfg['baselines']['holdout'].update(manifest_path=str(data / 'manifest.json'), data_root=str(data))
    cfg['baselines']['bapo']['stage_validation_files'] = str(data / 'stage.parquet')
    cfg['actor_rollout_ref']['model']['path'] = MODEL
    cfg['actor_rollout_ref']['ref']['model_path'] = MODEL
    cfg['critic']['model']['tokenizer_path'] = MODEL
    cfg['reward_model']['model']['input_tokenizer'] = MODEL
    cfg['trainer']['default_local_dir'] = str(BASE / 'runs' / group / 'checkpoints')
    return cfg


def validate_projection(parent, actual, group):
    if actual != project_config(parent, group):
        raise ValueError('Only enumerated storage paths and actual allocated host may change')
    from release.recursive_bran_a.ablation150_v1.contracts import validate_config_dict
    validate_config_dict(actual)


def validate_parent(parent, entry, manifest, group):
    p = PROFILES[group]
    identity = manifest['identity']
    if (parent['group'] != group or parent['run_id'] != p['run_id']
            or parent['source_commit'] != PARENT_SOURCE or parent['config_sha256'] != p['config']
            or entry['allocation'] != p['job'] or entry['run_id'] != p['run_id']
            or entry['source_commit'] != PARENT_SOURCE or entry['config_sha256'] != p['config']
            or manifest['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity['completed_step'] != p['step'] or identity['actual_update_limit'] != 150
            or identity['job_id'] != p['job'] or identity['code_commit'] != PARENT_SOURCE
            or identity['config_sha256'] != p['config']
            or identity['data_identity'] != parent['data_identity']
            or manifest['driver']['next_step'] != p['step'] + 1):
        raise ValueError('Wrong explicit durable parent; no reset or earlier-state fallback')
    workers = manifest['worker_states']
    if len(workers) != 8 or {w['identity']['rank'] for w in workers} != set(range(8)):
        raise ValueError('Eight complete distinct native ranks required')
    totals = identity['guard_continue_accounting']['total_optimizer_calls_per_rank']
    for w in workers:
        n = totals[str(w['identity']['rank'])]
        if (any(w['identity'][k] != v for k, v in identity.items())
                or w['identity']['world_size'] != 8 or w['scheduler_last_epoch'] != p['step']
                or w['full_worker_state_finite'] is not True
                or w['actual_total_optimizer_calls'] != n or w['observed_adam_steps'] != [n]):
            raise ValueError('Native rank identity, finite state or observed Adam count differs')


def mapped_manifest(original, stage, members):
    """Project file locations only; tensor bytes/identities/cursors remain original."""
    value = copy.deepcopy(original)
    records = value['worker_states'] + [value['driver']]
    if len(members) != 9 or len({x['name'] for x in members}) != 9:
        raise ValueError('Require nine unique transferred members')
    for r in records:
        name = Path(r['path']).name
        matches = [x for x in members if x['name'] == name]
        if len(matches) != 1 or matches[0]['original'] != r['path'] or matches[0]['sha256'] != r['sha256']:
            raise ValueError('Transferred member differs from original source manifest')
        r['path'] = str(Path(stage) / name)
    return value
