import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from search_r1.baselines import quality_resume as quality
from search_r1.baselines.recovery import file_sha256


@pytest.fixture
def selection(tmp_path):
    candidates = json.loads(Path(quality.__file__).with_name('quality_candidates.json').read_text())
    records = [dict(candidate_step=c['step'], candidate=c['path'], status='GPU512_EVALUATED',
                    cleanup_residual_states={}, final_gpu_snapshot={'processes': []},
                    metrics={'val/qa_count/all': 512, 'val/f1/all': .3 if c['step'] in (25, 50) else .1,
                             'val/em/all': .2}) for c in candidates]
    summary = dict(status='ALL_CANDIDATES_GPU512_COMPLETE_UNREVIEWED', completed_candidates=records,
                   selected=records[2])
    path = tmp_path / 'selection.json'; path.write_text(json.dumps(summary))
    review = tmp_path / 'review.json'; review.write_text(json.dumps(dict(status='PASS',
        selection_sha256=file_sha256(path), selected_checkpoint=candidates[2]['path'],
        selected_manifest_sha256=candidates[2]['manifest_sha256'], evaluation_job='3232')))
    cfg = OmegaConf.create({'trainer': {'quality_resume': dict(enabled=True, selection_path=str(path),
        selection_sha256=file_sha256(path), selection_review_path=str(review), selection_review_sha256=file_sha256(review))}})
    return cfg, path, review


def test_tie_latest_and_optimizer_offset(selection):
    cfg, _, _ = selection
    origin = quality.selected_origin(cfg)
    assert origin['step'] == 50
    assert quality.optimizer_steps(cfg, 50) == 1000
    assert quality.optimizer_steps(cfg, 55) == 1050
    assert quality.optimizer_steps(cfg, 300) == 3500
    with pytest.raises(ValueError): quality.optimizer_steps(cfg, 49)


@pytest.mark.parametrize('case', ['missing', 'nan', 'wrong_count', 'residual', 'wrong_winner', 'wrong_review'])
def test_selection_rejects_incomplete_or_unbound(selection, case):
    cfg, path, review = selection
    data = json.loads(path.read_text()); record = data['completed_candidates'][0]
    if case == 'missing': data['completed_candidates'].pop()
    if case == 'nan': record['metrics']['val/f1/all'] = float('nan')
    if case == 'wrong_count': record['metrics']['val/qa_count/all'] = 64
    if case == 'residual': record['final_gpu_snapshot']['processes'] = ['foreign']
    if case == 'wrong_winner': data['selected'] = record
    if case == 'wrong_review':
        r = json.loads(review.read_text());r['selected_manifest_sha256'] = 'bad';review.write_text(json.dumps(r))
        cfg.trainer.quality_resume.selection_review_sha256 = file_sha256(review)
    else:
        path.write_text(json.dumps(data));cfg.trainer.quality_resume.selection_sha256=file_sha256(path)
    with pytest.raises(ValueError): quality.selected_origin(cfg)


def test_migration_rejects_algorithm_lr_and_micro_changes():
    old = dict(actor_rollout_ref={'actor': {'ppo_mini_batch_size': 128, 'ppo_micro_batch_size':16,
                                          'optim': {'lr':1e-6}}}, baselines={'gigpo':{'gamma':.95}}, trainer={})
    new = json.loads(json.dumps(old));new['actor_rollout_ref']['actor']['ppo_mini_batch_size']=256
    new['trainer'].update(quality_resume={'enabled':True}, trajectory_logging=True)
    assert quality.migration_diff(old,new)
    for key in ('ppo_micro_batch_size', 'lr', 'gamma'):
        bad=json.loads(json.dumps(new))
        if key=='ppo_micro_batch_size':bad['actor_rollout_ref']['actor'][key]=32
        elif key=='lr':bad['actor_rollout_ref']['actor']['optim'][key]=1e-5
        else:bad['baselines']['gigpo'][key]=1
        with pytest.raises(ValueError):quality.migration_diff(old,bad)


def test_quality_warning_requires_full512_confirmation(tmp_path,monkeypatch):
    from search_r1.baselines import instruct300
    monkeypatch.setattr(quality,'selected_origin',lambda _:dict(step=50))
    monkeypatch.setattr(instruct300,'record_evaluation',lambda *args:None)
    trainer=SimpleNamespace(config=OmegaConf.create({'trainer': {'default_local_dir':str(tmp_path/'checkpoints')}, 'data': {'val_files':'validation.parquet'}}),
        quality_health=dict(baseline512=.3,baseline64=.3,recent_rewards=[0]*4),global_steps=55)
    saved=[];trainer._save_checkpoint=lambda:saved.append(True)
    trainer._validate=lambda:{'val/f1/all':.29}
    # Low training reward and a small64 fluctuation are warnings, not quality failure.
    quality.monitor_health(trainer,{'critic/rewards/mean':0},stage_metrics={'val/f1/all':0})
    assert not saved
    trainer.global_steps=60;trainer._validate=lambda:{'val/f1/all':0}
    with pytest.raises(RuntimeError,match='QUALITY_COLLAPSE'):
        quality.monitor_health(trainer,{'critic/rewards/mean':0},stage_metrics={'val/f1/all':0})
    assert saved == [True]


def test_declared_restore_path_must_be_selected_path(selection):
    from search_r1.baselines.instruct_resume import start_step
    from omegaconf import open_dict
    cfg, _, _ = selection
    origin=quality.selected_origin(cfg)
    with open_dict(cfg):
        cfg.trainer.instruct300=dict(restore_checkpoint='wrong-path',restore_manifest_sha256=origin['manifest'])
        cfg.trainer.acceptance_step=55;cfg.trainer.val_before_train=False
        cfg.trainer.resume=dict(enabled=False);cfg.trainer.total_training_steps=300
        cfg.actor_rollout_ref=dict(actor=dict(joint_loss_scaling=True))
    with pytest.raises(ValueError):start_step(cfg)
    cfg.trainer.instruct300.restore_checkpoint=origin['checkpoint']
    assert start_step(cfg)==50


def test_quality_yaml_resolves_numeric_gate_and_preserves_common_budget(monkeypatch):
    from hydra import compose, initialize_config_dir
    root=Path(__file__).resolve().parents[2]
    for key,value in dict(GIGPO_E1_ROOT='/tmp/root',GIGPO_E1_RUN='/tmp/run',GIGPO_QUALITY_GATE='30',
        GIGPO_QUALITY_CHECKPOINT='/tmp/checkpoint25',GIGPO_QUALITY_MANIFEST_SHA256='a'*64,
        GIGPO_QUALITY_SELECTION='/tmp/selection',GIGPO_QUALITY_SELECTION_SHA256='b'*64,
        GIGPO_QUALITY_REVIEW='/tmp/review',GIGPO_QUALITY_REVIEW_SHA256='c'*64).items():monkeypatch.setenv(key,value)
    with initialize_config_dir(config_dir=str(root/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='ppo_gigpo_instruct300_quality10')
        resolved=OmegaConf.to_container(cfg,resolve=True)
    assert resolved['trainer']['acceptance_step']==30 and isinstance(resolved['trainer']['acceptance_step'],int)
    assert resolved['actor_rollout_ref']['actor']['ppo_mini_batch_size']==256
    assert resolved['actor_rollout_ref']['actor']['ppo_micro_batch_size']==16
    assert resolved['actor_rollout_ref']['actor']['optim']['lr_warmup_steps']==57
    assert resolved['trainer']['total_training_steps']==300
