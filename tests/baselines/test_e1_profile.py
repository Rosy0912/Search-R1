import copy
import hashlib
import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.e1_profile import compose_profile,validate_profile,pinned_json
from search_r1.baselines.sharedholdout import unique_json

@pytest.fixture
def cfg(monkeypatch,tmp_path):
    values={'OPSD_E1_OFFICIAL_MODEL':str(tmp_path/'model'),'OPSD_E1_COMMON_MANIFEST':str(tmp_path/'manifest.json'),
      'OPSD_E1_COMMON_DATA_ROOT':str(tmp_path/'data'),'SEARCH_R1_BASELINE_RUN_ROOT':str(tmp_path/'run'),
      'OPSD_E1_REVIEW_PUBLIC_KEY':str(tmp_path/'public.json'),'OPSD_E1_REVIEW_PUBLIC_KEY_SHA256':'a'*64,
      'OPSD_E1_BINDING_MANIFEST':str(tmp_path/'binding.json'),'OPSD_E1_BINDING_MANIFEST_SHA256':'b'*64}
    for key,value in values.items():monkeypatch.setenv(key,value)
    return compose_profile()

def test_resolved_profile_actual_counts(cfg):
    d=validate_profile(cfg)
    actor=d['actor_rollout_ref']['actor'];roll=d['actor_rollout_ref']['rollout']
    assert 512*roll['n_agent']==2560
    assert 2560//actor['ppo_mini_batch_size']==20
    assert actor['ppo_micro_batch_size']//8==2
    assert actor['ppo_mini_batch_size']//8==16
    assert actor['ppo_mini_batch_size']//actor['ppo_micro_batch_size']==8
    assert d['baselines']['distillation']['teacher_path']==d['actor_rollout_ref']['model']['path']
    assert d['baselines']['bapo']['stage_validation_files'] is None

@pytest.mark.parametrize('key,value',[
 ('actor_rollout_ref.actor.grad_clip',0.5),('actor_rollout_ref.actor.clip_ratio',0.3),('actor_rollout_ref.actor.ppo_micro_batch_size',8),('actor_rollout_ref.actor.ppo_mini_batch_size',256),
 ('actor_rollout_ref.actor.optim.lr_warmup_steps',58),('trainer.total_training_steps',301),
 ('baselines.distillation.teacher_device','cpu'),('baselines.distillation.component_clip',.05),
 ('baselines.distillation.teacher_prompt_mode','legacy'),('actor_rollout_ref.actor.use_kl_loss',True),
 ('algorithm.kl_ctrl.kl_coef',.001),('actor_rollout_ref.rollout.n',5),
 ('baselines.local_runtime_only',False),('baselines.opsd_e1.gate_required',False),
 ('baselines.opsd_e1.stage_num',32),('data.val_data_num',64),
 ('retriever.url','http://foreign-cluster:8000/retrieve'),('baselines.distillation.teacher_path','/trained-hf5'),('baselines.opsd_e1.train_log_path','/foreign/train.log')])
def test_drift_refused(cfg,key,value):
    OmegaConf.update(cfg,key,value)
    with pytest.raises(ValueError):validate_profile(cfg)

def test_external_pin_and_duplicate_key(tmp_path):
    p=tmp_path/'binding.json';p.write_text('{"version":1}')
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert pinned_json(p,sha)=={'version':1}
    for bad in ['','a'*64]:
        with pytest.raises(ValueError):pinned_json(p,bad)
    p.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError,match='Duplicate'):pinned_json(p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_official_identity_manifest_is_original():
    p=Path(__file__).resolve().parents[3]/'e1-300/official-model-manifest-original.json'
    from search_r1.baselines.e1_profile import OFFICIAL_SHA,REVISION
    d=pinned_json(p,OFFICIAL_SHA)
    assert d['revision']==REVISION and len(d['files'])==11

def test_launcher_refuses_unknown_and_overrides_without_gpu(tmp_path):
    import subprocess
    script=Path(__file__).resolve().parents[2]/'reproduction/baselines/run_opsd_e1_300.sh'
    for args in [['--typo'],['--run','actor_rollout_ref.actor.ppo_micro_batch_size=8']]:
        result=subprocess.run(['bash',str(script)]+args,text=True,capture_output=True)
        assert result.returncode==2 and 'Only --cfg' in result.stderr

def test_boolean_integer_profile_drift_refused(cfg):
    cfg.baselines.opsd_e1.stage_num=True
    with pytest.raises(ValueError):validate_profile(cfg)


def test_official_pinned_manifest_metadata_only_and_marker_not_acceptance(tmp_path,monkeypatch):
    from search_r1.baselines import e1_profile
    root=tmp_path/'model';root.mkdir();records=[]
    for i in range(11):
        p=root/('official-'+str(i));p.write_bytes(bytes([i]))
        records.append(dict(file=p.name,bytes=1,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),revision=e1_profile.REVISION))
    mp=root/'VERIFIED-MODEL-MANIFEST.json'
    mp.write_text(json.dumps(dict(repo='Qwen/Qwen2.5-3B-Instruct',revision=e1_profile.REVISION,files=records)))
    monkeypatch.setattr(e1_profile,'OFFICIAL_SHA',hashlib.sha256(mp.read_bytes()).hexdigest())
    (root/'VERIFIED').write_text('self_reported_passed=true')
    assert e1_profile.verify_official_assets(root,mp)['files']==records
    extra=root/'arbitrary-metadata.json';extra.write_text('{}')
    with pytest.raises(ValueError,match='Unmanifested'):e1_profile.verify_official_assets(root,mp)
    extra.unlink();(root/'official-0').write_bytes(b'x')
    with pytest.raises(ValueError,match='file SHA'):e1_profile.verify_official_assets(root,mp)

def test_same_name_unpinned_metadata_not_allowed(tmp_path,monkeypatch):
    from search_r1.baselines import e1_profile
    root=tmp_path/'model';root.mkdir();records=[]
    for i in range(11):
        p=root/('official-'+str(i));p.write_bytes(b'0')
        records.append(dict(file=p.name,bytes=1,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),revision=e1_profile.REVISION))
    mp=tmp_path/'outside-official-manifest.json';mp.write_text(json.dumps(dict(repo='Qwen/Qwen2.5-3B-Instruct',revision=e1_profile.REVISION,files=records)))
    monkeypatch.setattr(e1_profile,'OFFICIAL_SHA',hashlib.sha256(mp.read_bytes()).hexdigest())
    (root/'VERIFIED-MODEL-MANIFEST.json').write_text('untrusted same name')
    with pytest.raises(ValueError,match='Unmanifested'):e1_profile.verify_official_assets(root,mp)
