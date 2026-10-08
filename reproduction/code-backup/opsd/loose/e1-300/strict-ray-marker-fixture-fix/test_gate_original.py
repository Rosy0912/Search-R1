"""Signed-consumer and real trainer-tail CPU tests; rank evidence is fixture-only."""
import ast
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import pytest
from search_r1.baselines import checkpoint_gate as gate
from search_r1.baselines.full_state import StatefulSampler, rng_state, state_digest


def metrics():
    values = {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0,
        'actor/samples': 320, 'actor/loss_mask_valid': 1, 'actor/opsd_loss_finite': 1,
        'actor/grad_norm_finite': 1, 'teacher/trainable_parameters_per_rank': 0,
        'teacher/gradient_tensors_per_rank': 0, 'teacher/student_optimizer_membership_per_rank': 0,
        'teacher/storage_is_cpu_per_rank': 1, 'teacher/all_parameter_dtype_is_fp32_per_rank': 1,
        'teacher/attention_is_eager_per_rank': 1, 'actor/allocator_is_native': 1,
        'actor/allocator_configuration_is_default': 1}
    return {key + '/rank_' + str(rank): value for rank in range(8) for key,value in values.items()}


@pytest.fixture
def rsa(tmp_path):
    private = tmp_path / 'external-review-private.pem'
    subprocess.run(['openssl','genrsa','-out',str(private),'2048'], check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    modulus = subprocess.check_output(['openssl','rsa','-in',str(private),'-noout','-modulus'],
                                     stderr=subprocess.PIPE,text=True).strip().split('=',1)[1]
    key = {'algorithm':'RSA-SHA256','n_hex':modulus,'e':65537}
    def sign(payload):
        # External CPU reviewer fixture: production contains no signing/private key.
        sig = subprocess.check_output(['openssl','dgst','-sha256','-sign',str(private)],
                                      input=gate.canonical(payload))
        return {'payload':payload,'signature_hex':sig.hex()}
    return key,sign


def test_actual_signature_and_changed_payload_rejected(rsa):
    key,sign = rsa
    payload = {'status':'PASS','driver_pid':345,'nonce':'fresh'}
    signed = sign(payload)
    gate.verify_rsa_signature(payload,signed['signature_hex'],key)
    with pytest.raises(ValueError):
        gate.verify_rsa_signature(dict(payload,driver_pid=346),signed['signature_hex'],key)
    with pytest.raises(ValueError):
        gate.verify_rsa_signature(payload,'00'*256,key)


@pytest.fixture
def signed_gate(tmp_path,rsa,monkeypatch):
    root = tmp_path / 'owned-run';root.mkdir()
    key,sign = rsa
    def record(name,data=b'evidence'):
        path=root/name;path.write_bytes(data)
        return {'path':str(path),'sha256':gate.sha256(path)}
    key_record = record('public-key.json',gate.canonical(key))
    g=gate.CheckpointGate.__new__(gate.CheckpointGate)
    g.root=root;g.pid=os.getpid();g.uid=os.getuid();g.job='cpu-fixture-job'
    g.source='new-source';g.config_sha='resolved-sha';g.public_key=key;g.key_record=key_record
    g.current_source=lambda:g.source
    monkeypatch.setenv('SLURM_JOB_ID',g.job)
    g.receipt=root/'release.json';g.settings=SimpleNamespace(train_log_path=str(root/'train.log'))
    files={'checkpoint':record('native-fixture.pt')}
    m={'version':1,'method':'pure_opsd','run_root':str(root),'source_commit':g.source,
       'config_sha256':g.config_sha,'global_step':5,'actual_outer_updates':5,'next_global_step':6,
       'horizon':300,'warmup_steps':57,'world_size':8,'driver_pid':g.pid,'uid':g.uid,
       'job_id':g.job,'pause_nonce':'fresh-unpredictable-fixture-nonce','files':files}
    path=root/'manifest.json';path.write_bytes(gate.canonical(m))
    rows=[dict(actual_outer_updates=i,global_step=i,driver_pid=g.pid,job_id=g.job,
          source_commit=g.source,config_sha256=g.config_sha,metrics=metrics()) for i in range(1,6)]
    text=''.join('OPSD_E1_OUTER_JSON='+json.dumps(row)+'\n' for row in rows).encode()
    snapshot=record('snapshot.log',text);record('train.log',text+b'later Ray output\n')
    payload={k:m[k] for k in ('method','run_root','source_commit','config_sha256','global_step',
         'actual_outer_updates','next_global_step','horizon','warmup_steps','world_size',
         'driver_pid','uid','job_id','pause_nonce')}
    payload.update(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',
        checkpoint_manifest_sha256=gate.sha256(path),checkpoint_files=files,
        independent_review=record('independent-review.json'),
        evidence={role:record(role+'.json') for role in ('resources','retrieval','hf_reload','rank_audit')},
        train_log_prefix={'live_path':g.settings.train_log_path,'snapshot_path':snapshot['path'],
                          'sha256':snapshot['sha256'],'bytes':len(text)})
    return g,path,m,payload,sign


def test_valid_release_and_changed_live_prefix(signed_gate):
    g,path,m,payload,sign=signed_gate;g.receipt.write_bytes(gate.canonical(sign(payload)))
    g.verify_receipt(g.receipt,path,m)
    live=Path(g.settings.train_log_path);raw=live.read_bytes();live.write_bytes(b'X'+raw[1:])
    with pytest.raises(ValueError,match='prefix changed'):g.verify_receipt(g.receipt,path,m)


@pytest.mark.parametrize('field,value',[('driver_pid',-1),('source_commit','old'),
    ('config_sha256','old'),('pause_nonce','stale'),('actual_outer_updates',6),
    ('job_id','other'),('horizon',301),('warmup_steps',85)])
def test_even_signed_wrong_identity_rejected(signed_gate,field,value):
    g,path,m,payload,sign=signed_gate;payload[field]=value
    g.receipt.write_bytes(gate.canonical(sign(payload)))
    with pytest.raises(ValueError,match='identity mismatch'):g.verify_receipt(g.receipt,path,m)


def test_changed_checkpoint_bytes_and_alias_rejected(signed_gate):
    g,path,m,payload,sign=signed_gate;g.receipt.write_bytes(gate.canonical(sign(payload)))
    Path(m['files']['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):g.verify_receipt(g.receipt,path,m)
    alias=g.root/'alias.json';alias.symlink_to(g.receipt)
    with pytest.raises(ValueError,match='physical'):g.verify_receipt(alias,path,m)


def test_no_receipt_blocks_before_sixth_fetch(signed_gate):
    g,path,m,payload,sign=signed_gate;calls=[];errors=[]
    def driver():
        try:
            calls.extend(range(1,6));g.wait(path,m);calls.append(6)
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=driver);thread.start();time.sleep(.1)
    assert calls==[1,2,3,4,5] and thread.is_alive()
    g.receipt.write_bytes(gate.canonical(sign(payload)));thread.join(4)
    assert not thread.is_alive() and not errors and calls==[1,2,3,4,5,6]


def test_driver_rng_sampler_disk_roundtrip_next_row(tmp_path):
    sampler=StatefulSampler(50,shuffle=True);iterator=iter(sampler)
    first=[next(iterator) for _ in range(5)];sampler.commit_consumed(5)
    state={'version':1,'method':'pure_opsd','global_step':5,'distillation':{}}
    def restore(actual,step):assert actual==state
    controller=SimpleNamespace(opsd_state_dict=lambda step:state.copy(),load_opsd_state_dict=restore)
    g=gate.CheckpointGate.__new__(gate.CheckpointGate);g.root=tmp_path
    g.trainer=SimpleNamespace(e1_sampler=sampler,baseline_controller=controller)
    g.pid=os.getpid();g.uid=os.getuid();g.job='fixture';g.source='source';g.config_sha='config'
    before_rng=state_digest(rng_state());before_sampler=state_digest(sampler.state_dict())
    expected=sampler.order[5];directory=g.save_driver(5,0,True)
    assert gate.load_json(directory/'driver.json')['status']=='EXACT_DRIVER_STATE_RELOAD_OK'
    assert state_digest(rng_state())==before_rng and state_digest(sampler.state_dict())==before_sampler
    assert next(iterator)==expected and expected not in first
    with pytest.raises(ValueError,match='Outstanding'):sampler.state_dict()


def test_twenty_nonempty_minis_required_and_save_points():
    good=metrics();gate.validate_outer(good)
    good['actor/optimizer_steps/rank_7']=19;good['actor/skipped_empty_minibatches/rank_7']=1
    with pytest.raises(RuntimeError):gate.validate_outer(good)
    assert [i for i in range(1,301) if gate.should_save(i)]==[5]+list(range(25,301,25))
    assert not gate.should_save(301)


def test_real_fit_tail_300_oracle():
    source=Path(__file__).resolve().parents[2]/'verl/trainer/ppo/ray_trainer.py'
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RayPPOTrainer')
    fit=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='fit')
    epoch=next(x for x in fit.body if isinstance(x,ast.For));batch=next(x for x in epoch.body if isinstance(x,ast.For))
    start=next(i for i,x in enumerate(batch.body) if isinstance(x,ast.If) and isinstance(x.test,ast.Name)
        and x.test.id=='e1' and any(isinstance(y,ast.Constant) and y.value=='OPSD_E1_OUTER_JSON=' for y in ast.walk(x)))
    fn=ast.FunctionDef(name='tail',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in
         ('self','metrics','e1','logger')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=batch.body[start:],decorator_list=[])
    namespace=dict(os=os,json=json,validate_evaluation=gate.validate_evaluation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
    pauses=[];finals=[];logs=[];logger=SimpleNamespace(log=lambda data,step:logs.append(step))
    g=SimpleNamespace(global_steps=1,actual_outer_updates=0,total_training_steps=300,
        e1_gate=SimpleNamespace(job='fixture',source='source',config_sha='config',wait=lambda *args:pauses.append(g.global_steps)),
        e1_pause_manifest=('path',{}),val_reward_fn=object(),_prepare_final_validation=lambda:finals.append(g.global_steps),
        _validate=lambda:{'val/qa_count/all':51713,'val/evaluated_count/all':51713})
    for consumed in range(1,301):
        assert g.global_steps==consumed;g.actual_outer_updates+=1
        namespace['tail'](g,{},True,logger)
    assert pauses==[5] and finals==[300] and logs==[300]
    assert g.global_steps==300 and g.actual_outer_updates==300


def test_manifest_requires_eight_distinct_shards_hf_and_real_roundtrip(tmp_path):
    g=gate.CheckpointGate.__new__(gate.CheckpointGate)
    g.root=tmp_path;g.pid=os.getpid();g.uid=os.getuid();g.job='fixture'
    g.source='source';g.current_source=lambda:g.source;g.config={};g.config_sha='config'
    def record(name,data=b'bytes'):
        path=tmp_path/name;path.write_bytes(data)
        return {'path':str(path),'sha256':gate.sha256(path)}
    g.bindings={'launch':record('launch.json')};g.binding_record=record('binding.json');g.key_record=record('public-key.json')
    local=tmp_path/'output';actor=local/'actor/global_step_5';native=actor/'training_state';native.mkdir(parents=True)
    for name in ('config.json','tokenizer.json','tokenizer_config.json','model-00001.safetensors'):
        (actor/name).write_bytes(b'fixture, not model contents')
    (actor/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model-00001.safetensors'}}))
    for rank in range(8):
        path=native/f'rank_{rank}.pt';path.write_bytes(('rank'+str(rank)).encode())
        report=dict(version=3,algorithm='pure_opsd',status='EXACT_FULL_STATE_RELOAD_OK',rank=rank,
          world_size=8,step=5,path=str(path),state_sha256=gate.sha256(path),model_state_dict_type='SHARDED_STATE_DICT',
          model_dtensor_entries=1,optimizer_state_entries=1,scheduler_last_epoch=5,pid=1000+rank)
        (native/f'rank_{rank}.json').write_text(json.dumps(report))
    driver=tmp_path/'full_state/global_step_5';driver.mkdir(parents=True)
    (driver/'driver.pt').write_bytes(b'fixture-driver')
    (driver/'driver.json').write_text(json.dumps(dict(status='EXACT_DRIVER_STATE_RELOAD_OK',step=5,
        pid=g.pid,state_sha256=gate.sha256(driver/'driver.pt'))))
    algorithm=local/'algorithm/global_step_5';algorithm.mkdir(parents=True);(algorithm/'state.json').write_text('{}')
    g.trainer=SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(local))))
    path,m=g.manifest(5,driver,actor,metrics())
    assert 'actor/training_state/rank_7.pt' in m['files'] and 'algorithm' in m['files']
    report=gate.load_json(native/'rank_7.json');report['pid']=1000
    (native/'rank_7.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='alias worker'):g.manifest(5,driver,actor,metrics())
    report['pid']=1007;report['status']='FULL_STATE_SERIALIZED_OK'
    (native/'rank_7.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='roundtrip is incomplete'):g.manifest(5,driver,actor,metrics())


def test_duplicate_json_nonfinite_and_path_escape_rejected(tmp_path):
    path=tmp_path/'bad.json';path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='Duplicate'):gate.load_json(path)
    path.write_text('{"x":NaN}')
    with pytest.raises(ValueError,match='Nonfinite'):gate.load_json(path)
    root=tmp_path/'owned';root.mkdir()
    with pytest.raises(ValueError):gate.owned_file(path,root)


def test_import_closure_rejects_live_function_patch_even_same_file_sha(tmp_path, monkeypatch):
    module_path=Path(gate.__file__).resolve();repo=module_path.parents[2]
    closure={'modules':{'search_r1.baselines.checkpoint_gate':{
        'path':str(module_path),'sha256':gate.sha256(module_path)}}}
    path=tmp_path/'imports.json';path.write_bytes(gate.canonical(closure))
    record={'path':str(path),'sha256':gate.sha256(path)}
    required={'search_r1.baselines.checkpoint_gate':('CheckpointGate.verify_receipt',)}
    gate.verify_import_closure(record,tmp_path,repo,required=required)
    monkeypatch.setattr(gate.CheckpointGate,'verify_receipt',lambda *args: {'forged':'pass'})
    with pytest.raises(ValueError,match='implementation differs'):
        gate.verify_import_closure(record,tmp_path,repo,required=required)


def test_explicit_signed_driver_recovery_restores_next_row_rng_and_rejects_wrong_destination(signed_gate):
    import torch
    g,_,_,_,sign=signed_gate
    source=StatefulSampler(50,shuffle=True);iterator=iter(source)
    for _ in range(5):next(iterator)
    source.commit_consumed(5)
    algorithm={'version':1,'method':'pure_opsd','step':5}
    controller=SimpleNamespace(opsd_state_dict=lambda step:algorithm.copy(),
        load_opsd_state_dict=lambda value,step:None)
    g.trainer=SimpleNamespace(e1_sampler=source,baseline_controller=controller)
    directory=g.save_driver(5,0,False)
    state_path=directory/'driver.pt';raw=torch.load(state_path,weights_only=False)
    record={'path':str(state_path),'sha256':gate.sha256(state_path)}
    authorization={'version':1,'authorization':'EXPLICIT_OPSD_E1_DRIVER_RECOVERY',
        'checkpoint':record,'destination_pid':g.pid,'destination_uid':g.uid,
        'destination_job_id':g.job,'source_commit':g.source,'config_sha256':g.config_sha,
        'source_pid':raw['pid'],'source_job_id':raw['job_id'],
        'state_digest':state_digest(raw),'step':5}
    path=g.root/'explicit-recovery.json'
    authorization['destination_pid']=-1;path.write_bytes(gate.canonical(sign(authorization)))
    auth_record={'path':str(path),'sha256':gate.sha256(path)}
    with pytest.raises(ValueError,match='destination'):g.restore_driver(record,auth_record)
    authorization['destination_pid']=g.pid;path.write_bytes(gate.canonical(sign(authorization)))
    auth_record['sha256']=gate.sha256(path)
    g.trainer.e1_sampler=StatefulSampler(50,shuffle=True);torch.rand(10)
    result=g.restore_driver(record,auth_record)
    assert result['next_global_step']==6 and result['actual_outer_updates']==5
    assert result['worker_recovery_verified'] is False
    assert next(iter(g.trainer.e1_sampler))==source.order[5]
    assert state_digest(rng_state())==state_digest(raw['rng'])
    # A first-five receipt is not a restart authorization.
    path.write_bytes(gate.canonical(sign({'status':'PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'})))
    auth_record['sha256']=gate.sha256(path)
    with pytest.raises(ValueError,match='schema'):g.restore_driver(record,auth_record)


def test_actual_all_seven_module_import_closure(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    path=tmp_path/'all-seven-modules.json'
    gate.atomic_json(path,{'modules':{name:{'path':str(repo/(name.replace('.','/')+'.py')),
        'sha256':gate.sha256(repo/(name.replace('.','/')+'.py'))} for name in gate.IMPORT_OBJECTS}})
    gate.verify_import_closure({'path':str(path),'sha256':gate.sha256(path)},tmp_path,repo)


def test_full_checker_uses_exact_external_bytes_and_rejects_changed_source(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'reproduction.baselines.e1_import_bootstrap',None)
    repo=tmp_path/'repo';helper=repo/'reproduction/baselines/e1_import_bootstrap.py'
    helper.parent.mkdir(parents=True)
    helper.write_text("def verify_imports(repo, closure, required):\n assert required == ()\n assert closure['modules']\n")
    path=tmp_path/'externally-reviewed-closure.json'
    gate.atomic_json(path,{'modules':{'reproduction.baselines.e1_import_bootstrap':
        {'path':str(helper),'sha256':gate.sha256(helper)}}})
    record={'path':str(path),'sha256':gate.sha256(path)}
    result=gate.full_import_checker(record,tmp_path,repo)
    assert result['checker_path']==str(helper) and result['closure_sha256']==record['sha256']
    helper.write_text(helper.read_text()+'# changed bytes\n')
    with pytest.raises(ValueError,match='source changed'):gate.full_import_checker(record,tmp_path,repo)
    helper.unlink();helper.symlink_to(path)
    with pytest.raises(ValueError,match='path/schema'):gate.full_import_checker(record,tmp_path,repo)


def test_actual_init_workers_propagates_checker_before_resource_pool(monkeypatch):
    from omegaconf import OmegaConf
    source=Path(__file__).resolve().parents[2]/'verl/trainer/ppo/ray_trainer.py'
    cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='RayPPOTrainer')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='init_workers')
    # Execute the actual prefix through pool creation, without starting Ray.
    stop=next(i for i,x in enumerate(method.body) if isinstance(x,ast.Expr) and
        isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and
        x.value.func.attr=='create_resource_pool')
    method.body=method.body[:stop+1]
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
    pins={'checker_path':'/owned/repo/checker.py','checker_sha256':'external-sha',
          'closure_path':'/owned/run/closure.json','closure_sha256':'external-closure-sha'}
    monkeypatch.setattr(gate,'worker_import_checker_config',lambda config:pins.copy())
    cfg=OmegaConf.create({'baselines':{'opsd_e1':{'enabled':True}},'actor_rollout_ref':{'actor':{}}})
    seen=[];obj=SimpleNamespace(config=cfg,resource_pool_manager=SimpleNamespace(
        create_resource_pool=lambda:seen.append(OmegaConf.to_container(cfg.actor_rollout_ref.actor.e1_import_checker))))
    namespace['init_workers'](obj)
    assert seen==[pins]
