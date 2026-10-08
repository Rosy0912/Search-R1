"""Small local CPU fixtures only. Never claims actual3304/200/GPU acceptance."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import types
import pytest
import torch

HERE=Path(__file__).resolve().parent


def module(name):
    p=HERE/(name+'.py');spec=importlib.util.spec_from_file_location('cpu_fixture_'+name,p)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


C=module('common');N=module('inspect_final200_native');HF=module('load_final200_hf');A=module('check_final200_archives')
SOURCE=HERE.parent.parent/'opsd-unclipped-multitraj200-teacherrecompute'


@pytest.fixture(scope='module')
def helper():
    value,_=C.exact_module(HERE.parent/'first5-native-dc1905562f2f/inspect_payloads.py',C.MATH_HELPER,'fixture_unchanged_storage_math')
    return value


def rank(rank=0):
    return dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=rank,world_size=8,step=200,status='FULL_STATE_SERIALIZED_OK',scheduler_last_epoch=200,optimizer_state_entries=37,model_dtensor_entries=435,pid=100+rank,object_ids=dict(model=1,optimizer=2,scheduler=3,teacher=4))


def inventory():
    return dict(version=1,kind=C.KIND,source_commit=C.SOURCE,config_sha256=C.CONFIG,job_id='3304',run_root=str(C.RUN),source_root=str(C.REPO),profile='multitraj200',checkpoint_step=200,horizon=200,warmup_steps=57,world_size=8,
        checkpoint_files={key:{} for key in C.checkpoint_paths()},archive_files={key:{} for key in C.archive_paths()},rank_reports=[rank(i) for i in range(8)],driver_report=dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,job_id='3304',uid=os.getuid()),raw_statuses=dict(ranks=['FULL_STATE_SERIALIZED_OK']*8,driver='DRIVER_STATE_SERIALIZED_OK'),accepted5={})


def test_inventory_exact_counters_and_members():
    x=inventory();assert C.validate_inventory(x)==x
    assert len(x['checkpoint_files'])==31 and len(x['archive_files'])==4000 and C.clock()['adam_steps']==2000


@pytest.mark.parametrize('key,value',[('checkpoint_step',5),('checkpoint_step',200.0),('version',True),('profile','author200'),('horizon',300),('job_id','3261'),('config_sha256','0'*64),('source_commit','188f'+'0'*36)])
def test_inventory_scope_drift_rejected(key,value):
    x=inventory();x[key]=value
    with pytest.raises(ValueError):C.validate_inventory(x)


@pytest.mark.parametrize('target',['checkpoint_files','archive_files'])
def test_missing_member_rejected(target):
    x=inventory();x[target].pop(next(iter(x[target])))
    with pytest.raises(ValueError):C.validate_inventory(x)


@pytest.mark.parametrize('key,value',[('rank',False),('step',200.0),('world_size',8.0),('pid',True),('optimizer_state_entries',37.0),('status','EXACT_FULL_STATE_RELOAD_OK')])
def test_rank_serialized_typed_negatives(key,value):
    r=rank();r[key]=value
    with pytest.raises(ValueError):C.rank_report(r,0,200,'FULL_STATE_SERIALIZED_OK')


def test_same_fd_bytes_and_no_atime_false_alarm(tmp_path):
    p=tmp_path/'actual.json';p.write_bytes(b'{"x":1}\n');obj,rec=C.read_json(p)
    assert obj=={'x':1} and C.verify_record(rec)==p
    os.utime(p,ns=(p.stat().st_atime_ns+10000,p.stat().st_mtime_ns))
    # Explicitly changing atime also changes ctime, so it must still be rejected;
    # ordinary repeated reads only change atime and preserve accepted identity.
    with pytest.raises(ValueError):C.verify_record(rec)


def test_symlink_hardlink_member_alias_and_wrong_pin(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'a');q=tmp_path/'q';q.symlink_to(p)
    with pytest.raises(ValueError):C.read_file(q)
    with pytest.raises(ValueError):C.read_file(p,'0'*64)
    raw,rec=C.read_file(p,keep_bytes=True);assert raw==b'a'
    rec['uid']=not rec['uid']
    with pytest.raises(ValueError):C.verify_record(rec)


def test_replace_or_rewrite_during_read_rejected(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_bytes(b'original');real=C.os.read;changed=False
    def read(fd,n):
        nonlocal changed
        value=real(fd,n)
        if value and not changed:
            changed=True;p.write_bytes(b'modified');p.write_bytes(b'original')
        return value
    monkeypatch.setattr(C.os,'read',read)
    with pytest.raises(ValueError,match='changed'):C.read_file(p,keep_bytes=True)


def test_duplicate_nonfinite_json_rejected():
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
        with pytest.raises(ValueError):C.strict_json(raw)


def test_compile_same_bytes_ignores_fake_pyc(tmp_path):
    p=tmp_path/'v.py';p.write_text('VALUE=42\n');(tmp_path/'__pycache__').mkdir();(tmp_path/'__pycache__/v.fake.pyc').write_bytes(b'not source')
    value,rec=C.exact_module(p,hashlib.sha256(p.read_bytes()).hexdigest(),'fixture_exact_source')
    assert value.VALUE==42
    p.write_text('raise RuntimeError("should not execute")\n')
    with pytest.raises(ValueError):C.exact_module(p,rec['sha256'],'fixture_exact_source_wrong_pin')


def test_exclusive_result_not_overwritten(tmp_path):
    p=tmp_path/'result.json';rec=C.write_exclusive(p,{'scope':'SYNTHETIC_CPU_FIXTURE_ONLY'})
    assert C.read_json(p,rec['sha256'])[0]['scope']=='SYNTHETIC_CPU_FIXTURE_ONLY'
    with pytest.raises(FileExistsError):C.write_exclusive(p,{'scope':'changed'})


def identity_args():
    stat='10 (fixture worker) '+' '.join(['S']+['0']*18+['77']+['0']*8)
    return dict(job='901',step='3',platform='linux',node='h100-10',env=dict(SLURM_JOB_ID='901',SLURM_STEP_ID='3',CUDA_VISIBLE_DEVICES=''),uid=0,pid=10,stat_raw=stat,status='Uid:\t0 0 0 0\n',cgroup='0::/slurm/uid_0/job_901/step_3/task_0\n')


def test_new_cpu_identity_separate_from_stored_training():
    value=C.parse_execution_identity(**identity_args());assert value['execution_job_id']=='901' and value['stored_training_job_id']=='3304' and value['start_ticks']==77


@pytest.mark.parametrize('key,value',[('platform','darwin'),('node','h100-11'),('job','90'),('step','4'),('step','../3'),('status','Uid: 0 0 0\n'),('cgroup','0::/slurm/job_9010/step_3\n'),('cgroup','0::/slurm/job_901/step_4\n')])
def test_cpu_identity_negatives(key,value):
    x=identity_args();x[key]=value
    with pytest.raises(ValueError):C.parse_execution_identity(**x)


def test_missing_environment_cannot_fill_training_job():
    x=identity_args();x['env']={}
    with pytest.raises(ValueError):C.parse_execution_identity(**x)


def opt():
    return dict(param_groups=[dict(lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False,maximize=False,params=[0])],state={0:dict(step=torch.tensor(2000.),exp_avg=torch.ones(8),exp_avg_sq=torch.ones(8))})


def test_adam200_real_cpu_moments_not_old_clocks(helper):
    value=N.adam200(opt(),helper,C,8);assert value['steps']==[2000] and len(value['entries'])==1


@pytest.mark.parametrize('part',['clock','lr','negative_moment','nonfinite_moment','boolid','local_numel'])
def test_adam200_negatives(helper,part):
    x=opt();numel=8
    if part=='clock':x['state'][0]['step']=torch.tensor(50.)
    if part=='lr':x['param_groups'][0]['lr']=1e-6*5/57
    if part=='negative_moment':x['state'][0]['exp_avg_sq'][0]=-1
    if part=='nonfinite_moment':x['state'][0]['exp_avg'][0]=float('nan')
    if part=='boolid':x['param_groups'][0]['params']=[False]
    if part=='local_numel':numel=16
    with pytest.raises(ValueError):N.adam200(x,helper,C,numel)


def schedule():return dict(last_epoch=200,_step_count=201,base_lrs=[1e-6],_last_lr=[1e-6],lr_lambdas=[None])


def test_scheduler200_true_clock():assert N.scheduler200(schedule(),C)['last_epoch']==200


@pytest.mark.parametrize('key,value',[('last_epoch',200.0),('_step_count',200),('_last_lr',[5e-6]),('base_lrs',[0]),('lr_lambdas',[{}])])
def test_scheduler200_negatives(key,value):
    x=schedule();x[key]=value
    with pytest.raises(ValueError):N.scheduler200(x,C)


def test_actual_frozen_sampler_writer_and_parameterized_cursor(helper):
    state,_=C.exact_module(SOURCE/'search_r1/baselines/full_state.py',C.FULL_STATE,'fixture_real_sampler_source')
    sampler=state.StatefulSampler(169028,shuffle=True);iterator=iter(sampler)
    for _ in range(102400):next(iterator)
    sampler.commit_consumed(102400);raw=sampler.state_dict()
    before=torch.get_rng_state().clone();result=helper.sampler_content(raw,dict(data=dict(shuffle_train_dataloader=True)),state,size=169028,cursor=102400)
    assert result['seed'] is None and result['cursor']==102400 and torch.equal(torch.get_rng_state(),before)
    for field,value in [('seed',42),('cursor',102400.),('order',[0]*169028)]:
        bad=copy.deepcopy(raw);bad[field]=value
        with pytest.raises(ValueError):helper.sampler_content(bad,dict(data=dict(shuffle_train_dataloader=True)),state,size=169028,cursor=102400)


def test_actual_receipt_writer_has_no_payload_version():
    writerpath=HERE.parent/'first5-1b16c2227caf/receipt_payload.py';writer,_=C.exact_module(writerpath,hashlib.sha256(writerpath.read_bytes()).hexdigest(),'fixture_actual_frozen_receipt_writer')
    manifest=dict(version=1,method='pure_opsd',run_root=str(C.RUN),source_commit=C.SOURCE,config_sha256=C.CONFIG,global_step=5,actual_outer_updates=5,next_global_step=6,horizon=200,warmup_steps=57,world_size=8,job_id='3304',uid=os.getuid(),driver_pid=100,pause_nonce='0'*64,files={})
    payload={key:manifest[key] for key in writer.IDENTITY};payload.update(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',checkpoint_manifest_sha256=C.ACCEPTED5,checkpoint_files={})
    assert 'version' not in payload;C.validate_accepted_headers(manifest,payload)
    bad=copy.deepcopy(payload);bad['global_step']=5.0
    with pytest.raises(ValueError):C.validate_accepted_headers(manifest,bad)


def tiny_export(tmp_path,helper):
    tensor=torch.arange(64,dtype=torch.float32).reshape(16,4);data=tensor.numpy().tobytes();header=json.dumps({'tiny.weight':dict(dtype='F32',shape=[16,4],data_offsets=[0,len(data)])}).encode();name='model-00001-of-00001.safetensors'
    (tmp_path/name).write_bytes(struct.pack('<Q',len(header))+header+data);(tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'tiny.weight':name}}))
    paths={'actor/model.safetensors.index.json':tmp_path/'model.safetensors.index.json','actor/'+name:tmp_path/name}
    return helper.Export(paths,{'tiny.weight'}),tensor


def test_unchanged_hf_slice_reader_actual_tiny_binary(tmp_path,helper):
    export,tensor=tiny_export(tmp_path,helper)
    for rank in range(8):assert export.equal_slice('tiny.weight',rank,tensor[rank*2:(rank+1)*2])
    with pytest.raises(ValueError):export.equal_slice('tiny.weight',0,tensor[:2]+1)


def test_native_hf_typed_clock_alias_rejected():
    x=inventory();r=dict(version=1,scope=N.SCOPE,passed=True,source_commit=C.SOURCE,config_sha256=C.CONFIG,job_id='3304',run_root=str(C.RUN),checkpoint_step=200,inventory_sha256='0'*64,inspector_source_sha256='1'*64,cuda_is_initialized=False,hf_all_rank_slices_exact=True,clock=C.clock(),accepted5={},original_raw_statuses=x['raw_statuses'],ranks=[])
    r['clock']['epoch']=False
    with pytest.raises(ValueError,match='clocks'):HF.validate_native(r,x,'0'*64,'1'*64,C)


def test_original_common_parquet_row_join_strict():
    archive=types.SimpleNamespace(plain=lambda x:x,E1_INSTRUCTION='fixture instruction')
    row=dict(prompt=[dict(role='user',content='fixture instruction\nQuestion: hello?')],extra_info=dict(index='nq:train:3'),data_source='nq',reward_model=dict(ground_truth=dict(target=['yes'])))
    q=dict(dataset_row_position=9,question_position=0,question_uid='nq:train:3',data_source='nq',original_prompt=row['prompt'],accepted_answers=['yes'],accepted_answers_provenance='common_training_reward_model.ground_truth.target',question='hello?')
    A.validate_question_row(q,row,0,9,archive,C)
    for field,value in [('dataset_row_position',9.0),('question_position',False),('accepted_answers',['leaked']),('question_uid','nq:train:4'),('question','different')]:
        bad=copy.deepcopy(q);bad[field]=value
        with pytest.raises(ValueError):A.validate_question_row(bad,row,0,9,archive,C)


def test_all_entrypoints_use_exact_common_source():
    pin=hashlib.sha256((HERE/'common.py').read_bytes()).hexdigest()
    for name in ('observe_final200','inspect_final200_native','load_final200_hf','check_final200_archives'):
        entry=module(name);loaded=entry.load_common(pin);assert loaded.SOURCE==C.SOURCE
        with pytest.raises(ValueError):entry.load_common('0'*64)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('cgroup',[
    '1:freezer:/slurm/uid_0/job_901/step_4/task_0\n0::/slurm/uid_0/job_902/step_3/task_0\n',
    '0::/slurm/uid_0/job_901/step_4/task_0/job_902/step_3/task_0\n',
    '0::/slurm/uid_0/job_9010/step_3/task_0\n',
    '0::/slurm/uid_0/job_901/step_30/task_0\n',
    '0::/slurm/uid_0/job_901\n/step_3/task_0\n',
])
def test_execution_job_step_cannot_join_different_paths(cgroup):
    args=identity_args();args['cgroup']=cgroup
    with pytest.raises(ValueError,match='cgroup'):C.parse_execution_identity(**args)


def test_actual_freezer_style_cgroup_is_valid():
    args=identity_args();args['cgroup']='1:freezer:/slurm_h100-10/uid_0/job_901/step_3\n0::/system.slice/slurmd.service\n'
    assert C.parse_execution_identity(**args)['execution_step_id']=='3'


@pytest.fixture
def teacher_fixture(helper,tmp_path):
    state,_=C.exact_module(SOURCE/'search_r1/baselines/full_state.py',C.FULL_STATE,'fixture_real_teacher_state_digest')
    config=dict(model_type='qwen2',hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,vocab_size=16,tie_word_embeddings=True)
    base=types.SimpleNamespace(directory=tmp_path,config=config)
    buffers={'rotary_emb.inv_freq':torch.arange(4,dtype=torch.float32)}
    value=dict(version=1,parameter_count=C.NUMEL,storage_device='cpu',model_content_digest='b'*64,execution_device='cuda:0',asset_directory=str(tmp_path),asset_files={'fixture-only':'a'*64},counters={k:0 for k in helper.COUNTERS},model_config=config,caches={'rotary_emb':dict(max_seq_len_cached=8192,original_max_seq_len=32768)},buffers=buffers,buffer_content_digest=state.state_digest(buffers),model_bytes=C.NUMEL*4+16)
    value['counters'].update(cuda_update_entries=200,scored_tokens=32)
    return value,base,state


def call_teacher_fixture(value,base,state,helper):
    report=helper.teacher_report_projection({k:v for k,v in value.items() if k!='buffers'})
    return N.teacher200(value,report,base,value['asset_files'],'b'*64,helper,state,C)


def test_teacher200_history_buffers_have_real_digest(helper,teacher_fixture):
    value,base,state=teacher_fixture;result=call_teacher_fixture(value,base,state,helper)
    assert result['cuda_update_entries']==200 and result['buffer_content_digest']==state.state_digest(value['buffers'])


@pytest.mark.parametrize('mutation',['clock_bool','negative_cache','cache_float','unknown_cache','nonfinite_buffer','buffer_digest','model_bytes_bool','parameter_count_float','execution_cpu'])
def test_teacher200_cache_content_and_typed_negative(helper,teacher_fixture,mutation):
    value,base,state=teacher_fixture
    if mutation=='clock_bool':value['counters']['cuda_update_entries']=True
    if mutation=='negative_cache':value['caches']['rotary_emb']['max_seq_len_cached']=-1
    if mutation=='cache_float':value['caches']['rotary_emb']['max_seq_len_cached']=8192.0
    if mutation=='unknown_cache':value['caches']['rotary_emb']['untracked']=0
    if mutation=='nonfinite_buffer':value['buffers']['rotary_emb.inv_freq'][0]=float('nan')
    if mutation=='buffer_digest':value['buffer_content_digest']='0'*64
    if mutation=='model_bytes_bool':value['model_bytes']=True
    if mutation=='parameter_count_float':value['parameter_count']=float(C.NUMEL)
    if mutation=='execution_cpu':value['execution_device']='cpu'
    with pytest.raises(ValueError):call_teacher_fixture(value,base,state,helper)


def test_actual_archive_writer_six_fields_join200(tmp_path,monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(SOURCE))
    archive=importlib.import_module('search_r1.baselines.trajectory_archive')
    assert Path(archive.__file__).resolve()==SOURCE/'search_r1/baselines/trajectory_archive.py'
    p=tmp_path/'paired-manifest.json';p.write_bytes(b'{"scope":"synthetic local writer-consumer fixture only"}\n')
    rec=C.read_file(p)[1]
    writer=object.__new__(archive.TrajectoryArchive);writer.completed_outer=200
    writer.last_manifest=dict(path=str(p),sha256=rec['sha256'],bytes=rec['bytes'])
    raw=writer.state_dict(200)
    inventory={'archive_files':{'trajectory_outer_000200/paired-manifest.json':rec}}
    monkeypatch.setattr(C,'RUN',tmp_path)
    assert N.archive_progress200(dict(trajectory_archive=raw),inventory,C)==raw
    for mutation in ('scope','scope_missing','extra','clock_float'):
        bad=copy.deepcopy(raw)
        if mutation=='scope':bad['scope']='EXACT_RELOAD'
        if mutation=='scope_missing':bad.pop('scope')
        if mutation=='extra':bad['invented']=True
        if mutation=='clock_float':bad['completed_outer']=200.0
        with pytest.raises(ValueError):N.archive_progress200(dict(trajectory_archive=bad),inventory,C)
