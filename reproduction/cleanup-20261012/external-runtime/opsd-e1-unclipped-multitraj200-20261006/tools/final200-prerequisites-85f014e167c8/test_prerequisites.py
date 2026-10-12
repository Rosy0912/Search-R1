"""Local synthetic CPU semantic fixtures; never signs or qualifies actual200.

Production main, historical accepted_context and invocation are not patched.
Tests of the complete 200 report schema use explicitly synthetic in-memory
objects. Physical byte-race tests use ordinary temporary CPU files only.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import types
import pytest
HERE=Path(__file__).resolve().parent
H=HERE.parent


def load(path,name):
    raw=path.read_bytes();value=types.ModuleType(name);value.__file__=str(path)
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),value.__dict__)
    return value


V=load(HERE/'verify_actual200_prerequisites.py','fixture_prerequisite')
C=load(HERE/'common.py','fixture_common')
HF=load(H/'final200-candidate/load_final200_hf.py','fixture_unchanged_hf')
assert hashlib.sha256((HERE/'common.py').read_bytes()).hexdigest()==V.PINS['common.py']
assert hashlib.sha256((H/'final200-candidate/load_final200_hf.py').read_bytes()).hexdigest()==V.PINS['load_final200_hf.py']


def rec(path='/fixture/explicit-synthetic-only',sha='a'*64,inode=1):
    return dict(physical_path=path,sha256=sha,bytes=1,device=1,inode=inode,uid=os.getuid(),mtime_ns=1,ctime_ns=1)


def eid(pid=900):
    return dict(node='h100-10',pid=pid,uid=0,uids=[0]*4,start_ticks=10000+pid,cgroup='1:freezer:/slurm_h100-10/uid_0/job_901/step_3\n',
      execution_job_id='901',execution_step_id='3',stored_training_job_id='3304',python='/fixture/native-python',python_version='3.10.0 fixture')


def context():
    return dict(accepted5={},driver_identity=dict(pid=10,start='100'),service_identity=dict(pid=11,start='101'),
      workers=[dict(historical_process=dict(pid=20+i,start=str(200+i))) for i in range(8)],records={})


def rank(i):
    return dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=i,world_size=8,step=200,status='FULL_STATE_SERIALIZED_OK',scheduler_last_epoch=200,optimizer_state_entries=37,model_dtensor_entries=435,pid=20+i,object_ids=dict(model=1,optimizer=2,scheduler=3,teacher=4))


def inventory():
    roles=list(C.checkpoint_paths().items())+list(C.archive_paths().items())
    files={k:rec(str(p),inode=i+1) for i,(k,p) in enumerate(roles)}
    return dict(version=1,kind=C.KIND,source_commit=C.SOURCE,config_sha256=C.CONFIG,job_id='3304',run_root=str(C.RUN),source_root=str(C.REPO),profile='multitraj200',checkpoint_step=200,horizon=200,warmup_steps=57,world_size=8,
      checkpoint_files={k:files[k] for k in C.checkpoint_paths()},archive_files={k:files[k] for k in C.archive_paths()},rank_reports=[rank(i) for i in range(8)],
      driver_report=dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,job_id='3304',uid=os.getuid()),raw_statuses=dict(ranks=['FULL_STATE_SERIALIZED_OK']*8,driver='DRIVER_STATE_SERIALIZED_OK'),accepted5={})


def payload():
    # The validator supplies the exact expected bindings, but the fixture remains
    # unsigned and cannot run main or emit actual production eligibility.
    b=dict(training_commit=V.SOURCE,config_sha256=V.CONFIG,stored_training_job_id='3304',run_root=str(V.RUN),actual_outer_updates=200,hf_step=200,horizon=200,optimizer_steps_per_rank=2000,world_size=8,
      accepted5_receipt_sha256=V.RECEIPT5,common_manifest_sha256=V.COMMON,final_parquet_sha256=V.FINAL,final_question_ids_sha256=V.IDS,final_questions=51713,final_source_counts=V.COUNTS,official_revision=V.REVISION,
      official_model_manifest_sha256=V.MODEL,official_base=str(V.BASE),actual_hf200=str(V.HF200),evaluation_budget=copy.deepcopy(V.BUDGET),evaluation_budget_sha256=hashlib.sha256(C.canonical(V.BUDGET)).hexdigest())
    return dict(version=1,kind=V.KIND,status=V.ELIGIBLE,bindings=b,tools={k:rec('/fixture/tools/'+k,v) for k,v in V.PINS.items()}|{'verify_actual200_prerequisites.py':rec('/fixture/verifier.py')},
       reports={k:rec('/fixture/'+k+'.json') for k in V.ROLES},independent_qualification=rec('/fixture/review.json'),literal_terminal={},cpu_executions={k:eid(900+i) for i,k in enumerate(('native','hf','archives'))},
       eligibility=dict(stored_checkpoint=True,paired_raw_eval_may_launch=True,evaluation_allocation='REQUIRES_NEW_OWNED_SAME_H10010_ALLOCATION',same_receipt_for_both_roles=True),effect_completion=None)


def reports(p):
    inv=inventory();out={'inventory':inv}
    for role in ('native','hf','archives'):
        src={'native':'inspect_final200_native.py','hf':'load_final200_hf.py','archives':'check_final200_archives.py'}[role]
        out[role]=dict(version=1,scope=V.SCOPES[role],passed=True,source_commit=V.SOURCE,config_sha256=V.CONFIG,job_id='3304',run_root=str(V.RUN),checkpoint_step=200,inventory_sha256=p['reports']['inventory']['sha256'],cuda_is_initialized=False,
          common_source_sha256=V.PINS['common.py'],inspector_source_sha256=V.PINS[src],accepted5={},execution_identity=p['cpu_executions'][role],verified_input_records={r['physical_path']:r for r in inv['checkpoint_files'].values()})
    n=out['native'];n.update(clock=C.clock(),original_raw_statuses=inv['raw_statuses'],hf_all_rank_slices_exact=True,math_helper_sha256=C.MATH_HELPER,cpu_runtime=dict(torch_version='2.4.0+cu121'),base_named_parameter_digest='b'*64,hf_files={})
    n['ranks']=[dict(rank=i,pid=20+i,object_ids=inv['rank_reports'][i]['object_ids'],raw_saved_status='FULL_STATE_SERIALIZED_OK',model_dtensor_entries=435,model_hf_slice_exact=True,optimizer_state_entries=37,optimizer_step_values=[2000],
       model_layouts={str(j):dict(mesh_coordinate=[i],local_dtype='torch.float32',global_dtype='torch.float32') for j in range(435)},model_hf_slice_keys=[str(j) for j in range(435)]) for i in range(8)]
    for role in ('hf','archives'):out[role].update(native_report_sha256=p['reports']['native']['sha256'],native_inspector_source_sha256=V.PINS['inspect_final200_native.py'])
    out['hf'].update(checkpoint=str(V.HF200),official_base=str(V.BASE),hf_helper_sha256=C.HF_HELPER,actual_unique_parameters=C.NUMEL,all_loaded_parameters_finite=True,actual_cpu_forward_finite=True,tied_embeddings=True,hf_map={str(i):'fixture' for i in range(435)},canonical_state_keys=[str(i) for i in range(434)],base_named_parameter_digest=n['base_named_parameter_digest'],hf_files={},cpu_runtime=dict(torch_version='2.4.0'))
    a=out['archives'];a.update(outer_count=200,member_count=4000,student_count=512000,teacher_count=102400,unique_common_question_count=102400,archived_questions_exact_stored200_sampler_and_common_parquet=True,official_tokenizer_replayed=True,
      outers=[dict(outer=i,student_count=2560,teacher_count=512,actual_worker_bindings_verified=True,paired_manifest_sha256=inv['archive_files'][f'trajectory_outer_{i:06d}/paired-manifest.json']['sha256']) for i in range(1,201)])
    a['verified_input_records'].update({r['physical_path']:r for r in inv['archive_files'].values()})
    whole=dict(status='FAILED_WHOLE_RESOURCE_OWNERSHIP_NOT_UPGRADED_BY_TRAINING_PHASE',bad66={'line_number':66},unresolved_or_bad_rows=[{'line_number':66}])
    terminal=dict(training_child_exit=dict(status='OBSERVED_LITERAL_EXIT',returncode=0),wrapper_exit=dict(status='OBSERVED_LITERAL_EXIT',returncode=1),scheduler=dict(status='OBSERVED_LITERAL_SCHEDULER_TERMINAL',state='FAILED',exit_code='1:0'))
    p['literal_terminal']=dict(**terminal,whole_resource_layer=whole)
    layer=dict(status='VERIFIED_LOG_EVENTS_ONLY_NOT_FINAL_ACCEPTANCE',actual_outer_updates=200,adam_calls_per_rank=2000,no_executed_outer201_in_read_originals=True,log_writer_lifetime_qualification='BLOCKED_REQUIRE_INDEPENDENT_WRITER_EXIT_AND_FD_CLOSURE',outer=[{}]*200,evaluations=[{}]*11)
    out['logs_terminal']=dict(version=1,kind='FINAL200_LOG_TERMINAL_LAYER_REPORT_NOT_COMPLETION_RECEIPT',status='VERIFIED_LOGS_WITH_INDEPENDENT_TERMINAL_LAYERS_FINAL_ACCEPTANCE_BLOCKED',source_commit=V.SOURCE,config_sha256=V.CONFIG,stored_training_job_id='3304',actual_final_completion=None,
      consumer_source=rec('/fixture/logs.py',V.PINS['consume_final200_logs_terminal.py']),common_source=rec('/fixture/common.py',V.PINS['common.py']),accepted5={},log_layers=dict(driver=copy.deepcopy(layer),aggregate=copy.deepcopy(layer)),
      dataset=dict(final=dict(rows=51713,question_ids_sha256=V.IDS,source_counts=V.COUNTS,file=rec('/fixture/final.parquet',V.FINAL)),validation=dict(rows=512),stage=dict(rows=64)),terminal_layers=terminal,whole_resource_layer=whole,input_records={})
    return out


def review(p):
    rawroles=set().union(*V.REQUIRED_RAW.values())
    return dict(version=1,kind=V.REVIEW_KIND,status='QUALIFIED_ARTIFACTS_AND_EVAL_ELIGIBILITY_NOT_EFFECT_COMPLETION',source_commit=V.SOURCE,config_sha256=V.CONFIG,stored_training_job_id='3304',
      report_sha256={k:v['sha256'] for k,v in p['reports'].items()},qualifications={k:dict(status='QUALIFIED_INDEPENDENT_RAW_EVIDENCE_REVIEW',evidence_roles=sorted(V.REQUIRED_RAW[k])) for k in V.QUALIFICATIONS},
      evidence_records={k:rec('/fixture/raw/'+k,inode=5000+i) for i,k in enumerate(rawroles)},
      cpu_execution_witnesses={role:dict(execution_identity=value,**{k:rec('/fixture/'+role+'/'+k) for k in ('command_record','started_record','wait_record','stdout_record','stderr_record')}) for role,value in p['cpu_executions'].items()},literal_terminal=p['literal_terminal'],effect_completion=None)


def validate(p,r):V.validate_contents(p,r,C,context(),HF.validate_native)


def test_unsigned_semantic_fixture_has_no_overall_pass():
    p=payload();r=reports(p);assert V.header(p,C) is p;validate(p,r);V.qualify_review(review(p),p,C)
    assert p['effect_completion'] is None and r['logs_terminal']['terminal_layers']['wrapper_exit']['returncode']==1


@pytest.mark.parametrize('key,val',[('actual_outer_updates',5),('actual_outer_updates',200.0),('hf_step',500),('stored_training_job_id','3007'),('config_sha256','0'*64),('training_commit','188f'+'0'*36),('common_manifest_sha256','0'*64),('final_parquet_sha256','0'*64),('final_question_ids_sha256','0'*64),('final_questions',51712),('official_revision','main'),('official_base','/other'),('actual_hf200',str(V.RUN/'checkpoints/actor/global_step_5')),('optimizer_steps_per_rank',80),('horizon',300)])
def test_old_or_substituted_binding_rejected(key,val):
    p=payload();p['bindings'][key]=val
    with pytest.raises(ValueError):V.header(p,C)


@pytest.mark.parametrize('part',['status','effect','model_only','cohort','budget','budget_hash','tool_missing'])
def test_blocked_aggregate_or_budget_cannot_be_eligibility(part):
    p=payload()
    if part=='status':p['status']='BLOCKED'
    if part=='effect':p['effect_completion']='PASS'
    if part=='model_only':p['reports'].pop('archives')
    if part=='cohort':p['bindings']['final_source_counts']={**V.COUNTS,'nq':3611}
    if part=='budget':p['bindings']['evaluation_budget']['n']=5
    if part=='budget_hash':p['bindings']['evaluation_budget_sha256']='0'*64
    if part=='tool_missing':p['tools'].pop('observe_final200.py')
    with pytest.raises(ValueError):V.header(p,C)


@pytest.mark.parametrize('role,key,value',[
 ('native','checkpoint_step',5),('native','scope','NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE'),('native','cuda_is_initialized',True),('native','math_helper_sha256','0'*64),
 ('hf','inspector_source_sha256','0'*64),('hf','native_report_sha256','0'*64),('hf','actual_unique_parameters',True),('hf','all_loaded_parameters_finite',False),
 ('archives','outer_count',5),('archives','member_count',3999),('archives','student_count',102400),('archives','official_tokenizer_replayed',False),
 ('hf','cpu_runtime',dict(torch_version='2.8.0'))])
def test_wrong_actual_content_proof_rejected(role,key,value):
    p=payload();r=reports(p);r[role][key]=value
    with pytest.raises(ValueError):validate(p,r)


def test_full_native_tensor_schema_not_a_label():
    p=payload();r=reports(p);r['native']['ranks'][7]['model_layouts'].pop('1')
    with pytest.raises(ValueError):validate(p,r)


def test_actual_adam2000_typed_not50_or_bool():
    for value in ([50],[2000.0],[True]):
        p=payload();r=reports(p);r['native']['ranks'][0]['optimizer_step_values']=value
        with pytest.raises(ValueError):validate(p,r)


@pytest.mark.parametrize('part',['passed','missing31','missing4000','rawstatus','ranktype','hfkey','archivegroup','bad66','wrapper','scheduler','no201','aggregate','count'])
def test_all_files_logs_and_literal_failure_required(part):
    p=payload();r=reports(p)
    if part=='passed':r['inventory']['passed']=True
    if part=='missing31':r['inventory']['checkpoint_files'].pop(next(iter(r['inventory']['checkpoint_files'])))
    if part=='missing4000':r['inventory']['archive_files'].pop(next(iter(r['inventory']['archive_files'])))
    if part=='rawstatus':r['inventory']['raw_statuses']['ranks'][0]='RELOADED'
    if part=='ranktype':r['inventory']['rank_reports'][0]['rank']=False
    if part=='hfkey':r['hf']['hf_map'].pop('1')
    if part=='archivegroup':r['archives']['outers'][179]['student_count']=512
    if part=='bad66':r['logs_terminal']['whole_resource_layer']['status']='PASS'
    if part=='wrapper':p['literal_terminal']=copy.deepcopy(p['literal_terminal']);p['literal_terminal']['wrapper_exit']['returncode']=0
    if part=='scheduler':r['logs_terminal']['terminal_layers']['scheduler']['status']='BLOCKED_MISSING_ACTUAL_ORIGINAL'
    if part=='no201':r['logs_terminal']['log_layers']['driver']['no_executed_outer201_in_read_originals']=False
    if part=='aggregate':r['logs_terminal']['log_layers']['aggregate']['outer']=[]
    if part=='count':r['logs_terminal']['dataset']['final']['rows']=51713.0
    with pytest.raises(ValueError):validate(p,r)


@pytest.mark.parametrize('key,value',[('pid',10),('uid',False),('execution_job_id',None),('execution_step_id','../3'),('stored_training_job_id','901'),('node','h100-11'),('cgroup','0::/job_901/step_4\n1:/job_902/step_3\n'),('cgroup','0::/job_9010/step_3\n'),('cgroup','0::/job_901/step_30\n')])
def test_execution_identity_is_real_separate_and_same_path(key,value):
    x=eid();x[key]=value
    if key=='pid':x['start_ticks']=100
    with pytest.raises(ValueError):V.execution(x,C,context())


def test_same_numeric_pid_with_new_start_is_not_historical_process():
    x=eid(10);assert V.execution(x,C,context())==x


@pytest.mark.parametrize('part',['missing_release','bare_pass','review_wrong_report','review_oldsource','no_original_roles','effect','wrongcpu'])
def test_independent_qualification_is_not_self_reported_pass(part):
    p=payload();reports(p);q=review(p)
    if part=='missing_release':q['qualifications'].pop('IndependentResourceRelease')
    if part=='bare_pass':q['qualifications']['IndependentResourceRelease']['status']='PASS'
    if part=='review_wrong_report':q['report_sha256']['native']='0'*64
    if part=='review_oldsource':q['source_commit']='188f'+'0'*36
    if part=='no_original_roles':q['qualifications']['IndependentResourceRelease']['evidence_roles']=['native_wait']
    if part=='effect':q['effect_completion']='PASS'
    if part=='wrongcpu':q['cpu_execution_witnesses']['native']=copy.deepcopy(q['cpu_execution_witnesses']['native']);q['cpu_execution_witnesses']['native']['execution_identity']['pid']=1
    with pytest.raises(ValueError):V.qualify_review(q,p,C)


def test_missing_full_input_identity_and_conflicting_samepath_reject():
    p=payload();r=reports(p);q=review(p);r['archives']['verified_input_records'].pop(next(iter(r['inventory']['archive_files'].values()))['physical_path'])
    with pytest.raises(ValueError):V.collect_inputs(p,r,q,context(),C)
    records={};a=rec('/fixture/x');V.merge(records,a,C);b=copy.deepcopy(a);b['inode']=2
    with pytest.raises(ValueError):V.merge(records,b,C)


def test_symlink_replace_fullsha_and_duplicate_json_boundaries(tmp_path,monkeypatch):
    p=tmp_path/'raw.json';p.write_bytes(b'{"x":1}\n');q=tmp_path/'q';q.symlink_to(p)
    with pytest.raises(ValueError):C.read_json(q)
    with pytest.raises(ValueError):C.read_json(p,'0'*64)
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
        with pytest.raises(ValueError):C.strict_json(raw)
    real=C.os.read;done=False
    def race(fd,n):
        nonlocal done
        raw=real(fd,n)
        if raw and not done:done=True;p.write_bytes(b'{"x":2}\n');p.write_bytes(b'{"x":1}\n')
        return raw
    monkeypatch.setattr(C.os,'read',race)
    with pytest.raises(ValueError):C.read_json(p)


def test_actual_signed5_verifies_but_is_not_actual200():
    raw=H/'job3304-first5-complete-originals-v3/raw'
    receipt=H/'job3304-release-live-originals-v8/acceptance-5-receipt.json';public=raw/'bindings/public.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==V.RECEIPT5
    obj=C.strict_json(receipt.read_bytes());key=C.strict_json(public.read_bytes())
    assert hashlib.sha256(public.read_bytes()).hexdigest()==C.PUBLIC
    gate_path=H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py'
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest()==C.GATE
    gate=load(gate_path,'actual_frozen_gate_signature_only')
    gate.verify_rsa_signature(obj['payload'],obj['signature_hex'],key)
    with pytest.raises(ValueError):V.header(obj['payload'],C)
    changed=copy.deepcopy(obj['payload']);changed['actual_outer_updates']=200
    with pytest.raises(ValueError):gate.verify_rsa_signature(changed,obj['signature_hex'],key)


def write(tmp_path,name,value):
    p=tmp_path/name;p.write_bytes(C.canonical(value)+b'\n');return C.read_file(p)[1]


def wait_fixture(tmp_path):
    p=payload();q=review(p)
    for role,entry in q['cpu_execution_witnesses'].items():
        who=entry['execution_identity'];inspector={'native':'inspect_final200_native.py','hf':'load_final200_hf.py','archives':'check_final200_archives.py'}[role]
        argv=[who['python'],'-B',p['tools'][inspector]['physical_path'],'--common-sha256',V.PINS['common.py'],'--inventory',p['reports']['inventory']['physical_path'],
          '--inventory-sha256',p['reports']['inventory']['sha256'],'--accepted5-receipt-sha256',V.RECEIPT5,'--output',p['reports'][role]['physical_path'],
          '--execution-job-id',who['execution_job_id'],'--execution-step-id',who['execution_step_id']]
        if role!='native':argv+=['--native-report',p['reports']['native']['physical_path'],'--native-report-sha256',p['reports']['native']['sha256'],'--native-inspector',p['tools']['inspect_final200_native.py']['physical_path'],'--native-inspector-sha256',V.PINS['inspect_final200_native.py']]
        parent=dict(pid=800,start_ticks=88);started=dict(argv=argv,child_pid=who['pid'],parent_pid=800,child=who,parent=parent,capture_errors=[])
        entry['command_record']=write(tmp_path,role+'-command',dict(argv=argv))
        entry['started_record']=write(tmp_path,role+'-started',started)
        entry['wait_record']=write(tmp_path,role+'-wait',dict(returncode=0,waited=True,waited_by_pid=800,child_pid=who['pid'],capture_errors=[]))
        entry['stdout_record']=write(tmp_path,role+'-stdout',dict(scope=V.SCOPES[role],passed=True,output=p['reports'][role],stored_training_job_id='3304',execution_identity=who))
        entry['stderr_record']=write(tmp_path,role+'-stderr',{'scope':'EXPLICIT_SYNTHETIC_LOCAL_FIXTURE_ONLY'})
    return p,q


def test_original_controller_writer_wait_schema_fixture(tmp_path):
    p,q=wait_fixture(tmp_path);V.validate_wait_originals(q,p,C)


@pytest.mark.parametrize('part',['notwaited','badexit','boolexit','otherpid','captureerror','wrongsource','wronginventory','wrongstep','stdoutold5','stdoutrecord','parentbool'])
def test_actual_wait_argv_stdout_cannot_be_labels(tmp_path,part):
    p,q=wait_fixture(tmp_path);e=q['cpu_execution_witnesses']['native']
    if part in ('notwaited','badexit','boolexit','otherpid','captureerror'):
        x=C.read_json(e['wait_record']['physical_path'])[0]
        if part=='notwaited':x['waited']=False
        if part=='badexit':x['returncode']=1
        if part=='boolexit':x['returncode']=False
        if part=='otherpid':x['child_pid']=901
        if part=='captureerror':x['capture_errors']=[{'unknown':True}]
        e['wait_record']=write(tmp_path,'mutated-wait',x)
    if part in ('wrongsource','wronginventory','wrongstep'):
        x=C.read_json(e['command_record']['physical_path'])[0]
        if part=='wrongsource':x['argv'][2]='/wrong/inspect_final200_native.py'
        if part=='wronginventory':x['argv'][x['argv'].index('--inventory')+1]='/wrong/old5.json'
        if part=='wrongstep':x['argv'][x['argv'].index('--execution-step-id')+1]='4'
        e['command_record']=write(tmp_path,'mutated-command',x)
        started=C.read_json(e['started_record']['physical_path'])[0];started['argv']=x['argv'];e['started_record']=write(tmp_path,'mutated-started',started)
    if part in ('stdoutold5','stdoutrecord'):
        x=C.read_json(e['stdout_record']['physical_path'])[0]
        if part=='stdoutold5':x['scope']='NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE'
        if part=='stdoutrecord':x['output']['sha256']='0'*64
        e['stdout_record']=write(tmp_path,'mutated-stdout',x)
    if part=='parentbool':
        x=C.read_json(e['started_record']['physical_path'])[0];x['parent_pid']=True;e['started_record']=write(tmp_path,'mutated-started',x)
    with pytest.raises(ValueError):V.validate_wait_originals(q,p,C)


def test_scope_specific_raw_bindings_match_unchanged_log_spec(tmp_path):
    p=payload();r=reports(p);q=review(p)
    spec=dict(terminal=dict(training_child_exit=rec('/fixture/child'),wrapper_exit=rec('/fixture/wrapper'),scheduler_original=rec('/fixture/scheduler')),
       logs=dict(driver=dict(file=rec('/fixture/driver')),aggregate=dict(file=rec('/fixture/aggregate'))),resources=rec('/fixture/whole'))
    r['logs_terminal']['input_spec']=write(tmp_path,'spec.json',spec)
    mapping=dict(child_exit=spec['terminal']['training_child_exit'],wrapper_exit=spec['terminal']['wrapper_exit'],scheduler_original=spec['terminal']['scheduler_original'],driver_complete=spec['logs']['driver']['file'],aggregate_complete=spec['logs']['aggregate']['file'],whole_resources=spec['resources'],resources_complete=spec['resources'])
    q['evidence_records'].update(mapping)
    q['evidence_records'].update({role+'_wait':x['wait_record'] for role,x in q['cpu_execution_witnesses'].items()})
    V.qualify_original_bindings(q,r,C)
    q['evidence_records']['whole_resources']=rec('/fixture/filtered-good-training-slice')
    with pytest.raises(ValueError):V.qualify_original_bindings(q,r,C)


def test_pinned_bootstrap_rejects_modified_helper_before_execution(tmp_path):
    copied=tmp_path/'verify_actual200_prerequisites.py';copied.write_bytes((HERE/'verify_actual200_prerequisites.py').read_bytes())
    common=tmp_path/'common.py';common.write_text('raise RuntimeError("unapproved helper executed")\n')
    candidate=load(copied,'fixture_unapproved_helper')
    with pytest.raises(ValueError,match='Pinned common'):candidate.load_common()


def test_no_signed_actual200_template_or_production_proof():
    blocked=json.loads((HERE/'BLOCKED_LAUNCH.json').read_bytes())
    assert blocked['actual_outer_updates'] is None and blocked['actual_signature'] is None and blocked['evaluation_job_id'] is None
    assert 'BLOCKED' in blocked['status'] and blocked['effect_completion'] is None
    assert len(blocked['strict_stdout_keys'])==9
