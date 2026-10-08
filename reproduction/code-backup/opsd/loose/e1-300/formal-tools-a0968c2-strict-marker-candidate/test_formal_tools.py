"""CPU negative tests; allocation/CUDA interfaces are fixtures, never actual acceptance."""
import copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
DIR=Path(__file__).resolve().parent;sys.path.insert(0,str(DIR))
import formal_supervisor as f

def inv():return {'inventory':[{'index':i,'uuid':f'GPU-{i}','name':'NVIDIA H100 80GB HBM3','used':1000.,'total':80000.,'util':0.} for i in range(8)],'applications':[]}
def test_actual_slurm_shape_and_typed_failures():
 raw='JobId=123 JobState=RUNNING NodeList=h100-10 UserId=root(0) NumNodes=1 NumCPUs=192 OverSubscribe=NO TresPerNode=gres:gpu:h100:8 MinMemoryNode=1T TimeLimit=7-00:00:00'
 assert f.slurm(raw,'123',0)['NumCPUs']=='192'
 for before,after in [('RUNNING','COMPLETED'),('h100-10','h100-9'),('root(0)','root(1)'),('h100:8','h100:4'),('h100:8','8'),('NumNodes=1','NumNodes=2'),('192','80'),('OverSubscribe=NO','OverSubscribe=YES'),('1T','64G'),('7-00:00:00','1-00:00:00')]:
  with pytest.raises(ValueError):f.slurm(raw.replace(before,after),'123',0)
 with pytest.raises(ValueError):f.slurm(raw,'999',0)

@pytest.mark.parametrize('key,value',[('used',float('nan')),('total',float('inf')),('util',-1),('util',101),('name','A800'),('total',0),('used',80001)])
def test_bad_inventory(key,value):
 d=inv();d['inventory'][0][key]=value
 with pytest.raises(ValueError):f.validate_snapshot(d,idle=True)

def test_uuid_exact_service_and_15_percent():
 d=inv();uuids=[r['uuid'] for r in d['inventory']];f.validate_snapshot(d,uuids,idle=True)
 with pytest.raises(ValueError):f.validate_snapshot(d,uuids[:-1])
 d['applications']=[{'pid':1,'uuid':u} for u in uuids];f.validate_snapshot(d,uuids,1,idle=True)
 with pytest.raises(ValueError):f.validate_snapshot(d,uuids)
 d['applications'][3]['pid']=2
 with pytest.raises(ValueError):f.validate_snapshot(d,uuids,1)
 d=inv();d['inventory'][0]['used']=12001
 with pytest.raises(ValueError):f.validate_snapshot(d,idle=True)

@pytest.fixture
def rsa(tmp_path):
 keypath=tmp_path/'test-review-only-private.pem';subprocess.run(['openssl','genrsa','-out',str(keypath),'2048'],check=True,capture_output=True)
 n=subprocess.check_output(['openssl','rsa','-in',str(keypath),'-noout','-modulus'],stderr=subprocess.PIPE,text=True).strip().split('=',1)[1]
 public={'algorithm':'RSA-SHA256','n_hex':n,'e':65537}
 def sign(payload):return {'payload':payload,'signature_hex':subprocess.check_output(['openssl','dgst','-sha256','-sign',str(keypath)],input=f.canonical(payload)).hex()}
 return public,sign

@pytest.fixture
def ready_case(tmp_path,rsa,monkeypatch):
 run=tmp_path/'run';(run/'bindings').mkdir(parents=True);key,sign=rsa
 f.emit(run/'bindings/public.json',key);review=run/'review.md';review.write_text('CPU reviewer fixture; no actual GPU acceptance')
 monkeypatch.setenv('SLURM_JOB_ID','CPU123')
 ready={'source_commit':f.SOURCE,'job_id':'CPU123','uid':os.getuid(),'pid':os.getpid(),'process_start_ticks':'fake-own-start','run_root':str(run),'nonce':'newnonce','ready_sha256':'a'*64,'binding_sha256':'b'*64,'public_key_sha256':f.sha(run/'bindings/public.json'),'process':{'pid':os.getpid(),'start':'fake-own-start'}}
 record={'path':str(review),'sha256':f.sha(review)}
 payload=f.approval_expected(ready,record)
 monkeypatch.setattr(f,'same_process',lambda p:p['start']=='fake-own-start')
 def write(envelope):
  path=run/'bindings/gpu-init-approval.json';path.write_bytes(f.canonical(envelope));(run/'bindings/gpu-init-approval.sha256').write_text(f.sha(path))
 write(sign(payload));return run,ready,payload,sign,write

def test_signed_current_job_release_and_changes(ready_case):
 run,ready,payload,sign,write=ready_case;assert len(f.check_approval(run,ready))==64
 for field,value in [('job_id','oldjob'),('pid',os.getpid()+1),('nonce','oldnonce'),('source_commit','0'*40),('ready_sha256','0'*64),('binding_sha256','0'*64),('authorization','PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW')]:
  bad=dict(payload,**{field:value});write(sign(bad))
  with pytest.raises(ValueError):f.check_approval(run,ready)
 write(sign(payload));Path(payload['independent_review']['path']).write_text('modified')
 with pytest.raises(ValueError):f.check_approval(run,ready)

def test_unsigned_signature_or_stale_process_rejected(ready_case,monkeypatch):
 run,ready,payload,sign,write=ready_case
 write({'payload':payload,'signature_hex':'00'*256})
 with pytest.raises(ValueError):f.check_approval(run,ready)
 write(sign(payload));monkeypatch.setattr(f,'same_process',lambda _:False)
 with pytest.raises(ValueError):f.check_approval(run,ready)

def test_external_approval_hash_required(ready_case):
 run,ready,*_=ready_case;(run/'bindings/gpu-init-approval.sha256').write_text('0'*64)
 with pytest.raises(ValueError):f.check_approval(run,ready)

def test_copy_exact_bytes_no_overwrite_or_symlink(tmp_path):
 p=tmp_path/'source.json';p.write_text('{"modules":{}}');dest=tmp_path/'copy.json';record=f.copy_pin(p,f.sha(p),dest);assert dest.read_bytes()==p.read_bytes() and record['sha256']==f.sha(p)
 with pytest.raises(FileExistsError):f.copy_pin(p,f.sha(p),dest)
 link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(ValueError):f.copy_pin(link,f.sha(p),tmp_path/'other.json')
 with pytest.raises(ValueError):f.copy_pin(p,'0'*64,tmp_path/'wrong.json')

@pytest.mark.parametrize('raw',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'])
def test_bad_json(raw):
 with pytest.raises(ValueError):f.unique(raw)

def test_binding_hash_change_missing_role_and_escape(tmp_path):
 run=tmp_path/'run';(run/'bindings').mkdir(parents=True);files={}
 for role in f.ROLES:
  p=run/'bindings'/f'{role}.json';p.write_text('{}');files[role]={'path':str(p),'sha256':f.sha(p)}
 path=run/'bindings/manifest.json'
 def write(d):path.write_bytes(f.canonical(d));return {'binding_sha256':f.sha(path)}
 binding={'source_commit':f.SOURCE,'run_root':str(run),'files':files};ready=write(binding);f.verify_binding(run,ready)
 bad=copy.deepcopy(binding);del bad['files']['allocation']
 with pytest.raises(ValueError):f.verify_binding(run,write(bad))
 bad=copy.deepcopy(binding);outside=tmp_path/'outside';outside.write_text('{}');bad['files']['source']['path']=str(outside)
 with pytest.raises(ValueError):f.verify_binding(run,write(bad))
 ready=write(binding);Path(files['source']['path']).write_text('{"changed":true}')
 with pytest.raises(ValueError):f.verify_binding(run,ready)

class FakeProcess:
 def __init__(self):self.signals=[]
 def poll(self):return None
 def terminate(self):self.signals.append('TERM')
 def wait(self,timeout):return 0

def test_cleanup_never_signals_stale_or_foreign(monkeypatch):
 p=FakeProcess();monkeypatch.setattr(f,'same_process',lambda _:False)
 with pytest.raises(ValueError):f.terminate_owned(p,{'pid':999,'start':'old'})
 assert p.signals==[]
 monkeypatch.setattr(f,'same_process',lambda _:True);f.terminate_owned(p,{'pid':999,'start':'current'});assert p.signals==['TERM']

def resource_row(timestamp):
 uuids=[f'GPU-{i}' for i in range(8)];apps=[{'pid':100,'uuid':u,'uid':str(os.getuid()),'in_slurm_job':True} for u in uuids]+[{'pid':200+i,'uuid':u,'uid':str(os.getuid()),'in_slurm_job':True} for i,u in enumerate(uuids)]
 return {'time':timestamp,'job_id':'cpujob','gpus':[{'uuid':u,'util':0.,'used_mib':20000.,'total_mib':80000.} for u in uuids],'gpu_processes':apps}

def test_resource_monotonic_real_coexistence(tmp_path):
 p=tmp_path/'samples';a=resource_row('2026-10-04T00:00:00+00:00');b=resource_row('2026-10-04T00:00:02+00:00');p.write_text('\n'.join(json.dumps(v) for v in [a,b]));result=f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100);assert result['eight_worker_plus_service_samples']==2
 for rows in ([a,a],[b,a],[dict(a,time='2026-10-04T00:00:00'),b],[dict(a,job_id='foreign'),b],[dict(a,gpu_processes=[dict(a['gpu_processes'][0],in_slurm_job=False)]),b]):
  p.write_text('\n'.join(json.dumps(v) for v in rows))
  with pytest.raises(ValueError):f.validate_resources(p,'cpujob',[f'GPU-{i}' for i in range(8)],100)

def test_frozen_entry_no_hydra_overrides_static():
 text=(DIR/'formal_supervisor.py').read_text();shell=(DIR/'formal_job.sh').read_text()
 assert "str(repo/'reproduction/baselines/run_opsd_e1_300.sh'),'--run'" in text
 assert 'baselines.method=' not in text and 'component_clip=' not in text and 'n_agent=' not in text
 assert 'ray stop' not in shell and 'pkill' not in shell and 'sbatch ' not in shell
 assert 'E1_NATIVE_IMPORT_CLOSURE_SHA256' in shell and 'OPSD_E1_REVIEW_PUBLIC_KEY_SHA256' in shell
 assert 'OPSD_E1_EXTERNAL_PRE_GPU_APPROVAL_READY=' in text


def test_sampler_is_reviewed_identical():
 expected='0f7f23ff556f9ef59f4fbee1a81f79bffa25ff7e98f4d2c207f8009b2d2eda36'
 assert f.sha(DIR/'resource_sample.py')==expected


def test_full300_raw_records_and_real_rank_validator(tmp_path,monkeypatch):
 source=DIR.parents[1]/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
 # Standard-library module with actual frozen-source validate_outer/evaluation;
 # no model/worker/GPU execution occurs in this fixture.
 spec=importlib.util.spec_from_file_location('search_r1.baselines.checkpoint_gate',source);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);monkeypatch.setitem(sys.modules,'search_r1.baselines.checkpoint_gate',gate)
 expected={'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0,'actor/samples':320,'actor/loss_mask_valid':1,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0,'teacher/storage_is_cpu_per_rank':1,'teacher/all_parameter_dtype_is_fp32_per_rank':1,'teacher/attention_is_eager_per_rank':1,'actor/allocator_is_native':1,'actor/allocator_configuration_is_default':1}
 metrics={k+'/rank_'+str(r):v for r in range(8) for k,v in expected.items()}
 def evaluation(n):return {'val/qa_count/all':n,'val/evaluated_count/fixture':n}
 rows=[]
 for step in range(1,301):
  m=dict(metrics)
  if step%50==0:m.update(evaluation(512))
  if step==5 or step%50==0:m.update({'opsd_stage/'+k:v for k,v in evaluation(64).items()})
  rows.append({'actual_outer_updates':step,'global_step':step,'driver_pid':100,'job_id':'cpujob','source_commit':f.SOURCE,'config_sha256':'a'*64,'metrics':m})
 path=tmp_path/'train.log'
 monkeypatch.setattr(f,'training_log_context',lambda run,job:(gate,dict(driver_pid=100,config_sha256='a'*64),tuple(range(200,208))))
 def write(rs,final=51713):path.write_text('\n'.join('(main_task pid=100) OPSD_E1_OUTER_JSON='+json.dumps(r) for r in rs)+'\nSEARCH_R1_FINAL_METRICS_JSON='+json.dumps(evaluation(final)))
 write(rows);assert f.check_training_log(path,'cpujob')['actual_outer_updates']==300
 for badrows in [rows[:-1],rows+[rows[-1]],list(reversed(rows))]:
  write(badrows)
  with pytest.raises(ValueError):f.check_training_log(path,'cpujob')
 bad=copy.deepcopy(rows);bad[299]['metrics']['actor/optimizer_steps/rank_7']=19;write(bad)
 with pytest.raises(RuntimeError):f.check_training_log(path,'cpujob')
 bad=copy.deepcopy(rows);bad[4]['metrics']['opsd_stage/val/qa_count/all']=63;write(bad)
 with pytest.raises(RuntimeError):f.check_training_log(path,'cpujob')
 write(rows,64)
 with pytest.raises(RuntimeError):f.check_training_log(path,'cpujob')
 bad=copy.deepcopy(rows);bad[1]['driver_pid']=101;write(bad)
 with pytest.raises(ValueError):f.check_training_log(path,'cpujob')


def test_cpu_prepare_binds_actual_bytes_then_external_release_before_exec(tmp_path,rsa,monkeypatch):
 """Full prepare flow with declared CPU interface fixtures; no real assets/GPU."""
 run=tmp_path/'run';run.mkdir();repo=tmp_path/'repo';repo.mkdir();tools=tmp_path/'tools';tools.mkdir();key,sign=rsa
 monkeypatch.setattr(f,'ROOT',tmp_path);model=tmp_path/'model';model.mkdir();monkeypatch.setattr(f,'MODEL',model)
 originals={}
 for name,data in [('native',{'modules':{}}),('common',{'cpu_fixture_only':True}),('model',{'cpu_fixture_only':True}),('assets',{'cpu_fixture_only':True}),('public',key)]:
  if name=='model':path=model/'VERIFIED-MODEL-MANIFEST.json'
  elif name=='public':path=tmp_path/'review-signing/public.json';path.parent.mkdir()
  else:path=tmp_path/f'{name}.json'
  f.emit(path,data);originals[name]=path
 for name,var in [('model','MODEL_SHA'),('common','COMMON_SHA'),('assets','ASSETS_SHA')]:monkeypatch.setattr(f,var,f.sha(originals[name]))
 env={'SLURM_JOB_ID':'CPU123','SLURM_JOB_NODELIST':'h100-10','SLURM_CPUS_PER_TASK':'192','CUDA_VISIBLE_DEVICES':','.join(map(str,range(8))),'E1_FORMAL_MANIFEST_SHA256':'f'*64,'E1_NATIVE_IMPORT_CLOSURE':str(originals['native']),'E1_NATIVE_IMPORT_CLOSURE_SHA256':f.sha(originals['native']),'OPSD_E1_COMMON_MANIFEST':str(originals['common']),'OPSD_E1_REVIEW_PUBLIC_KEY':str(originals['public']),'OPSD_E1_REVIEW_PUBLIC_KEY_SHA256':f.sha(originals['public']),'E1_TOOL_ASSET_MANIFEST':str(originals['assets'])}
 for k,v in env.items():monkeypatch.setenv(k,v)
 monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF',raising=False);monkeypatch.delenv('PYTORCH_ALLOC_CONF',raising=False)
 monkeypatch.setattr(f.os,'getuid',lambda:0) # local file owned_file tests use original os module below
 # physical(...owned) uses same mocked os.getuid; override only fixture ownership
 original_physical=f.physical
 monkeypatch.setattr(f,'physical',lambda p,owned=False:original_physical(p,False))
 monkeypatch.setattr(f,'under',lambda p,r:Path(p) if Path(p).is_relative_to(r) else (_ for _ in ()).throw(ValueError('escape')))
 owner={'pid':os.getpid(),'uid':0,'start':'fixture-start','state':'R','cgroup':'0::/job_CPU123'}
 events=[];monkeypatch.setattr(f,'sources',lambda t:(events.append('source-verified') or (repo,{'source_files':{'fixture':'a'*64}})))
 monkeypatch.setattr(f,'own',lambda pid:owner);monkeypatch.setattr(f,'same_process',lambda r:True);monkeypatch.setattr(f.socket,'gethostname',lambda:'h100-10');monkeypatch.setattr(f.time,'sleep',lambda _:None)
 raw='JobId=CPU123 JobState=RUNNING NodeList=h100-10 UserId=root(0) NumNodes=1 NumCPUs=192 OverSubscribe=NO TresPerNode=gres:gpu:h100:8 MinMemoryNode=1T TimeLimit=7-00:00:00'
 monkeypatch.setattr(f,'command',lambda args:raw if args[0]=='scontrol' else '{"cpu_fixture_only":true}')
 monkeypatch.setattr(f,'snapshot',lambda:(events.append('nvml-read-only') or inv()))
 monkeypatch.setattr(f,'verify_tool_assets',lambda a:events.append('assets-content-verification-fixture'))
 original_run=subprocess.run
 def cpu_call(args,stdout,**kw):
  if args[0]=='openssl':return original_run(args,stdout=stdout,**kw)
  events.append('cpu-subprocess-fixture');stdout.write(b'{}');return None
 monkeypatch.setattr(f.subprocess,'run',cpu_call)
 original_emit=f.emit
 def external_reviewer(path,data):
  original_emit(path,data)
  if path.name=='approval-ready.json':
   events.append('ready');actual=dict(data,ready_sha256=f.sha(path));review=run/'independent-fixture.md';review.write_text('External reviewer fixture; NOT real GPU review');record={'path':str(review),'sha256':f.sha(review)}
   original_emit(run/'bindings/gpu-init-approval.json',sign(f.approval_expected(actual,record)));(run/'bindings/gpu-init-approval.sha256').write_text(f.sha(run/'bindings/gpu-init-approval.json'));events.append('external-signature')
 monkeypatch.setattr(f,'emit',external_reviewer)
 def checked_exec(executable,args):
  assert events.index('ready')<events.index('external-signature')
  assert args[2:5]==['--mode','preflight','--'];events.append('first-lease-exec')
 monkeypatch.setattr(f.os,'execv',checked_exec)
 f.prepare(run,tools)
 assert events[-1]=='first-lease-exec' and 'external-signature' in events
 binding=f.unique((run/'bindings/manifest.json').read_bytes());assert set(binding['files'])==f.ROLES
 for role in ('import_closure','data_manifest','model_manifest','tool_manifest'):
  rec=binding['files'][role];assert Path(rec['path']).parent==run/'bindings' and f.sha(rec['path'])==rec['sha256']
 assert not (run/'train.log').exists() and not (run/'resolved-profile.json').exists()


@pytest.fixture
def signed_context(tmp_path,monkeypatch):
 import search_r1.baselines.checkpoint_gate as gate
 source=DIR.parents[1]/'opsd-e1-300/tests/baselines/test_opsd_e1_checkpoint_gate.py'
 spec=importlib.util.spec_from_file_location('actual_source_signature_fixture',source);fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
 key,sign=fixtures.rsa.__wrapped__(tmp_path)
 g,path,m,payload,_=fixtures.signed_gate.__wrapped__(tmp_path,(key,sign),monkeypatch)
 m['source_commit']=f.SOURCE;payload['source_commit']=f.SOURCE
 dest=g.root/'full_state/global_step_5';dest.mkdir(parents=True);path=dest/'manifest.json';path.write_bytes(gate.canonical(m));payload['checkpoint_manifest_sha256']=gate.sha256(path)
 (g.root/'acceptance-5-receipt.json').write_bytes(gate.canonical(sign(payload)))
 (g.root/'bindings').mkdir();public=g.root/'bindings/public.json';public.write_bytes(gate.canonical(key))
 ready={'public_key_sha256':f.sha(public)};(g.root/'approval-ready.json').write_bytes(gate.canonical(ready))
 # Existing wrapper suite independently checks preGPU approval. This unit isolates
 # the real first5 RSA signature/native-file join, not a full actual authorization.
 monkeypatch.setattr(f,'check_approval',lambda run,ready:None);monkeypatch.setattr(f,'verify_binding',lambda run,ready:None)
 return g.root,m,payload,sign,gate,path


def test_signed_context_uses_actual_eight_native_reports(signed_context):
 root,m,payload,sign,gate,path=signed_context
 _,actual,workers=f.training_log_context(root,m['job_id'])
 assert actual==m and workers==set(range(1000,1008))
 for foreign in ['foreign-job','3198']:
  with pytest.raises(ValueError):f.training_log_context(root,foreign)
 Path(m['files']['actor/training_state/rank_0.pt']['path']).write_bytes(b'changed raw state')
 with pytest.raises(ValueError):f.training_log_context(root,m['job_id'])


def test_signed_context_rejects_unsigned_manifest_or_bad_signature(signed_context):
 root,m,payload,sign,gate,path=signed_context
 m['driver_pid']+=1;path.write_bytes(gate.canonical(m))
 with pytest.raises(ValueError):f.training_log_context(root,m['job_id'])
 payload['checkpoint_manifest_sha256']=gate.sha256(path);payload['driver_pid']=m['driver_pid']
 envelope=sign(payload);envelope['signature_hex']='00'*256
 (root/'acceptance-5-receipt.json').write_bytes(gate.canonical(envelope))
 with pytest.raises(ValueError):f.training_log_context(root,m['job_id'])


def test_original3198_row4_shared_parser_and_malicious_suffixes(monkeypatch,tmp_path):
 import search_r1.baselines.checkpoint_gate as gate
 log=DIR.parent/'formal3198-first5-observed-20261004T134348Z/run/train.log'
 line=next(l for l in log.read_text().splitlines() if 'OPSD_E1_OUTER_JSON=' in l and '\x1b[36m(WorkerDict pid=470520)' in l)
 marker='OPSD_E1_OUTER_JSON=';raw=line.split(marker,1)[1];expected,end=json.JSONDecoder().raw_decode(raw)
 with pytest.raises(json.JSONDecodeError):json.loads(raw)
 warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'
 actual=gate.parse_log_marker(line,marker,driver_pid=expected['driver_pid'],job_id='3198',worker_pids=(470520,),checkpoint_warning_path=warning_path)
 assert actual==expected
 for bad in [line+' arbitrary',line+' {}',line+' OPSD_E1_OUTER_JSON={}',line.replace('470520','999999'),line.replace('checkpoint.py:1399','checkpoint.py:1400'),line.replace('FutureWarning:','Warning:'),line.replace('\x1b[36m','\x1b[31m')]:
  with pytest.raises(ValueError):gate.parse_log_marker(bad,marker,driver_pid=expected['driver_pid'],job_id='3198',worker_pids=(470520,),checkpoint_warning_path=warning_path)
 # Invoke the actual wrapper reader and record its shared parser result. This
 # historical four-row excerpt must fail completion; it never accepts 3198.
 seen=[];original=gate.parse_log_marker
 def parse(*args,**kwargs):
  kwargs['checkpoint_warning_path']=warning_path;value=original(*args,**kwargs);seen.append(value);return value
 monkeypatch.setattr(gate,'parse_log_marker',parse)
 monkeypatch.setattr(f,'training_log_context',lambda run,job:(gate,{'driver_pid':expected['driver_pid'],'config_sha256':expected['config_sha256']},(470520,)))
 excerpt=tmp_path/'train.log';excerpt.write_text(line+'\n')
 with pytest.raises(ValueError,match='Missing/duplicate'):f.check_training_log(excerpt,'3198')
 assert seen==[expected]
