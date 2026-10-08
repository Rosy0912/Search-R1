"""Local CPU byte/size/state fixtures; synthetic metadata is never GPU/first5 acceptance."""
import base64,copy,gzip,hashlib,importlib.util,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
s=importlib.util.spec_from_file_location('candidate',H/'collect_first5_originals_candidate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ns={};exec(compile(m.reader_constants()+m.REMOTE_LIBRARY,'actual_generated_library_cpu','exec'),ns)
REAL=json.loads((H/'job3304-first5-live-binding-original/manifest.json').read_bytes())
SPEC=json.loads((H/m.BINDING_NAME).read_bytes())
REQUEST={'run_root':m.RUN,'repo':m.REPO,'source_commit':m.SOURCE,'job_id':m.JOB,'source_files':SPEC['source_files'],'formal_sha':SPEC['formal']['manifest_sha256'],'checkpoint_manifest_sha256':'a'*64,'pause_nonce':REAL['pause_nonce'],'driver_pid':REAL['driver_pid'],'driver_start_ticks':'100','driver_stdout_physical':'/tmp/OWN_LOCAL_FIXTURE_ONLY.out'}

def enc(b):return json.dumps(b,sort_keys=True,allow_nan=False).encode()
def row(name,b,i=1):
 packed=gzip.compress(b,mtime=0);physical=REQUEST['repo']+'/'+name[7:] if name.startswith('source/')else REQUEST['run_root']+'/'+name
 if name=='driver-current-prefix.out':physical=REQUEST['driver_stdout_physical']
 if name==m.NATIVE_WARNING_NAME:physical=m.TORCH_PATH+'/distributed/fsdp/fully_sharded_data_parallel.py'
 prefix=name in ('train.log','driver-current-prefix.out') or name.endswith('.log') or 'stdout' in Path(name).name or 'stderr' in Path(name).name
 return {'physical':physical,'bytes':len(b),'sha256':m.sha(b),'gzip_bytes':len(packed),'gzip_sha256':m.sha(packed),'gzip_base64':base64.b64encode(packed).decode(),'uid':0,'device':1,'inode':i,'stat_before':[1,i,0,len(b),10,10],'stat_after':[1,i,0,len(b),10,10],'reading_scope':'VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED' if prefix else 'STABLE_FULL_BYTES_AT_OBSERVATION_NOT_ACCEPTANCE','line_count':b.count(b'\n'),'ends_lf':b.endswith(b'\n'),'link_count':2 if name==m.NATIVE_WARNING_NAME else 1,'link_count_after':2 if name==m.NATIVE_WARNING_NAME else 1,'third_party_hardlink_source_pin':m.NATIVE_WARNING_SHA if name==m.NATIVE_WARNING_NAME else None}
def logs(manifest,request):
 gate=types.ModuleType('actual_gate_cpu');gate.__file__=m.REPO+'/search_r1/baselines/checkpoint_gate.py';raw=(H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py').read_bytes();exec(compile(raw,gate.__file__,'exec'),gate.__dict__)
 keys={'actor/optimizer_steps':10,'actor/skipped_empty_minibatches':0,'actor/samples':320,'actor/loss_mask_valid':1,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0,'teacher/storage_is_cpu_per_rank':1,'teacher/all_parameter_dtype_is_fp32_per_rank':1,'teacher/attention_is_eager_per_rank':1,'actor/allocator_is_native':1,'actor/allocator_configuration_is_default':1}
 metrics={k+'/rank_'+str(r):v for r in range(8)for k,v in keys.items()};out=[]
 for n in range(1,6):out.append(b'OPSD_E1_OUTER_JSON='+enc({'actual_outer_updates':n,'global_step':n,'driver_pid':manifest['driver_pid'],'job_id':m.JOB,'source_commit':m.SOURCE,'config_sha256':manifest['config_sha256'],'metrics':metrics})+b'\n')
 pause={'manifest_path':m.RUN+'/full_state/global_step_5/manifest.json','manifest_sha256':request['checkpoint_manifest_sha256'],'pause_nonce':manifest['pause_nonce'],'driver_pid':manifest['driver_pid'],'job_id':m.JOB};out.append(b'OPSD_E1_STEP5_PAUSED_JSON='+enc(pause)+b'\n');body=b''.join(out);join=ns['check_logs'](body,gate,manifest,request,list(range(1000,1008)));return body,join

def fixture():
 manifest=copy.deepcopy(REAL);request=copy.deepcopy(REQUEST);bodies={}
 for role,name in m.CHECKPOINT_MAP.items():
  if name in m.TENSOR_NAMES:continue
  if name.endswith('student.jsonl'):b=b'{"fixture":"LOCAL_SYNTHETIC_TRAJECTORY_BYTES_NOT_GPU"}\n'*2560
  elif name.endswith('.jsonl'):b=b'{"fixture":"LOCAL_SYNTHETIC_DIAGNOSTIC_BYTES_NOT_GPU"}\n'*64
  elif name.endswith('questions.json'):b=enc({'questions':[{'fixture_question':i}for i in range(512)]})
  elif name.endswith('paired-manifest.json'):b=enc({'student_count':2560,'teacher_count':512})
  elif name.endswith('student-manifest.json'):b=enc({'student_count':2560,'recorded_before_balancing':True})
  elif '/training_state/rank_'in name:b=enc({'pid':1000+int(Path(name).stem.split('_')[1])})
  else:b=b'{}'
  bodies[name]=b
 for name in m.REQUIRED:bodies.setdefault(name,b'{}')
 for name in request['source_files']:bodies['source/'+name]=(H.parent/'opsd-unclipped-multitraj200-teacherrecompute'/name).read_bytes()
 bodies[m.NATIVE_WARNING_NAME]=(H.parent/'e1-author200/actual-fsdp-warning-source-20261005T235824197037Z/fully_sharded_data_parallel.py').read_bytes()
 bodies['bindings/source.json']=enc({'clean':True,'source_commit':m.SOURCE,'repo':m.REPO,'source_files':request['source_files'],'formal_manifest_sha256':request['formal_sha']})
 bound={'source_commit':m.SOURCE,'run_root':m.RUN,'files':{r:{'path':m.RUN+'/bindings/'+r+'.json','sha256':m.sha(bodies['bindings/'+r+'.json'])}for r in ('allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest')}};bodies['bindings/manifest.json']=enc(bound)
 bodies['approval-ready.json']=enc({'source_commit':m.SOURCE,'job_id':m.JOB,'run_root':m.RUN,'binding_sha256':m.sha(bodies['bindings/manifest.json']),'public_key_sha256':m.sha(bodies['bindings/public.json'])})
 for role,name in m.CHECKPOINT_MAP.items():
  if name not in m.TENSOR_NAMES:manifest['files'][role]['sha256']=m.sha(bodies[name])
 manifest['config_sha256']=m.sha(bodies['full_state/global_step_5/resolved-config.json']);bodies['full_state/global_step_5/manifest.json']=enc(manifest);request['checkpoint_manifest_sha256']=m.sha(bodies['full_state/global_step_5/manifest.json']);bodies['train.log'],join=logs(manifest,request);bodies['driver-current-prefix.out']=bodies['train.log']
 rows={n:row(n,b,i)for i,(n,b)in enumerate(bodies.items(),1)};metadata={}
 for i,name in enumerate(m.TENSOR_NAMES,10000):
  role=next(k for k,v in m.CHECKPOINT_MAP.items()if v==name);metadata[name]={'physical':m.RUN+'/'+name,'bytes':987,'uid':0,'device':1,'inode':i,'stat_before':[1,i,0,987,10,10],'stat_after':[1,i,0,987,10,10],'downloaded':False,'link_count':1,'link_count_after':1,'declared_sha256':manifest['files'][role]['sha256'],'reading_scope':'METADATA_ONLY_DECLARED_CHECKPOINT_SHA_NOT_FRESH_FULL_SHA'}
 driver={'pid':manifest['driver_pid'],'start_ticks':'100','uid':0,'cgroup':'/job_3304/step_0','stdout_physical':REQUEST['driver_stdout_physical'],'status_original':'LOCAL_FIXTURE_NOT_ACTUAL_PROC'}
 r={'version':1,'kind':m.SCOPE,'scope':m.SCOPE,'job_id':m.JOB,'source_commit':m.SOURCE,'run_root':m.RUN,'uid':0,'hostname':'h100-10','files':rows,'tensor_metadata':metadata,'driver_before':driver,'driver_after':driver,'train_prefix_join':join,'driver_prefix_join':join,'full_archive_members':100,'raw_bytes':sum(x['bytes']for x in rows.values()),'gzip_bytes':sum(x['gzip_bytes']for x in rows.values()),'missing_optional':list(m.OPTIONAL),'scope_fixture':'LOCAL_SYNTHETIC_METADATA_BYTE_READER_ONLY'}
 return r,request
class Tests(unittest.TestCase):
 def test_compile_same_generated_production_reader(self):compile(m.remote_program(REQUEST),'actual_generated','exec')
 def test_exact_full_roles_and_limits(self):
  self.assertEqual(len(m.CHECKPOINT_MAP),141);self.assertEqual(len(m.TENSOR_NAMES),12);self.assertEqual(sum(k.startswith('trajectory_outer')for k in m.CHECKPOINT_MAP),100);self.assertEqual((m.RAW_LIMIT,m.GZIP_LIMIT,m.WIRE_LIMIT),(3*1024**3,192*1024**2,256*1024**2));self.assertFalse(any('resource'in n for n in m.REQUIRED+m.OPTIONAL))
 def test_bounds_real_actual_size_allowed_not_sampling(self):
  ns['budget'](1874673584,100*1024**2)
  for a,b in ((m.RAW_LIMIT+1,1),(1,m.GZIP_LIMIT+1),(True,1),(1,False),(-1,1)):
   with self.assertRaises(ValueError):ns['budget'](a,b)
 def test_whole_synthetic_100_archive_byte_save_pending_only(self):
  r,req=fixture()
  with tempfile.TemporaryDirectory()as d:
   out=Path(d).resolve()
   with patch.object(m,'decompressed',side_effect=AssertionError('No full JSONL in memory decode')):v=m.save_snapshot(enc(r),req,out)
   self.assertEqual(v['cpu_pipeline_originals']['status'],'CPU_PIPELINE_PENDING_ORIGINALS_NOT_COMPLETION');self.assertEqual(len(list((out/'raw/trajectories').rglob('*jsonl'))),45);self.assertFalse(list((out/'raw').rglob('*.pt')));self.assertFalse(list((out/'raw').rglob('*.safetensors')))
 def test_manifest_bad_path_and_unknown_role_reject(self):
  for transform in (lambda v:v['files']['algorithm'].update(path=m.RUN+'/../other/state.json'),lambda v:v['files'].update({'fake':{}})):
   value=copy.deepcopy(REAL);transform(value)
   with self.assertRaises(ValueError):ns['check_manifest'](value,dict(REQUEST,driver_pid=value['driver_pid']))
 def test_bool_and_unknown_physical_metadata_rejected(self):
  for change in ({'uid':False},{'bytes':True},{'physical':m.RUN+'/../other/train.log'},{'reading_scope':'FULL_LIFETIME_CLOSED'}):
   r=row('train.log',b'x');r.update(change)
   with self.assertRaises(ValueError):m.validate_row('train.log',r,REQUEST)
 def test_stream_roundtrip_multiple_output_chunks(self):
  body=(bytes(range(256))*8192)+b'\n';r=row('student.jsonl',body)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'raw';saved=m.save_streamed_member(r,p);self.assertEqual(p.read_bytes(),body);self.assertEqual(saved['sha256'],m.sha(body))
 def test_stream_corruption_truncation_concat_overflow_preserved(self):
  for variant in ('sha','truncated','concat','shortdeclared'):
   r=row('student.jsonl',b'abcdef\n'*100);packed=base64.b64decode(r['gzip_base64'])
   if variant=='sha':r['sha256']='0'*64
   if variant=='truncated':packed=packed[:-3]
   if variant=='concat':packed+=gzip.compress(b'extra')
   if variant=='shortdeclared':r['bytes']-=1
   r.update(gzip_bytes=len(packed),gzip_sha256=m.sha(packed),gzip_base64=base64.b64encode(packed).decode())
   with tempfile.TemporaryDirectory()as d:
    p=Path(d)/'partial'
    with self.assertRaises(ValueError):m.save_streamed_member(r,p)
    self.assertTrue(p.exists())
 def test_stream_bool_lengths_and_nooverwrite(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'raw';r=row('student.jsonl',b'a\n');m.save_streamed_member(r,p)
   with self.assertRaises(FileExistsError):m.save_streamed_member(r,p)
   for key in ('bytes','gzip_bytes'):
    bad=copy.deepcopy(r);bad[key]=True
    with self.assertRaises(ValueError):m.save_streamed_member(bad,Path(d)/key)
 def remote_member(self,path,**kwargs):return ns['member'](path,uid=os.getuid(),**kwargs)
 def test_remote_same_fd_whole_gzip_and_tensor_metadata_zero_reads(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d).resolve()/'raw.jsonl';p.write_bytes(b'{}\n'*128);r=self.remote_member(p);self.assertEqual(gzip.decompress(base64.b64decode(r['gzip_base64'])),p.read_bytes())
   with patch.object(os,'read',side_effect=AssertionError('TENSOR CONTENT MUST NOT BE READ')):meta=self.remote_member(p,metadata=True,expected='a'*64)
   self.assertFalse(meta['downloaded']);self.assertEqual(meta['declared_sha256'],'a'*64);self.assertNotIn('sha256',meta)
 def test_remote_symlink_hardlink_and_wronguid_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d).resolve()/'a';p.write_bytes(b'{}');q=p.with_name('alias');q.symlink_to(p)
   with self.assertRaises(ValueError):self.remote_member(q)
   q.unlink();os.link(p,q)
   with self.assertRaises(ValueError):self.remote_member(p)
   q.unlink()
   with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid()+1)
 def mutate_member(self,mode,prefix):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d).resolve()/'raw.log';p.write_bytes(b'first\n');original=os.read;calls=[]
   def read(fd,n):
    raw=original(fd,n)
    if not calls:
     if mode=='growth':
      with p.open('ab')as f:f.write(b'appended\n')
     elif mode=='replace':
      q=p.with_name('new');q.write_bytes(b'first\n');os.replace(q,p)
     else:p.write_bytes(b'other\n')
    calls.append(1);return raw
   with patch.object(os,'read',read):return self.remote_member(p,prefix=prefix)
 def test_remote_append_prefix_only_and_mutation_rejected(self):
  r=self.mutate_member('growth',True);self.assertEqual(r['bytes'],6);self.assertEqual(r['stat_after'][3],15)
  for mode,prefix in (('growth',False),('replace',True),('overwrite',True)):
   with self.assertRaises(ValueError):self.mutate_member(mode,prefix)
 def test_log_no_release_or_sixth_outer(self):
  r,req=fixture();manifest=m.unique(gzip.decompress(base64.b64decode(r['files']['full_state/global_step_5/manifest.json']['gzip_base64'])));body,join=logs(manifest,req);gate=types.ModuleType('g');gate.__file__='local';exec(compile((H.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py').read_bytes(),'g','exec'),gate.__dict__)
  for suffix in (b'OPSD_E1_STEP5_RELEASED_JSON={"driver_pid":'+str(manifest['driver_pid']).encode()+b'}\n',body.splitlines(keepends=True)[4].replace(b'"actual_outer_updates": 5',b'"actual_outer_updates": 6')):
   with self.assertRaises(ValueError):ns['check_logs'](body+suffix,gate,manifest,req,list(range(1000,1008)))
 def test_nonzero_cpu_exit_never_completed(self):
  bodies={m.CPU_PREFIX+'native-exit-original.json':enc({'returncode':7})};self.assertEqual(m.cpu_status(bodies,REAL,REQUEST)['status'],'CPU_PIPELINE_RECORDED_NONZERO_ORIGINALS')
  bodies[m.CPU_PREFIX+'native-exit-original.json']=enc({'returncode':False})
  with self.assertRaises(ValueError):m.cpu_status(bodies,REAL,REQUEST)
 def test_completed_originals_fixed_v3_and_true_wait_join(self):
  launch={'job_id':m.JOB,'source_commit':m.SOURCE,'checkpoint_manifest_sha256':REQUEST['checkpoint_manifest_sha256'],'pause_nonce':REAL['pause_nonce'],'driver_pid':REAL['driver_pid'],'driver_start_ticks':REQUEST['driver_start_ticks'],'controller_sha256':m.CONTROLLER_SHA,'scope':m.CONTROLLER_SCOPE,'contract':m.CONTROLLER_CONTRACT,'output':m.RUN+'/'+m.CPU_PREFIX.rstrip('/')}
  values={'launch-original.json':launch,'controller-start.original.json':{'process':{'pid':666,'start_ticks':'200'}},'controller-child-start.original.json':{'child_pid':666,'child':{'pid':666,'start_ticks':'200'}},'srun-child-start.original.json':{'child_pid':777,'parent':{'pid':666}},'srun-exit.original.json':{'returncode':0,'waited':True,'waited_by_pid':666,'child_pid':777},'controller-exit.original.json':{'returncode':0,'waited':True,'waited_by_pid':555,'child_pid':666},'native-exit-original.json':{'returncode':0},'hf-exit-original.json':{'returncode':0}}
  for name,scope in (('native.json','NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE'),('hf.json','ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA')):values[name]={'scope':scope,'source_commit':m.SOURCE,'job_id':m.JOB,'run_root':m.RUN,'checkpoint_manifest_sha256':REQUEST['checkpoint_manifest_sha256'],'cuda_is_initialized':False,'passed':True}
  cg='/job_3304/step_1';values['cpu-step-original.json']={'pid':777,'uid':0,'hostname':'h100-10','job_id':m.JOB,'cuda_visible_devices':'','cuda_is_initialized':False,'manifest_sha256':REQUEST['checkpoint_manifest_sha256'],'cgroup':cg};values['cpu-step-before-exec.original.json']={'job_id':m.JOB,'cuda_visible_devices':'','omp_num_threads':'1','step_id':'1','process':{'pid':777,'start_ticks':'300','uid':0,'cgroup':cg}}
  done={'scope':'CPU_STORED_STATE_AND_HF_INSPECTION_ONLY_NOT_SIGNATURE_OR_RELEASE','source_commit':m.SOURCE,'job_id':m.JOB,'manifest_sha256':REQUEST['checkpoint_manifest_sha256'],'cuda_is_initialized':False,'files':{n:m.sha(enc(values[n]))for n in ('native.json','hf.json','native-exit-original.json','hf-exit-original.json','cpu-step-original.json')}};values['inspection-completed-review-required.json']=done
  bodies={m.CPU_PREFIX+n:enc(v)for n,v in values.items()};self.assertEqual(m.cpu_status(bodies,REAL,REQUEST)['status'],'CPU_PIPELINE_WAITED_COMPLETED_ORIGINALS_REVIEW_REQUIRED')
  for name,change in (('launch-original.json',{'controller_sha256':'868f145430dc152fb9d4788c07db1e3c83393c5e705f35196a837a9961d81c01'}),('srun-exit.original.json',{'waited':False}),('srun-exit.original.json',{'waited_by_pid':999}),('inspection-completed-review-required.json',{'files':{}})):
   bad=dict(bodies);value=dict(values[name]);value.update(change);bad[m.CPU_PREFIX+name]=enc(value)
   with self.assertRaises(ValueError):m.cpu_status(bad,REAL,REQUEST)
 def test_transport_nonzero_raw_preserved_no_retry(self):
  with tempfile.TemporaryDirectory()as d:
   out=Path(d).resolve()/'fresh';calls=[];transport=types.ModuleType('transport')
   def remote(*a,**k):calls.append(1);raise subprocess.CalledProcessError(255,['LOCAL_ONLY'],output=b'partial-wire',stderr=b'local-failure')
   transport.remote=remote;argv=['collector','--binding',str(H/m.BINDING_NAME),'--output',str(out),'--checkpoint-manifest-sha256','a'*64,'--pause-nonce','b'*64,'--driver-pid','999','--driver-start-ticks','100']
   with patch.dict(sys.modules,{'transport':transport}),patch.object(m,'local_request',lambda a:dict(REQUEST)),patch.object(sys,'argv',argv):
    with self.assertRaises(subprocess.CalledProcessError):m.main()
   self.assertEqual(calls,[1]);self.assertEqual((out/'transport.stdout.original').read_bytes(),b'partial-wire');self.assertEqual(json.loads((out/'transport-exit.original.json').read_bytes())['returncode'],255);self.assertFalse((out/'inventory.json').exists())
 def test_wire_bool_uid_missing_archive_and_unknown_member_reject(self):
  changes=(lambda r:r.update(uid=False),lambda r:r['files'].pop('trajectories/outer_000005/student.jsonl'),lambda r:r['files'].update({'resources.ndjson':row('resources.ndjson',b'{}')}))
  for change in changes:
   r,req=fixture();change(r)
   with tempfile.TemporaryDirectory()as d:
    with self.assertRaises(ValueError):m.save_snapshot(enc(r),req,Path(d))
 def test_current_submit_join_cpu_only(self):
  transport=types.ModuleType('transport');transport.remote=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No SSH'))
  a=types.SimpleNamespace(binding=H/m.BINDING_NAME,driver_pid=REAL['driver_pid'],driver_start_ticks='100',checkpoint_manifest_sha256='a'*64,pause_nonce=REAL['pause_nonce'])
  with patch.dict(sys.modules,{'transport':transport}):request=m.local_request(a)
  self.assertEqual(request['job_id'],'3304');self.assertEqual(request['source_commit'],m.SOURCE)
if __name__=='__main__':unittest.main(verbosity=2)
