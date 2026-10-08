#!/usr/bin/env python3
"""Standard Search-R1 only: one owned same-node retriever and original trainer."""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,secrets,socket,struct,sys,threading,time
sys.dont_write_bytecode=True
from allocator_policy import bootstrap,check_config,check_environment
if __name__=='__main__':bootstrap()
from pipeline import atomic,verify_files,validate_peer
from student_binding import check as check_source_v4
from checkpoint_inventory import sha,inventory,tokenizer_compatible
from ipc import identity,read,send
from resource_audit import verify_ready,verify_updates,charged_memory
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')

def bound(ref):
 p=pathlib.Path(ref['path'])
 if p.is_symlink() or ROOT not in p.resolve().parents or sha(p)!=ref['sha256']:raise ValueError('Bound source/evidence bytes changed')
 return json.loads(p.read_text())
def canonical(c):return hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='independent_gate'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_recipe(c):
 if c['mode']!='topd300' or c['expected_outer_steps']!=300:raise ValueError('Only fresh student300')
 command=c['training_overrides'];args={}
 for value in command:
  key,val=value.split('=',1);key=key.lstrip('+')
  if key in args:raise ValueError('Duplicate training override '+key)
  args[key]=val
 required={'baselines.method':'topd','baselines.topd.teacher_backend':'local_gpu_bf16_tp8_prompt_logprobs','baselines.topd.alpha':'0.1','baselines.distillation.privileged_answers':'false','actor_rollout_ref.actor.ppo_mini_batch_size':'128','actor_rollout_ref.actor.ppo_micro_batch_size':'16','actor_rollout_ref.actor.ppo_epochs':'1','actor_rollout_ref.actor.loss_agg_mode':'sequence_mean','actor_rollout_ref.actor.optim.lr':'1e-6','actor_rollout_ref.actor.optim.lr_warmup_steps':'57','actor_rollout_ref.actor.entropy_coeff':'0','actor_rollout_ref.rollout.n_agent':'5','actor_rollout_ref.rollout.n':'1','actor_rollout_ref.rollout.tensor_model_parallel_size':'1','trainer.total_training_steps':'301','trainer.acceptance_step':'5','trainer.test_freq':'0','trainer.val_before_train':'false','actor_rollout_ref.actor.use_kl_loss':'true','actor_rollout_ref.actor.kl_loss_coef':'0.001','actor_rollout_ref.actor.kl_loss_type':'low_var_kl','max_turns':'4','force_final_answer':'true'}
 for key,value in required.items():
  if args.get(key)!=value:raise ValueError('Student shared protocol drift '+key)
 if args.get('baselines.distillation.teacher_path')!=c['teacher_path'] or args.get('actor_rollout_ref.model.path')!=c['base_model']:raise ValueError('Actual teacher/student paths differ')
 return 300

def gate(c):
 # Human-authorized direct training: no pre-training proof/profile admission.
 return {'scope':'HUMAN_AUTHORIZED_DIRECT_GRPO100_NO_PRETRAINING_PROOF'}

def support(ref):
 if sha(ref['path'])!=ref['sha256']:raise ValueError('Owned lifecycle source drift')
 spec=importlib.util.spec_from_file_location('r1_owned_support',ref['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def gpu_baseline(tools,snapshot):
 rows=[]
 for line in snapshot['processes']:
  u,p,*_=line.split(',');r=tools.process_identity(int(p))
  if 'starttime_ticks' not in r:raise ValueError('Original GPU identity unavailable')
  rows.append({'uuid':u.strip(),'pid':int(p),'identity':r})
 return rows

def baseline_gpu_missing(tools,snapshot,records):
 present={(l.split(',')[0].strip(),int(l.split(',')[1])) for l in snapshot['processes']}
 return [r for r in records if (r['uuid'],r['pid']) not in present or tools.process_identity(r['pid']).get('starttime_ticks')!=r['identity']['starttime_ticks']]

def numeric_visible(inventory,leased):
 mapping={}
 for line in inventory:
  pieces=[v.strip() for v in line.split(',')];index=int(pieces[0]);uuid=pieces[1]
  if uuid in mapping or index in mapping.values():raise ValueError('GPU inventory duplicate')
  mapping[uuid]=index
 if len(leased)!=8 or len(set(leased))!=8 or not set(leased)<=set(mapping):raise ValueError('Foreign/nonallocated numeric GPU mapping')
 return ','.join(str(mapping[uuid]) for uuid in leased)

def healthy(url,child,guardian,timeout=2400):
 import requests
 session=requests.Session();session.trust_env=False;deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  guardian.check()
  if child.poll() is not None:raise RuntimeError('Retriever exited before health')
  try:
   r=session.get(url,timeout=3)
   if r.status_code==200:return {'url':url,'status_code':200,'content_type':r.headers.get('content-type','')}
  except requests.RequestException:pass
  time.sleep(1)
 raise TimeoutError('Source-bound retriever docs not healthy')

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();os.umask(0o077)
 c=json.loads(pathlib.Path(a.config).read_text());check_environment();steps=verify_recipe(c)
 check_source_v4(c,read_native=False)
 for key,root in [('source_manifest',c['candidate']),('tools_manifest',c['tools'])]:
  manifest=bound(c[key]);records=dict(manifest['files']);f=pathlib.Path(c[key]['path']);records[str(f.relative_to(root))]={'size':f.stat().st_size,'sha256':c[key]['sha256']};verify_files(root,records)
 if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<128 or int(os.environ.get('SLURM_MEM_PER_NODE','0'))<768*1024:raise ValueError('Only own h11 8GPU/128CPU/768GiB allocation')
 pre=support(c['preflight_source']);pre.validate_inherited_gpu_leases(8)
 leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);fds=tuple(int(r['fd']) for r in leases)
 if len(leases)!=8 or len({r['uuid'] for r in leases})!=8:raise ValueError('Actual8 leases absent')
 c.update(effective_cuda_allocator_environment=check_environment(),job=os.environ['SLURM_JOB_ID'],uuids=[r['uuid'] for r in leases],cgroup=identity(os.getpid())['cgroup'])
 # CPU-only fresh source/assets; no actor model construction here.
 from retrieval_gate import verify_retrieval
 from compiler_headers import verify as headers
 asset=verify_retrieval(ROOT,c['retrieval_asset_binding'],c['retriever_command'][3]);header=headers(c['compiler_headers'])
 base=inventory(c['base_model'],expected_role='student3b')
 from official_student import verify as verify_official_student
 student_identity=verify_official_student(base,c['student_model_identity'])
 teacher=inventory(c['teacher_path']);expected=bound(c['teacher_actual_result'])['export']
 if expected.get('files') is not None and teacher['files']!=expected['files']:raise ValueError('Approved teacher full-HF export bytes differ')
 if expected.get('files') is None:
  approval=bound(c['teacher_actual_result'])
  if approval.get('status')!='USER_APPROVED_TRAINED_CHECKPOINT' or approval.get('checkpoint_step')!=85 or teacher['directory']!=approval['export']['directory']:raise ValueError('Teacher checkpoint approval differs')
 for ref in c['input_bindings']:
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Common train/validation split drift')
 # No second quality cutoff; actual teacher100 source is already bound.
 root=pathlib.Path(c['evidence_root']);root.mkdir(mode=0o700,exist_ok=False)
 atomic(root/'teacher-hf-fresh.json',teacher);atomic(root/'effective-runtime-config.json',c);atomic(root/'retrieval-fresh-inventory.json',asset);atomic(root/'compiler-headers-fresh.json',header);atomic(root/'base-hf-fresh.json',base);atomic(root/'official-student-identity.json',student_identity)
 lock=os.open(ROOT/'tools/search-r1-32b-grpo100.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tools=support(c['support']);scope,relative=tools.cgroup_scope(c['job']);owned=tools.OwnedProcesses(scope,relative)
 baseline={pid:tools.process_identity(pid) for pid in owned.baseline};gpu_original=gpu_baseline(tools,tools.gpu_sample())
 from window_guardian import Guardian
 guardian=Guardian(ROOT,c['telemetry_policy'],root,scope,owned);error=None;listener=None;connection=None;sockpath=None;terminal_drained=False;roles=[];observed={};stop=threading.Event();sampler_errors=[];trainer=None;report=None
 def sampler():
  with open(root/'own-resource-samples.jsonl','x') as f:
   while not stop.is_set():
    try:
     pids=tools.cgroup_pids(scope)-owned.baseline;rows=[]
     for pid in sorted(pids):
      from process_observation import observe
      r=observe(tools,pid,observed.get(pid));rows.append(r)
      if 'identity' in r:observed[pid]=r['identity']
     raw={'epoch':time.time(),'cgroup_memory':charged_memory(scope),'processes':rows,'rss_scope':'SUMMED_PROCESS_RSS_INCLUDES_SHARED_PAGES_SAMPLED_NOT_UNIQUE_RAM','gpu':tools.gpu_sample()};f.write(json.dumps(raw)+'\n');f.flush()
    except BaseException as e:sampler_errors.append(repr(e));stop.set()
    stop.wait(1)
   os.fsync(f.fileno())
 monitor=threading.Thread(target=sampler,name='r1-resource-sampler')
 try:
  monitor.start()
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CPATH=header['cpath'],PYTHONPATH=str(pathlib.Path(__file__).parent)+':'+c['teacher_backend']+':'+c['candidate']+':'+str(ROOT/'retriever-gpu-lib')+':'+str(ROOT/'test-tools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NO_PROXY='127.0.0.1,localhost')
  cache=root/'cache';cache.mkdir(mode=0o700)
  for k,name in [('VLLM_CACHE_ROOT','vllm'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TORCH_EXTENSIONS_DIR','torch-extensions'),('TMPDIR','tmp')]:
   d=cache/name;d.mkdir(mode=0o700);env[k]=str(d)
  env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
  for k in list(env):
   if k.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(k)
  guardian.prepare_retriever_start(relative)
  retriever=owned.spawn(c['retriever_command'],root/'retriever.log',env=env,pass_fds=fds);roles.append({'role':'retriever','pid':retriever.pid});guardian.attach_retriever(retriever.pid);guardian.start()
  atomic(root/'retriever-docs-ready.json',healthy(c['retriever_health_url'],retriever,guardian));guardian.resident_retriever(retriever.pid,relative)
  # Preserve原Search-R1 actual retrieval response protocol, before any training.
  import requests
  session=requests.Session();session.trust_env=False;r=session.post(c['retriever_url'],json={'queries':['Who was the first president of the United States?'],'topk':3,'return_scores':True},timeout=120);r.raise_for_status();body=r.json()
  if len(body.get('result',[]))!=1 or len(body['result'][0])!=3 or not all(d['document']['contents'] for d in body['result'][0]):raise ValueError('Actual local fullwiki top3 response invalid')
  atomic(root/'retrieval-standard-client-ready.json',{'http_status':r.status_code,'request_return_scores':True,'response':body})
  listener=socket.socket(socket.AF_UNIX);sockpath='/dev/shm/r1-'+c['job']+'-'+str(os.getpid())+'.sock';listener.bind(sockpath);os.chmod(sockpath,0o600);listener.listen(1);listener.settimeout(1)
  binding={'socket':sockpath,'server_identity':identity(os.getpid()),'capability':secrets.token_hex(32)};atomic(root/'terminal-binding.json',binding)
  guardian.begin_phase('teacher-tp8-init')
  teacher_process=owned.spawn([str(ROOT/'env/bin/python'),'-m','topd_vllm_teacher.daemon','--teacher',c['teacher_path'],'--directory',str(root/'teacher'),'--preflight',c['preflight_source']['path'],'--gpu-memory-utilization',str(c['teacher_gpu_memory_utilization'])],root/'teacher.log',env=env,pass_fds=fds)
  roles.append({'role':'frozen-teacher-tp8','pid':teacher_process.pid})
  deadline=time.monotonic()+2400
  while not (root/'teacher/ready.json').is_file():
   guardian.check()
   if teacher_process.poll() is not None:raise RuntimeError('Teacher startup failed; original log retained')
   if time.monotonic()>=deadline:raise TimeoutError('Teacher ready timeout')
   time.sleep(1)
  ready_teacher=json.loads((root/'teacher/ready.json').read_text())
  if ready_teacher['status']!='FROZEN_BF16_TP8_WEIGHTS_RESIDENT_KV_RELEASED':raise RuntimeError('Teacher actual initial KV release missing')
  atomic(root/'actual-teacher-ready.json',ready_teacher)
  guardian.begin_phase('actor-rollout-init')
  train_env=dict(env,CUDA_VISIBLE_DEVICES=numeric_visible(tools.gpu_sample()['gpus'],c['uuids']))
  atomic(root/'actual-vllm-numeric-device-map.json',{'leased_uuids':c['uuids'],'numeric_visible':train_env['CUDA_VISIBLE_DEVICES'],'scope':'PINNED_VLLM063_PHYSICAL_INDEX_MAPPING_SAME_8_LEASES'})
  trainer=owned.spawn([str(ROOT/'env/bin/python'),str(pathlib.Path(__file__).with_name('train_entry.py')),'--config',str(root/'effective-runtime-config.json'),'--binding',str(root/'terminal-binding.json')],root/'trainer.log',env=train_env,pass_fds=fds);roles.append({'role':'original-standard-main-ppo','pid':trainer.pid});ready=False
  while True:
   guardian.check()
   if sampler_errors:raise RuntimeError('Actual resource sampler failed '+repr(sampler_errors))
   if not ready:
    receipt=verify_ready(root/'actor-resource-audit',c,trainer.pid,tools,guardian.safety.nvml_snapshot,validate_peer)
    if receipt:atomic(root/'global-worker-ready.json',receipt);guardian.sample();guardian.close_ready();ready=True
   try:connection,_=listener.accept()
   except socket.timeout:
    if trainer.poll() is not None:raise RuntimeError('Original trainer exited before terminal-held')
    continue
   connection.settimeout(30);peer,uid,_=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));validate_peer(peer,uid,trainer.pid,c['cgroup']);request=read(connection)
   if not ready or request.get('capability')!=binding['capability'] or request.get('operation')!='TERMINAL_HELD' or request.get('identity')!=identity(peer) or request.get('job')!=c['job'] or request.get('effective_config_sha256')!=sha(root/'effective-runtime-config.json'):raise ValueError('Terminal identity/capability/source differs')
   actual=bound(request['actual_outer_evidence'])
   if request['actual_outer_evidence']['path']!=c['actual_outer_receipt_path'] or actual['actual_completed_outer_steps']!=list(range(1,steps+1)) or actual['export_step']!=steps:raise ValueError('Actual standard outer count differs')
   records=[json.loads((root/'actor-resource-audit'/('actor-updates-rank%d-outer%d.json'%(rank,steps))).read_text()) for rank in range(8)];updates=verify_updates(records,steps)
   exported=inventory(c['final_checkpoint'],expected_role='student3b');tokenizer=tokenizer_compatible(c['final_checkpoint'],c['base_model'])
   guardian.finish();guardian.close();stop.set();monitor.join(45)
   if monitor.is_alive() or sampler_errors:raise RuntimeError('Terminal resource monitor not drained')
   guardian.check();terminal_drained=True
   held=root/'trainer-terminal-held.json'
   send(connection,{'status':'AUTHORIZED_OWN_SHUTDOWN','held_sha256':sha(held)})
   if trainer.wait(timeout=180)!=0:raise RuntimeError('Original trainer own shutdown failed')
   report={'scope':'ACTUAL_TOPD300_UNREVIEWED_NOT_ACCEPTANCE_SIGNOFF','job':c['job'],'evidence_root':str(root),'actual_update_evidence':updates,'actual_outer_evidence':request['actual_outer_evidence'],'export':exported,'tokenizer':tokenizer,'effective_config_sha256':sha(root/'effective-runtime-config.json'),'base_hf_fresh':{'path':str(root/'base-hf-fresh.json'),'sha256':sha(root/'base-hf-fresh.json')},'sampled_parameter_change_not_full_semantic_approval':True}
   break
 except BaseException as e:error=e
 finally:
  from final_cleanup import finalize_owned_pipeline
  error,cleanup=finalize_owned_pipeline(tools=tools,owned=owned,scope=scope,baseline=baseline,gpu_original=gpu_original,baseline_gpu_missing=baseline_gpu_missing,roles=roles,observed=observed,guardian=guardian,sampler_stop=stop,monitor=monitor,sampler_errors=sampler_errors,terminal_drained=terminal_drained,terminal_connection=connection,listener=listener,root=root,primary_error=error)
  if sockpath:
   try:os.unlink(sockpath)
   except FileNotFoundError:pass
   except OSError as unlink_error:
    cleanup['socket_unlink_error']=repr(unlink_error);error=error or unlink_error
  atomic(root/'owned-cleanup.json',cleanup);report=report or {'job':c['job'],'evidence_root':str(root)};report.update(status='FAILED' if error else 'COMPLETE_UNREVIEWED',cleanup_passed=error is None,cleanup={'path':str(root/'owned-cleanup.json'),'sha256':sha(root/'owned-cleanup.json')},error=repr(error) if error else None);atomic(root/'train-result.json',report);os.close(lock)
 if error:raise error
 print(json.dumps({'status':report['status'],'job':c['job'],'result':str(root/'train-result.json')}))
if __name__=='__main__':main()
