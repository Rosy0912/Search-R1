#!/usr/bin/env python3
"""Owned G1 initial calibration and synchronous refresh supervisor. No auto-submit."""
import argparse,fcntl,importlib.util,hashlib,json,os,pathlib,secrets,socket,struct,sys,time,threading,urllib.parse
from pipeline import Supervisor,atomic,original_round,verify_overlay,validate_peer
from ipc import identity,read,send
from checkpoint_inventory import sha
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')

def bound_json(binding):
 p=pathlib.Path(binding['path'])
 if p.is_symlink() or sha(p)!=binding['sha256']:raise ValueError('bound receipt bytes changed')
 return json.loads(p.read_text())

def reviewed_recipe_sha(config):
 unsigned={k:v for k,v in config.items() if k!='independent_gate'}
 return hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify_gate(config,mode):
 """Source/CPU preparation never silently becomes GPU calibration/training approval."""
 gate=bound_json(config['independent_gate'])
 if gate.get('reviewed_static_recipe_sha256')!=reviewed_recipe_sha(config):raise ValueError('complete reviewed static recipe drift')
 expected_scope='G1_32B_INITIAL_CALIBRATION_SOURCE_AND_ALLOCATION_GATE' if mode=='initial-only' else 'G1_32B_CALIBRATION_AND_UPDATE_RESOURCE_PROOF'
 if gate.get('status')!='PASS' or gate.get('scope')!=expected_scope:raise ValueError('missing new32B actual resource proof')
 for key in ('candidate_manifest_sha256','producer_manifest_sha256','pipeline_manifest_sha256','base64_result_sha256','base_model_inventory_sha256'):
  if gate.get(key)!=config[key]:raise ValueError('independent execution gate source binding changed')
 if gate.get('actual_gpu_count')!=8 or gate.get('host_memory_request_gib')!=768 or gate.get('global_actor_microbatch')!=64:raise ValueError('new32B actual resource protocol mismatch')
 for key in ('actual_raw_evidence','independent_review'):
  evidence=bound_json(gate[key])
  if key=='independent_review' and evidence.get('status')!='PASS':raise ValueError('independent review did not pass')
 if mode!='initial-only' and (gate.get('actual_initial_calibration_complete') is not True or gate.get('actual_actor_update_complete') is not True):raise ValueError('CPU proofs cannot authorize actual100 outer training')
 return gate

def own_path(path):
 p=pathlib.Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('foreign path')
 return p

def process_arguments(config,binding):
 command=list(config['training_command'])
 if command[:3]!=[str(ROOT/'env/bin/python'),'-m','verl.trainer.main_ppo_eco']:raise ValueError('unbound training entry')
 overrides={x.split('=',1)[0].lstrip('+'):x.split('=',1)[1] for x in command[3:] if '=' in x}
 required={'actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','actor_rollout_ref.rollout.n_agent':'5','data.train_batch_size':'128','trainer.n_gpus_per_node':'8','trainer.nnodes':'1','trainer.total_training_steps':'101','trainer.save_freq':'25','trainer.g1_refresh_freq':'25','actor_rollout_ref.actor.optim.foreach':'false','actor_rollout_ref.actor.fsdp_config.param_offload':'true','actor_rollout_ref.actor.fsdp_config.grad_offload':'true','actor_rollout_ref.actor.fsdp_config.optimizer_offload':'true','actor_rollout_ref.ref.fsdp_config.param_offload':'true','data.max_prompt_length':'4096','data.max_response_length':'500','data.max_start_length':'2048','data.max_obs_length':'500','max_turns':'4','actor_rollout_ref.rollout.gpu_memory_utilization':'0.60'}
 for k,v in required.items():
  if overrides.get(k)!=v:raise ValueError('production recipe drift '+k)
 for key,value in {'actor_rollout_ref.model.path':config['base_model'],'trainer.default_local_dir':config['checkpoint_root'],'data.train_files':config['train_file'],'data.val_files':config['val_file'],'algorithm.adv_estimator':'grpo','actor_rollout_ref.actor.state_masking':'true'}.items():
  if overrides.get(key)!=value:raise ValueError('actual input/caller protocol drift '+key)
 return command+['+trainer.g1_owned_pipeline_binding='+str(binding),'+trainer.g1_actual_outer_receipt='+config['actual_outer_receipt_path'],'+trainer.g1_probe_phase_url='+config['probe_url'],'+trainer.g1_probe_manifest_path='+str(pathlib.Path(config['refit_args']['manifest_root'])/('live_'+config['experiment'])/'manifest.json')]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',choices=['initial-only'],required=True);a=ap.parse_args();os.umask(0o077)
 config=json.loads(pathlib.Path(a.config).read_text());verify_gate(config,a.mode)
 from pipeline import verify_files
 if {x['kind'] for x in config['source_bindings']}!={'candidate','producer','pipeline'}:raise ValueError('three exact isolated source bindings required')
 for source in config['source_bindings']:
  if source['manifest']['sha256']!=config[source['kind']+'_manifest_sha256']:raise ValueError('source manifest differs from independent approval')
  manifest=bound_json(source['manifest']);records=manifest['files'];relative=str(pathlib.Path(source['manifest']['path']).relative_to(source['root']))
  records=dict(records);records[relative]={'size':pathlib.Path(source['manifest']['path']).stat().st_size,'sha256':source['manifest']['sha256']}
  verify_files(source['root'],records)
 if config['refit_args']['layers']!='21,24,27' or config['refit_args']['group_size']!=5 or config['refit_args']['commit_position']!='Xa' or config['refit_args']['commit_label']!='flip':raise ValueError('original calibration defaults drift')
 if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('own8GPU h11 allocation required')
 if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<128 or int(os.environ.get('SLURM_MEM_PER_NODE','0'))<768*1024:raise ValueError('actual Slurm CPU/RAM allocation below128/768GiB')
 preflight=config['preflight_source'];preflight_path=own_path(preflight['path'])
 if sha(preflight_path)!=preflight['sha256']:raise ValueError('common preflight bytes drift')
 spec=importlib.util.spec_from_file_location('common_preflight',preflight_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.validate_inherited_gpu_leases(8)
 config['job']=os.environ['SLURM_JOB_ID'];config['owner_context']['config']['job']=config['job'];config['owner_context']['config']['leases']=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
 config['owner_context']['config']['uuids']=[r['uuid'] for r in config['owner_context']['config']['leases']]
 for key in ('retriever_url','retriever_health_url','probe_url','probe_health_url'):
  parsed=urllib.parse.urlsplit(config[key])
  if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or not parsed.port:raise ValueError('own loopback URL required: '+key)
 producer=own_path(config['producer']);sys.path.insert(0,str(producer))
 from owner import support,leases,gpu_baseline,baseline_gpu_missing
 from gates import conditional_base64
 route=conditional_base64(config['owner_context']['config']['base64_binding'])
 if route['correct']>=26:raise ValueError('quality branch does not authorize G1 fallback')
 tools=support(producer/'owner_support.py',config['owner_context']['support_sha256']);scope,relative=tools.cgroup_scope(str(config['job']))
 config['owner_context']['config']['cgroup']=relative
 fds=leases(config['owner_context']['config']['leases']);config['uuids']=[x['uuid'] for x in config['owner_context']['config']['leases']]
 if set(config['uuids'])!=set(config['owner_context']['config']['uuids']):raise ValueError('lease config identity mismatch')
 config['cgroup']=identity(os.getpid())['cgroup'];root=own_path(config['evidence_root']);root.mkdir(exist_ok=False)
 atomic(root/'effective-runtime-config.json',config)
 lock=os.open(ROOT/'tools/g1-production.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 verify_overlay(config['overlay'],config['overlay_receipt']['path'],config['overlay_receipt']['sha256'])
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join([str(own_path(config['candidate'])),str(own_path(config['overlay'])),str(ROOT/'test-tools')]),NO_PROXY='127.0.0.1,localhost')
 for key in list(env):
  if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key)
 env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['RETRIEVER_URL']=config['retriever_url']
 for key,value in {'PROBE_URL':config['probe_url'],'PROBE_MANIFEST_PATH':str(pathlib.Path(config['refit_args']['manifest_root'])/('live_'+config['experiment'])/'manifest.json'),'ECO_TRUST_VARIANT':'balanced','ECO_TRUST_SIGNAL':'probe','ECO_PROBE_GATE':'1','ECO_BAL_GROUND':'0.6','ECO_BAL_NOSEARCH_BONUS':'0.5','ECO_W_COST':'0.02','ECO_WRONG_GROUND_SCALE':'0.0','ECO_USE_CF':'0'}.items():env[key]=value
 owned=tools.OwnedProcesses(scope,relative);baseline={pid:tools.process_identity(pid) for pid in owned.baseline};gpu_original=gpu_baseline(tools,tools.gpu_sample());error=None;trainer=None;listener=None;roles=[];observed={};sampler_stop=threading.Event();sampler_errors=[]
 def sampler():
  try:
   with open(root/'own-process-resource-samples.jsonl','x') as stream:
    while not sampler_stop.is_set():
     pids=tools.cgroup_pids(scope)-owned.baseline;samples=[tools.process_sample(pid) for pid in sorted(pids)]
     for pid in pids:observed[pid]=tools.process_identity(pid)
     stream.write(json.dumps({'epoch':time.time(),'own_process_samples':samples,'rss_scope':'SUMMED_PROCESS_RSS_INCLUDES_SHARED_PAGES_SAMPLED_NOT_UNIQUE_RAM','gpu':tools.gpu_sample()},sort_keys=True)+'\n');stream.flush();sampler_stop.wait(1)
    os.fsync(stream.fileno())
  except BaseException as exc:sampler_errors.append(str(exc))
 monitor=threading.Thread(target=sampler,name='own-resource-sampler')
 from window_guardian import Guardian,poll_phase_controls
 guardian=Guardian(ROOT,config['telemetry_policy'],root,scope,owned)
 control=root/'phase-control';control.mkdir(mode=0o700)
 env['G1_GUARDIAN_CONTROL']=str(control)
 guardian.phase_poll=lambda:poll_phase_controls(guardian,config,root,tools)
 monitor.start()
 def spawn(role,command):
  child_env=dict(env)
  if role=='retriever':
   child_env['PYTHONPATH']=str(ROOT/'retriever-gpu-lib')+os.pathsep+str(ROOT/'repo')
   child_env.pop('SEARCH_R1_GPU_LEASES',None);child_env.pop('SEARCH_R1_GPU_PREFLIGHT',None)
  child=owned.spawn(command,root/(role+'.log'),env=child_env,pass_fds=fds if role!='retriever' else ());roles.append({'role':role,'pid':child.pid,'identity':tools.process_identity(child.pid)});return child
 def http(path):
  import requests
  session=requests.Session();session.trust_env=False
  response=session.post(config['probe_url'].rsplit('/',1)[0]+'/'+path,timeout=2400);response.raise_for_status();receipt=response.json()
  if path=='barrier_pause':
   actual_probe=next(r for r in roles if r['role']=='probe')
   if receipt.get('pid')!=actual_probe['pid'] or identity(receipt['pid'])['starttime_ticks']!=actual_probe['identity']['starttime_ticks'] or receipt.get('cgroup')!=config['cgroup']:raise ValueError('HTTP probe actual owned identity drift')
  return receipt
 def healthy(url,child):
  import requests
  session=requests.Session();session.trust_env=False;start=time.monotonic()
  while time.monotonic()-start<2400:
   if child.poll() is not None:raise RuntimeError('owned service exited before health')
   try:
    response=session.get(url,timeout=5)
    if response.status_code==200 and (url!=config['probe_health_url'] or response.json().get('ok') is True):return response.json()
   except requests.RequestException:pass
   time.sleep(1)
  raise TimeoutError('own service health timeout')
 try:
  # Retrieval startup must use separately source-bound fullwiki GPU launcher, not CPU fallback.
  from checkpoint_inventory import inventory
  if inventory(config['base_model'])['sha256']!=config['base_model_inventory_sha256']:raise ValueError('official base model differs from quality gate inventory')
  for item in config['input_bindings']:
   if sha(own_path(item['path']))!=item['sha256']:raise ValueError('actual data bytes changed')
  retrieval=spawn('retriever',config['retriever_command']);healthy(config['retriever_health_url'],retrieval)
  guardian.resident_retriever(retrieval.pid,relative);guardian.start();guardian.check()
  initial=root/'initial';initial.mkdir();original_round(config,own_path(config['base_model']),0,env,initial)
  guardian.check()
  atomic(root/'initial-complete.json',{'scope':'ACTUAL_INITIAL_CALIBRATION_UNREVIEWED','frozen_sha256':sha(initial/'frozen-calibration.json'),'before_actor_init':True})
  if a.mode=='train100':
   probe=spawn('probe',config['probe_command']);healthy(config['probe_health_url'],probe)
   socketpath=root/'owner.sock';listener=socket.socket(socket.AF_UNIX);listener.bind(str(socketpath));os.chmod(socketpath,0o600);listener.listen(1);listener.settimeout(1)
   capability=secrets.token_hex(32);binding=root/'owned-binding.json'
   atomic(binding,{'socket':str(socketpath),'capability':capability,'server_identity':identity(os.getpid()),'uuids':config['uuids']})
   trainer_command=[str(ROOT/'env/bin/python'),str(pathlib.Path(__file__).with_name('run_candidate.py')),'--directory',config['candidate'],'--']+process_arguments(config,binding)
   trainer=spawn('trainer',trainer_command);supervisor=Supervisor(config,env,http)
   while trainer.poll() is None:
    for pid in tools.cgroup_pids(scope)-owned.baseline:observed[pid]=tools.process_identity(pid)
    try:connection,_=listener.accept()
    except socket.timeout:continue
    with connection:
     pid,uid,gid=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));validate_peer(pid,uid,trainer.pid,config['cgroup']);request=read(connection)
     if not secrets.compare_digest(request.pop('capability',''),capability) or request.get('operation')!='REFRESH':raise ValueError('unauthorized IPC request')
     response=supervisor.refresh(request);send(connection,response)
   if trainer.returncode!=0 or supervisor.completed!=[25,50,75]:raise RuntimeError('100 outer trainer/refresh failed')
   # Full arbitrary-shard export and all tokenizer files, not just config.json.
   from checkpoint_inventory import inventory,tokenizer_compatible
   final=own_path(config['final_checkpoint'])
   if final!=own_path(config['checkpoint_root'])/'actor/global_step_100':raise ValueError('terminal100 export path drift')
   export=inventory(final);compat=tokenizer_compatible(final,config['base_model'])
   outer_path=own_path(config['actual_outer_receipt_path']);outer=json.loads(outer_path.read_text())
   if outer.get('actual_completed_outer_steps')!=list(range(1,101)) or outer.get('export_step')!=100:raise ValueError('missing actual100 outer evidence')
   atomic(root/'train-result.json',{'scope':'ACTUAL_G1_100_OUTER_UNREVIEWED_NOT_TOPD_ACCEPTANCE','export':export,'tokenizer_compatibility':compat,'actual_outer_evidence':{'path':str(outer_path),'sha256':sha(outer_path)}})
 except BaseException as exc:error=exc
 finally:
  if listener:listener.close()
  try:guardian.finish()
  except BaseException as exc:
   if error is None:error=exc
  guardian.close()
  sampler_stop.set();monitor.join(45)
  if monitor.is_alive() or sampler_errors:
   if error is None:error=RuntimeError('owned sampler did not finish cleanly: '+str(sampler_errors))
  for pid in tools.cgroup_pids(scope)-owned.baseline:observed[pid]=tools.process_identity(pid)
  try:owned.cleanup()
  except BaseException as exc:
   if error is None:error=exc
  remaining=sorted(tools.cgroup_pids(scope)-owned.baseline);missing=[pid for pid,r in baseline.items() if tools.process_identity(pid).get('starttime_ticks')!=r.get('starttime_ticks')];gpu=tools.gpu_sample();gmissing=baseline_gpu_missing(tools,gpu,gpu_original)
  own_gpu=[line for line in gpu['processes'] if int(line.split(',')[1]) in observed]
  atomic(root/'cleanup.json',{'scope':'ACTUAL_OWNED_PIPELINE_EXIT_UNREVIEWED','roles':roles,'identities':observed,'own_processes_remaining':remaining,'baseline_identity_missing':missing,'original_gpu_baseline_missing':gmissing,'own_gpu_records_remaining':own_gpu,'cleanup':getattr(owned,'cleanup_record',None),'parent_torch_imported':'torch' in sys.modules,'parent_cuda_initialized':bool(sys.modules['torch'].cuda.is_initialized()) if 'torch' in sys.modules else False,'parent_gpu_records':[line for line in gpu['processes'] if int(line.split(',')[1])==os.getpid()],'sampler_errors':sampler_errors,'guardian_errors':guardian.errors,'guardian_protocol_state':getattr(guardian,'protocol_state',{}),'error':str(error) if error else None})
  if remaining or missing or gmissing or own_gpu or any(int(line.split(',')[1])==os.getpid() for line in gpu['processes']) or ('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()):
   if error is None:error=RuntimeError('actual owned pipeline cleanup failed')
 if error:raise error
if __name__=='__main__':main()
