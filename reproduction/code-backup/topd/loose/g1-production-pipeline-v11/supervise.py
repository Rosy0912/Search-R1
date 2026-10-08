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
 expected_scope={'initial-only':'G1_32B_INITIAL_CALIBRATION_SOURCE_AND_ALLOCATION_GATE','profile2':'G1_32B_TWO_OUTER_RESOURCE_PROFILE_SOURCE_AND_ALLOCATION_GATE','train100':'G1_32B_CALIBRATION_AND_UPDATE_RESOURCE_PROOF'}[mode]
 if gate.get('status')!='PASS' or gate.get('scope')!=expected_scope:raise ValueError('missing new32B actual resource proof')
 for key in ('candidate_manifest_sha256','producer_manifest_sha256','pipeline_manifest_sha256','base64_result_sha256','base_model_inventory_sha256'):
  if gate.get(key)!=config[key]:raise ValueError('independent execution gate source binding changed')
 if gate.get('actual_gpu_count')!=8 or gate.get('host_memory_request_gib')!=768 or gate.get('global_actor_microbatch')!=64:raise ValueError('new32B actual resource protocol mismatch')
 proof_documents={}
 for key in ('actual_raw_evidence','independent_review'):
  evidence=bound_json(gate[key]);proof_documents[key]=evidence
  if key=='independent_review' and evidence.get('status')!='PASS':raise ValueError('independent review did not pass')
 if mode=='train100' and (gate.get('actual_initial_calibration_complete') is not True or gate.get('actual_actor_update_complete') is not True):raise ValueError('CPU proofs cannot authorize actual100 outer training')
 if mode=='train100':
  raw=proof_documents['actual_raw_evidence'];review=proof_documents['independent_review'];updates=raw.get('actual_update_evidence',{})
  if raw.get('scope')!='ACTUAL_G1_PROFILE2_UNREVIEWED_NOT_100_OR_TOPD' or any(updates.get(k)!=v for k,v in {'actual_outer_steps':2,'optimizer_steps_per_rank':6,'micro_forwards_per_rank':20,'ranks':8}.items()):raise ValueError('Formal100 needs actual two-outer geometry, not monitor source approval')
  if review.get('reviewed_profile2_result_sha256')!=gate['actual_raw_evidence']['sha256']:raise ValueError('Actual profile2 independent review binding differs')
  reuse=config.get('initial_calibration_reuse',{})
  if review.get('reviewed_initial_frozen_sha256')!=reuse.get('sha256') or not reuse.get('sha256'):raise ValueError('Formal100 profile2 must use the same actual round0 heads')
  outer=bound_json(raw['actual_outer_evidence'])
  if outer.get('actual_completed_outer_steps')!=[1,2] or outer.get('export_step')!=2:raise ValueError('Actual profile2 outer raw receipt incomplete')
 return gate

def own_path(path):
 p=pathlib.Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('foreign path')
 return p

def process_arguments(config,binding,mode='train100'):
 command=list(config['training_command'])
 if command[:3]!=[str(ROOT/'env/bin/python'),'-m','verl.trainer.main_ppo_eco']:raise ValueError('unbound training entry')
 overrides={x.split('=',1)[0].lstrip('+'):x.split('=',1)[1] for x in command[3:] if '=' in x}
 required={'actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','actor_rollout_ref.rollout.n_agent':'5','data.train_batch_size':'128','trainer.n_gpus_per_node':'8','trainer.nnodes':'1','trainer.total_training_steps':'101','trainer.save_freq':'25','trainer.g1_refresh_freq':'25','actor_rollout_ref.actor.optim.foreach':'false','actor_rollout_ref.actor.fsdp_config.param_offload':'true','actor_rollout_ref.actor.fsdp_config.grad_offload':'true','actor_rollout_ref.actor.fsdp_config.optimizer_offload':'true','actor_rollout_ref.ref.fsdp_config.param_offload':'true','data.max_prompt_length':'4096','data.max_response_length':'500','data.max_start_length':'2048','data.max_obs_length':'500','max_turns':'4','actor_rollout_ref.rollout.gpu_memory_utilization':'0.60'}
 if mode=='profile2':required.update({'trainer.total_training_steps':'3','trainer.save_freq':'2'})
 for k,v in required.items():
  if overrides.get(k)!=v:raise ValueError('production recipe drift '+k)
 for key,value in {'actor_rollout_ref.model.path':config['base_model'],'trainer.default_local_dir':config['checkpoint_root'],'data.train_files':config['train_file'],'data.val_files':config['val_file'],'algorithm.adv_estimator':'grpo','actor_rollout_ref.actor.state_masking':'true'}.items():
  if overrides.get(key)!=value:raise ValueError('actual input/caller protocol drift '+key)
 return command+['+trainer.g1_owned_pipeline_binding='+str(binding),'+trainer.g1_actual_outer_receipt='+config['actual_outer_receipt_path'],'+trainer.g1_probe_phase_url='+config['probe_url'],'+trainer.g1_probe_manifest_path='+str(pathlib.Path(config['refit_args']['manifest_root'])/('live_'+config['experiment'])/'manifest.json')]

def health_response(response,*,probe):
 if response.status_code!=200:return None
 if probe:
  data=response.json();return data if data.get('ok') is True else None
 return {'scope':'SOURCE_BOUND_RETRIEVER_DOCS_HTTP_READY_NOT_QUALITY','status_code':200,'content_type':response.headers.get('content-type','')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',choices=['initial-only','profile2','train100'],required=True);a=ap.parse_args();os.umask(0o077)
 config=json.loads(pathlib.Path(a.config).read_text());verify_gate(config,a.mode);config['runtime_mode']=a.mode
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
 from retrieval_gate import verify_retrieval
 atomic(root/'retrieval-fresh-inventory.json',verify_retrieval(ROOT,config['retrieval_asset_binding'],config['retriever_command'][3]))
 from compiler_headers import verify as verify_headers
 header_receipt=verify_headers(config['compiler_headers']);atomic(root/'compiler-headers-fresh.json',header_receipt)
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join([str(own_path(config['candidate'])),str(own_path(config['overlay'])),str(ROOT/'test-tools')]),NO_PROXY='127.0.0.1,localhost')
 for key in list(env):
  if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key)
 env['PYTHONPATH']=str(pathlib.Path(__file__).parent)+os.pathsep+env['PYTHONPATH']
 env['CPATH']=header_receipt['cpath']
 env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['RETRIEVER_URL']=config['retriever_url']
 for key,value in {'PROBE_URL':config['probe_url'],'PROBE_MANIFEST_PATH':str(pathlib.Path(config['refit_args']['manifest_root'])/('live_'+config['experiment'])/'manifest.json'),'ECO_TRUST_VARIANT':'balanced','ECO_TRUST_SIGNAL':'probe','ECO_PROBE_GATE':'1','ECO_BAL_GROUND':'0.6','ECO_BAL_NOSEARCH_BONUS':'0.5','ECO_W_COST':'0.02','ECO_WRONG_GROUND_SCALE':'0.0','ECO_USE_CF':'0'}.items():env[key]=value
 owned=tools.OwnedProcesses(scope,relative);baseline={pid:tools.process_identity(pid) for pid in owned.baseline};gpu_original=gpu_baseline(tools,tools.gpu_sample());error=None;trainer=None;listener=None;terminal_connection=None;terminal_drained=False;roles=[];observed={};sampler_stop=threading.Event();sampler_errors=[]
 def sampler():
  try:
   with open(root/'own-process-resource-samples.jsonl','x') as stream:
    while not sampler_stop.is_set():
     pids=tools.cgroup_pids(scope)-owned.baseline;samples=[tools.process_sample(pid) for pid in sorted(pids)]
     for pid in pids:observed[pid]=tools.process_identity(pid)
     from resource_audit import charged_memory
     stream.write(json.dumps({'epoch':time.time(),'cgroup_memory':charged_memory(scope),'own_process_samples':samples,'rss_scope':'SUMMED_PROCESS_RSS_INCLUDES_SHARED_PAGES_SAMPLED_NOT_UNIQUE_RAM','gpu':tools.gpu_sample()},sort_keys=True)+'\n');stream.flush();sampler_stop.wait(1)
    os.fsync(stream.fileno())
  except BaseException as exc:sampler_errors.append(str(exc))
 monitor=threading.Thread(target=sampler,name='own-resource-sampler')
 from window_guardian import Guardian,poll_phase_controls
 guardian=Guardian(ROOT,config['telemetry_policy'],root,scope,owned)
 control=root/'phase-control';control.mkdir(mode=0o700)
 env['G1_GUARDIAN_CONTROL']=str(control)
 def poll_with_http_audit():
  from retrieval_http_audit import require_healthy_log
  require_healthy_log(root/'retriever.log',persist=lambda r:atomic(root/'retrieval-http-failure.json',r))
  poll_phase_controls(guardian,config,root,tools)
 guardian.phase_poll=poll_with_http_audit
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
    result=health_response(response,probe=(url==config['probe_health_url']))
    if result is not None:return result
   except requests.RequestException:pass
   time.sleep(1)
  raise TimeoutError('own service health timeout')
 try:
  # Retrieval startup must use separately source-bound fullwiki GPU launcher, not CPU fallback.
  from checkpoint_inventory import inventory
  if inventory(config['base_model'])['sha256']!=config['base_model_inventory_sha256']:raise ValueError('official base model differs from quality gate inventory')
  for item in config['input_bindings']:
   if sha(own_path(item['path']))!=item['sha256']:raise ValueError('actual data bytes changed')
  guardian.prepare_retriever_start(relative)
  with guardian.lock:
   retrieval=spawn('retriever',config['retriever_command']);guardian.attach_retriever(retrieval.pid)
  guardian.start();healthy(config['retriever_health_url'],retrieval);guardian.check()
  guardian.resident_retriever(retrieval.pid,relative);guardian.check()
  # Both original G1(defaultFalse) and Search-R1/eval(True) schemas must
  # actually work before admitting any of the expensive calibration rows.
  from verify_api import verify
  import requests
  api_session=requests.Session();api_session.trust_env=False
  def persist_api(record):atomic(root/('retrieval-api-'+str(record['index'])+'.json'),record)
  try:
   api_receipt=verify(config['retriever_url'],api_session,persist=persist_api,original_dump=pathlib.Path(config['candidate'])/'ts_probe_dump.py')
   atomic(root/'retrieval-api-verified.json',api_receipt)
  except BaseException as api_failure:
   atomic(root/'retrieval-api-failure.json',{'scope':'ACTUAL_RPC_FAILURE_NO_CALIBRATION_ADMISSION','error':repr(api_failure)})
   raise
  guardian.check()
  initial=root/'initial';initial.mkdir()
  reuse=config.get('initial_calibration_reuse')
  if reuse:
   from pipeline import reuse_initial
   frozen=reuse_initial(config,reuse,initial)
  else:original_round(config,own_path(config['base_model']),0,env,initial)
  guardian.check()
  from retrieval_http_audit import require_healthy_log
  atomic(root/'retrieval-http-initial-complete.json',require_healthy_log(root/'retriever.log',persist=lambda r:atomic(root/'retrieval-http-failure-final.json',r)))
  atomic(root/'initial-complete.json',{'scope':'ACTUAL_INITIAL_CALIBRATION_UNREVIEWED','frozen_sha256':sha(initial/'frozen-calibration.json'),'before_actor_init':True})
  if a.mode in ('profile2','train100'):
   guardian.begin_phase('probe-and-actor-init')
   probe=spawn('probe',config['probe_command']);healthy(config['probe_health_url'],probe)
   socketpath=root/'owner.sock';listener=socket.socket(socket.AF_UNIX);listener.bind(str(socketpath));os.chmod(socketpath,0o600);listener.listen(1);listener.settimeout(1)
   capability=secrets.token_hex(32);binding=root/'owned-binding.json'
   atomic(binding,{'socket':str(socketpath),'capability':capability,'server_identity':identity(os.getpid()),'uuids':config['uuids']})
   command=process_arguments(config,binding,a.mode);commandfile=root/'actual-training-command.json';atomic(commandfile,command)
   trainer_command=[str(ROOT/'env/bin/python'),str(pathlib.Path(__file__).with_name('train_entry.py')),'--config',str(root/'effective-runtime-config.json'),'--command-json',str(commandfile)]
   trainer=spawn('trainer',trainer_command);supervisor=Supervisor(config,env,http);global_ready=False;pending_refresh_step=None
   while trainer.poll() is None:
    guardian.check()
    if not global_ready:
     from resource_audit import verify_ready
     ready=verify_ready(root/'actor-resource-audit',config,trainer.pid,tools,guardian.safety.nvml_snapshot,validate_peer)
     if ready is not None:
      guardian.sample();guardian.close_ready();atomic(root/'actual-global-workers-ready.json',ready);global_ready=True
    if pending_refresh_step is not None:
     from resource_audit import verify_ready
     resumed_ready=verify_ready(root/'actor-resource-audit',config,trainer.pid,tools,guardian.safety.nvml_snapshot,validate_peer,prefix='round%d-'%pending_refresh_step)
     if resumed_ready is not None:
      guardian.sample();guardian.close_ready();atomic(root/('actual-refresh%d-workers-restored.json'%pending_refresh_step),resumed_ready);pending_refresh_step=None
    for pid in tools.cgroup_pids(scope)-owned.baseline:observed[pid]=tools.process_identity(pid)
    try:connection,_=listener.accept()
    except socket.timeout:continue
    try:
     pid,uid,gid=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));validate_peer(pid,uid,trainer.pid,config['cgroup']);request=read(connection)
     if not secrets.compare_digest(request.pop('capability',''),capability):raise ValueError('unauthorized IPC request')
     if request.get('operation')=='TERMINAL_HELD':
      if pid!=trainer.pid or request.get('identity')!=identity(pid) or request.get('job')!=config['job']:raise ValueError('actual terminal trainer identity changed')
      held=root/'trainer-terminal-held.json'
      if held.is_symlink() or json.loads(held.read_text())!=request:raise ValueError('durable terminal request drift')
      terminal_connection=connection;break
     if request.get('operation')!='REFRESH':raise ValueError('unauthorized IPC operation')
     if pending_refresh_step is not None:raise RuntimeError('prior actual worker resume evidence incomplete')
     guardian.begin_phase('refresh-'+str(request['outer_step'])+'-controlled-transition')
     response=supervisor.refresh(request);pending_refresh_step=request['outer_step'];send(connection,response)
    finally:
     if connection is not terminal_connection:connection.close()
   expected_refresh=[] if a.mode=='profile2' else [25,50,75]
   if not global_ready or pending_refresh_step is not None or terminal_connection is None or trainer.poll() is not None or supervisor.completed!=expected_refresh:raise RuntimeError('100 outer trainer/refresh failed')
   # Full arbitrary-shard export and all tokenizer files, not just config.json.
   from checkpoint_inventory import inventory,tokenizer_compatible
   final=own_path(config['final_checkpoint'])
   if final!=own_path(config['checkpoint_root'])/('actor/global_step_2' if a.mode=='profile2' else 'actor/global_step_100'):raise ValueError('terminal100 export path drift')
   export=inventory(final);compat=tokenizer_compatible(final,config['base_model'])
   outer_path=own_path(config['actual_outer_receipt_path']);outer=json.loads(outer_path.read_text())
   if outer.get('actual_completed_outer_steps')!=list(range(1,3 if a.mode=='profile2' else 101)) or outer.get('export_step')!=(2 if a.mode=='profile2' else 100):raise ValueError('missing actual100 outer evidence')
   from resource_audit import verify_updates
   steps=2 if a.mode=='profile2' else 100
   updates=[json.loads((root/'actor-resource-audit'/('actor-updates-rank%d-outer%d.json'%(rank,steps))).read_text()) for rank in range(8)]
   update_evidence=verify_updates(updates,steps)
   from terminal import drain_monitors
   from retrieval_http_audit import require_healthy_log
   atomic(root/'retrieval-http-training-complete.json',require_healthy_log(root/'retriever.log',persist=lambda r:atomic(root/'retrieval-http-failure-final.json',r)))
   guardian.sample()
   monitor_drain=drain_monitors(guardian,sampler_stop,monitor,sampler_errors);terminal_drained=True
   if request['actual_outer_evidence']!={'path':str(outer_path),'sha256':sha(outer_path)} or request['effective_config_sha256']!=sha(root/'effective-runtime-config.json'):raise ValueError('held terminal source/outer receipt drift')
   authorization={'status':'AUTHORIZED_OWN_SHUTDOWN','held_sha256':sha(root/'trainer-terminal-held.json'),'parent_identity':identity(os.getpid()),'job':config['job'],'monitors':monitor_drain,'actual_outer_evidence':{'path':str(outer_path),'sha256':sha(outer_path)}}
   atomic(root/'trainer-terminal-authorization.json',authorization);send(terminal_connection,authorization);terminal_connection.close();terminal_connection=None
   if trainer.wait(timeout=300)!=0:raise RuntimeError('actual owned trainer shutdown failed')
   atomic(root/('profile2-result.json' if a.mode=='profile2' else 'train-result.json'),{'scope':'ACTUAL_G1_PROFILE2_UNREVIEWED_NOT_100_OR_TOPD' if a.mode=='profile2' else 'ACTUAL_G1_100_OUTER_UNREVIEWED_NOT_TOPD_ACCEPTANCE','actual_update_evidence':update_evidence,'export':export,'tokenizer_compatibility':compat,'actual_outer_evidence':{'path':str(outer_path),'sha256':sha(outer_path)},'terminal_supervision_evidence':{'path':str(root/'trainer-terminal-authorization.json'),'sha256':sha(root/'trainer-terminal-authorization.json')}})
 except BaseException as exc:error=exc
 finally:
  from final_cleanup import finalize_owned_pipeline
  error,cleanup_record=finalize_owned_pipeline(tools=tools,owned=owned,scope=scope,baseline=baseline,gpu_original=gpu_original,baseline_gpu_missing=baseline_gpu_missing,roles=roles,observed=observed,guardian=guardian,sampler_stop=sampler_stop,monitor=monitor,sampler_errors=sampler_errors,terminal_drained=terminal_drained,terminal_connection=terminal_connection,listener=listener,root=root,primary_error=error)
  atomic(root/'cleanup.json',cleanup_record)

 if error:raise error
if __name__=='__main__':main()
