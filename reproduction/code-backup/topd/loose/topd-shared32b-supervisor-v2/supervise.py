"""Owned same-node8lease lifecycle. No approvals or remote submission generated."""
import argparse,json,os,sys,socket,time,threading,subprocess
from pathlib import Path
from contracts import *
from dependencies import gate,load
from lifecycle import publish,wait,StartupPhases,join_monitor

def allocated():
 require(os.environ.get('AGENT_ROOT')==str(ROOT) and socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Own allocated h11 only')
 require(int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==8 and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==8 and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=128 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=384*1024,'Actual8GPU/128CPU/384GiB allocation required')
 from search_r1.baselines.preflight import validate_inherited_gpu_leases
 validate_inherited_gpu_leases(8)
 leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);return leases

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=('profile','formal500'),required=True);a=p.parse_args()
 config=json.loads(Path(a.config).read_text());selected=gate(config,a.mode)
 sys.path.insert(0,str(own(config['repo'])));leases=allocated();uuids=[r['uuid'] for r in leases];require(uuids==config['expected_uuids'],'Actual allocationUUID differs from approved scope')
 import fcntl
 run=own(config['run_directory']);require(ROOT/'logs' in run.parents and not run.exists(),'Unique actual record directory required');run.mkdir(mode=0o700)
 lock=(ROOT/'logs/shared32b-supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 helper=load('owned_support',config['owner_support_source']);scope,relative=helper.cgroup_scope(os.environ['SLURM_JOB_ID']);owned=helper.OwnedProcesses(scope,relative)
 ipcgate=load('import_gate',config['owner_import_gate']);ipcgate.verified_ipc()
 from topd32b_shared_pipeline.protocol import process_identity
 # Load sealed eval module before its runtime import; source must be separately pinned.
 load('eval64',config['sealed_eval_source']);sealed=load('sealed_runtime',config['sealed_runtime_source']);telemetry=load('active_telemetry',config['telemetry_source'])
 baseline=sealed.nvml_snapshot();guard=telemetry.StartupGuard(sealed,expected_path=config['sealed_runtime_source']['path'],trace_path=run/'active-telemetry.jsonl');guard.closed=True
 phases=StartupPhases(guard);guard.ready_check=phases.ready_check;errors=[];stop=threading.Event();runtime=None;monitor=None;result=dict(status='UNKNOWN_BEFORE_START',mode=a.mode,baseline=baseline,source_config_sha256=digest(config),leased_uuids=uuids,events=[])
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=','.join(uuids),PYTHONPATH=str(Path(__file__).parent)+os.pathsep+str(own(config['repo']))+os.pathsep+str(ROOT/'tools/topd-32b-shared-owner-candidate-v2'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 for name in list(env):
  if name.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(name)
 fds=tuple(r['fd'] for r in leases)
 def event(stage):result['events'].append(dict(stage=stage,epoch=time.time()));publish(run/'progress.json',result)
 def sample():
  while not stop.is_set():
   try:
    snapshot=guard.active_gate(runtime,pre_model=False)
    samples=[helper.process_sample(pid) for pid in sorted(helper.cgroup_pids(scope)-owned.baseline)]
    row=dict(epoch=time.time(),stage=phases.state,gpu=snapshot,own_processes=samples,rss_scope='SUMMED_PROCESS_RSS_INCLUDES_SHARED_PAGES_SAMPLED_NOT_UNIQUE_RAM')
    with (run/'resource-timeseries.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
    if phases.teacher_ready:
     import requests
     session=requests.Session();session.trust_env=False
     response=session.post('http://127.0.0.1:18085/retrieve',json={'queries':['capital of France'],'topk':3,'return_scores':True},timeout=120);response.raise_for_status()
     with (run/'retrieval-health.jsonl').open('a') as stream:stream.write(json.dumps(dict(epoch=time.time(),status=response.status_code,teacher_stage='SCORING_ONLY',response_sha256=digest(response.json())))+'\n')
   except BaseException as exc:errors.append(repr(exc));stop.set()
   stop.wait(1)
 try:
  launcher=load('retriever_launcher',config['retriever_launcher_source']);command,retriever_env=launcher.gpu_retriever_spec();retriever=owned.spawn(command,run/'retriever.log',env=retriever_env);launcher.wait_for_gpu_idle(retriever,run/'retriever.log')
  runtime=dict(baseline=baseline,retriever_identity=sealed.pid_identity(retriever.pid),cgroup=relative)
  guard.active_gate(runtime,pre_model=False);event('ACTUAL_E5_GPU8_READY_IDLE15')
  driver=owned.spawn([sys.executable,str(Path(__file__).with_name('train_entry.py')),'--config',a.config,'--mode',a.mode,'--run',str(run),'--ipc',str(run/'ipc')],run/'driver.log',env=env,pass_fds=fds)
  hello=wait(run/'driver-hello.json',[driver,retriever],errors)
  require(hello['driver']==dict(process_identity(driver.pid),uid=os.getuid()),'Owned actual driver HELLO mismatch')
  manifest=dict(artifact(config['formal_manifest'] if a.mode=='formal500' else config['profile_manifest']))
  require(manifest['kind']==('NATIVE_32B_TOPD_SHARED_PIPELINE' if a.mode=='formal500' else 'NATIVE_HARDWARE_PROBE_NO_TRAINING'),'Wrong execution scope')
  manifest.update({k:hello[k] for k in ('run_id','code_commit','config_sha256','data_identity','source_sha256')});manifest.update(driver=hello['driver'],commit=D23,parent_cgroup=relative,leased_uuids=uuids,outer_updates=500 if a.mode=='formal500' else 2)
  binding=dict(manifest,fp32_parameter_sha256=selected['provenance']['expected_fp32_parameter_sha256'],teacher_path=selected['model'],ipc_directory=str(run/'ipc'),student_model_directory=hello['student_model_directory'])
  publish(run/'binding.json',binding);publish(run/'manifest-template.json',manifest)
  ownerenv=dict(env,TOPD32B_OWNER_PARENT=str(os.getpid()))
  owner=owned.spawn([sys.executable,str(Path(__file__).with_name('owner_entry.py')),'--config',a.config,'--mode',a.mode,'--template',str(run/'manifest-template.json'),'--directory',str(run/'ipc')],run/'owner.log',env=ownerenv,pass_fds=fds)
  cpu=wait(run/'ipc/owner-cpu-ready.json',[owner,driver,retriever],errors);require(cpu['owner']==dict(process_identity(owner.pid),uid=os.getuid()),'Owned teacher identity mismatch')
  require(cpu['initial_cpu']['parameter_sha256']==binding['fp32_parameter_sha256'],'Actual selected teacher CPU SHA')
  phases.arm();monitor=threading.Thread(target=sample,name='owned-full-lifecycle-monitor');monitor.start()
  permission=dict(pid=owner.pid,starttime_ticks=cpu['owner']['starttime'],job=os.environ['SLURM_JOB_ID'],provenance_sha256=manifest['teacher_receipt_sha256'])
  publish(run/'ipc/allow-context.json',dict(permission,phase='CONTEXT_ONLY'))
  context=wait(run/'ipc/owner-context-ready.json',[owner,driver,retriever],errors)
  ownerpolicy=load('context_policy',config['owner_context_source']);canonicalgate=load('context_canonical_gate',config['teacher_gate_source']);policies=ownerpolicy.policies(Path(config['teacher_gate_source']['path']).with_name('caller_contracts.py'),canonicalgate)
  policies['context_receipt'](context,dict(pid=owner.pid),uuids)
  publish(run/'ipc/allow-service.json',dict(permission,phase='SCORING_ONLY'))
  service=wait(run/'ipc/owner-service-ready.json',[owner,driver,retriever],errors);require(service['source_commit']==D23 and service['initial_cpu']['parameter_sha256']==binding['fp32_parameter_sha256'],'Teacher service source/frozen identity drift');phases.teacher();event('TEACHER_READY_GLOBAL_STARTUP_STILL_OPEN')
  publish(run/'actor-start-authorized.json',dict(actual_teacher_ready_sha256=sha(run/'ipc/owner-service-ready.json')))
  workers=wait(run/'global-worker-ready.json',[owner,driver,retriever],errors)
  phases.workers(workers['receipts'],lambda rows:actual_worker_ready(rows,uuids,os.environ['SLURM_JOB_ID'],process_identity));event('ACTUAL_GLOBAL_ACTOR_VLLM_READY')
  publish(run/'global-worker-authorized.json',dict(actual_workers_sha256=sha(run/'global-worker-ready.json')))
  accepted=False
  while not (run/'driver-algorithm-returned.json').exists():
   require(driver.poll() is None,'Original driver exited before controlled terminal boundary')
   require(not errors,'Strict active monitor failed '+repr(errors))
   require(retriever.poll() is None,'Own retriever exited')
   if a.mode=='formal500' and not accepted:
    try:receipt=actual_acceptance(run,5,binding)
    except (FileNotFoundError,RuntimeError,KeyError):pass
    else:publish(run/'actual-first5-unreviewed.json',receipt);accepted=True;event('FIRST5_ACTUAL_EVIDENCE_CONTINUING500')
   time.sleep(1)
  returned=wait(run/'driver-algorithm-returned.json',[driver,retriever],errors)
  require(returned['pid']==driver.pid and returned['job']==os.environ['SLURM_JOB_ID'],'Actual terminal driver identity mismatch')
  if a.mode=='formal500':result['actual_algorithm_evidence']=actual_acceptance(run,500,binding)
  else:
   result['hardware_profile']=artifact(dict(path=str(run/'profile-driver-result.json'),sha256=sha(run/'profile-driver-result.json')))
   result['actual_resource_updates']=actual_acceptance(run,2,binding)
   workers={rank:read_complete_jsonl(run/'workers'/f'{rank}.jsonl') for rank in range(8)}
   observed=next_rollout_rebuilt(result['hardware_profile']['records'],workers)
   health=read_complete_jsonl(run/'retrieval-health.jsonl')
   require(all(any(r['scoring_started_epoch']<=h['epoch']<=r['scoring_completed_epoch'] and h['status']==200 for h in health) for r in result['hardware_profile']['records']),'Actual loopback HTTP during both teacher score intervals missing')
   result['profile_verified_observations']=dict(next_rollout_kv_rebuilt=observed,retrieval_http_during_score=health,all8_release_ack=[release_and_ack(binding['ipc_directory'],step,binding) for step in (1,2)])
  require(not errors,'Strict active telemetry failed before terminal drain')
  result['terminal_active_snapshot']=guard.active_gate(runtime,pre_model=False)
  join_monitor(stop,monitor,errors)
  event('VERIFIED_ALL_UPDATES_BEGIN_CONTROLLED_OWNED_EXIT')
  publish(run/'authorized-terminal-exit.json',dict(job=os.environ['SLURM_JOB_ID'],parent_pid=os.getpid(),actual_driver_return_sha256=sha(run/'driver-algorithm-returned.json'),phase='NO_MORE_GPU_WORK_OWNED_EXIT'))
  require(driver.wait(timeout=300)==0,'Original driver/Ray exit failed')
  require(owner.wait(timeout=300)==0,'Owner did not return after exact round completion')
  result['status']='COMPLETE_UNREVIEWED';result['owner_completed_rounds_only']=json.loads((run/'ipc/owner-done.json').read_text())
 except BaseException as exc:result.update(status='FAIL_CLOSED_NO_ACCEPTANCE',error=repr(exc));raise
 finally:
  stop.set()
  if monitor:
   monitor.join(timeout=180)
   if monitor.is_alive():result['monitor_stop_error']='MONITOR_STILL_LIVE';errors.append(result['monitor_stop_error'])
  cleanup_error=None
  try:owned.cleanup()
  except BaseException as exc:cleanup_error=repr(exc)
  remaining=sorted(helper.cgroup_pids(scope)-owned.baseline);final=sealed.nvml_snapshot();missing=[p for p,identity in baseline['identities'].items() if final['identities'].get(p)!=identity]
  result.update(cleanup_error=cleanup_error,actual_own_remaining=remaining,own_cleanup_passed=not remaining and cleanup_error is None and not (monitor and monitor.is_alive()),original_baseline_identity_missing=missing,original_baseline_preserved=not missing,final_gpu_snapshot=final,cleanup_record=getattr(owned,'cleanup_record',None))
  require(not __import__('torch').cuda.is_initialized(),'Supervisor must never create CUDA context');result['parent_cuda_initialized']=False
  if remaining or cleanup_error or missing or (monitor and monitor.is_alive()):result['status']='FAIL_CLOSED_CLEANUP'
  publish(run/'result.json',result);require(result['own_cleanup_passed'] and result['original_baseline_preserved'],'Strict owned cleanup/baseline failed');fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
