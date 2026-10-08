"""Real Slurm8 leases; tiny singleton worker only. NOT_CAPACITY_NOT_TRAINING."""
import os,sys,pathlib,json,hashlib,importlib.util,time,socket,signal,threading,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');HERE=pathlib.Path(__file__).resolve().parent

def need(v,m):
 if not v:raise RuntimeError(m)
def load(name,path,digest):
 need(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Source drift '+str(path))
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(p,v):
 t=p.with_name(p.name+'.private')
 with t.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.link(t,p);t.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def wait_file(p,child,seconds=120,errors=None):
 end=time.monotonic()+seconds
 while not p.is_file():
  need(not errors,'Foreign guardian failed '+repr(errors))
  need(child.poll() is None,'Worker exited before durable receipt');need(time.monotonic()<end,'Bounded worker receipt timeout');time.sleep(.05)
 return json.loads(p.read_text())
def main():
 need(socket.gethostname()=='h100-11' and HERE==ROOT/'tools/entropy-device-proof-v2-r1','Own exact node/path required')
 manifest=json.loads((HERE/'files.json').read_text())
 source=ROOT/'tools/search-r1-32b-grpo100-source-v5';source_manifest=source/'r1-files.json'
 need(hashlib.sha256(source_manifest.read_bytes()).hexdigest()=='72c5fd163b9cbe6bfb502d098b7bc14e7dfa386e5c008ffd7a202eebd14b57b4','SourceV5 manifest drift')
 source_records=json.loads(source_manifest.read_text())['files']
 for name,record in source_records.items():need(not (source/name).is_symlink() and (source/name).stat().st_size==record['size'] and hashlib.sha256((source/name).read_bytes()).hexdigest()==record['sha256'],'WholeSourceV5 payload drift '+name)
 for name,digest in manifest.items():need(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,'Parent payload drift')
 for path,digest in [(ROOT/'repo-terminal-fixed413/search_r1/baselines/preflight.py','cc9445265a3835e1623c0de9d95380717f0e44d4a303d45e3fe1f559431e1605'),(ROOT/'repo-terminal-fixed413/search_r1/baselines/topd_local_gpu.py','4861cb7cd20fe9277a89dca21eff77e7e6107d91940e0f6591b6b75d3e5f6a89')]:need(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Original8lease import drift')
 sys.path.insert(0,str(ROOT/'repo-terminal-fixed413'))
 from search_r1.baselines.preflight import validate_inherited_gpu_leases
 from search_r1.baselines.topd_local_gpu import lease_ownership
 def leases():
  rows=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);need(len(rows)==8,'All8 inherited leases required')
  for row in rows:lease_ownership(row)
  return rows
 validate_inherited_gpu_leases(expected_count=8) # Initial aggregate qualification only, before own CUDA.
 rows=leases();uuids={r['uuid'] for r in rows};need(len(uuids)==8,'8UUID required')
 tools=load('entropy_owned_cleanup',ROOT/'tools/topd-short-run.py','d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09')
 load('eval64',ROOT/'tools/qwen25-32b-terminal-fix-refreeze/eval64.py','24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee')
 safety=load('entropy_window_runtime',ROOT/'tools/qwen25-32b-window-safety-candidate-v6/runtime_safety.py','d9c495d8f2eb5bf9b1ff5680b6a8c0c2626d1e5375e92ae4971a78a9bf3dac70')
 partial=load('entropy_partial_original',ROOT/'tools/search-r1-32b-grpo100-tools-v14/retrieval_startup.py','1f401e123b97e240f1040beb9ee02fbcb525d86fa90b9494951f231f9003ecf4')
 scope,relative=tools.cgroup_scope(os.environ['SLURM_JOB_ID']);owned=tools.OwnedProcesses(scope,relative)
 folder=ROOT/'logs'/('entropy-device-proof-'+os.environ['SLURM_JOB_ID']);folder.mkdir(exist_ok=False)
 baseline=safety.nvml_snapshot();need(len({p['pid'] for p in baseline['applications']})==8,'Protected8 baseline required')
 result=dict(status='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING',job=os.environ['SLURM_JOB_ID'],baseline=baseline,leased_uuids=sorted(uuids),worker_source_sha256=manifest['worker.py'],source_manifest_sha256='72c5fd163b9cbe6bfb502d098b7bc14e7dfa386e5c008ffd7a202eebd14b57b4')
 child=None;stop=threading.Event();thread=None;errors=[];service=None;lock=threading.Lock()
 def raw(record):
  with (folder/'foreign-raw.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
 def measure():
  with lock:
   return partial.bounded_sample(lambda:partial.sample(safety,baseline,uuids,service,relative,lease_reader=lambda:{r['uuid'] for r in leases()},raw_writer=raw),progress_service=service,progress_uuids=uuids)
 def monitor():
  while not stop.is_set():
   try:measure()
   except BaseException as e:errors.append(repr(e));stop.set()
   stop.wait(.25)
 def interrupted(n,frame):raise RuntimeError('Own device probe interrupted '+str(n))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  measure() # Complete baseline foreign dual15 before any own CUDA.
  contracts=ROOT/'tools/entropy-source-v5-contracts';need(hashlib.sha256((contracts/'files.json').read_bytes()).hexdigest()=='4697569d9509aaa86cad24dc5316b8c6385d0af565ef11ed81c89ea6ef6d232e','Contracts manifest drift')
  cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',R1_ENTROPY_SOURCE=str(source),PYTHONPATH=str(ROOT/'test-tools')+':'+str(ROOT/'repo-terminal-fixed413'))
  with (folder/'native-four-contracts.log').open('xb') as cpu_log:subprocess.run([sys.executable,'-B',str(contracts/'native_entry.py')],env=cpuenv,stdout=cpu_log,stderr=subprocess.STDOUT,check=True,timeout=180)
  cpu_result=ROOT/'logs'/('r1-entropy-source-v5-cpu-'+os.environ['SLURM_JOB_ID'])/'result.json';cpu=json.loads(cpu_result.read_text());need(cpu['tests']==4 and cpu['skips']==0 and cpu['failures']==0 and cpu['exit_code']==0 and not cpu['cuda_initialized'],'Exact native CPU4 failed')
  result['native_cpu_four']={'path':str(cpu_result),'sha256':hashlib.sha256(cpu_result.read_bytes()).hexdigest(),'log':{'path':str(folder/'native-four-contracts.log'),'sha256':hashlib.sha256((folder/'native-four-contracts.log').read_bytes()).hexdigest()}}
  env=dict(os.environ,TORCHINDUCTOR_CACHE_DIR=str(folder/'inductor-cache'),TRITON_CACHE_DIR=str(folder/'triton-cache'),CUDA_VISIBLE_DEVICES=rows[0]['uuid'],R1_DEVICE_PROOF_FOLDER=str(folder),R1_DEVICE_PROOF_OUTPUT=str(folder/'worker-result.json'),R1_STREAM_SOURCE=str(ROOT/'tools/search-r1-32b-grpo100-source-v5'),R1_DEVICE_PROOF_FULL8_LEASES_HELD='true',R1_APPROVED_DEVICE_CORRECTNESS_SCOPE='ORIGINAL_ENTROPY_CUDA_DEVICE_CORRECTNESS_NOT_CAPACITY',R1_DEVICE_PROOF_LEASE_FD=str(rows[0]['fd']),R1_DEVICE_PROOF_EXPECTED_UUID=rows[0]['uuid'])
  child=owned.spawn([sys.executable,'-B',str(HERE/'hold_worker.py')],folder/'worker.log',env,pass_fds=tuple(int(r['fd']) for r in rows))
  service=safety.pid_identity(child.pid)
  thread=threading.Thread(target=monitor);thread.start()
  ready=wait_file(folder/'child-context-ready.json',child,errors=errors)
  need(ready['pid']==child.pid and ready['allocated']==ready['reserved']==0,'Real empty own context required')
  result['worker_identity']=service;result['context_foreign_gate']=measure()
  need(not errors,'Foreign guardian failed '+repr(errors));(folder/'run-authorized').touch(exist_ok=False)
  wait_file(folder/'child-work-complete.json',child,seconds=600,errors=errors);need(not errors,'Foreign guardian failed '+repr(errors))
  result['after_worker_foreign_gate']=measure();stop.set();thread.join(35);need(not thread.is_alive() and not errors,'Guardian drain failed')
  proof=json.loads((folder/'worker-result.json').read_text());need(proof['status']=='ENTROPY_DEVICE_CORRECTNESS_AND_BENCH_SUPPRESSION_PASS_NOT_CAPACITY_NOT_TRAINING' and proof['pid']==child.pid and proof['source_manifest_sha256']==result['source_manifest_sha256'] and proof['allocator_after_release']['allocated']==proof['allocator_after_release']['reserved']==0,'Worker device/math/release proof failed')
  result['worker_result']={'path':str(folder/'worker-result.json'),'sha256':hashlib.sha256((folder/'worker-result.json').read_bytes()).hexdigest()}
  (folder/'exit-authorized').touch(exist_ok=False);need(child.wait(timeout=15)==0,'Worker actual exit failed');result['status']='ENTROPY_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING'
 except BaseException as e:result['error']=repr(e)
 finally:
  stop.set()
  try:
   if thread:thread.join(35)
  except BaseException as e:result['guardian_join_error']=repr(e);result['status']='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING'
  try:
   if thread and thread.is_alive():result['guardian_join_error']='Guardian thread still querying';result['status']='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING'
   owned.cleanup()
  except BaseException as e:result['cleanup_error']=repr(e);result['status']='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING'
  result['cleanup']=getattr(owned,'cleanup_record',None);result['guardian_errors']=errors
  try:
   final=safety.nvml_snapshot();result['final']=final;need(result['cleanup']['actual_cgroup_empty'] and result['cleanup']['remaining']==[],'Own descendants remain')
   need(final['identities']==baseline['identities'] and {(r['uuid'],r['pid']) for r in final['applications']}=={(r['uuid'],r['pid']) for r in baseline['applications']},'Final foreign baseline changed')
   need(all(g['sm']<=15 and sum(a['used_mib'] for a in final['applications'] if a['uuid']==g['uuid'])<=.15*g['total_mib'] for g in final['gpus']),'Final foreign dual15 failed')
   leases();import torch;need(not torch.cuda.is_initialized(),'Parent initialized CUDA');result['parent_cuda_initialized']=False;result['baseline_preserved']=True
  except BaseException as e:result['final_error']=repr(e);result['status']='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING'
  write(folder/'parent-result.json',result)
 need(result['status']=='ENTROPY_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING',str(result))
if __name__=='__main__':main()
