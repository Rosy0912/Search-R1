"""Preparation-only formal300 launcher. No signing, resume, algorithm or receipt production."""
import argparse,datetime,hashlib,hmac,json,math,os,re,secrets,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime import sha,unique,physical,pinned,emit,command,own,same_process,snapshot,verify_tool_assets
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004')
OLD=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
SOURCE='89beac273f783d8e48d7d3e34da040f753514d90'
MODEL=ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COMMON_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
ASSETS_SHA='87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
TOOLS={'formal_job.sh','formal_supervisor.py','owned_runtime.py','resource_sample.py','retriever_entry.py','ray_cpu_check.py','hardware_rank.py','cpu_import_check.py'}
SOURCE_FILES={'reproduction/baselines/run_opsd_e1_300.sh','reproduction/baselines/e1_import_bootstrap.py','reproduction/h20/retriever_local.py','search_r1/search/retrieval_server.py','verl/trainer/config/ppo_opsd_e1_300.yaml','search_r1/baselines/main.py','search_r1/baselines/preflight.py','search_r1/baselines/e1_profile.py','search_r1/baselines/controller.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/ray_trainer.py'}
ROLES={'allocation','source','import_closure','data_manifest','model_manifest','tool_manifest','environment'}
QUERIES=['What is the capital of France?','Who wrote Hamlet?','When was Apollo 11 launched?','Where is Mount Everest?','Who discovered penicillin?','What is the capital of Japan?','Who painted the Mona Lisa?','Where was Albert Einstein born?']*3

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def fail(condition,message):
 if not condition:raise ValueError(message)
def under(path,root):
 p=physical(path,True);p.relative_to(root);return p

def copy_pin(path,pin,destination):
 src=physical(path,True);pinned(src,pin)
 raw=src.read_bytes();fail(hashlib.sha256(raw).hexdigest()==pin,'Source changed during copy')
 with destination.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fail(sha(destination)==pin,'Copied binding changed');return {'path':str(destination),'sha256':pin}

def validate_snapshot(value,expected=None,service_pid=None,idle=False):
 rows=value['inventory'];fail(len(rows)==8 and len({v['uuid'] for v in rows})==8,'Need eight physical UUIDs')
 fail(all(isinstance(v['index'],int) and 'H100' in v['name'] and all(math.isfinite(v[k]) for k in ('used','total','util')) and v['total']>0 and 0<=v['used']<=v['total'] and 0<=v['util']<=100 for v in rows),'Invalid H100 inventory')
 if expected is not None:fail({v['uuid'] for v in rows}==set(expected),'Allocated UUID set changed')
 apps=value['applications']
 if service_pid is None:fail(not apps,'GPU compute application already exists')
 else:fail({v['pid'] for v in apps}=={service_pid} and {v['uuid'] for v in apps}==set(expected),'Service must be own exact PID on all eight UUIDs')
 if idle:fail(all(v['used']<=.15*v['total'] and v['util']<=15 for v in rows),'15% startup guard failed')

def slurm(raw,job,uid):
 fields=dict(v.split('=',1) for v in raw.split() if '=' in v)
 fail(fields.get('JobId')==job and fields.get('JobState')=='RUNNING' and fields.get('NodeList')=='h100-10','Wrong live allocation')
 fail(re.fullmatch(r'\S+\('+str(uid)+r'\)',fields.get('UserId','')) is not None,'Wrong Slurm UID')
 fail(fields.get('TresPerNode')=='gres:gpu:h100:8' and fields.get('NumNodes')=='1','Typed eight H100 allocation required')
 fail(int(fields.get('NumCPUs','0'))>=192 and fields.get('OverSubscribe')=='NO','Exclusive 192 CPU allocation required')
 fail(fields.get('MinMemoryNode') in ('1T','1024G','1048576M') and fields.get('TimeLimit')=='7-00:00:00','Requested 1T/seven-day formal allocation required')
 return fields

def sources(tools):
 fail(tools.is_absolute() and tools.resolve(strict=True)==tools and tools.parent==ROOT/'tools' and tools.stat().st_uid==os.getuid(),'Physical own tools directory required')
 manifest=pinned(tools/'manifest.json',os.environ['E1_FORMAL_MANIFEST_SHA256'])
 fail(manifest['source_commit']==SOURCE and set(manifest['tools'])==TOOLS and set(manifest['source_files'])==SOURCE_FILES,'Wrong formal tool manifest')
 for name,pin in manifest['tools'].items():fail(sha(under(tools/name,tools))==pin,'Tool changed: '+name)
 fail(Path(__file__)==tools/'formal_supervisor.py','Wrong supervisor physical path')
 fail(physical(os.environ['E1_ENTRY_SCRIPT'],True)==tools/'formal_job.sh','Wrong actual physical shell entry')
 repo=Path(os.environ['SEARCH_R1_REPO']);fail(repo==ROOT/('repo-'+SOURCE[:12]) and repo.resolve(strict=True)==repo,'Wrong new source path')
 fail(command(['git','-C',str(repo),'rev-parse','HEAD']).strip()==SOURCE and not command(['git','-C',str(repo),'status','--porcelain']).strip(),'Source is not clean frozen HEAD')
 for name,pin in manifest['source_files'].items():fail(sha(physical(repo/name))==pin,'Frozen source file changed: '+name)
 return repo,manifest

def environment():
 names=('PATH','PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','RAY_ADDRESS','TMPDIR','RAY_TMPDIR','SEARCH_R1_REPO','OPSD_E1_OFFICIAL_MODEL','OPSD_E1_COMMON_DATA_ROOT','OPSD_E1_COMMON_MANIFEST','OPSD_E1_RETRIEVER_URL','SEARCH_R1_BASELINE_RUN_ROOT','OPSD_E1_BINDING_MANIFEST','OPSD_E1_BINDING_MANIFEST_SHA256','OPSD_E1_REVIEW_PUBLIC_KEY','OPSD_E1_REVIEW_PUBLIC_KEY_SHA256','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','WANDB_MODE','NCCL_NVLS_ENABLE','VLLM_ATTENTION_BACKEND')
 fail(not os.environ.get('PYTORCH_CUDA_ALLOC_CONF') and not os.environ.get('PYTORCH_ALLOC_CONF'),'Experimental allocator forbidden')
 return {'scope':'Actual wrapper environment; not future Ray actor PID/environment proof','python':sys.executable,'python_version':sys.version,'values':{k:os.environ.get(k) for k in names}}

def verify_rsa(payload,signature_hex,key):
 fail(set(key)=={'algorithm','n_hex','e'} and key['algorithm']=='RSA-SHA256' and key['e']==65537,'Invalid review public key')
 n=int(key['n_hex'],16);fail(2048<=n.bit_length()<=8192 and n%2==1,'RSA modulus invalid')
 size=(n.bit_length()+7)//8;s=bytes.fromhex(signature_hex);fail(len(s)==size and int.from_bytes(s,'big')<n,'Signature invalid size')
 tail=bytes.fromhex('3031300d060960864801650304020105000420')+hashlib.sha256(canonical(payload)).digest()
 expected=b'\x00\x01'+b'\xff'*(size-len(tail)-3)+b'\x00'+tail
 fail(hmac.compare_digest(pow(int.from_bytes(s,'big'),65537,n).to_bytes(size,'big'),expected),'External signature failed')

def approval_expected(ready,review):
 return {'version':1,'authorization':'EXPLICIT_OPSD_E1_GPU_INITIALIZATION_ONLY','source_commit':SOURCE,'job_id':ready['job_id'],'uid':ready['uid'],'pid':ready['pid'],'process_start_ticks':ready['process_start_ticks'],'run_root':ready['run_root'],'nonce':ready['nonce'],'ready_sha256':ready['ready_sha256'],'binding_sha256':ready['binding_sha256'],'public_key_sha256':ready['public_key_sha256'],'independent_review':review}

def check_approval(run,ready):
 fail(ready['source_commit']==SOURCE and ready['job_id']==os.environ['SLURM_JOB_ID'] and ready['uid']==os.getuid() and ready['run_root']==str(run) and ready['pid']==os.getpid(),'Ready is not the actual current allocation/process')
 path=under(run/'bindings/gpu-init-approval.json',run);pinpath=under(run/'bindings/gpu-init-approval.sha256',run)
 pin=pinpath.read_text().strip();envelope=pinned(path,pin);fail(set(envelope)=={'payload','signature_hex'},'Approval schema differs')
 p=envelope['payload'];review=p['independent_review'];fail(set(review)=={'path','sha256'},'Review binding invalid')
 fail(sha(under(review['path'],run))==review['sha256'],'Independent review bytes changed')
 fail(p==approval_expected(ready,review),'Approval is stale or belongs to another job/process')
 key=pinned(run/'bindings/public.json',ready['public_key_sha256']);verify_rsa(p,envelope['signature_hex'],key)
 fail(same_process(ready['process']) and os.getpid()==ready['pid'],'Actual approving process changed')
 return pin

def verify_binding(run,ready):
 binding=pinned(run/'bindings/manifest.json',ready['binding_sha256'])
 fail(binding['source_commit']==SOURCE and binding['run_root']==str(run) and set(binding['files'])==ROLES,'Wrong launch binding')
 for rec in binding['files'].values():fail(set(rec)=={'path','sha256'} and sha(under(rec['path'],run))==rec['sha256'],'Bound artifact changed')
 return binding

def prepare(run,tools):
 repo,manifest=sources(tools);job=os.environ['SLURM_JOB_ID']
 fail(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=192,'srun step must expose192 CPUs')
 fail(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10' and os.environ.get('SLURM_JOB_NODELIST')=='h100-10','Wrong UID/node')
 raw=command(['scontrol','show','job',job,'-o']);slurm(raw,job,os.getuid());owner=own(os.getpid())
 selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',');fail(len(selected)==len(set(selected))==8,'Eight allocated selectors required')
 # NVML queries are not a CUDA-context initializer. Any application rejects the job.
 first=snapshot();validate_snapshot(first,idle=True);time.sleep(2);second=snapshot();validate_snapshot(second,{r['uuid'] for r in first['inventory']},idle=True)
 for port in (16384,18084):
  with socket.socket() as s:s.bind(('127.0.0.1',port))
 (run/'bindings').mkdir();(run/'evidence').mkdir()
 emit(run/'bindings/allocation.json',{'version':1,'job_id':job,'uid':os.getuid(),'hostname':socket.gethostname(),'wrapper_process':owner,'scontrol':raw,'selected_cuda_devices':selected,'nvml_double_guard':[first,second],'scope':'Actual wrapper allocation before CUDA; driver and rank identities recorded later by frozen trainer'})
 emit(run/'bindings/source.json',{'source_commit':SOURCE,'repo':str(repo),'clean':True,'source_files':manifest['source_files'],'formal_manifest_sha256':os.environ['E1_FORMAL_MANIFEST_SHA256']})
 roles={n:{'path':str(run/'bindings'/f'{n}.json'),'sha256':sha(run/'bindings'/f'{n}.json')} for n in ('allocation','source')}
 for role,path,pin in [('import_closure',os.environ['E1_NATIVE_IMPORT_CLOSURE'],os.environ['E1_NATIVE_IMPORT_CLOSURE_SHA256']),('data_manifest',os.environ['OPSD_E1_COMMON_MANIFEST'],COMMON_SHA),('model_manifest',str(MODEL/'VERIFIED-MODEL-MANIFEST.json'),MODEL_SHA),('tool_manifest',os.environ['E1_TOOL_ASSET_MANIFEST'],ASSETS_SHA)]:
  roles[role]=copy_pin(path,pin,run/'bindings'/f'{role}.json')
 closure=pinned(roles['import_closure']['path'],roles['import_closure']['sha256'])
 fail(set(closure)=={'modules'},'Native inventory must use frozen checker schema')
 for name,rec in closure['modules'].items():
  fail(set(rec)=={'path','sha256'} and sha(physical(rec['path']))==rec['sha256'],'Import inventory file changed')
  if name.startswith(('search_r1.','verl.','reproduction.')):fail(repo in Path(rec['path']).parents,'Import inventory points to another source')
 fail(Path(os.environ['OPSD_E1_REVIEW_PUBLIC_KEY'])==ROOT/'review-signing/public.json','Only task-local review public key may enter the job')
 key=copy_pin(os.environ['OPSD_E1_REVIEW_PUBLIC_KEY'],os.environ['OPSD_E1_REVIEW_PUBLIC_KEY_SHA256'],run/'bindings/public.json');verify_rsa_key=pinned(key['path'],key['sha256']);fail(verify_rsa_key.get('algorithm')=='RSA-SHA256','Only task review public key allowed')
 os.environ.pop('OPSD_E1_BINDING_MANIFEST_SHA256',None)
 os.environ.update(OPSD_E1_COMMON_MANIFEST=roles['data_manifest']['path'],OPSD_E1_BINDING_MANIFEST=str(run/'bindings/manifest.json'),OPSD_E1_REVIEW_PUBLIC_KEY=key['path'],OPSD_E1_OFFICIAL_MODEL=str(MODEL),OPSD_E1_OFFICIAL_MANIFEST=str(MODEL/'VERIFIED-MODEL-MANIFEST.json'),OPSD_E1_RETRIEVER_URL='http://127.0.0.1:18084/retrieve')
 # Environment binding precedes the binding-manifest hash: its field documents the
 # actual stage and cannot be a self-referential hash. Training env fills it later.
 emit(run/'bindings/environment.json',environment());roles['environment']={'path':str(run/'bindings/environment.json'),'sha256':sha(run/'bindings/environment.json')}
 emit(run/'bindings/manifest.json',{'version':1,'source_commit':SOURCE,'run_root':str(run),'files':roles});binding_sha=sha(run/'bindings/manifest.json');os.environ['OPSD_E1_BINDING_MANIFEST_SHA256']=binding_sha
 # CPU-only strict current imports plus complete official/common assets verification.
 # The checker stage compiles pinned source, never consumes stale pyc.
 with (run/'evidence/pre-gpu-import-check.json').open('xb') as out:
  subprocess.run([sys.executable,str(tools/'cpu_import_check.py')],stdout=out,check=True,env=dict(os.environ),timeout=180)
 with (run/'evidence/pre-gpu-profile.json').open('xb') as out:
  subprocess.run([sys.executable,str(repo/'reproduction/baselines/e1_import_bootstrap.py'),'--mode','validate'],stdout=out,check=True,env=dict(os.environ))
 assets=pinned(roles['tool_manifest']['path'],ASSETS_SHA);verify_tool_assets(assets)
 emit(run/'evidence/pre-gpu-python-environment.json',{'packages':command([sys.executable,'-c','import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ["torch","transformers","ray","vllm","flash-attn","hydra-core","click"]},sort_keys=True))']),'cuda_context_initialization':'No model, CUDA context or service launched by wrapper before approval; torch CPU import/check explicitly observed cuda_is_initialized false'})
 ready={'version':1,'source_commit':SOURCE,'job_id':job,'uid':os.getuid(),'pid':os.getpid(),'process_start_ticks':owner['start'],'process':owner,'run_root':str(run),'nonce':secrets.token_hex(32),'binding_sha256':binding_sha,'public_key_sha256':key['sha256'],'nvml_uuids':[r['uuid'] for r in first['inventory']],'scope':'Await external pre-GPU review; not first5/300/effect approval'}
 emit(run/'approval-ready.json',ready);ready['ready_sha256']=sha(run/'approval-ready.json')
 print('OPSD_E1_EXTERNAL_PRE_GPU_APPROVAL_READY='+json.dumps(ready,sort_keys=True),flush=True)
 deadline=time.monotonic()+int(os.environ.get('E1_GPU_APPROVAL_TIMEOUT_SECONDS','7200'))
 while not ((run/'bindings/gpu-init-approval.json').exists() and (run/'bindings/gpu-init-approval.sha256').exists()):
  fail(time.monotonic()<deadline,'External pre-GPU approval timed out');time.sleep(2)
 check_approval(run,ready);verify_binding(run,ready);sources(tools)
 # exec preserves PID and parent-held FD9; frozen preflight acquires leases and
 # execs the same supervisor PID. Never invoke a second independent RL job.
 os.execv(sys.executable,[sys.executable,str(repo/'reproduction/baselines/e1_import_bootstrap.py'),'--mode','preflight','--',sys.executable,str(tools/'formal_supervisor.py'),'--run'])

def double_idle(uuids,service=None):
 deadline=time.monotonic()+120;rows=[]
 while len(rows)<2:
  value=snapshot();validate_snapshot(value,uuids,service)
  if all(v['used']<=.15*v['total'] and v['util']<=15 for v in value['inventory']):rows.append(value)
  else:rows=[]
  fail(time.monotonic()<deadline,'Double 15% guard timed out');time.sleep(2)
 return rows

def http24(run,pid):
 import urllib.request
 t=time.perf_counter();request=urllib.request.Request('http://127.0.0.1:18084/retrieve',data=json.dumps({'queries':QUERIES,'topk':3,'return_scores':True}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as response:data=unique(response.read())
 fail(len(data['result'])==24,'Retrieval count differs')
 for row in data['result']:
  fail(len(row)==3,'Retrieval top3 differs')
  for v in row:fail(isinstance(v['document']['contents'],str) and bool(v['document']['contents']) and math.isfinite(float(v['score'])),'Invalid live retrieval response')
 emit(run/'evidence/http-top3-24.json',{'job_id':os.environ['SLURM_JOB_ID'],'service':own(pid),'queries':QUERIES,'seconds':time.perf_counter()-t,'response':data,'scope':'Live local GPU API conformance; eight unique queries, not numerical policy or RL correctness proof'})

def terminate_owned(proc,owner):
 if proc is None or proc.poll() is not None:return
 fail(owner is not None and same_process(owner),'Refusing to terminate stale/foreign PID')
 proc.terminate()
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:
  fail(same_process(owner),'Refusing to kill stale/foreign PID');proc.kill();proc.wait(timeout=10)

def check_training_log(path,job):
 rows=[];final=[]
 for line in Path(path).read_text().splitlines():
  # Ray adds PID prefixes; the immutable marker still starts the JSON payload.
  for marker,target in [('OPSD_E1_OUTER_JSON=',rows),('SEARCH_R1_FINAL_METRICS_JSON=',final)]:
   if marker in line:target.append(unique(line.split(marker,1)[1]))
 fail(len(rows)==300 and [v['actual_outer_updates'] for v in rows]==list(range(1,301)),'Missing/duplicate actual300 outer records')
 fail(len({v['driver_pid'] for v in rows})==1 and len({v['config_sha256'] for v in rows})==1,'Driver/config changed')
 from search_r1.baselines.checkpoint_gate import validate_outer,validate_evaluation
 for i,row in enumerate(rows,1):
  fail(row['source_commit']==SOURCE and row['job_id']==job and row['global_step']==i,'Wrong actual training source/job/step');validate_outer(row['metrics'])
  if i%50==0:validate_evaluation(row['metrics'],512)
  if i==5 or i%50==0:validate_evaluation({k.removeprefix('opsd_stage/'):v for k,v in row['metrics'].items() if k.startswith('opsd_stage/')},64)
 fail(len(final)==1,'Final evaluation missing/duplicated');validate_evaluation(final[0],51713)
 return {'actual_outer_updates':300,'driver_pid':rows[0]['driver_pid'],'config_sha256':rows[0]['config_sha256'],'final_count':51713,'scope':'Process/log count audit only; independent first5 and final scientific review remain separate'}

def validate_resources(path,job,uuids,service_pid):
 count=errors=coexist=0;previous=None;first=None;last=None;maxgap=0.0
 with Path(path).open() as handle:
  for line in handle:
   row=unique(line);fail(row['job_id']==job,'Sampler job changed')
   timestamp=datetime.datetime.fromisoformat(row['time']);fail(timestamp.tzinfo is not None and timestamp.utcoffset()==datetime.timedelta(0),'Sampler timestamp must be UTC-aware')
   if previous is not None:
    gap=(timestamp-previous).total_seconds();fail(gap>0,'Sampler duplicate/backward timestamp');maxgap=max(maxgap,gap)
   previous=timestamp;first=first or timestamp;last=timestamp;count+=1
   if 'query_error' in row:errors+=1;continue
   gpus=row['gpus'];fail(len(gpus)==8 and {r['uuid'] for r in gpus}==set(uuids),'Sampler UUID count changed')
   for v in gpus:fail(all(math.isfinite(v[k]) for k in ('util','used_mib','total_mib')) and 0<=v['util']<=100 and 0<=v['used_mib']<=v['total_mib'] and v['total_mib']>0,'Invalid sampled GPU values')
   apps=row['gpu_processes']
   for app in apps:fail(not app.get('unavailable') and app.get('in_slurm_job') is True and str(app.get('uid'))==str(os.getuid()) and app['uuid'] in set(uuids),'Foreign/unavailable sampled GPU application')
   pids={v['pid'] for v in apps}
   workers=[v for v in apps if v['pid']!=service_pid]
   if len(pids)==9 and service_pid in pids and {v['uuid'] for v in workers}==set(uuids) and all(len({v['uuid'] for v in workers if v['pid']==pid})==1 for pid in pids-{service_pid}):coexist+=1
 fail(count>=2 and coexist>=2,'No two real eight-worker plus service observations')
 return {'samples':count,'query_errors':errors,'eight_worker_plus_service_samples':coexist,'first_timestamp':first.isoformat(),'last_timestamp':last.isoformat(),'max_gap_seconds':maxgap,'sha256':sha(path),'scope':'NVML sampled values and per-PID observed CPU lower bounds; missing queries/gaps retained, not allocation release or exclusive teacher cost. RSS sums double-count shared pages; cgroup scope may be shared Slurmd, not job peak.'}

def run_training(run,tools):
 repo,m=sources(tools);ready=unique((run/'approval-ready.json').read_bytes());ready['ready_sha256']=sha(run/'approval-ready.json')
 check_approval(run,ready);verify_binding(run,ready)
 os.environ.update(OPSD_E1_BINDING_MANIFEST=str(run/'bindings/manifest.json'),OPSD_E1_BINDING_MANIFEST_SHA256=ready['binding_sha256'],OPSD_E1_REVIEW_PUBLIC_KEY=str(run/'bindings/public.json'),OPSD_E1_REVIEW_PUBLIC_KEY_SHA256=ready['public_key_sha256'])
 from search_r1.baselines.preflight import validate_inherited_gpu_leases
 validate_inherited_gpu_leases(8);leases=unique(os.environ['SEARCH_R1_GPU_LEASES']);uuids=[v['uuid'] for v in leases];fail(set(uuids)==set(ready['nvml_uuids']),'Lease/allocated UUID mismatch');fds=tuple([9]+[int(v['fd']) for v in leases]);os.fstat(9)
 emit(run/'evidence/leases.json',leases);emit(run/'evidence/post-lease-double15.json',double_idle(uuids))
 runtime=Path('/tmp')/('o4e-'+os.environ['SLURM_JOB_ID']);fail(not runtime.exists() and not runtime.is_symlink(),'Short runtime already exists');runtime.mkdir(mode=0o700);(runtime/'ray').mkdir();(run/'runtime').symlink_to(runtime,target_is_directory=True)
 os.environ.update(TMPDIR=str(runtime),RAY_TMPDIR=str(runtime/'ray'),RAY_ADDRESS='127.0.0.1:16384',RAY_USAGE_STATS_ENABLED='0',VLLM_NO_USAGE_STATS='1',WANDB_MODE='offline')
 children=[];service=None;service_owner=None
 try:
  fail(not (run/'resources.ndjson').exists(),'Resource output already exists')
  sampler=subprocess.Popen([sys.executable,str(tools/'resource_sample.py'),str(run/'resources.ndjson'),str(os.getpid())],pass_fds=fds);children.append((sampler,own(sampler.pid)))
  with (run/'evidence/ray-start.log').open('xb') as out:
   subprocess.run(['ray','start','--head','--port=16384','--num-gpus=8','--num-cpus=80','--object-store-memory=8589934592','--include-dashboard=false','--min-worker-port=30000','--max-worker-port=30999','--temp-dir='+str(runtime/'ray')],check=True,stdout=out,stderr=subprocess.STDOUT,pass_fds=fds,timeout=120)
  subprocess.run([sys.executable,str(tools/'ray_cpu_check.py')],check=True,pass_fds=fds,timeout=90)
  with (run/'evidence/hardware-torchrun.log').open('xb') as out:
   p=subprocess.Popen(['torchrun','--standalone','--nproc_per_node=8',str(tools/'hardware_rank.py')],pass_fds=fds,stdout=out,stderr=subprocess.STDOUT);children.append((p,own(p.pid)));rc=p.wait(timeout=180);emit(run/'evidence/hardware-exit.json',{'returncode':rc});fail(rc==0,'Actual8 NCCL/Flash gate failed')
  ranks=[unique((run/'evidence'/f'hardware-rank-{i}.json').read_bytes()) for i in range(8)];fail({v['uuid'] for v in ranks}==set(uuids) and {v['rank'] for v in ranks}==set(range(8)),'Actual8 hardware UUID differs')
  assets=pinned(run/'bindings/tool_manifest.json',ASSETS_SHA);results=assets['results'];env=dict(os.environ,PYTHONPATH=str(repo)+':'+str(OLD/'retriever-gpu-lib'))
  with (run/'retriever.log').open('xb') as out:
   service=subprocess.Popen([sys.executable,'-u',str(tools/'retriever_entry.py'),'--index_path',results['index']['path'],'--corpus_path',results['corpus']['path'],'--retriever_model',str(Path(results['e5'][0]['path']).parent),'--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18084','--threads','8','--topk','3'],env=env,pass_fds=fds,stdout=out,stderr=subprocess.STDOUT)
  service_owner=own(service.pid);children.append((service,service_owner));emit(run/'evidence/service.json',{'owner':service_owner,'command':service.args,'uuids':uuids,'pythonpath':env['PYTHONPATH']})
  deadline=time.monotonic()+600
  while True:
   fail(service.poll() is None and same_process(service_owner),'Own retriever died');raw=command(['ss','-lntp']);lines=[v for v in raw.splitlines() if re.search(r'127\.0\.0\.1:18084\s',v)]
   if lines:
    fail(len(lines)==1 and re.search(r'pid='+str(service.pid)+r'[,)]',lines[0]),'Listener is not own exact service PID');(run/'evidence/service-listener.txt').write_text(raw);break
   fail(time.monotonic()<deadline,'Own retriever startup timeout');time.sleep(2)
  http24(run,service.pid);emit(run/'evidence/pre-training-double15.json',double_idle(uuids,service.pid))
  emit(run/'evidence/actual-training-wrapper-environment.json',environment())
  # Frozen entry performs CPU validate, actual full import/live-code audit and
  # inherited lease verification. It creates train.log/resolved-profile itself.
  p=subprocess.Popen(['bash',str(repo/'reproduction/baselines/run_opsd_e1_300.sh'),'--run'],pass_fds=fds,env=dict(os.environ));children.append((p,own(p.pid)))
  while p.poll() is None:
   fail(same_process(service_owner) and service.poll() is None,'Own local retrieval lost during training');time.sleep(5)
  emit(run/'evidence/train-process-exit.json',{'returncode':p.returncode});fail(p.returncode==0,'Training process failed')
  terminate_owned(sampler,children[0][1]);emit(run/'evidence/sampler-stop.json',{'pid':sampler.pid,'returncode':sampler.returncode,'scope':'Stopped our sampler before final NDJSON read'})
  resource_summary=validate_resources(run/'resources.ndjson',os.environ['SLURM_JOB_ID'],uuids,service.pid);emit(run/'evidence/resource-data-audit.json',resource_summary)
  summary=check_training_log(run/'train.log',os.environ['SLURM_JOB_ID']);fail((run/'acceptance-5-receipt.json').is_file(),'Independent first5 receipt absent');sources(tools);verify_binding(run,ready)
  emit(run/'process-finished-review-required.json',summary)
 finally:
  # Only our exact Popen children can receive a signal. Ray daemon descendants
  # are left to the Slurm job teardown; never use global ray stop/pkill.
  for p,owner in reversed(children):terminate_owned(p,owner)
  emit(run/'evidence/owned-child-cleanup.json',{'children':[v for _,v in children],'ray_daemons':'Slurm cgroup teardown; no global kill','runtime_logs':str(runtime/'ray'),'scope':'Not evidence of resource release; capture terminal scontrol separately'})

def main():
 fail(sys.flags.optimize==0,'Optimized Python disables upstream asset asserts')
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args();fail(a.prepare!=a.run,'Choose exactly one mode')
 run=Path(os.environ['E1_FORMAL_RUN']);fail(os.environ['SEARCH_R1_BASELINE_RUN_ROOT']==str(run),'Training runroot differs')
 fail(run==ROOT/'runs'/('formal-'+os.environ['SLURM_JOB_ID']) and run.resolve(strict=True)==run and run.stat().st_uid==os.getuid(),'Wrong physical own runroot')
 tools=Path(os.environ['E1_FORMAL_TOOLS']);sources(tools)
 prepare(run,tools) if a.prepare else run_training(run,tools)
if __name__=='__main__':main()
