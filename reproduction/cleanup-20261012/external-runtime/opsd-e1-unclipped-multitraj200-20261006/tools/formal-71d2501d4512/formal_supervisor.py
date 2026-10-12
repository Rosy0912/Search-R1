"""Preparation-only author200 launcher. No signing, resume, algorithm or receipt production."""
import argparse,datetime,hashlib,hmac,json,math,os,re,secrets,signal,socket,subprocess,sys,time
from pathlib import Path
from owned_runtime import sha,unique,physical,pinned,emit,command,own,same_process,snapshot,verify_tool_assets
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
ASSET_ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004')
OLD=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
MODEL=ASSET_ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COMMON_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
ASSETS_SHA='87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
TOOLS={'formal_job.sh','formal_supervisor.py','owned_runtime.py','resource_sample.py','retriever_entry.py','ray_cpu_check.py','hardware_rank.py','cpu_import_check.py'}
SOURCE_FILES={'reproduction/baselines/run_opsd_e1_multitraj200.sh','reproduction/baselines/e1_import_bootstrap.py','reproduction/h20/retriever_local.py','search_r1/search/retrieval_server.py','verl/trainer/config/ppo_opsd_e1_multitraj200.yaml','search_r1/baselines/main.py','search_r1/baselines/preflight.py','search_r1/baselines/e1_profile.py','search_r1/baselines/controller.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py', 'search_r1/baselines/trajectory_archive.py', 'search_r1/baselines/rollout.py', 'reproduction/baselines/e1-multitraj200/microbatch_probe.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/ray_trainer.py','verl/utils/torch_functional.py'}
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
 fail(re.fullmatch(r'[0-9a-f]{40}',SOURCE) is not None,'Unbound source candidate cannot launch')
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
  subprocess.run([sys.executable,str(repo/'reproduction/baselines/e1_import_bootstrap.py'),'--mode','validate','--profile','multitraj200'],stdout=out,check=True,env=dict(os.environ))
 assets=pinned(roles['tool_manifest']['path'],ASSETS_SHA);verify_tool_assets(assets)
 emit(run/'evidence/pre-gpu-python-environment.json',{'packages':command([sys.executable,'-c','import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ["torch","transformers","ray","vllm","flash-attn","hydra-core","click"]},sort_keys=True))']),'cuda_context_initialization':'No model, CUDA context or service launched by wrapper before approval; torch CPU import/check explicitly observed cuda_is_initialized false'})
 ready={'version':1,'source_commit':SOURCE,'job_id':job,'uid':os.getuid(),'pid':os.getpid(),'process_start_ticks':owner['start'],'process':owner,'run_root':str(run),'nonce':secrets.token_hex(32),'binding_sha256':binding_sha,'public_key_sha256':key['sha256'],'nvml_uuids':[r['uuid'] for r in first['inventory']],'scope':'Await external pre-GPU review; not first5/200/effect approval'}
 emit(run/'approval-ready.json',ready);ready['ready_sha256']=sha(run/'approval-ready.json')
 print('OPSD_E1_EXTERNAL_PRE_GPU_APPROVAL_READY='+json.dumps(ready,sort_keys=True),flush=True)
 deadline=time.monotonic()+int(os.environ.get('E1_GPU_APPROVAL_TIMEOUT_SECONDS','7200'))
 while not ((run/'bindings/gpu-init-approval.json').exists() and (run/'bindings/gpu-init-approval.sha256').exists()):
  fail(time.monotonic()<deadline,'External pre-GPU approval timed out');time.sleep(2)
 check_approval(run,ready);verify_binding(run,ready);sources(tools)
 # exec preserves PID and parent-held FD9; frozen preflight acquires leases and
 # execs the same supervisor PID. Never invoke a second independent RL job.
 os.execv(sys.executable,[sys.executable,str(repo/'reproduction/baselines/e1_import_bootstrap.py'),'--mode','preflight','--profile','multitraj200','--',sys.executable,str(tools/'formal_supervisor.py'),'--run'])

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

def validate_capacity_rank(r,job):
 fail(r['passed'] is True and r['source_commit']==SOURCE and r['job_id']==job and r['world_size']==8 and r['optimizer_steps']==2 and r['component_clip'] is None and r['response_only_logits'] is False and type(r['teacher_forward_micro_batch']) is int and r['teacher_forward_micro_batch']==4 and r['teacher_logits_recompute'] is True,'Wrong actual native-head/full-teacher4 capacity source/config/update evidence')
 fail(r['local_micro']==4 and r['input_width']==6144 and r['response_width']==4096 and r['minimum_required_free_fraction']==.15 and r['estimated_peak_device_used_bytes']<=.85*r['gpu_total_bytes'],'Requested worst-width capacity/headroom not passed')
 numeric=r['short_native_teacher_logits_recompute_equivalence']
 fail(numeric['comparison_kind']=='NATIVE_FULL_HEAD_FULL_TEACHER4_STORED_VS_RECOMPUTED_TARGETS' and numeric['response_only_logits'] is False and type(numeric['stored_teacher_forward_micro_batch']) is int and numeric['stored_teacher_forward_micro_batch']==4 and type(numeric['recomputed_teacher_forward_micro_batch']) is int and numeric['recomputed_teacher_forward_micro_batch']==4,'Wrong native full-head/full-teacher4 stored-versus-recomputed target path')
 fail(numeric['stored_teacher_logits_recompute'] is False and numeric['recomputed_teacher_logits_recompute'] is True,'Wrong target recomputation flags')
 calls=numeric['teacher_projection_calls']
 fail(set(calls)=={'stored_forward','stored_backward','recomputed_forward','recomputed_backward'} and all(type(value) is int for value in calls.values()) and calls==dict(stored_forward=2,stored_backward=0,recomputed_forward=2,recomputed_backward=2),'Wrong teacher projection forward/backward calls')
 forward_features=numeric['teacher_projection_forward_feature_sha256'];backward_features=numeric['teacher_projection_backward_feature_sha256']
 fail(type(forward_features) is list and len(forward_features)==2 and all(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value) for value in forward_features) and backward_features==forward_features,'Teacher backward feature identity/order differs')
 raw=numeric['shard_gradient_cosine_raw'];norm=numeric['shard_gradient_cosine'];rounded=numeric['shard_gradient_cosine_roundoff_applied']
 fail(type(raw) in (int,float) and math.isfinite(raw) and type(norm) in (int,float) and math.isfinite(norm) and type(rounded) is bool and numeric['maximum_cosine_roundoff']==1e-12,'Wrong diagnostic raw cosine evidence')
 expected=min(1.,max(-1.,raw));is_rounded=not -1<=raw<=1
 fail(abs(raw-expected)<=1e-12 and norm==expected and rounded is is_rounded,'Cosine roundoff normalization exceeds diagnostic tolerance')
 forward_q=numeric['teacher_q_forward_mean_probability_l1_difference'];backward_q=numeric['teacher_q_backward_mean_probability_l1_difference']
 fail(all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in (forward_q,backward_q)) and numeric['teacher_q_mean_probability_l1_difference']==max(forward_q,backward_q),'Wrong separate forward/backward teacher q comparison')
 values={key:numeric[key] for key in ('global_native_stored_squared_gradient_norm','global_native_recomputed_squared_gradient_norm','shard_gradient_relative_l2','shard_gradient_cosine','loss_relative_difference','teacher_q_mean_probability_l1_difference')}
 fail(all(type(value) in (int,float) and math.isfinite(value) for value in values.values()),'Nonfinite or nonnumeric native target recomputation comparison')
 fail(all(values[key]>=0 for key in values if key!='shard_gradient_cosine') and -1<=values['shard_gradient_cosine']<=1,'Out-of-range native target recomputation comparison')
 fail(numeric['passed'] is True and numeric['global_native_stored_squared_gradient_norm']>0 and numeric['global_native_recomputed_squared_gradient_norm']>0 and numeric['shard_gradient_relative_l2']<=.02 and numeric['shard_gradient_cosine']>=.9999 and numeric['loss_relative_difference']<=.002 and numeric['teacher_q_mean_probability_l1_difference']<=1e-4,'Native full-head/full-teacher4 target recomputation comparison failed')
 lp=r['old_logprob_capacity'];fail(lp['passed'] is True and lp['local_micro']==8 and lp['global_micro']==64 and lp['logprob_shape']==[8,4096] and lp['entropy_shape']==[8,4096] and lp['logprob_finite'] is True and lp['entropy_finite'] is True and lp['estimated_peak_device_used_bytes']<=.85*r['gpu_total_bytes'],'Actual requested old-logprob micro8 GPU capacity/numeric gate failed')

def terminate_owned(proc,owner):
 if proc is None or proc.poll() is not None:return
 fail(owner is not None and same_process(owner),'Refusing to terminate stale/foreign PID')
 proc.terminate()
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:
  fail(same_process(owner),'Refusing to kill stale/foreign PID');proc.kill();proc.wait(timeout=10)

def training_log_context(run,job):
 # Context is an independently signed actual step5 checkpoint, not inferred
 # from a transport prefix or a guessed future worker list.
 from search_r1.baselines import checkpoint_gate as gate
 manifest_path=under(run/'full_state/global_step_5/manifest.json',run)
 manifest=gate.load_json(manifest_path)
 receipt=gate.load_json(under(run/'acceptance-5-receipt.json',run))
 fail(set(receipt)=={'payload','signature_hex'},'Wrong first5 signed envelope')
 ready=unique(under(run/'approval-ready.json',run).read_bytes())
 ready['ready_sha256']=sha(run/'approval-ready.json')
 check_approval(run,ready);verify_binding(run,ready)
 key=pinned(run/'bindings/public.json',ready['public_key_sha256'])
 gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],key)
 payload=receipt['payload']
 fail(payload['checkpoint_manifest_sha256']==sha(manifest_path),'Signed first5 manifest changed')
 fail(payload['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW','Wrong first5 authorization')
 for key in ('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce'):
  fail(payload[key]==manifest[key],'First5 signed identity changed: '+key)
 fail(manifest['method']=='pure_opsd' and manifest['source_commit']==SOURCE and manifest['job_id']==job and manifest['run_root']==str(run) and manifest['uid']==os.getuid(),'First5 checkpoint is foreign')
 fail((manifest['global_step'],manifest['actual_outer_updates'],manifest['next_global_step'],manifest['horizon'],manifest['warmup_steps'],manifest['world_size'])==(5,5,6,200,57,8),'First5 budget differs')
 fail(payload['checkpoint_files']==manifest['files'],'Signed checkpoint files differ')
 for record in manifest['files'].values():gate.verify_file(record,run)
 workers=gate.manifest_worker_pids(manifest,run)
 return gate,manifest,workers

def check_training_log(path,job):
 gate,manifest,workers=training_log_context(Path(path).parent,job)
 rows=[];final=[]
 for line in Path(path).read_text().splitlines():
  # Ray adds PID prefixes; the immutable marker still starts the JSON payload.
  for marker,target in [('OPSD_E1_OUTER_JSON=',rows),('SEARCH_R1_FINAL_METRICS_JSON=',final)]:
   if marker in line:target.append(gate.parse_log_marker(line,marker,driver_pid=manifest['driver_pid'],job_id=job,worker_pids=workers))
 fail(len(rows)==200 and [v['actual_outer_updates'] for v in rows]==list(range(1,201)),'Missing/duplicate actual200 outer records')
 fail(len({v['driver_pid'] for v in rows})==1 and {v['config_sha256'] for v in rows}=={manifest['config_sha256']},'Driver/config changed')
 validate_outer=gate.validate_outer;validate_evaluation=gate.validate_evaluation
 for i,row in enumerate(rows,1):
  fail(row['source_commit']==SOURCE and row['job_id']==job and row['global_step']==i,'Wrong actual training source/job/step');validate_outer(row['metrics'], profile='multitraj200')
  lr=row['metrics'].get('actor/lr');fail(type(lr) in (int,float) and math.isfinite(lr) and math.isclose(lr,1e-6*min(i/57,1),rel_tol=1e-12,abs_tol=1e-18),'Actual outer post-update warmup LR differs')
  used_lr=1e-6*min((i-1)/57,1)
  for rank in range(8):
   actual=row['metrics'].get(f'actor/optimizer_lr_used_mean/rank_{rank}')
   fail(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,used_lr,rel_tol=1e-12,abs_tol=1e-18) and row['metrics'].get(f'actor/optimizer_lr_used_finite/rank_{rank}')==1,'Actual rank optimizer-used LR differs from outer warmup clock')
  if i%50==0:validate_evaluation(row['metrics'],512)
  if i==5 or i%50==0:validate_evaluation({k.removeprefix('opsd_stage/'):v for k,v in row['metrics'].items() if k.startswith('opsd_stage/')},64)
 fail(len(final)==1,'Final evaluation missing/duplicated');validate_evaluation(final[0],51713)
 return {'actual_outer_updates':200,'driver_pid':rows[0]['driver_pid'],'config_sha256':rows[0]['config_sha256'],'final_count':51713,'scope':'Process/log count audit only; independent first5 and final scientific review remain separate'}

def validate_resources(path,job,uuids,service_pid):
 count=errors=coexist=resolved_exits=0;previous=None;first=None;last=None;maxgap=0.0;prior_owned={}
 with Path(path).open() as handle:
  for line in handle:
   row=unique(line);fail(row['job_id']==job,'Sampler job changed')
   timestamp=datetime.datetime.fromisoformat(row['time']);fail(timestamp.tzinfo is not None and timestamp.utcoffset()==datetime.timedelta(0),'Sampler timestamp must be UTC-aware')
   if previous is not None:
    gap=(timestamp-previous).total_seconds();fail(gap>0,'Sampler duplicate/backward timestamp');maxgap=max(maxgap,gap)
   previous=timestamp;first=first or timestamp;last=timestamp;count+=1
   if 'query_error' in row:
    # Ordinary missing NVML samples are reported as gaps. An application
    # already observed but whose ownership is unresolved is a hard failure.
    for app in row.get('gpu_processes',[]):
     fail(not app.get('unavailable') and app.get('in_slurm_job') is True and str(app.get('uid'))==str(os.getuid()) and app.get('uuid') in set(uuids),'NVML confirmation failed with unresolved/foreign GPU application')
    errors+=1;continue
   gpus=row['gpus'];fail(len(gpus)==8 and {r['uuid'] for r in gpus}==set(uuids),'Sampler UUID count changed')
   for v in gpus:fail(all(math.isfinite(v[k]) for k in ('util','used_mib','total_mib')) and 0<=v['util']<=100 and 0<=v['used_mib']<=v['total_mib'] and v['total_mib']>0,'Invalid sampled GPU values')
   apps=row['gpu_processes']
   live_apps=[]
   for app in apps:
    fail(app['uuid'] in set(uuids),'Sampled application on foreign GPU')
    key=(app['uuid'],app['pid'])
    if app.get('unavailable'):
     resolution=app.get('exit_race_resolution',{})
     expected=prior_owned.get(key)
     raw=row.get('nvml_apps_recheck_original')
     fail(expected is not None and app.get('read_error',{}).get('errno')==2 and resolution.get('status')=='EXITED_AFTER_NVML_PRIOR_OWNED' and resolution.get('prior')==expected and resolution.get('proc_exists_after') is False and resolution.get('absent_from_confirmed_nvml') is True and type(raw) is str,'Unresolved unavailable GPU application')
     import csv
     confirmed_rows=list(csv.reader(raw.splitlines()));fail(all(len(v)==3 for v in confirmed_rows),'Malformed confirmation NVML application record')
     confirmed={(v[0].strip(),int(v[1])) for v in confirmed_rows}
     fail(key not in confirmed,'Exited PID still in confirmed NVML')
     resolved_exits+=1
     continue
    fail(app.get('in_slurm_job') is True and str(app.get('uid'))==str(os.getuid()),'Foreign sampled GPU application')
    prior_owned[key]={'time':row['time'],'record':app}
    live_apps.append(app)
   apps=live_apps
   pids={v['pid'] for v in apps}
   workers=[v for v in apps if v['pid']!=service_pid]
   if len(pids)==9 and service_pid in pids and {v['uuid'] for v in workers}==set(uuids) and all(len({v['uuid'] for v in workers if v['pid']==pid})==1 for pid in pids-{service_pid}):coexist+=1
 fail(count>=2 and coexist>=2,'No two real eight-worker plus service observations')
 return {'samples':count,'query_errors':errors,'resolved_exit_races':resolved_exits,'eight_worker_plus_service_samples':coexist,'first_timestamp':first.isoformat(),'last_timestamp':last.isoformat(),'max_gap_seconds':maxgap,'sha256':sha(path),'scope':'NVML sampled values and per-PID observed CPU lower bounds; missing queries/gaps retained, not allocation release or exclusive teacher cost. RSS sums double-count shared pages; cgroup scope may be shared Slurmd, not job peak.'}

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
  # Isolated synthetic workers exit before the fresh training process starts;
  # neither their weights nor Adam state can enter official initialization.
  capacity=run/'evidence/capacity-micro4';capacity.mkdir()
  cenv=dict(os.environ,OPSD_EVIDENCE=str(capacity),OPSD_E1_OFFICIAL_MODEL=str(MODEL),OPSD_E1_OFFICIAL_MANIFEST=str(MODEL/'VERIFIED-MODEL-MANIFEST.json'),E1_EXPECTED_SOURCE=SOURCE)
  with (capacity/'torchrun.log').open('xb') as out:
   probe=subprocess.Popen(['torchrun','--standalone','--nproc_per_node=8',str(repo/'reproduction/baselines/e1-multitraj200/microbatch_probe.py'),'--micro','4','--input-width','6144','--response-width','4096','--output-dir',str(capacity)],pass_fds=fds,stdout=out,stderr=subprocess.STDOUT,env=cenv)
   children.append((probe,own(probe.pid)));capacity_rc=probe.wait();emit(capacity/'exit.json',{'returncode':capacity_rc,'rl_outer_updates':0})
  fail(capacity_rc==0,'Exact requested micro4 capacity probe failed; no fresh training started')
  capacity_result=unique((capacity/'micro-4.json').read_bytes())
  fail(capacity_result['passed'] is True and capacity_result['local_micro']==4 and len(capacity_result['ranks'])==8,'Incomplete requested micro4 all-rank capacity')
  fail({r['rank'] for r in capacity_result['ranks']}==set(range(8)),'Missing capacity rank')
  for r in capacity_result['ranks']:
   validate_capacity_rank(r,os.environ['SLURM_JOB_ID'])
  fail(same_process(service_owner) and service.poll() is None,'Own retriever lost during capacity probe')
  emit(run/'evidence/pre-training-double15-after-capacity.json',double_idle(uuids,service.pid))
  emit(run/'evidence/actual-training-wrapper-environment.json',environment())
  # Frozen entry performs CPU validate, actual full import/live-code audit and
  # inherited lease verification. It creates train.log/resolved-profile itself.
  p=subprocess.Popen(['bash',str(repo/'reproduction/baselines/run_opsd_e1_multitraj200.sh'),'--run'],pass_fds=fds,env=dict(os.environ));children.append((p,own(p.pid)))
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
