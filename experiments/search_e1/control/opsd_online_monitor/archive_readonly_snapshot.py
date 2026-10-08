from pathlib import Path
import hashlib,json,os,re,subprocess,time,math,socket
B=Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
R=Path('/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004')
RUN='f04fe593830c4831918ede872812c016'; UID=1001
started=time.time()
def stat(p):
 s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jread(p):
 p=Path(p);before=stat(p);raw=p.read_bytes();after=stat(p)
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),before=before,after=after,read_stable=before==after,data=json.loads(raw))
def kernel_uid(d):
 text=(d/'status').read_text();line=next(x for x in text.splitlines() if x.startswith('Uid:'));return int(line.split()[2])
def identity(pid):
 d=Path('/proc')/str(pid);raw=(d/'stat').read_text().rsplit(') ',1)[1].split();args=(d/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
 return dict(pid=pid,uid=kernel_uid(d),starttick=int(raw[19]),ppid=int(raw[1]),pgid=int(raw[2]),state=raw[0],argv=args)
manifest_roots=[('archive-runtime.sha256.json','runtime/archive'),('archive-jit-runtime.sha256.json','runtime/archive-jit-20261005'),('archive-jit-env-v2-runtime.sha256.json','runtime/archive-jit-20261005-env-v2'),('archive-unused-ray-path-v1-runtime.sha256.json','runtime/archive-unused-ray-path-v1'),('archive-opsd-runtime.sha256.json','opsd-runtime/archive')]
checks=[]
for name,relative in manifest_roots:
 item=jread(B/'receipts'/name);manifest=item.pop('data');root=B/relative;errors=[];unstable=[];actual={};total=0
 for rel,expected in sorted(manifest.items()):
  p=root/rel
  if not p.is_file():errors.append(dict(relative=rel,error='MISSING'));continue
  before=stat(p);raw=p.read_bytes();after=stat(p);observed=hashlib.sha256(raw).hexdigest();total+=len(raw);actual[rel]=observed
  if before!=after:unstable.append(rel)
  if observed!=expected:errors.append(dict(relative=rel,expected=expected,actual=observed))
 checks.append(dict(manifest=item,runtime_root=str(root),expected_files=len(manifest),actual_files_hashed=len(actual),bytes_hashed=total,all_match=not errors and not unstable and len(actual)==669,errors=errors,unstable=unstable,actual_hash_map_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest(),opsd_py_sha256=actual.get('search_e1/opsd.py'),core_sha256=actual.get('search_e1/core.py')))
processes=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if kernel_uid(d)!=UID:continue
  args=(d/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
  env=dict(x.decode(errors='replace').split('=',1) for x in (d/'environ').read_bytes().split(b'\0') if b'=' in x)
  if env.get('E1_RUN_ID')!=RUN and not any(str(B) in x or str(R) in x for x in args):continue
  q=identity(int(d.name));q['selected_environment']={k:env[k] for k in ('E1_RUN_ID','E1_BASE','E1_PROFILE','E1_STAGE','CUDA_VISIBLE_DEVICES','PYTHONPATH','E1_PHYSICAL_MICRO','E1_PHYSICAL_SCORE','E1_JIT_OPTIMIZER') if k in env};processes.append(q)
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:processes.append(dict(pid=int(d.name),error='PERMISSION_DENIED_same_uid'))
receipts={}
for name in ('archive-round-1-grpo.json','archive-jit-memory-continuous.json','archive-post-jit-memory.json','archive-live-jit-optimizer.json','archive-live-physical-score1.json','shared-gpu-oracle.json','archive-control-atomic-transition.json'):
 p=B/'receipts'/name
 if p.exists():receipts[name]=jread(p)
for name in ('shared-memory-monitor.json','joined-backend-transition.json','gpu-oracle-acceptance.json'):
 p=R/'receipts'/name
 if p.exists():receipts['retriever/'+name]=jread(p)
log=B/'runs/archive/round-1-grpo/train.log';before=stat(log);lines=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace')).splitlines();after=stat(log)
updates=[];validations=[]
for line in lines:
 if re.search(r'\bstep:\s*\d+',line):
  metrics={k:float(v) for k,v in re.findall(r'([A-Za-z0-9_/.-]+):([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)',line)}
  sm=re.search(r'\bstep:\s*(\d+)',line);metrics['step']=int(sm.group(1))
  if 'actor/grad_norm' in metrics:updates.append(metrics)
  if any(k.startswith('val/test_score/') for k in metrics):validations.append(metrics)
errors=[dict(line=i+1,kind=word) for i,line in enumerate(lines) for word in ('Traceback','CUDA out of memory','FloatingPointError','illegal memory access','CUDA error') if word in line]
checkpoints=[]
actor=B/'runs/archive/round-1-grpo/actor'
for d in sorted(actor.glob('global_step_*')):
 checkpoints.append(dict(path=str(d),config_exists=(d/'config.json').exists(),files=len([p for p in d.iterdir() if p.is_file()]),mtime_ns=d.stat().st_mtime_ns))
raw_gpu=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
devices=[dict(zip(('index','uuid','actual_free_mib','used_mib','total_mib','utilization_percent'),[int(x.strip()) if i!=1 else x.strip() for i,x in enumerate(line.split(','))])) for line in raw_gpu.splitlines() if line.strip()]
uuids={x['uuid'] for x in devices}
raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
apps=[]
for line in raw_apps.splitlines():
 parts=[x.strip() for x in line.split(',')]
 if parts[0] not in uuids:continue
 a=dict(gpu_uuid=parts[0],pid=int(parts[1]),used_mib=int(parts[2]));
 try:a['identity']=identity(a['pid'])
 except (FileNotFoundError,ProcessLookupError):a['identity']='VANISHED_DURING_READ'
 apps.append(a)
control={}
for name,regions in [('pipeline.sh',[(62,71),(94,100),(110,116)]),('verify_runtime.py',[(4,9)]),('build_runtime.py',[(7,10),(90,92),(102,105)]),('train_profile.sh',[(4,17)])]:
 p=B/'source/experiments/search_e1/control'/name;ls=p.read_text().splitlines();control[name]=dict(path=str(p),sha256=sha(p),excerpts=[dict(line=i,text=line) for i,line in enumerate(ls,1) if any(a<=i<=b for a,b in regions)])
try:tmux=subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pane_start_command}'],text=True);tmux=[line for line in tmux.splitlines() if any(x in line for x in ('archive','e1-','search-e1'))]
except subprocess.CalledProcessError as e:tmux=dict(error=str(e))
validation_counts={}
for line in (B/'data/validation.jsonl').read_text().splitlines():
 row=json.loads(line);key=row.get('data_source',row.get('source'));validation_counts[key]=validation_counts.get(key,0)+1
markers=[line[:500] for line in lines if re.search(r'epoch(?:=|:|\s).*step|epoch:\s*\d+|step\s+\d+',line)]
opsmanifest=json.loads((B/'receipts/archive-opsd-runtime.sha256.json').read_text());v2manifest=json.loads((B/'receipts/archive-jit-env-v2-runtime.sha256.json').read_text());diff=[dict(relative=k,opsd=opsmanifest.get(k),grpo_v2=v2manifest.get(k)) for k in sorted(opsmanifest.keys()|v2manifest.keys()) if opsmanifest.get(k)!=v2manifest.get(k)]
output=dict(schema='archive-correct-manifest-and-live-readonly-audit-v1',started_unix=started,ended_unix=time.time(),base=str(B),run_id=RUN,expected_uid=UID,node=dict(hostname=socket.gethostname(),actual_euid=os.geteuid(),kernel_uid_basis='Uid effective field from proc/status'),validation_counts=validation_counts,latest_epoch_step_markers=markers[-4:],scope='Read-only CPU source hashes and private identity; nvidia-smi limited output to self GPU4-7; no model hashes/CUDA/HTTP/signals/foreign GPU changes.',manifest_mappings=checks,opsd_vs_grpo_v2_declared_differences=diff,opsd_runtime_exists=(B/'opsd-runtime/archive').is_dir(),opsd_mapping_conclusion='archive-opsd-runtime.sha256.json binds actual opsd-runtime/archive, not runtime/archive-jit-20261005-env-v2; pipeline/verify/build source explicitly use that separate directory for subsequent OPSD. Synthetic OPSD memory gate is historical; actual current stage and output evidence are reported separately.',control_path_evidence=control,actual_processes=processes,receipts=receipts,pipeline_stage=dict(path=str(B/'receipts/pipeline-stage.txt'),text=(B/'receipts/pipeline-stage.txt').read_text()),train_log=dict(path=str(log),before=before,after=after,latest_real_updates=updates[-4:],native_validation=validations[-4:],known_fatal_patterns=errors,last_real_update=updates[-1] if updates else None,latest_nonmetric_progress=[line[:500] for line in lines[-50:] if 'Active batch size' in line or 'Trajectory stats' in line or 'Generating Sequences' in line][-5:]),checkpoints=checkpoints,actual_gpu4_7_devices=devices,actual_gpu4_7_apps=apps,tmux=tmux)
print(json.dumps(output,indent=2))
