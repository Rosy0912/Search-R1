import subprocess,pathlib,json,time,os,argparse,posixpath
parser=argparse.ArgumentParser();parser.add_argument("job",type=int);args=parser.parse_args();assert args.job>0
job=args.job;root='/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-grpo100-common-v5-r1/evidence'
body=r'''import pathlib,json,time,subprocess,hashlib,os
known=KNOWN_SIGNATURES
known_stats=KNOWN_STATS
r=pathlib.Path(ROOT);out={'scope':'READ_ONLY_ACTUAL_STANDARD_SEARCH_R1_GRPO100_COMMON_V5_R1','job_id':ACTUAL_JOB,'epoch':time.time(),'files':[],'errors':[]}
for cmd,key in [(['scontrol','show','job',str(ACTUAL_JOB),'-o'],'scontrol'),(['squeue','-h','-j',str(ACTUAL_JOB),'-o','%i|%T|%M'],'queue')]:
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=15);out[key]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
if r.exists():
 paths=[]
 for walk_root,dirs,names in os.walk(r,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in ('cache','checkpoints') and not (pathlib.Path(walk_root)/d).is_symlink())
  paths.extend(pathlib.Path(walk_root)/n for n in sorted(names))
 for p in sorted(paths):
  if not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log'):continue
  try:
   st=p.stat();item={'relative_path':str(p.relative_to(r)),'size':st.st_size,'mtime_epoch':st.st_mtime}
   prior=known_stats.get(str(p))
   if ('-ready-rank' in p.name or p.name=='global-worker-ready.json') and prior and prior['size']==st.st_size and prior['mtime_epoch']==st.st_mtime:
    item.update(original_path=str(p),sha256=prior['sha256'],unchanged_data_reference_only=True,fresh_hash_computed=False,reference_basis='PRIOR_ORIGINAL_SHA_WITH_UNCHANGED_SIZE_MTIME_NOT_FRESH_HASH')
    out['files'].append(item);continue
   if p.suffix=='.json' and st.st_size<256*1024:
    b=p.read_bytes();item.update(sha256=hashlib.sha256(b).hexdigest(),original_path=str(p))
    if known.get(str(p))==item['sha256'] and p.name not in ('trainer-first-exception.json','train-result.json','owned-cleanup.json'):
     item['unchanged_data_reference_only']=True
    else:item['data']=json.loads(b)
   elif p.suffix=='.json' and '-ready-rank' in p.name:
    b=p.read_bytes();v=json.loads(b);a=v.get('cuda_allocator_evidence',{});snap=a.get('original_snapshot',{});segments=snap.get('segments',[])
    item.update(sha256=hashlib.sha256(b).hexdigest(),derived_summary_only=True,summary={k:v.get(k) for k in ('pid','starttime_ticks','cgroup','job','hostname','role','rank','gpu_uuids','allocated_bytes','reserved_bytes','ready_epoch','cuda_allocator_environment')},derived_allocator_summary={'environment':a.get('environment'),'backend':a.get('backend'),'original_parsed_settings':snap.get('allocator_settings'),'original_segment_count':len(segments),'original_expandable_segment_count':sum(x.get('is_expandable') is True for x in segments),'original_field_all_supported':bool(segments) and all(type(x.get('is_expandable')) is bool for x in segments)})
   elif p.suffix in ('.jsonl','.log'):
    with p.open('rb') as f:f.seek(max(0,st.st_size-1500));item['tail']=f.read().decode(errors='replace')
   out['files'].append(item)
  except Exception as e:out['errors'].append({'path':str(p),'error':repr(e)})
p=r.parent/'actual-outer.json'
if p.exists():
 try:
  assert not p.is_symlink();a=p.stat();b=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
  item={'relative_path':'../actual-outer.json','original_path':str(p),'size':len(b),'mtime_epoch':z.st_mtime,'sha256':hashlib.sha256(b).hexdigest()}
  if known.get(str(p))==item['sha256']:item['unchanged_data_reference_only']=True
  else:item['data']=json.loads(b)
  out['files'].append(item)
 except Exception as e:out['errors'].append({'path':str(p),'error':repr(e)})
student={'job_id':3224,'scope':'READ_ONLY_DEPENDENT_STUDENT_CONTROLLER_STATUS_NOT_TRAINING_ACCEPTANCE','epoch':time.time()}
for cmd,key in [(['scontrol','show','job','3224','-o'],'scontrol'),(['squeue','-h','-j','3224','-o','%i|%T|%M|%R'],'queue')]:
 try:
  q=subprocess.run(cmd,capture_output=True,text=True,timeout=15);student[key]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
 except (subprocess.TimeoutExpired,OSError) as e:student[key]={'status':'UNKNOWN','error':repr(e)}
out['dependent_student']=student
print(json.dumps(out))
'''.replace('ROOT',repr(root)).replace('ACTUAL_JOB',repr(job))
out=pathlib.Path(f'outputs/remote-evidence/r1-standard32b-direct100-{job}-readonly.jsonl');end=time.monotonic()+48*3600
known_signatures={}
known_stats={}
if out.exists():
 with out.open() as prior:
  for line in prior:
   try:previous=json.loads(line)
   except json.JSONDecodeError:continue  # Preserve incomplete old bytes without inventing observations.
   for item in previous.get('files',[]):
    if item.get('sha256') and ('data' in item or item.get('unchanged_data_reference_only') or item.get('derived_summary_only')):
     known_signatures[item.get('original_path') or posixpath.normpath(root+'/'+item['relative_path'])]=item['sha256']
     known_stats[item.get('original_path') or posixpath.normpath(root+'/'+item['relative_path'])]={k:item[k] for k in ('size','mtime_epoch','sha256')}
while time.monotonic()<end:
 native_body=body.replace('KNOWN_SIGNATURES',repr(known_signatures)).replace('KNOWN_STATS',repr(known_stats))
 p=None
 for attempt in range(1,4):
  try:
   candidate=subprocess.run(['python3','work/remote_topd.py'],input="python3 - <<'OBSERVE'\n"+native_body+"\nOBSERVE\n",text=True,capture_output=True,timeout=90)
   if candidate.returncode==0:
    json.loads(candidate.stdout)  # Incomplete stdout is UNKNOWN, never a native observation.
    p=candidate;break
   failure=dict(error_type='ReadOnlyTransportNonzero',returncode=candidate.returncode,stderr=candidate.stderr[-1500:])
  except (subprocess.TimeoutExpired,OSError,json.JSONDecodeError) as exc:
   failure=dict(error_type=type(exc).__name__,error=str(exc))
  failure.update(scope='LOCAL_READ_ONLY_TRANSPORT_FAILURE_NOT_NATIVE_STATE',status='UNKNOWN',job_id=job,local_epoch=time.time(),attempt=attempt)
  errorfile=pathlib.Path(f'outputs/remote-evidence/r1-standard32b-direct100-{job}-readonly-transport-failures-r2.jsonl')
  with errorfile.open('a') as f:f.write(json.dumps(failure)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(failure),flush=True)
  if attempt<3:time.sleep(min(5*attempt,max(0,end-time.monotonic())))
 if p is None:
  time.sleep(min(45,max(0,end-time.monotonic())));continue
 if p.returncode:
  print(json.dumps({'transport_error':p.stderr[-1500:],'returncode':p.returncode}),flush=True)
 else:
  data=json.loads(p.stdout)
  for item in data.get('files',[]):
   if item.get('sha256') and ('data' in item or item.get('unchanged_data_reference_only') or item.get('derived_summary_only')):
    known_signatures[item.get('original_path') or posixpath.normpath(root+'/'+item['relative_path'])]=item['sha256']
    known_stats[item.get('original_path') or posixpath.normpath(root+'/'+item['relative_path'])]={k:item[k] for k in ('size','mtime_epoch','sha256')}
  with out.open('a') as f:f.write(json.dumps(data)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps({'epoch':data['epoch'],'queue':data['queue'],'files':[f['relative_path'] for f in data['files']],'errors':data['errors'],'dependent_student':data.get('dependent_student')}),flush=True)
  if any('JobState='+state in data['scontrol']['stdout'] for state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED')):
   final=pathlib.Path(f'outputs/remote-evidence/r1-standard32b-direct100-{job}-actual-final.json')
   with final.open('x') as f:json.dump({k:v for k,v in data.items() if k!='files'},f,indent=2);f.flush();os.fsync(f.fileno())
   break
 time.sleep(45)
