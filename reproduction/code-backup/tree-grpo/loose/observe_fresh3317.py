"""Read only own job audit metadata through the existing h100-9 route."""
import datetime, hashlib, json, pathlib, shlex, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / 'work/tree-fresh200-freeze-53082e-v12/actual-zero-dispatch'
label = sys.argv[1]
assert label.isalnum()
code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
run=root/'runs/tree-fresh200-3317'
env={k:v for k,v in os.environ.items() if k not in {'SLURM_CLUSTERS','SLURM_CONF','SLURM_CONF_SERVER','SLURM_CLUSTER_NAME','SLURM_FEDERATION','SLURM_FEDERATION_CLUSTER'} and not k.startswith(('SBATCH_','SQUEUE_','SACCT_','SCONTROL_','SCANCEL_'))}
out={'job_id':3317,'run':str(run),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Own Slurm/log diagnostics and JSON audit metadata only; no GPU calls, weights, dataset, PT or raw trajectories'}
out['prior_formal_dispatch_intents']=[]
for p in sorted((root/'instruct300').glob('TREE_FRESH200_TRAIN_ONCE-*.json')):
 raw=p.read_bytes();assert len(raw)<1024*1024
 value=json.loads(raw)
 out['prior_formal_dispatch_intents'].append({'name':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'status':value.get('status'),'job_id':value.get('job_id'),'cluster_name':value.get('cluster_name')})
for name,args in [('squeue',['squeue','--local','-h','-j','3317','-o','%i|%j|%T|%N']),('scontrol',['scontrol','--local','show','job','3317'])]:
 r=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
 out[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out['files']={str(p.relative_to(run)):p.stat().st_size for d in [run,run/'output',run/'output/probability-calibration',run/'output/tree-audit',run/'output/tree-rank-audit',run/'output/validation-cost',run/'output/trajectories/outer-000001'] if d.is_dir() for p in d.iterdir() if p.is_file()}
out['native_json']={}
out['active_validation_progress']=[]
for ledger in (run/'output/validation-cost').glob('validation-*.jsonl'):
 records=[]
 with ledger.open() as stream:
  for line in stream:
   try:records.append(json.loads(line))
   except json.JSONDecodeError:pass
 if records:
  kinds={kind:sum(r.get('kind')==kind for r in records) for kind in ['generation_start','generation_completed','generation_failed','http_start','http_completed','http_failed','validation_completed']}
  out['active_validation_progress'].append({'ledger':ledger.name,'phase':records[0].get('phase'),'global_step':records[0].get('global_step'),'events':len(records),'dataset_rows':records[0].get('dataset_rows'),'real_sample_rows_submitted':sum(r.get('rows',0) for r in records if r.get('kind')=='real_sample_batch'),'event_counts':kinds,'generation_wall_seconds_completed':sum(r.get('wall_seconds',0) for r in records if r.get('kind')=='generation_completed'),'http_wall_seconds_completed':sum(r.get('wall_seconds',0) for r in records if r.get('kind')=='http_completed'),'age_of_last_event_seconds':time.time()-records[-1]['epoch']})
names=['run-result.json','cost.json','termination-cost.json','scheduler-runtime-identity.json','guard-runtime-identity.json','launch-manifest.json','resolved.yaml','input-config.yaml','output/fresh-initialization.json','output/probability-calibration/ready.json','output/trajectories/outer-000001/manifest.json']
names += [str(p.relative_to(run)) for p in (run/'output/validation-cost').glob('validation-*.json')]
names += ['output/probability-calibration/rank-%02d.json'%r for r in range(8)]
names += ['output/probability-calibration/rank-%02d-mode.json'%r for r in range(8)]
for name in names:
 p=run/name
 if p.is_file():
  raw=p.read_bytes()
  assert len(raw)<2*1024*1024
  out['native_json'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()}
for name,p in [('stage.log',run/'stage.log'),('slurm.out',root/'logs/tree-fresh200-mu-zero-3317.out'),('slurm.err',root/'logs/tree-fresh200-mu-zero-3317.err'),('guard.log',run/'guard.log')]:
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-40000)); lines=f.read().decode('utf8','replace').splitlines()
  safe=[]
  for line in lines:
   # Never forward sampled text: generic substring matching can hit an answer.
   plain=re.sub(r'\x1b\[[0-9;]*m','',line)
   plain=re.sub(r'^\([^)]*(?:pid|rank)[^)]*\)\s*','',plain).strip()
   if any(tag in plain.lower() for tag in ('<think','<search','<answer','</think','</search','</answer')):continue
   if re.match(r'^(Traceback|File \"|(?:Runtime|Value|Type|Assertion|Key|Import|ModuleNotFound|OSError|Memory)Error:|TREE_[A-Z_]+(?:=|\b)|Initial validation metrics:|Validation metrics:|STOP(?:\s|:)|ERROR(?:\s|:))',plain):safe.append(plain[:2500])
  out[name]={'bytes':p.stat().st_size,'diagnostic_lines':safe[-40:]}
p=run/'resource-samples.jsonl'
if p.is_file():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-40000)); lines=f.read().splitlines()
 if lines:
  last=json.loads(lines[-1]);out['latest_resource_sample']=last;out['latest_resource_age_seconds']=time.time()-last.get('epoch',time.time())
print(json.dumps(out,ensure_ascii=False))
'''
argv = ['python3','-B',str(ROOT/'work/remote.py'),'python3 -B -c '+shlex.quote(code)]
start = time.monotonic()
result = subprocess.run(argv,capture_output=True)
for suffix,data in [('stdout',result.stdout),('stderr',result.stderr)]:
    with (DEST/(label+'_RAW.'+suffix)).open('xb') as handle: handle.write(data)
receipt={'returncode':result.returncode,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'readonly':True}
(DEST/(label+'_FETCH_RESULT.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
if result.returncode:
    print(result.stderr.decode('utf8','replace'));sys.exit(result.returncode)
data=json.loads(result.stdout)
for name,entry in data.pop('native_json').items():
    path=DEST/'native3317'/name;path.parent.mkdir(parents=True,exist_ok=True)
    raw=entry['text'].encode();assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    data[name]={'sha256':entry['sha256'],'bytes':entry['bytes']}
(DEST/(label+'_OBSERVATION.json')).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ['latest_resource_sample','scontrol','files']},ensure_ascii=False))
print('file_inventory',json.dumps(data['files']))
