import pathlib,json,subprocess,datetime,os,hashlib
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-653712b-20261012')
receipt=json.loads((base/'preparation/native-submit-receipt.json').read_text());job=str(receipt['slurm_job_id'])
def read(name):
 p=base/name
 return json.loads(p.read_text()) if p.exists() else None
def tail(p,n=10000):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
o=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base=str(base),job_id=job)
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True);o['native_status']=r.stdout or r.stderr
for n in ('preparation/native-submit-intent.json','preparation/native-submit-receipt.json','logs/pipeline-status.json','logs/cleanup-status.json','logs/full-state-restored-workers.json','logs/full-state-restored-driver.json','logs/five-resumed-updates-acceptance.json'):
 v=read(n)
 if v is not None:
  if n.endswith('five-resumed-updates-acceptance.json'):
   v={k:v[k] for k in ('accepted','run_id','source_completed_outer','source_actual_optimizer_writes_per_rank','first_resumed_outer','last_resumed_outer','next_outer','total_budget','checked_unix')}
   v['full_receipt_sha256']=hashlib.sha256((base/n).read_bytes()).hexdigest()
  o[n]=v
run=read('logs/run-manifest.json')
if run:o['run']=dict(run_id=run['run_id'],source_commit=run['source_commit'],full_state_restore=run.get('full_state_restore'),initial_probe_version=run.get('initial_probe_version'),micro=run.get('actor_micro_batch_per_rank'),cache=run.get('vllm_gpu_memory_utilization'))
for name in ('actual-updates.jsonl','actual-attempts.jsonl'):
 p=base/'logs'/name;rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else [];o[name]=dict(count=len(rows),last=rows[-1] if rows else None)
rank=[]
for i in range(8):
 p=base/'logs'/f'optimizer-rank-{i}.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else [];writes=[v for v in rows if v['kind']=='OPTIMIZER_WRITE'];s=base/'logs'/f'scheduler-rank-{i}.jsonl';sch=[json.loads(l) for l in s.read_text().splitlines()] if s.exists() else []
 rank.append(dict(rank=i,suffix_write_rows=len(writes),actual_last_write_count=writes[-1].get('write_count') if writes else None,last_write_context=writes[-1].get('context') if writes else None,scheduler_last=sch[-1] if sch else None))
o['ranks']=rank
o['individual_restore_receipts']=[read(f'logs/full-state-restored-rank-{i}.json') for i in range(8)]
for n in ('logs/native-'+job+'.log','logs/search-g1-common8-training.log'):
 p=base/n;o[n]=dict(bytes=p.stat().st_size,tail=tail(p)) if p.exists() else None
live=read('artifacts/live/manifest.json')
if live:o['probe']=dict(version=live['version'],round=live['round'],restoration_origin=live.get('restoration_origin'))
evaluation=read('logs/formal-inline-development-after.json')
if evaluation:
 o['after5_validation_summary']={k:evaluation[k] for k in ('step','weighted_em','weighted_f1')}
 o['after5_validation_summary']['qa_count']=len(evaluation['rows'])
 o['after5_validation_summary']['source_artifact_sha256']=hashlib.sha256((base/'logs/formal-inline-development-after.json').read_bytes()).hexdigest()
import re
training_tail=tail(base/'logs/search-g1-common8-training.log',20000) or ''
matches=re.findall(r'main_task pid=(\d+)',training_tail)
if matches:
 pid=int(matches[-1]);proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
  o['actual_training_driver_process']=dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],cgroup=(proc/'cgroup').read_text())
v=os.statvfs(base);o['storage']=dict(ordinary_free_bytes=v.f_bavail*v.f_frsize,physical_free_including_reserved_bytes=v.f_bfree*v.f_frsize,not_exclusive_quota=True)
o['varlen_actual']=[read(f'logs/actual-varlen-execution-rank-{i}.json') for i in range(8)]
resolved=read('logs/resolved-training-config.json')
if resolved:o['resolved_config_binding']={k:resolved.get(k) for k in ('resolved_config_sha256','run_id')}
print(json.dumps(o))
