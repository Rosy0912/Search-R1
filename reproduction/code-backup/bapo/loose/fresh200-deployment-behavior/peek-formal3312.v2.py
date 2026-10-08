from pathlib import Path
import json,hashlib,datetime,subprocess
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006');JOB=3312;SOURCE='90755a561abd239fbb23882939410e643bb913ae'
receipt=json.loads((R/'scheduler-head/formal200-receipt.json').read_text());assert receipt['source']==SOURCE and str(receipt['job_id'])==str(JOB) and receipt['nonce']=='8c29851b54174f2083023a98cde9c19b'
run=R/'runs'/('formal200-'+receipt['nonce']+'-'+str(JOB));assert run.resolve()==run
out=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,source_commit=SOURCE,run_root=str(run),receipt=receipt,files=[p.name for p in run.iterdir()]if run.exists()else[],launch_approval_sha256=hashlib.sha256((R/'logs/launch-approval.json').read_bytes()).hexdigest())
for name in ['train.log','stage.log','resource-watch.jsonl','retriever.log','resource-monitor.jsonl']:
 p=run/name
 if p.is_file():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-120000));raw=f.read()
  text=raw.decode(errors='replace');out[name+'_bytes']=p.stat().st_size;out[name+'_tail']=text[-6000:]
  if name=='train.log':
   attempts=[]
   for line in text.splitlines():
    marker='FRESH200_ATTEMPT_JSON='
    if marker in line:
     try:attempts.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
     except ValueError:pass
   out['latest_driver_attempts']=attempts[-3:]
  if name=='resource-watch.jsonl':
   for line in reversed(text.splitlines()):
    try:out['last_complete_resource_sample']=json.loads(line);break
    except ValueError:pass
resource=run/'resource-watch.jsonl'
if resource.is_file():
 last=None;last_raw=None
 with resource.open('rb')as f:
  for line in f:
   if not line.endswith(b'\n'):continue
   try:last=json.loads(line);last_raw=line
   except ValueError:continue
 if last is not None:
  out['last_complete_resource_sample_summary']={k:last.get(k)for k in ['time','run','slurm_job_id','pause','reason','phase_pid','external_totals','exceeded']}
  out['last_complete_resource_sample_summary'].update(raw_complete_line_bytes=len(last_raw),raw_complete_line_sha256=hashlib.sha256(last_raw).hexdigest(),journal_path=str(resource),owned_process_count=len(last.get('owned_before_query',{})))
print(json.dumps(out,allow_nan=False))
