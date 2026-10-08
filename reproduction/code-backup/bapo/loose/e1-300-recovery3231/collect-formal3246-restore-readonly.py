import pathlib,json,time,subprocess,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005/runs/bapo-instruct300-formal300-3246')
raw=(r/'train.log').read_text(errors='replace');decoder=json.JSONDecoder();records={}
for marker in ['BAPO_RECOVERY_RANK_JSON=','BAPO_RECOVERY_DRIVER_JSON=']:
 records[marker]=[decoder.raw_decode(line.split(marker,1)[1].lstrip())[0] for line in raw.splitlines() if marker in line]
files={}
for n in ['resolved-config.yaml','hardware-receipt-check.log','recovery-permission-check.json','retriever-health.json']:
 p=r/n
 if p.is_file():
  b=p.read_bytes();files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
w=(r/'resource-watch.jsonl').read_text().splitlines();latest=json.loads(w[-1]);fields=['time','reason','slurm_job_id','run','phase_pid','paused','action','external_totals']
procs={}
for pid in [190996,188479]:
 p=pathlib.Path('/proc')/str(pid);procs[str(pid)]={'exists':p.exists(),'stat':(p/'stat').read_text() if p.exists() else None}
print(json.dumps({'job_id':3246,'time_unix':time.time(),'records':records,'files':files,'resource_samples':len(w),'last_resource':{k:latest.get(k)for k in fields},'processes':procs},allow_nan=False))
