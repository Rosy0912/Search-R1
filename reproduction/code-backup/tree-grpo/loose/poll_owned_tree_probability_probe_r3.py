"""Read-only scheduler and JSON/source metadata for one confirmed owned probe.

Does not load tensor files, construct models, call retrieval or change any job.
"""
import base64,gzip,hashlib,json,re,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
job=sys.argv[1];assert re.fullmatch('[1-9][0-9]*',job)
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
intent=root/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json'
journal=json.loads(intent.with_suffix(intent.suffix+'.result.json').read_bytes())
assert journal['status']=='SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE' and journal['job_id']==job
run=root/'runs'/('tree-probability-probe-'+job)
commands={}
for args in [['scontrol','show','job',job],['squeue','-h','-j',job,'-o','%i %j %T %M %N']]:
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    commands[' '.join(args)]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
files={}
def capture(path):
    if not path.is_file():return
    data=path.read_bytes()
    files[str(path)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
        gzip_base64=base64.b64encode(gzip.compress(data)).decode())
for name in ['initial-allocation.json','launch.json','torchrun-command.json','bundle.json',
             'termination-cost.json','result.json','resources.jsonl','process-resources.jsonl','guard.log','probe.log']:
    capture(run/name)
capture(root/'logs'/('tree-probability-probe-'+job+'.out'))
capture(root/'instruct300'/('probe-scheduler-observation-'+job+'.jsonl'))
summary=[]
for rank in range(8):
    path=run/'gpu-records'/('rank-%02d'%rank)
    jsons=sorted(path.glob('*.json'))
    for item in jsons:capture(item)
    records=[]
    for item in jsons:
        if item.name.startswith('micro'):
            row=json.loads(item.read_bytes());records.append(dict(name=item.name,micro=row['micro'],kind=row['selection']['kind'],
                excluded=row['duplicate_excluded_from_occurrence_totals'],metrics=row['metrics']))
    summary.append(dict(rank=rank,ready=(path/'ready.json').is_file(),json_files=len(jsons),comparisons=records))
print(json.dumps(dict(captured_epoch=time.time(),job_id=job,journal=journal,commands=commands,
    files=files,rank_summaries=summary,tensor_files_loaded=False,optimizer_writes_by_this_poll=0),sort_keys=True,allow_nan=False))
