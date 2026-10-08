"""One native read-only job observation, persisted in own logs; no control APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')


def observe(job):
    raw=subprocess.run(['scontrol','show','job','-dd',str(job)],capture_output=True,text=True,timeout=20)
    row=dict(job_id=job,epoch=time.time(),scontrol_stdout=raw.stdout,scontrol_stderr=raw.stderr,scontrol_returncode=raw.returncode)
    out=ROOT/'logs'/('qwen25-32b-eval64-'+str(job)); evaluation=out/'evaluation'
    files=[ROOT/'logs'/('qwen25-32b-startup-guard-'+str(job)+'.jsonl'),out/'supervisor-result.json',
           evaluation/'readiness.json',evaluation/'result.json',evaluation/'qa.jsonl',out/'evaluation.log']
    files+=sorted((ROOT/'logs').glob('*'+str(job)+'.log'))
    row['files']={}
    for path in dict.fromkeys(files):
        if path.exists():
            content=path.read_bytes(); text=content.decode('utf-8',errors='replace')
            row['files'][str(path)]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
                lines=len(text.splitlines()),tail=text.splitlines()[-12:])
    row['generation_files']=[p.name for p in sorted(evaluation.glob('generation-*.json'))]
    row['qa_rows']=len((evaluation/'qa.jsonl').read_text().splitlines()) if (evaluation/'qa.jsonl').exists() else 0
    row['readiness_exists']=(evaluation/'readiness.json').exists()
    row['final_state_observed']=any('JobState='+state in raw.stdout for state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'))
    target=ROOT/'logs'/('qwen25-32b-readonly-watch-'+str(job)+'.jsonl')
    with target.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
    print(json.dumps(row,sort_keys=True))
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=int,required=True);args=p.parse_args()
    if args.job<=0:raise ValueError('Positive actual job ID required')
    observe(args.job)
