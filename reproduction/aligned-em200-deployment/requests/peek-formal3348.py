from pathlib import Path
import json,os,subprocess,datetime
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');run=r/'runs/formal-aligned-em200-66cb18840a404586ab3333f8c393b8b7-3348'
v=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,job='3348',run=str(run),run_exists=run.is_dir(),scope='READONLY_SAME_FORMAL_JOB_NO_MUTATION')
for key,args in [('slurm',['scontrol','show','job','3348','-o']),('queue',['squeue','-h','-j','3348','-o','%i|%j|%T|%R|%k']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[key]=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
v['files']={}
for name in ['source-commit.txt','formal-supervisor.log','train.log','driver-time.log','retriever.log','ray-head.log','own-retriever-birth.json','own-process-cleanup.json','original-exit-code.txt','cleanup-exit-code.txt','resource-watch.jsonl']:
 p=run/name
 if p.is_file():v['files'][name]=dict(bytes=p.stat().st_size,tail=p.read_text(errors='replace')[-3000:])
print(json.dumps(v))
