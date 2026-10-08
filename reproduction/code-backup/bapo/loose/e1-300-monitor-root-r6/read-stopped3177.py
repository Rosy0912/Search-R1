import json,subprocess,time
from pathlib import Path
run=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177')
def command(args):
 r=subprocess.run(args,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
lines=(run/'train.log').read_text(errors='replace').splitlines()
outer=[json.loads(l.split('BAPO_TRAIN_METRICS_JSON=',1)[1]) for l in lines if 'BAPO_TRAIN_METRICS_JSON=' in l]
watch=[json.loads(l) for l in (run/'resource-watch.jsonl').read_text().splitlines() if l]
print(json.dumps(dict(time=time.time(),job=3177,run=str(run),actual_outer=outer[-1]['step'],last_outer=outer[-1],sacct=command(['sacct','-j','3177','--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList','-P']),scontrol=command(['scontrol','show','job','-dd','3177']),squeue=command(['squeue','-j','3177','-h','-o','%i %T %N']),gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),processes=command(['ps','-eo','pid,ppid,args']),watch_last=watch[-1],resource_samples=len(watch),resource_pauses=sum(bool(x.get('pause')) for x in watch),run_logs=sorted(p.name for p in (run/'logs').iterdir())),indent=2))
