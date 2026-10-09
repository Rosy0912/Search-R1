from pathlib import Path
import json,os,subprocess,datetime
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');run=r/'runs/hardware-aligned-em200-13e07285c97044fd97974d94f53fea1b-3345'
v=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,job='3345',run=str(run),run_exists=run.is_dir(),scope='READONLY_HARDWARE_JOB_NOT_FORMAL_OUTER')
for key,args in [('slurm',['scontrol','show','job','3345','-o']),('queue',['squeue','-h','-j','3345','-o','%i|%j|%T|%R|%k']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[key]=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
v['files']={}
for name in ['source-commit.txt','hardware.log','numeric-fixture.log','retrieval-golden.log','retriever.log','capture-completed-before-live.json','own-retriever-birth.json','live-retrieval-acceptance.json','own-process-cleanup.json','original-exit-code.txt','cleanup-exit-code.txt','resource-watch.jsonl']:
 p=run/name
 if p.is_file():v['files'][name]=dict(bytes=p.stat().st_size,tail=p.read_text(errors='replace')[-3000:])
v['rank_records']=[dict(file=p.name,source=json.loads(p.read_text())['source_commit'],status=json.loads(p.read_text())['status'],records=len(json.loads(p.read_text())['records'])) for p in sorted((run/'numeric-fixture').glob('rank-*.json'))]
print(json.dumps(v))
