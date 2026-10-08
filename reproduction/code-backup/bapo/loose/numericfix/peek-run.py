import json,sys,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric')
phase,job=sys.argv[1:]
assert phase in ['hardware','smoke2','accept5','formal500'] and job.isascii() and job.isdigit()
run=root/'runs'/('bapo-'+phase+'-'+job)
rec={'source_commit':subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),'run':str(run),'job':job,'phase':phase,'exists':run.exists()}
for name in ['hardware.log','phase.log','retriever.log','train.log','parity-gpu.log','parity-cpu.log']:
 p=run/name
 if p.exists():
  rec[name]={'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-1700:]}
p=run/'resource-watch.jsonl'
if p.exists():
 lines=p.read_text().splitlines();last=json.loads(lines[-1]);rec['resource_watch']={k:last.get(k) for k in ['time','pause','reason','slurm_job_id']};rec['resource_watch']['events']=len(lines)
rec['markers']={n:(run/n).exists() for n in ['HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY','TRAIN_RETURNED_SUCCESSFULLY','own-process-cleanup.json']}
p=run/'retrieval-comparison.json'
if p.exists():rec['comparison']=json.loads(p.read_text())
print(json.dumps(rec))
