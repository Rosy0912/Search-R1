import json,subprocess,time,hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004')
run=r/'runs/bapo-instruct300-formal300-3177'
slurm=subprocess.run(['scontrol','show','job','-dd','3177'],capture_output=True,text=True,check=True).stdout
assert 'JobId=3177 ' in slurm and 'BatchHost=h100-5' in slurm
assert 'Command='+str(r/'scheduler-head/formal300.sbatch') in slurm
assert (run/'source-commit.txt').read_text().strip()=='733097feeae371b4efd8fd48940c46ff36da839b'
records=[json.loads(l.split('BAPO_TRAIN_METRICS_JSON=',1)[1]) for l in (run/'train.log').read_text().splitlines() if 'BAPO_TRAIN_METRICS_JSON=' in l]
proof={'authorized_action':'User requested stop current training after ratio/metrics/warmup114 code repair; no replacement launch','job':3177,'node':'h100-5','run':str(run),'source':(run/'source-commit.txt').read_text().strip(),'before_slurm':slurm,'actual_completed_outer_before_stop':records[-1]['step'],'last_outer':records[-1],'time':time.time(),'train_log_sha256_before':hashlib.sha256((run/'train.log').read_bytes()).hexdigest()}
p=run/'logs/user-authorized-stop-20261005.json'; p.write_text(json.dumps(proof,indent=2))
cancel=subprocess.run(['scancel','3177'],capture_output=True,text=True)
proof['scancel_returncode']=cancel.returncode;proof['scancel_stdout']=cancel.stdout;proof['scancel_stderr']=cancel.stderr
p.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2));assert cancel.returncode==0
