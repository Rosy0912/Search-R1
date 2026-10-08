import subprocess
remote = r'''python3 - <<'INNER'
import hashlib,json,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
t=r/'tools/qwen25-32b-terminal-fix-refreeze'
assert subprocess.check_output(['git','-C',str(r/'repo-terminal-fixed413'),'rev-parse','HEAD'],text=True).strip()=='413a10c50f32febaafc83db8170e2b005e0e979f'
jobs=[('native-cpu.sbatch','2e0627622655e4085694acab27c1f7dd2d5be5cf1634bbf94a38ac3d6d6f0b14','qwen25-32b-terminal-fixed413-eval64-contracts'),('native-builder-cpu.sbatch','be981991ffd98f8048d5f71060261eec8fbafd74fff600c8b2f3f6203dfae901','qwen25-32b-binding-contracts')]
results=[]
for entry,sha,name in jobs:
 p=t/entry
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 record=r/'logs'/('terminal-fixed413-'+entry+'.submission.json')
 if record.exists():
  results.append(json.loads(record.read_text()));continue
 active=subprocess.check_output(['squeue','-h','-u',subprocess.check_output(['id','-un'],text=True).strip(),'-o','%i|%j'],text=True)
 assert not any(x.split('|',1)[-1]==name for x in active.splitlines()),active
 state=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(p),sha256=sha,name=name,epoch=time.time(),scope='CPU_ONLY_NO_GPU')
 record.write_text(json.dumps(state,indent=2)+'\n')
 output=subprocess.check_output(['sbatch','--parsable',str(p)],text=True).strip()
 job=output.split(';')[0];assert job.isdigit(),output
 state.update(status='SUBMITTED',job_id=int(job));record.write_text(json.dumps(state,indent=2)+'\n');results.append(state)
print(json.dumps(results))
INNER'''
result=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(result.stdout,end='');print(result.stderr,end='')
if result.returncode==0:
 import pathlib
 pathlib.Path('outputs/topd-terminal-fixed413-native-cpu-submissions.json').write_text(result.stdout)
raise SystemExit(result.returncode)
