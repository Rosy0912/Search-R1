import subprocess,pathlib
remote=r'''python3 - <<'INNER'
import hashlib,json,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');t=r/'tools/qwen25-32b-terminal-fix-refreeze'
assert subprocess.check_output(['git','-C',str(r/'repo-terminal-fixed413'),'rev-parse','HEAD'],text=True).strip()=='413a10c50f32febaafc83db8170e2b005e0e979f'
for name,digest in [('qwen25-32b-terminal-fixed413-eval64-native-3019.log','73d078e0b968833b49d4e30d8a210ea94fdcdbe3be10ac9c07263deeaf6ee31a'),('qwen25-32b-binding-native-3020.log','5fe2ffb72990eba6bc256d72e01d8a0483be3d32b309dcdf4564a8aab20badeb')]:
 assert hashlib.sha256((r/'logs'/name).read_bytes()).hexdigest()==digest
# Completed CPU jobs aged out of Slurm; independent scontrol exit0 snapshots are retained.
manifest=json.loads((t/'files.json').read_text())
for name,digest in manifest['source_files'].items():
 assert hashlib.sha256((r/'repo-terminal-fixed413'/name).read_bytes()).hexdigest()==digest
download=json.loads((r/'logs/qwen25-32b-download-h16proxy-state.json').read_text())
assert download['script_sha256']=='cedc4a5e7dd5f920646a5fde63960c3b49793e18c8fef6af6fb3e06509e7df9a'
assert not download['errors']
results=[]
def submit(entry,sha,name,dependency,args=()):
 p=t/entry;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 record=r/'logs'/('terminal-fixed413-'+entry+'.submission.json')
 if record.exists():
  state=json.loads(record.read_text());assert state['status']=='SUBMITTED',state;results.append(state);return state['job_id']
 active=subprocess.check_output(['squeue','-h','-u',subprocess.check_output(['id','-un'],text=True).strip(),'-o','%i|%j'],text=True)
 assert not any(x.split('|',1)[-1]==name for x in active.splitlines()),active
 state=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(p),sha256=sha,name=name,dependency=dependency,epoch=time.time(),args=list(args))
 record.write_text(json.dumps(state,indent=2)+'\n')
 output=subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+str(dependency),str(p),*args],text=True).strip()
 job=output.split(';')[0];assert job.isdigit(),output
 state.update(status='SUBMITTED',job_id=int(job));record.write_text(json.dumps(state,indent=2)+'\n');results.append(state);return int(job)
i=submit('inventory.sbatch','204da445541a0b86b0ae4cf13152b8f951053b56c0c6f57d4361f8fd7203d698','topd32b-terminal-fixed413-inventory',3016)
submit('eval64.sbatch','21aa558e8f6ce0ad3e719b56c91e8c32644add9dc28326400e1d3655e00bf0ba','qwen25-32b-eval64',i,(str(r/'logs/qwen25-32b-terminal-fixed413-binding.json'),))
print(json.dumps(results))
INNER'''
result=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(result.stdout,end='');print(result.stderr,end='')
if result.returncode==0:pathlib.Path('outputs/topd-terminal-fixed413-inventory-eval-submissions.json').write_text(result.stdout)
raise SystemExit(result.returncode)
