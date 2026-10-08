"""Additional complete queue precheck; invoke frozen one-shot formal controller."""
import pathlib,subprocess,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');d=root/'scheduler-head'
assert d==d.resolve() and not d.is_symlink()
assert not (d/'formal300-job-id.txt').exists() and not (d/'formal300-submit-intent.txt').exists(),'Existing formal outcome; reconcile, never repeat'
q=subprocess.check_output(['squeue','-h','-w','h100-5','-o','%i|%j'],text=True)
for line in q.splitlines():
 fields=line.split('|');assert len(fields)==2,'Malformed full queue snapshot'
 job,name=(x.strip()for x in fields)
 assert job not in ('3231','3237','3239') and 'bapo'not in name.lower(),'Owned BAPO job active; no duplicate'
print(json.dumps({'status':'ACTUAL_FULL_QUEUE_PRECHECK_PASS','queue':q,'next':'frozen original global-lock formal controller actual hardware gate'}),flush=True)
subprocess.run(['bash',str(d/'submit.sh'),'formal300'],check=True)
