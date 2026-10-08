"""Read-only local observer for the existing teacher85 student job."""
import hashlib
import json
import pathlib
import subprocess
import time

WORK = pathlib.Path(__file__).resolve().parent
OUT = WORK.parent / 'outputs/remote-evidence/topd3224-teacher85-readonly.jsonl'
OUT.parent.mkdir(exist_ok=True)
COMMAND = r'''python3 - <<'READONLY'
import pathlib,json,time,subprocess,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/topd-instruct300-teacher85-approved-v1')
out={'epoch':time.time(),'scope':'READ_ONLY_ACTUAL_STUDENT3224_NOT_COMPLETION','files':[]}
q=subprocess.run(['scontrol','show','job','3224','-o'],capture_output=True,text=True,timeout=15)
out['scontrol']={'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
for rel in ['resource-wait.json','teacher85-user-approval.json','evidence/actual-outer-updates.json','evidence/train-result.json','evidence/trainer-first-exception.json','evidence/actual-teacher-ready.json']:
 p=r/rel
 if p.is_file():
  b=p.read_bytes();d=json.loads(b)
  if rel.endswith('actual-outer-updates.json'):
   d={'actual_completed_outer_steps':d.get('actual_completed_outer_steps'),'export_step':d.get('export_step'),'records_last':d.get('records',[])[-1:]}
  out['files'].append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':d})
log=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/topd-instruct300-3224.log')
if log.is_file():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
READONLY
'''

if __name__ == '__main__':
    while True:
        try:
            r = subprocess.run(['python3', str(WORK / 'remote_topd_retained18.py')],
                               input=COMMAND, text=True, capture_output=True, timeout=90)
            if r.returncode:
                print(json.dumps({'status': 'UNKNOWN_TRANSPORT', 'epoch': time.time(),
                                  'returncode': r.returncode, 'stderr': r.stderr[-1000:]}), flush=True)
            else:
                data = json.loads(r.stdout)
                with OUT.open('a') as f:
                    f.write(r.stdout.strip() + '\n')
                    f.flush()
                print(json.dumps({'epoch': data['epoch'], 'files': [p['path'] for p in data['files']],
                                  'scontrol': data['scontrol']}), flush=True)
                if any('JobState=' + state in data['scontrol']['stdout']
                       for state in ['FAILED','COMPLETED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY']):
                    break
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as error:
            print(json.dumps({'status': 'UNKNOWN_TRANSPORT', 'epoch': time.time(), 'error': str(error)}), flush=True)
        time.sleep(60)
