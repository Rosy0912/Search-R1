import json,subprocess,time
from pathlib import Path
original=Path('work/read_search_g1_job3361_actual_v2.sh').read_text()
for version in range(3,123):
    path=Path('work')/('read_search_g1_job3361_actual_v'+str(version)+'.sh')
    with path.open('x') as f:f.write(original.replace('ACTUAL-20261011-v2.json','ACTUAL-20261011-v'+str(version)+'.json'))
    r=subprocess.run(['bash',str(path)],capture_output=True,text=True)
    if r.returncode:
        print(json.dumps({'version':version,'observation_error':r.stderr[-1500:],'stdout':r.stdout[-1500:]}),flush=True)
    else:
        try:
            data=json.loads(r.stdout)
            status=data.get('pipeline-status.json',{})
            logs=data.get('logs',{})
            sizes={name:record['bytes'] for name,record in logs.items()}
            errors={name:record['tail'][-3000:] for name,record in logs.items()
                    if any(s in record['tail'] for s in ['Traceback (most recent call last)','CUDA out of memory','AssertionError','RuntimeError:'])}
            summary={'version':version,'utc':data['utc'],'stage':status,'job':data['job']['stdout'].split('JobState=')[-1].split()[0],
                     'log_bytes':sizes,'staging_files':data.get('staging_files'), 'errors':errors}
            print(json.dumps(summary),flush=True)
            if summary['job'] not in ('RUNNING','PENDING','COMPLETING') or status.get('stage') in ('FAILED','OWNED_CLEANUP_FAILED'):
                break
        except Exception as e:print(json.dumps({'version':version,'parse_error':str(e),'stdout':r.stdout[-1200:]}),flush=True)
    time.sleep(30)
