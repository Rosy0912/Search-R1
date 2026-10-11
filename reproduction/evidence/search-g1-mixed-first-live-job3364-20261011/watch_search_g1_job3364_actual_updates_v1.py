from pathlib import Path
import subprocess,json,time,re
root=Path.cwd()
template=(root/'work/read_search_g1_job3364_actual_updates_template.sh').read_text()
for tag in range(3,10000):
 script=root/f'work/read_search_g1_job3364_actual_updates_v{tag}.sh'
 script.write_text(template.replace('ACTUAL-UPDATES-20261011-v1.json',f'ACTUAL-UPDATES-20261011-v{tag}.json'))
 try:
  r=subprocess.run(['bash',str(script)],capture_output=True,text=True,timeout=100)
  if r.returncode: print(json.dumps({'tag':tag,'observation_error':r.stderr[-300:],'stdout':r.stdout[-300:]}),flush=True)
  else:
   d=json.loads(r.stdout)
   match=re.search(r'JobState=(\S+)',d['job']['stdout'])
   state=match.group(1) if match else 'UNKNOWN'
   print(json.dumps({'tag':tag,'utc':d['utc'],'job_state':state,'complete':d['actual_updates'],'attempts':d['actual_attempts'],'rank_counts':{k:v['kind_counts'] for k,v in d['rank_write_counts'].items()},'pools':d['candidate_pools'],'checkpoint_files':d['checkpoint_files'],'errors':{k:v['tail'][-700:] for k,v in d['logs'].items() if any(e in v['tail'] for e in ['Traceback','RuntimeError:','AssertionError:','CUDA out of memory'])}}),flush=True)
   if state in ['FAILED','COMPLETED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']: break
 except Exception as e:print(json.dumps({'tag':tag,'observation_error':str(e)}),flush=True)
 time.sleep(55)
