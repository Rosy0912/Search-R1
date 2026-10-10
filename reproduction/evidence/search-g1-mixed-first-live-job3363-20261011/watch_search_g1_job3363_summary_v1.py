from pathlib import Path
import json,subprocess,time,re
root=Path.cwd();template=(root/'work/read_search_g1_job3363_summary_template.sh').read_text();previous=None
for i in range(2,10002):
 script=root/f'work/read_search_g1_job3363_summary_v{i}.sh';script.write_text(template.replace('SUMMARY-20261011-v1.json',f'SUMMARY-20261011-v{i}.json'))
 r=subprocess.run(['bash',str(script)],cwd=root,capture_output=True,text=True,timeout=100)
 if r.returncode:
  print(json.dumps({'tag':i,'observation_error':r.stderr[-1000:],'stdout':r.stdout[-1000:]}),flush=True)
 else:
  d=json.loads(r.stdout);job=d['job'];state=re.search(r'JobState=(\S+)',job['stdout']);state=state.group(1) if state else 'UNKNOWN'
  v={'tag':i,'utc':d['utc'],'job_state':state,'pipeline':d.get('pipeline-status.json'),'processes':d.get('process-ledger.json'),'method_created':d['method_created'],'staging_files':d.get('staging_files'),'candidate_pools':d['candidate_pools'],'optimizer_proof_files':len(d['optimizer_proofs']),'checkpoints':d['checkpoint_files'],'logs':d['logs']}
  print(json.dumps(v),flush=True)
  if state in ['FAILED','COMPLETED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:break
 time.sleep(30)
