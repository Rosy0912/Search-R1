"""Read only the submitted retained18 Tree job and its native JSON/log evidence.

No resubmission, cancellation, GPU query, tensor loading, or other-agent contact.
Missing controller handles are unknown unless actual native terminal evidence
exists. Preserved rolling4 is a recovery boundary, never a new completed update.
"""
import json
from pathlib import Path
import shlex
from datetime import datetime,timezone
import ssh_retained18 as ssh

HERE=Path(__file__).resolve().parent

def main():
    spec=json.loads((HERE/'deployment.json').read_text())
    if spec.get('remote_submitted') is not True:raise RuntimeError('No actual job receipt')
    code='''from pathlib import Path
import json,subprocess,time,hashlib
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');job=JOB;run=root/'runs'/('tree-current200-continue-'+str(job))
intent=Path(INTENT);journal=json.loads(intent.read_text())
assert journal['status']=='SUBMITTED' and journal['job_id']==job and journal['manifest_sha256']==MANIFEST_SHA
commands={'squeue':['squeue','--local','-h','-j',str(job),'-o','%i|%j|%T|%R|%N|%b'],
 'scontrol':['scontrol','--local','show','job',str(job)],
 'node_queue':['squeue','--local','-h','-w','h100-10','-o','%i|%j|%T|%R|%N|%b']}
out={'observed_epoch':time.time(),'job_id':job,'run_id':run.name,'run_exists':run.exists(),
 'source_commit':SOURCE,'manifest_sha256':MANIFEST_SHA,'journal_status':journal['status'],
 'preserved_parent_completed_outer':4,'native_parent_loaded':False,'scheduler':{},'native_json':{},
 'no_new_gpu_work_or_submission':True}
for name,command in commands.items():
 r=subprocess.run(command,text=True,capture_output=True,timeout=25)
 out['scheduler'][name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for relative in ['output/current-chain-migrated.json','output/fresh-initialization.json','launch-manifest.json',
 'run-result.json','cost.json','termination-cost.json','checkpoints/training_state/rolling-completed/latest.json']:
 p=run/relative
 if p.is_file():
  raw=p.read_bytes();out['native_json'][relative]={'sha256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
out['native_parent_loaded']='output/current-chain-migrated.json' in out['native_json']
decisions=[]
for p in sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json')):
 raw=p.read_bytes();value=json.loads(raw)
 ranks=value.get('all_rank_result',{}).get('ranks',[])
 decisions.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),
  'status':value.get('status'),'attempt_id':value.get('attempt_id'),
  'accepted_outer':value.get('accepted_outer'),
  'actual_completed_optimizer_calls_per_rank':value.get('actual_completed_optimizer_calls_per_rank'),
  'rank_updates':[{'rank':r.get('rank'),'optimizer_calls':r.get('optimizer_calls'),
   'opportunities_processed':r.get('opportunities_processed'),
   'opportunities_skipped':r.get('opportunities_skipped'),
   'scheduler_before':r.get('scheduler_before'),'scheduler_after':r.get('scheduler_after'),
   'direct_rank_record':r.get('direct_rank_record')} for r in ranks]})
out['attempt_result_summaries']=decisions
latest=out['native_json'].get('checkpoints/training_state/rolling-completed/latest.json',{}).get('value',{})
if latest.get('manifest_path'):
 p=Path(latest['manifest_path'])
 if p.is_file() and p.resolve().is_relative_to((run/'checkpoints/training_state/rolling-completed').resolve()):
  raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
  out['latest_rolling_manifest']={'path':str(p),'sha256':actual,
   'matches_latest_pointer':actual==latest.get('manifest_sha256'),'value':json.loads(raw)}
for relative in ['resource-samples.jsonl','stage.log','guard.log','retriever.log']:
 p=run/relative
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));raw=f.read()
  out[relative+'_tail']=raw.decode('utf-8',errors='replace')
audit=run/'output/tree-rank-audit';records=sorted(audit.glob('step-*-rank-*.json'))
out['direct_rank_record_file_count']=len(records)
out['latest_direct_rank_records']=[]
if records:
 last=max(p.name.split('-rank-')[0] for p in records)
 for p in records:
  if p.name.startswith(last+'-rank-'):out['latest_direct_rank_records'].append(json.loads(p.read_text()))
reports=[]
for p in sorted((run/'output/validation-cost').glob('validation-*.json')):
 v=json.loads(p.read_text())
 if v.get('status')=='VALIDATION_COST_COMPLETE':reports.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'value':v})
out['validation_reports']=reports
print(json.dumps(out,allow_nan=False))
'''
    for name,value in [('JOB',spec['job_id']),('INTENT',spec['once_intent']),('MANIFEST_SHA',spec['manifest_sha']),('SOURCE',spec['source_commit'])]:
        code=code.replace(name,repr(value))
    result=ssh.run('python3 -c '+shlex.quote(code),timeout=95)
    record=dict(observed_at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,connection_failure_not_interpreted_as_normal_or_stopped=True)
    output=HERE.parents[1]/'outputs/tree-fresh200/RETAINED18_CURRENT_LIVE.json'
    output.write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:print(json.dumps({'status':'THIS_OBSERVATION_UNKNOWN','stderr':result.stderr}));raise SystemExit(result.returncode)
    value=json.loads(result.stdout)
    print(json.dumps({'job_id':value['job_id'],'run_exists':value['run_exists'],
        'native_parent_loaded':value['native_parent_loaded'],
        'scheduler':value['scheduler']['scontrol'],'direct_rank_record_files':value['direct_rank_record_file_count'],
        'validation_reports':len(value['validation_reports'])}))

if __name__=='__main__':main()
