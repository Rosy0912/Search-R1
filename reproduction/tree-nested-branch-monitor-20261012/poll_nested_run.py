"""Small read-only retained18 nested Tree-GRPO run snapshot; no tensors or output downloads."""
import argparse,base64,importlib.util,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,os,pathlib,re,subprocess,time,math
JOB=JOB_VALUE
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
run=root/'runs'/('tree-nested-branch-fresh200-'+JOB)
def query(argv):
 try:
  r=subprocess.run(argv,text=True,capture_output=True,timeout=20)
  return dict(returncode=r.returncode,stdout=r.stdout[-16000:],stderr=r.stderr[-4000:])
 except Exception as e:return dict(error=repr(e))
def small_json(path):
 try:
  if path.stat().st_size>2*1024*1024:return dict(error='JSON_TOO_LARGE_FOR_SMALL_SNAPSHOT',bytes=path.stat().st_size)
  return json.loads(path.read_text())
 except Exception as e:return dict(error=repr(e))
def tail(path,limit=65536):
 try:
  with path.open('rb') as f:
   f.seek(0,2);n=f.tell();f.seek(max(0,n-limit));raw=f.read()
  return raw.decode('utf8','replace').splitlines()
 except FileNotFoundError:return []
def selected(value,keys):return {k:value[k] for k in keys if k in value}
result=dict(observed_epoch=time.time(),job_id=JOB,run=str(run),read_only=True,tensors_loaded=False,large_outputs_downloaded=False)
result['queue']=query(['squeue','--local','-h','-j',JOB,'-o','%i|%j|%T|%M|%R'])
result['sacct']=query(['sacct','--local','-n','-P','-j',JOB,'--format=JobID,State,ExitCode,Elapsed,NodeList'])
result['run_exists']=run.is_dir()
logs={}
for name,path in [('stage',run/'stage.log'),('guard',run/'guard.log'),('retriever',run/'retriever.log'),('ray',run/'ray.log'),('slurm_out',root/'instruct300/node-move-logs'/(JOB+'.out')),('slurm_err',root/'instruct300/node-move-logs'/(JOB+'.err'))]:
 lines=tail(path)
 logs[name]=dict(last_lines=[line[-1200:] for line in lines[-8:]],errors=[line[-1200:] for line in lines if re.search(r'Traceback|Error|Exception|FAILED|out of memory|No space left|STOP',line,re.I)][-8:])
result['logs']=logs
state=run/'checkpoints/training_state'
attempts=sorted((state/'attempt-results').glob('attempt-*.json'))
result['attempts']=dict(record_count=len(attempts),completed_outer_count=0,all_skipped_attempt_count=0,unknown_attempt_count=0,latest=[])
complete_totals={str(r):dict(adam_writes=0,opportunities=0,skips=0) for r in range(8)}
attempt_totals={str(r):dict(adam_writes=0,opportunities=0,skips=0) for r in range(8)}
accounting_errors=[];credited={};complete_counter=0
for path in attempts:
 v=small_json(path);status=v.get('status');ranks=v.get('all_rank_result',{}).get('ranks',[])
 if status=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
  result['attempts']['completed_outer_count']+=1
  outer=v.get('accepted_outer');attempt=v.get('attempt_id');credited[outer]=attempt
  if outer!=complete_counter+1:accounting_errors.append(dict(path=path.name,error='Completed outer sequence gap'))
  complete_counter=outer if type(outer) is int else complete_counter
 elif status=='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT':
  result['attempts']['all_skipped_attempt_count']+=1
  evidence=v.get('partial_evidence',{});partial=pathlib.Path(evidence.get('path','/missing'))
  if partial.resolve().is_relative_to(state.resolve()):
   saved=small_json(partial);workers=saved.get('worker_outcome',{}).get('worker_states',[])
   ranks=[dict(rank=w.get('rank'),optimizer_calls=w.get('committed_optimizer_calls'),opportunities_processed=len(w.get('optimizer_opportunity_ledger',[])),opportunities_skipped=sum(x.get('actual_writes')==0 for x in w.get('optimizer_opportunity_ledger',[]))) for w in workers]
  else:accounting_errors.append(dict(path=path.name,error='All-skip evidence path outside own state'))
 else:
  result['attempts']['unknown_attempt_count']+=1;accounting_errors.append(dict(path=path.name,error=v.get('error',status or 'Missing status')))
 if status in ('COMPLETED_OPPORTUNITIES_OUTER_CREDITED','ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT'):
  if len(ranks)!=8 or any(type(x.get('rank')) is not int for x in ranks) or sorted(x['rank'] for x in ranks)!=list(range(8)):accounting_errors.append(dict(path=path.name,error='Eight rank totals unavailable'))
  else:
   for x in ranks:
    vals=[x.get('optimizer_calls'),x.get('opportunities_processed'),x.get('opportunities_skipped')]
    if any(type(v) is not int or v<0 for v in vals) or vals[0]+vals[2]!=vals[1]:accounting_errors.append(dict(path=path.name,rank=x['rank'],error='Actual rank opportunity/write/skip fields invalid'));continue
    for name,num in zip(('adam_writes','opportunities','skips'),vals):
     attempt_totals[str(x['rank'])][name]+=num
     if status=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':complete_totals[str(x['rank'])][name]+=num
 if path in attempts[-8:]:
  row=dict(path=path.name,**selected(v,['status','attempt_id','accepted_outer','actual_completed_optimizer_calls_per_rank','error']))
  row['ranks']=[selected(x,['rank','optimizer_calls','opportunities_processed','opportunities_skipped','scheduler_before','scheduler_after']) for x in ranks];result['attempts']['latest'].append(row)
result['cumulative_accounting']=dict(complete_outer_only_per_rank=complete_totals,all_finalized_attempts_per_rank=attempt_totals,
 complete_outer_count=result['attempts']['completed_outer_count'],finalized_attempt_count=len(attempts),all_skipped_attempt_count=result['attempts']['all_skipped_attempt_count'],
 accounting_errors=accounting_errors,complete_totals_valid=not accounting_errors,source='Every finalized small attempt JSON and all-skip worker ledger, not last8',
 in_flight_or_unrecorded_partial_adam_writes=None,scope='Current attempt and unfinalized partial writes excluded and unknown; completed credits differ from consumed attempts')
rankroot=run/'output/fresh-state'
rankfiles=sorted(rankroot.glob('outer-*-attempt-*-rank*.json'))
result['rank_records']=dict(count=len(rankfiles),latest=[])
for path in rankfiles[-8:]:
 v=small_json(path);result['rank_records']['latest'].append(dict(path=path.name,**selected(v,['status','rank','observed_outer','attempt_id','actual_optimizer_calls','opportunities_processed','opportunities_skipped','scheduler_step_before','scheduler_step_after','local_before_error','local_scheduler_error'])))
result['fresh_initialization']=selected(small_json(run/'output/fresh-initialization.json'),['status','initialized_global_steps','parent_completed_step','parent_cursor','driver_completed_outer','checkpoint_loaded','code_commit','config_sha256','error'])
result['run_result']=selected(small_json(run/'run-result.json'),['status','error_type','error','training_body_exit_code','actual_completed_outer','wall_seconds','code_commit','error']) if (run/'run-result.json').exists() else None
validation=sorted((run/'output/validation-cost').glob('validation-*.json'),key=lambda p:p.stat().st_mtime)
result['validation']=dict(report_count=len(validation),latest=[])
for path in validation[-3:]:
 v=small_json(path);result['validation']['latest'].append(dict(path=path.name,**selected(v,['status','phase','actual_completed_outer','evaluated_rows','dataset_rows','metrics','error'])))
# Stream complete lines on retained18, extracting numbers before any truncation.
number=r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'
metric_pattern=re.compile(r'((?:tree_variance|tree_sampling|timing_s|training|rollout)/[A-Za-z0-9_./-]+):('+number+r')')
metrics_rows=[];latest_complete_timing=None
try:
 with (run/'stage.log').open(errors='replace') as stream:
  for line in stream:
   if not any(prefix in line for prefix in ('tree_variance/','tree_sampling/','timing_s/','rollout/')):continue
   step_match=re.search(r'\bstep:([0-9]+)',line)
   if not step_match:continue
   numbers={key:float(value) for key,value in metric_pattern.findall(line)}
   numbers={k:v for k,v in numbers.items() if math.isfinite(v)}
   step=int(step_match.group(1));variances={k:v for k,v in numbers.items() if k.startswith(('tree_variance/','tree_sampling/'))}
   timing={k:v for k,v in numbers.items() if k.startswith('timing_s/')}
   row=dict(logger_step=step,completed_outer=numbers.get('training/actual_outer_updates'),attempt_id=numbers.get('training/attempt_id'),metrics=variances)
   row['rollout_sampling_costs']={k:v for k,v in numbers.items() if k.startswith('rollout/')}
   if variances:metrics_rows.append(row)
   if 'timing_s/step' in timing and step in credited:
    latest_complete_timing=dict(completed_outer=step,attempt_id=credited[step],timing_seconds=timing,scope='Original trainer completed step timer including validation/checkpoint when scheduled; joined to credited attempt')
except FileNotFoundError:pass
result['variance_by_logged_iteration']=metrics_rows
result['latest_complete_outer_timing']=latest_complete_timing
if latest_complete_timing:
 cost=small_json(run/'output/training-rollout-cost'/('attempt-%06d.json'%latest_complete_timing['attempt_id']))
 latest_complete_timing['rollout_cost']=selected(cost,['status','attempt_id','candidate_outer','start_epoch','end_epoch','wall_seconds','costs','error'])
result['tree_audit']=dict(rollout_json_count=len(list((run/'output/tree-audit').glob('step_*_rollout_*.json'))))
result['checkpoints']=dict(milestones=[p.name for p in sorted(state.glob('global_step_*')) if p.is_dir()],rolling=[p.name for p in sorted((state/'rolling-completed').glob('global_step_*')) if p.is_dir()],partial_attempts=[p.name for p in sorted(state.glob('partial_attempt_*')) if p.is_dir()][-8:])
lines=tail(run/'resource-samples.jsonl')
try:
 g=json.loads(lines[-1]) if lines else {}
 result['guard']=dict(**selected(g,['epoch','job_id','event']),age_seconds=time.time()-g['epoch'] if 'epoch' in g else None,gpus=[selected(x,['uuid','total_used_memory_bytes','total_gpu_util','foreign_memory_fraction','foreign_sm_upper']) for x in g.get('gpus',[])])
except Exception as e:result['guard']=dict(error=repr(e))
try:
 s=os.statvfs(run if run.exists() else root/'runs');result['storage']=dict(f_bavail_bytes=s.f_bavail*s.f_frsize,f_bfree_bytes=s.f_bfree*s.f_frsize,realpath=str((run if run.exists() else root/'runs').resolve()))
except Exception as e:result['storage']=dict(error=repr(e))
print(json.dumps(result,allow_nan=False))
'''
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True,type=int);p.add_argument('--route-file',type=Path,default=Path('work/tree-speed-repair-20261008/ssh_retained18.py'));p.add_argument('--dryrun',action='store_true');a=p.parse_args()
 if a.job<1:raise ValueError('Positive exact own job required')
 spec=importlib.util.spec_from_file_location('retained18_route',a.route_file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 source=REMOTE.replace('JOB_VALUE',repr(str(a.job)))
 # Base64 protects Python/string quoting; source contains no user shell text.
 payload=base64.b64encode(source.encode()).decode()
 command="python3 -c \"import base64;exec(base64.b64decode('"+payload+"'))\""
 argv=module.route(command,'h100-10')
 if a.dryrun:print(json.dumps(dict(status='POLL_DRYRUN_NOT_EXECUTED',job_id=a.job,remote_source_bytes=len(source),read_only=True)));return
 r=subprocess.run(argv,text=True,capture_output=True,timeout=90)
 if r.returncode:raise RuntimeError(json.dumps(dict(returncode=r.returncode,stderr=r.stderr[-8000:],stdout=r.stdout[-2000:])))
 value=json.loads(r.stdout);print(json.dumps(value,indent=2,allow_nan=False))
if __name__=='__main__':main()
