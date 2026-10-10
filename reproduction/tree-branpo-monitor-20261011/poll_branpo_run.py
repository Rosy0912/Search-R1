"""Small read-only retained18 BranPO run snapshot; no tensors or output downloads."""
import argparse,base64,importlib.util,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,os,pathlib,re,subprocess,time
JOB=JOB_VALUE
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
run=root/'runs'/('tree-branpo-fresh200-'+JOB)
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
result['attempts']=dict(record_count=len(attempts),latest=[])
for path in attempts[-8:]:
 v=small_json(path);row=dict(path=path.name,**selected(v,['status','attempt_id','accepted_outer','actual_completed_optimizer_calls_per_rank','error']))
 ranks=v.get('all_rank_result',{}).get('ranks',[])
 row['ranks']=[selected(x,['rank','optimizer_calls','opportunities_processed','opportunities_skipped','scheduler_before','scheduler_after']) for x in ranks]
 result['attempts']['latest'].append(row)
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
result['variance_lines']=[line[-6000:] for line in tail(run/'stage.log',1024*1024) if 'tree_variance/' in line][-3:]
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
