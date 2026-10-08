import hashlib, importlib.util, json, os, tempfile, types
from pathlib import Path
from unittest.mock import patch
source=Path(__file__).resolve().parents[1]/'verify_instruct3154_terminal.py'
spec=importlib.util.spec_from_file_location('terminal_v2',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=['positive','gpu_slash','gpu_colon','gpu_typed','reserved_env','allocation_failed','wrong_cpu_job','wrong_cpu_node','cpu4','mem16G','wrong_parent_job','wrong_parent_node','body_failed','entry_false','wrong_source','wrong_run_job','missing_step','duplicate_step','ledger_missing','ledger_hash','observer_wrong','step300','wrong_run','sequence','pid_event','failed_event','coverage_short','summary_short','foreign_consistent_pid','guard_string','scheduler_exception']
results=[]
for case in cases:
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);run=root/'run';run.mkdir();(run/'output/validation-cost').mkdir(parents=True)
  checker=root/'checker.py';checker.write_text('fixture only')
  m.ROOT=root;m.RUN=run;m.BASE=root/'base';m.CHECKER=checker;m.CHECKER_SHA=hashlib.sha256(checker.read_bytes()).hexdigest()
  result=dict(status='ACTUAL_STAGE_COMPLETED_UNREVIEWED',final_guard_healthy=True,cleanup_residual={},job_id='3154',code_commit=m.COMMIT,training_body_exit_code=0,entry_config_verified=True)
  for name,key,value in [('body_failed','training_body_exit_code',1),('entry_false','entry_config_verified',False),('wrong_source','code_commit','other'),('wrong_run_job','job_id','9999'),('guard_string','final_guard_healthy','false')]:
   if case==name:result[key]=value
  (run/'run-result.json').write_text(json.dumps(result))
  steps=list(range(1,301))
  if case=='missing_step':steps.remove(64)
  if case=='duplicate_step':steps.append(300)
  (run/'stage.log').write_text('\n'.join(f'step:{i} - training/actual_outer_updates:{i} - actor/optimizer_steps:24' for i in steps))
  shared=dict(phase='final',global_step=301,pid=4048887,validation_sequence=8)
  events=[dict(shared,event=1,kind='validation_start',dataset_rows=51713,observer_source_sha256=m.OBSERVER_SHA,job_id='3154',run=str(run/'output')),dict(shared,event=2,kind='real_sample_batch',rows=51713),dict(shared,event=3,kind='validation_completed',evaluated_rows=51713)]
  summary=dict(shared,status='VALIDATION_COST_COMPLETE',dataset_rows=51713,evaluated_rows=51713,job_id='3154',observer_source_sha256=m.OBSERVER_SHA,run=str(run/'output'),counters=dict(real_sample_rows=51713))
  if case=='observer_wrong':summary['observer_source_sha256']='bad'
  if case=='step300':summary['global_step']=300
  if case=='wrong_run':summary['run']='elsewhere'
  if case=='sequence':events[1]['event']=4
  if case=='pid_event':events[1]['pid']=99
  if case=='failed_event':events.insert(2,dict(shared,event=3,kind='http_failed'));events[-1]['event']=4
  if case=='coverage_short':events[1]['rows']=51712
  if case=='summary_short':summary['evaluated_rows']=51712
  if case=='foreign_consistent_pid':
   summary['pid']=123
   for e in events:e['pid']=123
  ledger=run/'output/validation-cost/validation-4048887-8.jsonl';ledger.write_text('\n'.join(json.dumps(e) for e in events)+'\n');summary['ledger_sha256']=hashlib.sha256(ledger.read_bytes()).hexdigest()
  if case=='ledger_hash':summary['ledger_sha256']='wrong'
  if case=='ledger_missing':ledger.unlink()
  ledger.with_suffix('.json').write_text(json.dumps(summary))
  cpu='JobId=3999 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 NumCPUs=8 CPUs/Task=8 MinMemoryNode=32G TRES=cpu=8,mem=32G TresPerNode=gres:gpu:0'
  parent='JobId=3154 JobState=COMPLETED ExitCode=0:0 Restarts=0 NodeList=h100-9 BatchHost=h100-9'
  for name,old,new in [('gpu_slash','TRES=cpu=8,mem=32G','TRES=cpu=8,mem=32G,gres/gpu=8'),('gpu_colon','TresPerNode=gres:gpu:0','TresPerNode=gres:gpu:8'),('gpu_typed','TresPerNode=gres:gpu:0','TresPerNode=gres:gpu:h100:8'),('wrong_cpu_job','JobId=3999','JobId=3998'),('wrong_cpu_node','NodeList=h100-9','NodeList=h100-11'),('cpu4','NumCPUs=8','NumCPUs=4'),('mem16G','MinMemoryNode=32G','MinMemoryNode=16G')]:
   if case==name:cpu=cpu.replace(old,new)
  if case=='wrong_parent_job':parent=parent.replace('JobId=3154','JobId=3153')
  if case=='wrong_parent_node':parent=parent.replace('NodeList=h100-9','NodeList=h100-11')
  calls=[]
  def fake(command,**kw):
   calls.append(command)
   if command[0]=='scontrol':
    if case=='scheduler_exception':raise OSError('fixture scheduler executable failure')
    return types.SimpleNamespace(returncode=1 if command[-1]=='3999' and case=='allocation_failed' else 0,stdout=parent if command[-1]=='3154' else cpu,stderr='')
   Path(command[command.index('--output')+1]).write_text(json.dumps(dict(status='HF_RELOAD_PARAMETER_DELTA_ACCEPTED')))
   return types.SimpleNamespace(returncode=0)
  env={'SLURMD_NODENAME':'h100-9','SLURM_JOB_ID':'3999','CUDA_VISIBLE_DEVICES':''}
  if case=='reserved_env':env['SLURM_JOB_GPUS']='0'
  raised=None
  with patch.dict(os.environ,env,clear=True),patch.object(m.socket,'gethostname',lambda:'h100-9'),patch.object(m.subprocess,'run',fake):
   try:code=m.main()
   except Exception as e:raised=type(e).__name__;code=None
  reportpath=run/'terminal-cpu-verification/result.json'
  report=json.loads(reportpath.read_text()) if reportpath.exists() else None
  expected=0 if case in ('positive','foreign_consistent_pid','guard_string') else (None if case=='scheduler_exception' else 1)
  assert code==expected,(case,code,expected)
  results.append(dict(case=case,exit_code=code,exception=raised,result_recorded=report is not None,status=None if report is None else report['status'],hf_subprocess_called=any(c[0]!='scontrol' for c in calls)))
out=Path(__file__).parent/'V2_CPU_BOUNDARY_PROBES.json';out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Actual main with temporary root and mocked scheduler/HF, no real job/model',cases=results),indent=2)+'\n')
