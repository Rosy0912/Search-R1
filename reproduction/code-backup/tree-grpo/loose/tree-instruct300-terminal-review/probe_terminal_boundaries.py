"""Execute actual collector with temporary files and mocked scheduler/HF subprocess only."""
import importlib.util, hashlib, json, os, tempfile, types
from pathlib import Path
from unittest.mock import patch
source=Path(__file__).resolve().parents[1]/'verify_instruct3154_terminal.py'
spec=importlib.util.spec_from_file_location('terminal_candidate',source); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results=[]
for case in ('valid_summary_only_no_ledger','wrong_body_source_job','missing_cpu_allocation'):
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);run=root/'run';run.mkdir();(run/'output/validation-cost').mkdir(parents=True)
  checker=root/'checker.py';checker.write_text('fixture only')
  m.ROOT=root;m.RUN=run;m.BASE=root/'base';m.CHECKER=checker;m.CHECKER_SHA=hashlib.sha256(checker.read_bytes()).hexdigest()
  result=dict(status='ACTUAL_STAGE_COMPLETED_UNREVIEWED',final_guard_healthy=True,cleanup_residual={},job_id='3154',code_commit='f994af33c6c93b6b9844863be912cab2a4020dc1',training_body_exit_code=0,entry_config_verified=True)
  if case=='wrong_body_source_job':result.update(job_id='9999',code_commit='other',training_body_exit_code=1,entry_config_verified=False)
  (run/'run-result.json').write_text(json.dumps(result))
  (run/'stage.log').write_text('\n'.join(f'step:{i} - training/actual_outer_updates:{i} - actor/optimizer_steps:24' for i in range(1,301)))
  (run/'output/validation-cost/final.json').write_text(json.dumps(dict(phase='final',status='VALIDATION_COST_COMPLETE',dataset_rows=51713,evaluated_rows=51713,job_id='3154',global_step=0,observer_source_sha256='wrong',ledger_sha256='no ledger',run='elsewhere')))
  calls=[]
  def fake(command,**kw):
   calls.append(command)
   if command[0]=='scontrol':return types.SimpleNamespace(returncode=0,stdout='JobId=3154 JobState=COMPLETED ExitCode=0:0 Restarts=0 NodeList=other',stderr='')
   Path(command[command.index('--output')+1]).write_text(json.dumps(dict(status='HF_RELOAD_PARAMETER_DELTA_ACCEPTED')))
   return types.SimpleNamespace(returncode=0)
  with patch.dict(os.environ,{'SLURMD_NODENAME':'h100-9','CUDA_VISIBLE_DEVICES':''},clear=True),patch.object(m.subprocess,'run',fake):code=m.main()
  report=json.loads((run/'terminal-cpu-verification/result.json').read_text())
  assert code==0,'Probe expects current candidate vulnerability; update fixture after fix'
  results.append(dict(case=case,exit_code=code,status=report['status'],scheduler_queries=[c for c in calls if c[0]=='scontrol'],accepted_invalid_evidence=True))
# Genuine no-overwrite helper test.
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp)/'record.json';m.save(p,{'first':1})
 try:m.save(p,{'second':2})
 except FileExistsError:pass
 else:raise AssertionError('Existing record was overwritten')
 assert json.loads(p.read_text())=={'first':1}
results.append(dict(case='no_overwrite',passed=True))
output=Path(__file__).parent/'CPU_BOUNDARY_PROBES.json'
output.write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Actual source main with mocked scheduler and HF subprocess; no real job or model',cases=results),indent=2)+'\n')
