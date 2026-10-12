"""Observe exact3353 until real fifth pause or terminal; never submit/signal."""
import datetime,json,subprocess,time
from pathlib import Path
base=Path(__file__).resolve().parent
cwd=base.parents[3]
last=None
for index in range(120):
 out=cwd/'work/evidence'/f'node11-formal3353-first5-watch-{index:03d}.actual.json'
 p=subprocess.run(['python3',str(base/'inventory_transport.py'),'--target','h100-11','--python-file',str(base/'requests/read-node11-3353-first5-progress.py'),'--output',str(out),'--timeout','80'],cwd=cwd,capture_output=True,text=True,timeout=110)
 if p.returncode:
  print(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observation_exit':p.returncode,'scope':'OBSERVATION_FAILURE_NOT_JOB_TERMINAL','evidence':str(out),'stderr':p.stderr[-1000:]}),flush=True)
 else:
  x=json.loads(out.read_text());history=x['ledger'];metrics=history['metrics'];completed=max((v.get('metrics',{}).get('fresh200/completed_outers',0)for v in metrics if v.get('accepted_complete_outer')is True),default=0)
  logs=x['logs'];terminal=logs.get('original-exit-code.txt');paused=bool(history['pauses']);train=logs.get('train.log','');phase='MODEL_INITIALIZATION'
  if 'Processed prompts:'in train:phase='GENERATING'
  if history['evaluations']:phase='INITIAL_EVALUATION_COMPLETE'
  if completed:phase='TRAINING'
  if paused:phase='REAL_FIRST5_PAUSED_FOR_EVIDENCE'
  state=(completed,paused,terminal,phase,len(history['evaluations']))
  if state!=last:
   print(json.dumps({'utc':x['utc'],'job':x['job'],'completed_outer':completed,'phase':phase,'last_metrics':metrics[-1]if metrics else None,'last_training_em':history['training_em'][-1]if history['training_em']else None,'last_evaluation':history['evaluations'][-1]if history['evaluations']else None,'terminal_exit':terminal,'evidence':str(out)}),flush=True);last=state
  if paused or terminal is not None:raise SystemExit(0)
 time.sleep(30)
print('OBSERVATION_WINDOW_ENDED_NOT_JOB_TERMINAL',flush=True)
