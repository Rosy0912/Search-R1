"""Finite, read-only job checks; mirror only small audit summaries.

Run from the control workspace. Never submit, stop, restart, or alter GPU jobs.
Connection credentials are handled exclusively by hpc3_private_control.py.
"""
import argparse
import json
import pathlib
import subprocess
import sys

REMOTE = r'''
import datetime,json,pathlib,subprocess,time
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
OUT=BASE/'outputs/training-preparation'
JOB=(OUT/'topd-alpha1-job.id').read_text().strip() if (OUT/'topd-alpha1-job.id').exists() else '712204'
assert JOB.isdigit()
ACTOR=BASE/('runtime/logs/topd-hpc3-'+JOB+'/evidence')
FINAL=BASE/'runtime/logs/topd-final51713-recovery-707690'
def save(name,value):
 p=OUT/name;t=p.with_suffix('.json.tmp')
 t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
 print(json.dumps({'kind':'saved','name':name,'data':value}),flush=True)
def snapshot():
 progress=[json.loads((FINAL/('rank%d-progress.json'%r)).read_text()) for r in range(4)]
 actual=json.loads((ACTOR/'actual-outer-updates.json').read_text()) if (ACTOR/'actual-outer-updates.json').exists() else {'actual_completed_outer_steps':[],'records':[]}
 queue=subprocess.check_output(['squeue','-j',JOB+',712266','-h','-o','%i|%T|%M|%R'],text=True).splitlines()
 errors={}
 for p in [ACTOR/'trainer.log',BASE/('runtime/logs/topd-instruct300-'+JOB+'.log'),FINAL/'retriever.log',BASE/'runtime/logs/topd-final51713-712266.log']+list(FINAL.glob('rank*.log')):
  if not p.exists():continue
  with p.open('rb') as h:
   h.seek(max(0,p.stat().st_size-8000));tail=h.read().decode(errors='replace')
  matches=[line[-400:] for line in tail.splitlines() if any(x in line for x in ('Traceback','RuntimeError','CUDA error','OutOfMemoryError','Killed','500 Internal Server Error'))]
  if matches:errors[str(p)]=matches[-5:]
 restored=[json.loads((ACTOR/('resume-loaded-rank%d.json'%r)).read_text()) for r in range(4) if (ACTOR/('resume-loaded-rank%d.json'%r)).exists()]
 driver=json.loads((ACTOR/'resume-loaded-driver.json').read_text()) if (ACTOR/'resume-loaded-driver.json').exists() else None
 d={'epoch':time.time(),'queues':queue,'alpha1_job':int(JOB),'alpha1_full_outers':actual['actual_completed_outer_steps'],
    'alpha1_restore_loaded_ranks':restored,'alpha1_restore_loaded_driver':driver,
    'alpha1_exception_present':(ACTOR/'trainer-first-exception.json').exists(),
    'final_rank_progress':[{k:v for k,v in x.items() if k!='metrics'} for x in progress],
    'final_scored_rows':sum(x['rows'] for x in progress),'errors':errors,
    'final_complete':(FINAL/'complete.json').exists(),'goal_complete':False}
 live_ids={line.split('|')[0] for line in queue}
 if live_ids!={JOB,'712266'}:
  d['accounting']=subprocess.check_output(['sacct','-j',JOB+',712266','-n','-P','--format=JobID,State,ExitCode,Elapsed'],text=True).splitlines()
 return d,actual
def audit_outer(n,actual):
 ranks=[]
 for rank in range(4):
  q=json.loads((ACTOR/('actor-resource-audit/actor-updates-rank%d-outer%d.json'%(rank,n))).read_text())
  assert str(q['job'])==JOB and q['rank']==rank and q['actual_scheduler_last_epoch']==n and len(q['updates'])==n
  for index,u in enumerate(q['updates']):
   assert u['actual_outer_update_ordinal']==index+1 and len(u['optimizer_step_records'])==20 and len(u['micro_forward_records'])==160 and all(u['input_finite'].values())
   assert all(s['finite'] and s['actual_learning_rates'] and all(abs(lr-1e-6*min(index/57,1))<1e-15 for lr in s['actual_learning_rates']) for s in u['optimizer_step_records'])
  assert q['actual_learning_rates_after_outer'] and all(abs(lr-1e-6*min(n/57,1))<1e-15 for lr in q['actual_learning_rates_after_outer'])
  ranks.append({'rank':rank,'full_outers':n,'actual_optimizer_steps':20*n,'scheduler_outer':n,'actual_LR':q['actual_learning_rates_after_outer']})
 t=json.loads((ACTOR/('teacher/round-%04d.json'%n)).read_text())
 assert t['status']=='ACTUAL_TP4_KV_RELEASE_ACK_BEFORE_STUDENT_UPDATE' and t['rows']==2560 and t['no_gold'] and t['full_vocab_normalization']==152064 and t['generated_tokens_used_for_supervision']==0 and len(t['released_workers'])==4
 assert all(w['requires_grad_parameters']==0 and w['cache_engine_absent'] and w['gpu_cache_absent'] for w in t['released_workers'])
 assert actual['records'][-1]['metrics']['rollout/context_mismatches']==0
 return {'epoch':time.time(),'status':'FOUR_RANK_ACTUAL_UPDATES_LR_FROZEN_TEACHER_AND_CONTEXT_AUDIT_PASS','job':int(JOB),'outer':n,'proofs':ranks,'goal_complete':False}
previous=json.loads((OUT/'alpha1-and-final-current.json').read_text())
previous_turn=json.loads((OUT/'latest-goal-continuation.json').read_text())
samples=[]
for i in range(COUNT):
 d,actual=snapshot();samples.append(d)
 print(json.dumps({'kind':'sample','data':d}),flush=True)
 if i<COUNT-1:time.sleep(INTERVAL)
n=len(d['alpha1_full_outers']);proof_path=OUT/'alpha1-latest-outer-update-audit.json'
old_proof=json.loads(proof_path.read_text()) if proof_path.exists() else {'outer':0}
proof=None
if n>old_proof['outer'] or (n>5 and old_proof.get('job')!=int(JOB)):
 proof=audit_outer(n,actual);save(proof_path.name,proof)
v=json.loads((OUT/'topd-alpha1-and-stage512-status.json').read_text())
v.update(observed_at=datetime.datetime.fromtimestamp(d['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),alpha1_verified_full_outers=d['alpha1_full_outers'],final51713_scored_rows=d['final_scored_rows'],latest_observation_errors=d['errors'])
states={line.split('|')[0]:line.split('|')[1] for line in d.get('accounting',[]) if len(line.split('|'))>=2}
states.update({line.split('|')[0]:line.split('|')[1] for line in d['queues']})
v['alpha1_job']=int(JOB)
v['alpha1_state']=states.get(JOB,'UNKNOWN_NOT_IN_QUEUE')
if JOB!='712204':
 v['alpha1_phase']='authorized_step5_resume; awaiting_verified_new_outer6' if n<=5 else 'resumed_training_after_outer%d'%n
 v['alpha1_restore_choice_pending']=False
if v['alpha1_state']=='FAILED':
 v['alpha1_phase']='failed_during_stage50_validation; restore_decision_required' if JOB=='712204' else 'failed; inspect current job first exception and startup log'
 v['alpha1_stage50_checkpoint_present']=(ACTOR/'checkpoints/actor/global_step_50/resume-complete.json').is_file()
if proof:v.update(alpha1_rank_optimizer_steps=[20*n]*4,alpha1_phase='training_continues_after_outer%d'%n)
metrics=actual['records'][-1]['metrics'] if actual['records'] else {}
keys=['response_length/mean','response_length/max','env/finish_ratio','env/number_of_actions/mean','critic/score/mean','timing_s/gen','timing_s/adv','timing_s/update_actor','timing_s/step']
v.update(alpha1_latest_training_diagnostics={k:metrics[k] for k in keys if k in metrics},alpha1_latest_training_diagnostics_outer=n,training_trajectory_score_is_not_fixed512_or_final_EM=True)
live={line.split('|')[0] for line in d['queues']}=={JOB,'712266'}
classification='progress_new_full_outer_audit' if proof else ('verified_wait' if live and not d['errors'] and not d['alpha1_exception_present'] else 'requires_failure_or_accounting_review')
report={'previous_goal_turn_classification':previous_turn.get('current_goal_turn_classification','unknown'),
        'current_goal_turn_classification':classification,
        'samples':samples,'final_rows_since_previous_turn':d['final_scored_rows']-previous['final_scored_rows'],
        'final_rows_between_samples':d['final_scored_rows']-samples[0]['final_scored_rows'],'goal_complete':False}
save('alpha1-and-final-current.json',d)
save('topd-alpha1-and-stage512-status.json',v)
save('latest-goal-continuation.json',report)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2, choices=range(2, 7))
    parser.add_argument('--interval', type=int, default=55, choices=range(1, 56))
    args = parser.parse_args()
    workspace = pathlib.Path.cwd()
    helper = workspace / 'work/hpc3_private_control.py'
    code = 'COUNT=%d\nINTERVAL=%d\n' % (args.samples, args.interval) + REMOTE
    command = '/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python - <<\'REMOTE\'\n' + code + '\nREMOTE'
    process = subprocess.Popen([sys.executable, str(helper)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=None, text=True)
    process.stdin.write(command)
    process.stdin.close()
    output = workspace / 'outputs/training-preparation'
    for line in process.stdout:
        if not line.startswith('{'):
            print(line.rstrip(), flush=True)
            continue
        item = json.loads(line)
        if item['kind'] == 'saved':
            path = output / item['name']
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(item['data'], indent=2) + '\n')
            temporary.replace(path)
            print(json.dumps({'saved': item['name']}), flush=True)
        else:
            print(line.rstrip(), flush=True)
    if process.wait():
        raise SystemExit('Observation failed; never infer a job failure or restart from this alone.')


if __name__ == '__main__':
    main()
