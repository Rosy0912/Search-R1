import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);step=int(sys.argv[2]);d=json.loads(p.read_text())
while 'stdout' in d:d=json.loads(d['stdout'])
a=d['actual-attempts.jsonl']['last'];u=d['actual-updates.jsonl']['last'];r=d['ranks']
assert a['complete_outer'] and a['all_ranks_agreed'] and a['completed_outer']==u['step']==step
assert len(r)==8 and all(x['actual_last_write_count']==step*20 and x['scheduler_last']['scheduler_last_epoch_after']==step and x['scheduler_last']['actual_optimizer_writes']==20 and x['scheduler_last']['scheduler_advances']==1 for x in r)
f=dict(utc=d['utc'],job_id='3368',node='h100-8',run_id=a['run_id'],source_commit=r[0]['scheduler_last']['source_commit'],completed_outer=step,total_target=200,actual_adam_per_rank=[x['actual_last_write_count'] for x in r],complete_attempt=a,complete_update=u,scheduler_receipts=[x['scheduler_last'] for x in r],raw_observer_path=str(p.resolve()),raw_observer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),latest_complete_checkpoint_step=5,personal_followthrough_pending='6-10 complete and11 actually started')
out=Path('outputs/gigpo-e1-instruct200')/('SEARCH-G1-JOB3368-COMPLETED'+str(step)+'-ACTUAL-FACTS-20261011-v1.json')
with out.open('x') as fp:json.dump(f,fp,indent=2);fp.write('\n')
h=Path('work/SEARCH-G1-CURRENT-HANDOFF-20261011.md')
h.write_text('# Latest actual3368 full outer'+str(step)+'\n\nUTC'+d['utc']+' full'+str(step)+'/200, all8rank actual'+str(step*20)+'Adam/rank, each20writes/20opportunities/0skip, scheduler'+str(step)+' once. Same3368 h100-8 runtime475fbb6 healthy. Latest fullcheckpoint parent5; thisstep not saved fullHF/Adam. Personal followthrough6-10 complete +11 started pending, watcher92989. Facts '+str(out)+'; original readback retained.\n\nEarlier handoffs:\n\n'+h.read_text())
print(json.dumps(dict(facts=str(out),utc=d['utc'],complete=step)))
