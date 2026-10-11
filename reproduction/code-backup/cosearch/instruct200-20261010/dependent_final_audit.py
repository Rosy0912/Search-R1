"""CPU-only dependency stage. Never train, load models, submit, or allocate GPUs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def archived_equal(saved,archived):
    if saved.get('step')!=200 or archived.get('step')!=200 or archived.get('status')!='HKUST3_ALL_CHECKPOINT_SHA_PASS':raise RuntimeError('Missing verified final archive')
    def files(d):return {x['path'].split('/global_step_200/')[1]:(x['bytes'],x['sha256']) for x in d['files']}
    if len(files(saved))!=33 or files(saved)!=files(archived):raise RuntimeError('Final archive differs from checkpoint receipt')

def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('source_job');a=p.parse_args();root=Path(a.root)
    out=root/'dependent-final-audit';out.mkdir(exist_ok=True)
    def state(**kw):(out/'state.json').write_text(json.dumps(dict(epoch=time.time(),dependency_job=a.source_job,audit_job=os.environ.get('SLURM_JOB_ID'),**kw),indent=2)+'\n')
    try:
        state(status='VERIFYING_FINAL_ARTIFACTS')
        # afterany permits inspection of the known producer padding gate failure.
        # Missing training/evaluation/recovery remains a hard failure, never success.
        saved=json.loads((root/'checkpoint-receipt-200.json').read_text());archived=json.loads((root/'archive-verified-200.json').read_text());archived_equal(saved,archived)
        sys.path.insert(0,str(root/'tools'))
        from verify_updates import verify
        plan=json.loads((root/'recovery-plan.json').read_text())
        training=verify(root/'evidence',200,a.source_job,prior_job=plan['source_job'],prior_through=plan['step'],prior_history=plan.get('prior_history',{}))
        records=[]
        for path in (root/'evidence').rglob('*.jsonl'):
            for line in path.open():
                d=json.loads(line)
                if d.get('kind')=='recovery_state_equal' and str(d.get('job'))==a.source_job and '/global_step_200/' in d.get('path',''):records.append(d)
        if len(records)!=8 or {(d['role'],d['rank']) for d in records}!={(role,rank) for role in ('main','reranker') for rank in range(4)}:raise RuntimeError('Missing actual full8 final checkpoint recovery')
        data=json.loads((root/'data-recovery-200.json').read_text())
        if data.get('step')!=200 or data.get('equal') is not True:raise RuntimeError('Final data recovery differs')
        from verify_evaluation import verify as native_verify
        native=native_verify(root/'final-evaluation/200.jsonl',51713,200)
        sys.path.insert(0,str(root.parent/'common-score-v4-tools'))
        from score_saved_boundaries import evaluation
        common=evaluation(root,'common-final-evaluation',200,51713)
        if common['questions']!=51713 or len(common['datasets'])!=7:raise RuntimeError('Incomplete seven-dataset evaluation')
        state(status='COMPLETE_200_COMMON_FINAL_RECOVERY_ARCHIVE_PASS',training=training,native_final=native,common_final=common,actual_recovery_count=8,data_recovery=data,archive_path=archived['path'],no_iteration201=True)
    except BaseException as e:
        state(status='FINAL_ACCEPTANCE_FAILED_REQUIRES_RECOVERY_OR_REVIEW',error_type=type(e).__name__,error=str(e));raise
if __name__=='__main__':main()
