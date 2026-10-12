"""Preserve terminal OOM140 evidence and build immutable checkpoint100 provenance."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011')
def main():
    state=subprocess.check_output(['sacct','-X','-j','709923,710171','--format=JobID,State,ExitCode','-P'],universal_newlines=True)
    if '709923|FAILED|1:0' not in state or '710171|FAILED|1:0' not in state:raise RuntimeError('Original jobs are not terminal failures')
    failure=ROOT/'failed-job-709923'
    failure.mkdir(exist_ok=False)
    (failure/'slurm-terminal.txt').write_text(state)
    for name in ['logs','evidence','rollouts']:
        shutil.move(str(ROOT/name),str(failure/name));(ROOT/name).mkdir()
    for name in ['tools','author-cli.json','deployment-manifest.json','resolved-continue200.yaml','controller-state.json','recovery-plan.json','allocation.json','job.id','submit-claim.json','submission-result.json','slurm-709923.log','dependent-final-audit']:
        source=ROOT/name
        if source.is_dir():shutil.copytree(str(source),str(failure/name))
        elif source.exists():shutil.copy2(str(source),str(failure/name))
    source_manifest={}
    for path in (failure/'evidence').glob('*.jsonl'):
        if not (path.name.startswith('main-') or path.name.startswith('reranker-')):continue
        source_manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as src,(ROOT/'evidence'/path.name).open('x') as dst:
            for line in src:
                event=json.loads(line)
                if event.get('job')=='709923' and event.get('outer',0)>100:continue
                dst.write(line)
    training=ROOT/'common-training'
    for path in list(training.iterdir()):
        import re
        match=re.search(r'(?:outer-|step-)(\d+)',path.name)
        if match and int(match.group(1))>100:
            destination=failure/'common-training';destination.mkdir(exist_ok=True);shutil.move(str(path),str(destination/path.name))
    for name in ['submit-claim.json','submission-result.json','job.id']:
        if (ROOT/name).exists():shutil.move(str(ROOT/name),str(failure/(name+'.original')))
    for step in [100]:
        p=ROOT/('archive-delete-safe-'+str(step))
        if p.exists():shutil.move(str(p),str(failure/p.name))
    prior=json.loads((failure/'recovery-plan.json').read_text())
    saved=ROOT/'checkpoints/global_step_100/full-sha-receipt.json'
    receipt=json.loads((ROOT/'checkpoint-receipt-100.json').read_text())
    # The stage brings back the original receipt; its exact bytes are checked before launch.
    plan=dict(step=100,source_job='709923',receipt_sha256=None,
              prior_history=dict(prior_job=prior['source_job'],prior_through=prior['step'],prior_history=prior.get('prior_history',{})),
              failure_history=str(failure),recomputed_failed_interval=[101,140])
    (ROOT/'recovery100-plan-pending-stage.json').write_text(json.dumps(plan,indent=2)+'\n')
    (failure/'preservation-receipt.json').write_text(json.dumps(dict(epoch=time.time(),status='FAILED_EVIDENCE_PRESERVED_PREFIX100_SELECTED',source_evidence_sha256=source_manifest,latest_complete_outer=139,incomplete_outer=140,recovery_checkpoint=100,recompute=[101,140]),indent=2)+'\n')
    print('FAILED_EVIDENCE_PRESERVED_PREFIX100_SELECTED')
if __name__=='__main__':main()
