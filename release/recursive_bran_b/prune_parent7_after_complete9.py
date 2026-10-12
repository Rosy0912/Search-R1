"""Apply cross-run rolling retention only after complete9 and archived parent8.

Keep parent5, parent8 and every artifact of the current run untouched. Parent7
is the single obsolete rolling state eligible for removal by this entry.
"""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from search_r1.baselines.recursive_b_completed_resume import PARENT_RUN,PARENT_COMMIT

ROOT=Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
CHILD='2e3a9029051d4771bc61c95b28b60e21'
CHILD_COMMIT='65e96b32c8228513e600effa0de9810148f1f9b9'

def require(ok,msg):
    if not ok:raise ValueError('Parent7 rolling retention: '+msg)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    require(os.getuid()==1001,'owner')
    child=ROOT/('output-speed-resume3-'+CHILD)
    restored=json.loads((child/'sparse-schedule-migrated.json').read_text())
    require(restored['driver_actual_state_equal'] and restored['next_outer']==9 and
        all(w['actual_cold_native_state_equal'] and w['adam_steps']==[160]
            and w['sampling_stream']['actual_separate_sampling_cuda_stream_restored'] for w in restored['workers']),
        'full native8 restoration')
    m=json.loads((child/'checkpoints/training_state/global_step_9/manifest.json').read_text())
    require(m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and m['identity']['completed_step']==9
        and m['identity']['code_commit']==CHILD_COMMIT and m['identity']['job_id']=='hkust-direct-'+CHILD,
        'complete9 child identity')
    require(len(m['worker_states'])==4 and all(w['full_worker_state_finite'] and
        w['actual_total_optimizer_calls']==180 and w['observed_adam_steps']==[180]
        and w['scheduler_last_epoch']==9 for w in m['worker_states']),'complete9 rank state')
    parent=ROOT/('output-speed-resume3-'+PARENT_RUN)/'checkpoints/training_state'
    parent8=parent/'global_step_8/manifest.json'
    sha8=hashlib.sha256(parent8.read_bytes()).hexdigest()
    archive='/weka_3/shaoshitong/recursive-bran-b-hkust5-archive-20261012/'+PARENT_RUN+'/checkpoint-bundles/global_step_8/archive-complete.json'
    raw=subprocess.check_output(['ssh','-oBatchMode=yes','-oConnectTimeout=8','-oStrictHostKeyChecking=yes',
        'shaoshitong@10.123.4.11','cat '+archive],timeout=20)
    receipt=json.loads(raw)
    require(receipt['status']=='COMPLETE_CHECKPOINT_ARCHIVE_SHA_VERIFIED' and receipt['run']==PARENT_RUN
        and receipt['step']==8 and receipt['source_commit']==PARENT_COMMIT
        and receipt['manifest_sha256']==sha8 and len(receipt['files'])==10,'independent parent8 archive')
    for pid in [1024017,1026405,1026891,1027302,1027303,1027304]:
        p=Path('/proc')/str(pid)
        require(not p.exists() or ((p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
            and not list((p/'fd').glob('*'))),'old run still live')
    target=parent/'global_step_7';require(not target.is_symlink() and target.resolve()==target,'target path')
    report_path=ROOT/'receipts/pruned-parent7-after-complete9.json'
    if not target.exists():
        require(report_path.exists(),'missing target without prior receipt');print(report_path.read_text());return
    old=json.loads((target/'manifest.json').read_text())
    require(old['identity']['completed_step']==7 and old['identity']['code_commit']==PARENT_COMMIT
        and old['identity']['job_id']=='hkust-direct-'+PARENT_RUN,'obsolete rolling state identity')
    expected={'manifest.json','driver.pt'}|{'rank-%02d.pt'%rank for rank in range(4)}|{'sampling-rank-%02d.pt'%rank for rank in range(4)}
    require({p.name for p in target.iterdir()}==expected and all(p.is_file() and not p.is_symlink()
        and p.stat().st_uid==1001 for p in target.iterdir()),'closed owned native members only')
    report=dict(time=time.time(),target=str(target),bytes=sum(p.stat().st_size for p in target.iterdir()),
        parent7_manifest=old,parent8_archive_sha_verified=True,complete9_verified=True,
        retained_parent_steps=[5,8],child_artifacts_untouched=True,applied=a.apply)
    if a.apply:
        report_path.write_text(json.dumps(report,sort_keys=True));shutil.rmtree(target)
    print(json.dumps(report))

if __name__=='__main__':main()
