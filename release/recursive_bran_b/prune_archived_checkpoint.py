"""Reclaim an archived B milestone without removing current rolling states.

Only the task's stopped parent5 and current run50/100/150 are eligible. Keep
native parent8, current last2, final200, algorithm records and logs untouched.
"""
import argparse,hashlib,json,os,shlex,shutil,socket,subprocess,time
from pathlib import Path
from .archive_checkpoint import SOURCE_BASE,DEST_BASE,SSH,payload,pin_root,safe_dirs

PARENT='a177507c37c94e898616a194ed6b2a45'
PARENT_SOURCE='bc666df3a3fee9da7bb5d09f8da649cfa7165131'
CURRENT='2e3a9029051d4771bc61c95b28b60e21'
CURRENT_SOURCE='65e96b32c8228513e600effa0de9810148f1f9b9'

def require(ok,message):
    if not ok:raise ValueError('Archived B retention: '+message)

def complete_guard(step):
    require(10<=step<=200,'guard must be a complete current outer10–200')
    root=SOURCE_BASE/('output-speed-resume3-'+CURRENT)
    p=root/'checkpoints/training_state'/('global_step_%d'%step)/'manifest.json'
    safe_dirs(p.parent,SOURCE_BASE);require(not p.is_symlink(),'guard manifest symlink')
    m=json.loads(p.read_text());i=m['identity']
    require(m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and i['completed_step']==step
        and i['job_id']=='hkust-direct-'+CURRENT and i['code_commit']==CURRENT_SOURCE
        and i['actual_update_limit']==200 and m['driver']['next_step']==step+1,'guard identity/cursor')
    ws=m['worker_states'];require(len(ws)==4 and {w['identity']['rank'] for w in ws}==set(range(4)),
        'guard rank coverage')
    require(all(w['full_worker_state_finite'] and w['scheduler_last_epoch']==step
        and w['observed_adam_steps']==[w['actual_total_optimizer_calls']] for w in ws),'guard state/counters')
    restored=json.loads((root/'sparse-schedule-migrated.json').read_text())
    require(restored['driver_actual_state_equal'] and restored['next_outer']==9
        and restored['provenance']['parent_identity']['job_id']=='hkust-direct-'+PARENT
        and restored['provenance']['parent_identity']['code_commit']==PARENT_SOURCE
        and all(w['actual_cold_native_state_equal'] and w['sampling_stream']['actual_separate_sampling_cuda_stream_restored']
            for w in restored['workers']),'own parent lineage/full recovery')
    return m,hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--guard-completed',type=int,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    require(os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4','host/owner')
    eligible=(a.run==PARENT and a.step==5) or (a.run==CURRENT and a.step in (50,100,150))
    require(eligible,'only archived own parent5/current50/100/150 are eligible')
    guard,guard_sha=complete_guard(a.guard_completed)
    if a.run==CURRENT:require(a.step<=a.guard_completed-2,'never delete newest2')
    else:
        for pid in [1024017,1026405,1026891,1027302,1027303,1027304]:
            proc=Path('/proc')/str(pid)
            require(not proc.exists() or ((proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
                and not list((proc/'fd').glob('*'))),'stopped parent still owns resources')
    source=PARENT_SOURCE if a.run==PARENT else CURRENT_SOURCE
    root=SOURCE_BASE/('output-speed-resume3-'+a.run)/'checkpoints'
    report_dir=SOURCE_BASE/'receipts/archived-local-prunes'/a.run
    safe_dirs(report_dir,SOURCE_BASE);report_dir.mkdir(parents=True,exist_ok=True)
    report_path=report_dir/('global_step_%d.json'%a.step)
    native=root/'training_state'/('global_step_%d'%a.step)
    if not native.exists():
        require(report_path.exists(),'missing source without prior receipt')
        old=json.loads(report_path.read_text());require(old['status']=='PRUNE_AUTHORIZED_AFTER_INDEPENDENT_ARCHIVE',
            'prior receipt status');require(all(not Path(t).exists() for t in old['targets']),'partial prune')
        print(json.dumps(dict(already_pruned=True,receipt=str(report_path))));return
    data,targets=payload(root,a.run,a.step,source)
    require(not pin_root(a.run,a.step).exists(),'archive transfer still pinned')
    dest=DEST_BASE/a.run/'checkpoint-bundles'/('global_step_%d'%a.step)/'archive-complete.json'
    archived=json.loads(subprocess.check_output(SSH+['cat '+shlex.quote(str(dest))],text=True,timeout=20))
    require(archived['status']=='COMPLETE_CHECKPOINT_ARCHIVE_SHA_VERIFIED' and
        all(archived[key]==value for key,value in data.items()),'complete archive identity/inventory/checksums')
    identity=json.loads((native/'manifest.json').read_text())['identity']
    require(all(identity[k]==guard['identity'][k] for k in ('data_identity','model_identity','actual_update_limit')),
        'scientific identity differs')
    report=dict(status='PRUNE_AUTHORIZED_AFTER_INDEPENDENT_ARCHIVE',time=time.time(),run=a.run,step=a.step,
        source_commit=source,guard_completed=a.guard_completed,guard_manifest_sha256=guard_sha,
        archive_receipt=archived,targets=list(map(str,targets)),
        bytes=sum(r['bytes'] for r in data['files'].values()),applied=a.apply,
        scope='Only immutable archived native/HF directories; parent8/current latest2/final200/logs/algorithm untouched')
    if a.apply:
        require(not report_path.exists(),'existing receipt needs explicit partial-prune inspection')
        with report_path.open('x') as f:
            json.dump(report,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        fd=os.open(report_dir,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        for target in targets:shutil.rmtree(target)
    print(json.dumps(dict(applied=a.apply,run=a.run,step=a.step,bytes=report['bytes'],
        targets=report['targets'],guard_completed=a.guard_completed,receipt=str(report_path))))

if __name__=='__main__':main()
