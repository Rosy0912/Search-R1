"""Explicit shared-GPU user authority; retire only our pending3392 then start own unit once."""
import argparse,json,os,subprocess,time
from pathlib import Path
from eval_merge import atomic_immutable,digest

def main():
    p=argparse.ArgumentParser()
    for n in ('unit','tools','prefix','manifest','release','score-output','journal','review'):p.add_argument('--'+n,required=True)
    p.add_argument('--repair-predecessor-unit')
    p.add_argument('--repair-predecessor-journal')
    a=p.parse_args();root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    if os.uname().nodename.split('.')[0]!='h100-10' or not a.unit.startswith('sst-tree-eval-shared-') or not a.unit.endswith('.service'):
        raise RuntimeError('Own retained18 systemd identity required')
    journal=Path(a.journal)
    if journal.exists():raise RuntimeError('Existing shared once journal; inspect unit and score ledger instead of relaunching')
    release=json.loads(Path(a.release).read_text());review=json.loads(Path(a.review).read_text())
    if release['review_status']!='ACCEPTED' or review['review_status']!='ACCEPTED' or release['resource_protocol']!='memory_headroom_shared_v1':
        raise RuntimeError('Reviewed current memory-sharing authorization missing')
    for n,h in release['files'].items():
        if digest(Path(a.tools)/n)!=h or review['files'][n]!=h:raise RuntimeError('Shared source/review differs: '+n)
    if digest(a.prefix)!=release['prefix_sha256'] or digest(a.manifest)!=release['manifest_sha256']:
        raise RuntimeError('Saved metrics/assets changed')
    done=json.loads(Path(a.prefix+'.done.json').read_text())
    if done['sha256']!=release['prefix_sha256'] or done['actual_trained_outer']!=200:
        raise RuntimeError('Actual completed200 and durable prefix proof missing')
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SQUEUE_','SCANCEL_','SACCT_','SCONTROL_'))}
    active_ids=set(subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True,env=env).split())
    if a.repair_predecessor_unit:
        if a.repair_predecessor_unit not in ('sst-tree-eval-shared-1c15c49.service','sst-tree-eval-shared-b316a77.service') or not a.repair_predecessor_journal:
            raise RuntimeError('Only identified failed own predecessor may be repaired')
        prior=json.loads(Path(a.repair_predecessor_journal).read_text())
        original=json.loads((root/'runs/tree-branpo-fresh200-3360/output/eval-resume/shared-dispatch-once.json').read_text())
        if prior['unit']!=a.repair_predecessor_unit or original['unit']!='sst-tree-eval-shared-1c15c49.service' or original['cancel_own_queued_returncode']!=0 or prior['prefix_sha256']!=release['prefix_sha256'] or original['prefix_sha256']!=release['prefix_sha256']:
            raise RuntimeError('Durable prior cancellation/score identity absent')
        terminal=subprocess.check_output(['systemctl','show',a.repair_predecessor_unit,'--property=ActiveState,MainPID,ExecMainStatus'],text=True,env=env)
        if 'ActiveState=failed' not in terminal or 'MainPID=0' not in terminal or 'ExecMainStatus=1' not in terminal:
            raise RuntimeError('Previous own startup must be confirmed failed and terminal')
        if active_ids.intersection({'3360','3392'}):raise RuntimeError('Old evaluator still active; no repair duplicate')
        queued='Previously canceled own3392; identified failed shared startup: '+terminal
    else:
        queued=subprocess.check_output(['scontrol','--local','show','job','3392','-o'],text=True,env=env)
        if ('JobState=PENDING ' not in queued or 'JobName=tree-final-eval-resume ' not in queued
             or 'ReqNodeList=h100-10 ' not in queued or 'eval-resume-7933ec6' not in queued):
            raise RuntimeError('Only known queued own3392 can be superseded; inspect current state')
    state=subprocess.run(['systemctl','show',a.unit,'--property=LoadState,ActiveState'],text=True,capture_output=True,env=env)
    if 'LoadState=not-found' not in state.stdout:raise RuntimeError('Own service identity already exists; never duplicate')
    selected=json.loads((Path(a.tools)/'expected-gpu-uuids-retained18.json').read_text())
    value=dict(status='PREPARED_SHARED_UNIT_LAUNCH_UNKNOWN',epoch=time.time(),unit=a.unit,preserved_count=done['count'],
      prefix_sha256=release['prefix_sha256'],release_sha256=digest(a.release),review_sha256=digest(a.review),
      previous_own_queue=queued,physical_gpus=8,slurm_reserved_gpus=0,training_updates=200,foreign_jobs_modified=False,
      current_user_authorization='2026-10-13 explicit shared-GPU evaluation with memory headroom, four H100 nodes permitted; do not wait for exclusive emptiness')
    atomic_immutable(journal,value)
    def update():
        temporary=journal.with_name(journal.name+'.update')
        with temporary.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temporary,journal);fd=os.open(journal.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    if not a.repair_predecessor_unit:
        r=subprocess.run(['scancel','3392'],text=True,capture_output=True,env=env,timeout=30)
        value.update(cancel_own_queued_returncode=r.returncode,cancel_own_queued_stderr=r.stderr);update()
        if r.returncode:raise RuntimeError('Queued cancellation uncertain; inspect, no shared launch')
        for _ in range(20):
            if '3392' not in subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True,env=env).split():break
            time.sleep(.5)
        else:raise RuntimeError('Own exclusive queue still present; no concurrent evaluator')
    else:
        value.update(repaired_predecessor_unit=a.repair_predecessor_unit,predecessor_journal=a.repair_predecessor_journal,predecessor_terminal=terminal);update()
    argv=['systemd-run','--unit='+a.unit,'--property=CPUQuota=3200%','--property=MemoryMax=256G',
          '--property=TasksMax=4096','--property=KillMode=control-group','--property=TimeoutStopSec=60',
          '--property=Nice=5','--setenv=CUDA_VISIBLE_DEVICES='+','.join(selected),
          '--setenv=PYTHONPATH='+str(root/'instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/source/repo')+':'+str(root/'tools'),
          '--setenv=PYTHONOPTIMIZE=0',str(root/'env/bin/python'),str(Path(a.tools)/'shared_supervisor.py'),
          '--unit',a.unit,'--prefix',a.prefix,'--manifest',a.manifest,'--release',a.release,'--score-output',a.score_output]
    r=subprocess.run(argv,text=True,capture_output=True,env=env,timeout=30)
    value.update(systemd_start_returncode=r.returncode,systemd_start_stdout=r.stdout,systemd_start_stderr=r.stderr,
                 status='SHARED_UNIT_REQUESTED' if r.returncode==0 else 'SHARED_UNIT_LAUNCH_UNKNOWN');update()
    if r.returncode:raise RuntimeError('Shared unit launch unconfirmed; inspect known unit, never blind retry')
    print(json.dumps(value,sort_keys=True))

if __name__=='__main__':main()
