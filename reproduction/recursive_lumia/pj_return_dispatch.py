"""Unique CPU-only dispatcher: wait for complete files; then submit each group once."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .pj_return_contracts import BASE, PROFILES
from .pj_return import prepare, dump_new
from release.recursive_bran_a.ablation150_v1.contracts import sha
from release.recursive_bran_a.ablation150_v1.progress import replace_json

NAMES={'budget-5':'zhangjunming-ttt-b5-lumia85-v1','generation-2':'zhangjunming-ttt-d2-lumia137-v1'}
RESOURCES={'budget-5':('ADA6000','nvidia_rtx_6000_ada_generation'), 'generation-2':('L40S','nvidia_l40s')}
DIRECT=BASE/'migrations/pj-lumia-direct-20261017T1350'


def ready(group):
    stage=BASE/'migrations'/PROFILES[group]['stage']
    manifest=json.loads((stage/'parent-manifest.json').read_text())
    rows=manifest['worker_states']+[manifest['driver']]
    # A .part never counts. The transfer controller renames only after real SHA.
    return all((stage/Path(x['path']).name).is_file() and not (stage/Path(x['path']).name).is_symlink() for x in rows)


def submit(group,source):
    stage=BASE/'migrations'/PROFILES[group]['stage']; name=NAMES[group]
    if (stage/'return-submit-intent.json').exists() or (stage/'return-submission.json').exists():
        raise ValueError('Already submitted or uncertain; never submit again automatically')
    matches=subprocess.check_output(['squeue','-h','-u','jmzhang','--name',name,'-o','%i|%j'],text=True).strip()
    if matches:raise ValueError('An allocation with this unique name already exists')
    binding=prepare(group,source,DIRECT/'identity.json')
    stat=os.statvfs(BASE)
    if stat.f_bavail*stat.f_frsize<600_000_000_000:raise RuntimeError('Insufficient target write headroom')
    partition,gpu=RESOURCES[group]; runtime=BASE/'sources'/source
    command=['sbatch','--parsable','--hold','--no-requeue','--job-name',name,
        '--partition',partition,'--nodelist',PROFILES[group]['node'],'--nodes','1','--ntasks','1',
        '--cpus-per-task','64','--mem','600G','--gres','gpu:'+gpu+':8','--time','3-00:00:00',
        '--output',str(BASE/'logs'/('pj-return-'+group+'-%j.log')),
        '--error',str(BASE/'logs'/('pj-return-'+group+'-%j.log')),
        '--export',','.join(['ALL','RECURSIVE_LUMIA_BASE='+str(BASE),'RECURSIVE_GROUP='+group,
            'RECURSIVE_SOURCE_COMMIT='+source,'RECURSIVE_SOURCE_DIR='+str(runtime),
            'RECURSIVE_RETURN_BINDING='+str(binding)]),str(runtime/'reproduction/recursive_lumia/pj_return.sh')]
    dump_new(stage/'return-submit-intent.json',dict(at=time.time(),command=command,binding_sha256=sha(binding)))
    # Held job cannot start before the exact returned ID is durably recorded.
    result=subprocess.run(command,text=True,capture_output=True,timeout=45)
    dump_new(stage/'return-submit-response.json',dict(at=time.time(),rc=result.returncode,stdout=result.stdout,stderr=result.stderr))
    result.check_returncode();job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise ValueError('Ambiguous scheduler receipt; inspect held job')
    dump_new(stage/'return-submission.json',dict(at=time.time(),job_id=job,name=name,group=group,
        source=source,binding_sha256=sha(binding),state='SUBMITTED_HELD_PENDING_RELEASE'))
    release=subprocess.run(['scontrol','release',job],text=True,capture_output=True,timeout=30)
    dump_new(stage/'return-release.json',dict(at=time.time(),job_id=job,rc=release.returncode,stdout=release.stdout,stderr=release.stderr))
    release.check_returncode()
    return dict(job_id=job,name=name,status='RELEASED_TO_SCHEDULER')


def main():
    source=os.environ['RECURSIVE_SOURCE_COMMIT']
    if Path(__file__).resolve().parents[2] != BASE/'sources'/source:
        raise ValueError('Dispatcher must run the immutable published source')
    with (DIRECT/'return-dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump_new(DIRECT/'return-dispatch-intent.json',dict(at=time.time(),pid=os.getpid(),source=source))
        state=dict(at=time.time(),pid=os.getpid(),source=source,status='WAITING_FOR_VERIFIED_FULL_STATES',groups={})
        try:
            deadline=time.monotonic()+20*3600
            while time.monotonic()<deadline:
                for group in ('generation-2','budget-5'):
                    if group not in state['groups'] and ready(group):
                        state['groups'][group]=submit(group,source)
                state['at']=time.time()
                replace_json(DIRECT/'return-dispatch-status.json',state)
                if len(state['groups'])==2:
                    state['status']='BOTH_UNIQUE_CONTINUATIONS_SUBMITTED';break
                copy=json.loads((DIRECT/'copy-status.json').read_text())
                if copy['state'].startswith('FAILED'):raise RuntimeError('Copy failed; preserve partial, no automatic retry')
                time.sleep(45)
            else:raise TimeoutError('Full-state transfer not finished within20h; no blind submission')
        except BaseException as exc:
            state.update(status='NEEDS_INSPECTION_NO_AUTOMATIC_REPLAY',error=repr(exc));raise
        finally:
            state['at']=time.time();replace_json(DIRECT/'return-dispatch-status.json',state)

if __name__=='__main__':main()
