"""Synchronous worker drain -> owned calibration -> original worker resume."""
import hashlib,json,pathlib
from contextlib import ExitStack

def validate_worker_release(records,uuids,role):
    if not isinstance(records,list) or len(records)!=8:raise ValueError('eight actual '+role+' release receipts required')
    if {r.get('rank') for r in records}!=set(range(8)):raise ValueError('rank coverage mismatch')
    if {r.get('gpu_uuid') for r in records}!=set(uuids):raise ValueError('lease UUID coverage mismatch')
    for r in records:
        if r.get('status')!='RELEASED_ZERO' or r.get('allocated')!=0 or r.get('reserved')!=0:raise ValueError('nonzero actual worker allocator')
        if r.get('process_gpu_uuids')!=[r['gpu_uuid']]:raise ValueError('actual PID UUID mismatch')
        for group in ('parameter_devices','buffer_devices'):
            if not isinstance(r.get(group),dict) or any(any(not d.startswith('cpu') for d in devices) for devices in r[group].values()):raise ValueError('worker tensors not CPU')
        if not all(r.get(k) for k in ('pid','starttime_ticks','cgroup','job','host')):raise ValueError('actual worker identity incomplete')
    return records

def refresh_owned_pipeline(binding_path,checkpoint,step,actor,reference=None):
    """Unknown IPC outcome never resumes GPU or advances training. Supervisor aborts job."""
    from search_r1.g1_pipeline_ipc import client
    binding=json.loads(pathlib.Path(binding_path).read_text())
    if step not in (25,50,75):raise ValueError('original nonterminal refresh schedule')
    groups=[('actor_rollout',actor)]+([('ref',reference)] if reference is not None else [])
    # All groups offload BEFORE any per-process allocator check. Actor/ref may colocate.
    for _,group in groups:group.g1_calibration_quiesce()
    receipts={role:validate_worker_release(group.g1_calibration_release_receipt(),binding['uuids'],role) for role,group in groups}
    response=client(binding_path,{'operation':'REFRESH','outer_step':step,'checkpoint':str(checkpoint),'worker_releases':receipts})
    # Only the authenticated owner COMPLETE ACK permits original caller restoration.
    for _,group in groups:group.g1_calibration_resume()
    return response


def record_actual_outer(path,step,terminal,checkpoint_root):
    import os,time
    p=pathlib.Path(path)
    prior=json.loads(p.read_text()) if p.exists() else {'scope':'ACTUAL_TRAINER_COMPLETED_OUTER_UNREVIEWED','actual_completed_outer_steps':[],'export_step':None}
    if prior['actual_completed_outer_steps']!=list(range(1,step)):raise ValueError('actual completed outer order changed')
    prior['actual_completed_outer_steps'].append(step)
    if terminal:
        checkpoint=pathlib.Path(checkpoint_root)/'actor'/('global_step_'+str(step))
        if not (checkpoint/'config.json').is_file():raise ValueError('terminal original HF export absent')
        prior.update(export_step=step,export_path=str(checkpoint))
    prior.update(pid=os.getpid(),epoch=time.time());tmp=p.with_name(p.name+'.tmp')
    with open(tmp,'x') as f:os.chmod(tmp,0o600);json.dump(prior,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
