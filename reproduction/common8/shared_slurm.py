"""Explicit native CPU allocation with user-authorized physical eight-GPU sharing.

Shared physical devices are never labelled as a second Slurm GPU allocation.
The existing video reservation and device policy remain unchanged.
"""
import hashlib,json,os
from pathlib import Path
AUTHORIZATION='允许核实并改为Slurm共用，保留视频和8卡训练预算'
POLICY='authorized_physical8_with_native_cpu_memory_allocation'

def authorization(path,base):
    p=Path(path);base=Path(base).resolve()
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(base/'preparation')
    raw=p.read_bytes();a=json.loads(raw)
    assert a['authorization']==AUTHORIZATION and a['base']==str(base)
    assert a['physical_indices']==list(range(8)) and len(set(a['physical_gpu_uuids']))==8
    assert a['training_world_size']==8 and a['actor_micro_batch_per_rank']==2
    assert a['vllm_gpu_memory_utilization']==.1 and a['cpu_request']==96 and a['memory_request']=='512G'
    assert a['preserve_foreign_jobs'] is True and a['source_completed_outer']==5
    assert all(str(job).isdecimal() and birth['NodeList']=='h100-8' and birth['JobId']==str(job)
               for job,birth in a['allowed_native_jobs'].items())
    assert a['allowed_native_jobs'], 'Explicit native cohost birth is required'
    return a,hashlib.sha256(raw).hexdigest()

def bind(run,base):
    p=os.environ.get('SEARCH_G1_SLURM_SHARED_AUTHORIZATION')
    if not p:return
    from reproduction.common8.precision_contract import NATIVE_BF16_PROFILE
    assert run['precision_profile']==NATIVE_BF16_PROFILE and run.get('full_state_restore')
    a,sha=authorization(p,base)
    run.update(slurm_gpu_sharing_policy=POLICY,slurm_shared_authorization_path=str(Path(p).resolve()),
               slurm_shared_authorization_sha256=sha,shared_physical_gpu_indices=a['physical_indices'],
               shared_physical_gpu_uuids=a['physical_gpu_uuids'],shared_native_cohost_births=a['allowed_native_jobs'])

def physical_indices(allocation):
    if allocation.get('gpu_index_mapping_policy')=='slurm_global_minor_to_process_nvml_uuid_v1':
        assert allocation['training_world_size']==4 and len(set(allocation['actual_allocated_indices']))==4
        indices=allocation['actual_nvml_indices']
        assert len(indices)==len(set(indices))==4 and len(set(allocation['actual_cuda_runtime_uuids']))==4
        return indices
    if allocation.get('slurm_gpu_sharing_policy')==POLICY:
        assert allocation['actual_allocated_indices']==[] and allocation['slurm_reserved_gpu_count']==0
        indices=allocation['actual_shared_physical_indices']
        assert indices==list(range(8)) and len(set(allocation['actual_cuda_runtime_uuids']))==8
        return indices
    return allocation['actual_allocated_indices']

def validate(job,node,visible,active,gpus,applications,actual,allowed,run):
    assert run['slurm_gpu_sharing_policy']==POLICY
    a,sha=authorization(run['slurm_shared_authorization_path'],Path(run['checkpoint_dir']).parents[1])
    assert sha==run['slurm_shared_authorization_sha256']
    assert job['JobState']=='RUNNING' and job['NodeList']==node=='h100-8' and job['NumNodes']=='1'
    assert job.get('TresPerNode','(null)') in ('(null)','N/A')
    assert job['NumCPUs']=='96' and job['MinMemoryNode']=='512G' and job['OverSubscribe']!='NO'
    assert set(active)<=set(a['allowed_native_jobs'])|{job['JobId']} and job['JobId'] in active
    assert visible==list(map(str,range(8)))
    inventory={int(row[0]):row[1].strip() for row in gpus}
    assert set(inventory)==set(range(8)) and [inventory[i] for i in range(8)]==a['physical_gpu_uuids']==actual
    bound={(str(x[0]).strip(),str(x[1]).strip()) for x in allowed}
    assert all((str(x[0]).strip(),str(x[1]).strip()) in bound for x in applications)
    return True
