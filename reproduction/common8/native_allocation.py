"""Verify real native allocation or the explicitly assigned local A800 resources.

Slurm's requested GRES alone is insufficient: also check RUNNING/node binding,
visible CUDA devices, no other active allocation on this node, and no foreign
compute applications. The local branch binds only the authorized GPUs and actual
PID birth/host; foreign jobs outside that assignment do not affect admission.
The explicit shared8 native branch separately proves CPU/memory allocation and
actual shared CUDA UUIDs, without claiming Slurm reserved the shared GPUs twice.
This is resource evidence, not model/GRPO acceptance.
"""
import argparse
import csv
import io
import json
import os
import re
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def fields(text):
    result=dict(re.findall(r'(\w+)=(\S+)', text))
    # Newer Slurm distinguishes requested from actually allocated resources.
    # Never use ReqTRES to manufacture allocation evidence.
    if 'TRES' not in result and result.get('AllocTRES') not in (None,'(null)'):
        result['TRES']=result['AllocTRES']
    return result


def allocated_indices(detail, node, count=8):
    line=next((line for line in detail.splitlines() if 'Nodes='+node+' ' in line), '')
    m=re.search(r'GRES=gpu:(?:h100:)?(\d+)\(IDX:([^)]*)\)',line)
    assert count in (1,4,8) and m and int(m.group(1))==count, 'Actual Slurm allocation GPU count differs'
    indices=[]
    for item in m.group(2).split(','):
        if '-' in item:
            lo,hi=map(int,item.split('-'));indices.extend(range(lo,hi+1))
        else:indices.append(int(item))
    assert len(indices)==count and len(set(indices))==count, 'Invalid actual Slurm GPU indices'
    return indices


def add_gpu_minor_numbers(rows,xml):
    """Use nvidia-smi's verbose XML; minor_number is not a selective CSV field."""
    minor_by_uuid={}
    for gpu in ET.fromstring(xml).findall('gpu'):
        uuid=(gpu.findtext('uuid') or '').strip()
        minor=(gpu.findtext('minor_number') or '').strip()
        assert uuid.startswith('GPU-') and minor.isdecimal() and uuid not in minor_by_uuid, 'Incomplete or duplicate GPU XML identity'
        minor_by_uuid[uuid]=int(minor)
    assert set(minor_by_uuid)=={row[1].strip() for row in rows}, 'CSV and XML actual GPU inventories differ'
    assert len(set(minor_by_uuid.values()))==len(minor_by_uuid), 'Duplicate actual physical minor number'
    return [row+[str(minor_by_uuid[row[1].strip()])] for row in rows]


def cuda_uuids(count=8):
    if type(count) is not int or count < 1:
        raise ValueError('CUDA UUID count must be a positive integer')
    import ctypes
    import importlib.util
    import sysconfig
    import uuid
    path=Path(sysconfig.get_paths()['purelib'])/'nvidia/cuda_runtime/lib/libcudart.so.12'
    if not path.is_file():
        spec=importlib.util.find_spec('nvidia.cuda_runtime')
        assert spec is not None and spec.submodule_search_locations, 'Installed CUDA runtime package is unavailable'
        path=next((Path(root)/'lib/libcudart.so.12' for root in spec.submodule_search_locations
                   if (Path(root)/'lib/libcudart.so.12').is_file()),None)
        assert path is not None, 'Installed CUDA12 runtime shared library is unavailable'
    lib=ctypes.CDLL(str(path))
    # CUDA 12 exports properties_v2, not cudaDeviceGetUuid. Verify the
    # installed header's ABI prefix before reading its UUID field.
    header=(path.parents[1]/'include/driver_types.h').read_text()
    header=re.sub(r'/\*.*?\*/', '', header, flags=re.S)
    assert re.search(r'struct\s+(?:__device_builtin__\s+)?cudaDeviceProp\s*\{\s*char\s+name\[256\];\s*cudaUUID_t\s+uuid\s*;',header), 'Unknown CUDA property ABI prefix'
    call=lib.cudaGetDeviceProperties_v2;call.argtypes=[ctypes.c_void_p,ctypes.c_int];call.restype=ctypes.c_int
    result=[]
    for ordinal in range(count):
        # The complete CUDA-12 structure is much smaller; reserve ample space
        # because only the verified name/UUID prefix is needed here.
        value=ctypes.create_string_buffer(65536)
        assert call(value,ordinal)==0, 'Cannot obtain actual CUDA device properties'
        result.append('GPU-'+str(uuid.UUID(bytes=value.raw[256:272])))
    return result


def validate(job, node, visible, active_jobs, gpu_rows, applications, allocated, actual_cuda_uuids, allowed_applications=None):
    assert job['JobState']=='RUNNING', 'No active native allocation'
    assert job['NodeList']==node and job['NumNodes']=='1', 'Allocation is not this single node'
    assert job['TresPerNode'] in ('gres:gpu:h100:8','gres:gpu:8','gres/gpu:8'), 'Eight GPUs were not requested'
    assert len(visible)==8 and len(set(visible))==8, 'Expected eight unique allocated CUDA devices'
    assert set(active_jobs)=={job['JobId']}, 'Another native allocation shares the node'
    assert len(gpu_rows)==8 and len({r[1] for r in gpu_rows})==8, 'Expected eight physical GPUs'
    physical={int(r[0]):r[1] for r in gpu_rows}
    assert len(physical)==8
    expected=[physical[int(i)] if i.isdecimal() else i for i in visible]
    assert set(expected)=={physical[i] for i in allocated}, 'Visible GPUs differ from actual Slurm allocation'
    assert actual_cuda_uuids==expected, 'CUDA runtime UUID mapping differs from allocated visible devices'
    if allowed_applications is None:
        assert not applications, 'Foreign compute applications occupy this node'
    else:
        allowed = {(str(row[0]).strip(), str(row[1]).strip()) for row in allowed_applications}
        assert all((str(row[0]).strip(), str(row[1]).strip()) in allowed for row in applications), 'Unbound compute application appeared before shared launch'
    return True


def validate_native_four(job, node, visible, active_jobs, gpu_rows, applications, allocated, actual_cuda_uuids):
    """Four real reserved GPUs; foreign allocations may use other UUIDs only."""
    assert job['JobState']=='RUNNING' and job['NodeList']==node and job['NumNodes']=='1'
    assert job['TresPerNode'] in ('gres:gpu:h100:4','gres:gpu:4','gres/gpu:4')
    assert job['JobId'] in active_jobs and len(visible)==len(set(visible))==4
    physical={int(row[0]):row[1].strip() for row in gpu_rows}
    assert len(physical)==len(gpu_rows) and len(set(physical.values()))==len(physical)
    # Under ConstrainDevices, NVML indices are relative to the permitted
    # devices, while Slurm's global GRES IDs retain their device-file numbering.
    # Keep both namespaces and authenticate their common UUIDs. Never infer the
    # assigned physical GPUs from the cgroup-relative numeric indices alone.
    minors={int(row[3]):row[1].strip() for row in gpu_rows} if all(len(row)>3 for row in gpu_rows) else physical
    assert len(minors)==len(physical) and len(allocated)==len(set(allocated))==4 and set(allocated)<=set(minors), 'Assigned Slurm devices absent from actual minor-number inventory'
    expected=[physical[int(token)] if token.isdecimal() else token for token in visible]
    assigned={minors[index] for index in allocated}
    assert len(set(expected))==4 and set(expected)==assigned and actual_cuda_uuids==expected
    assert all(row[0].strip() not in assigned for row in applications), 'Assigned GPUs have foreign compute applications'
    return True


def admit(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('execution_backend','slurm')=='ssh_local':
        return admit_local(base,run)
    import torch
    assert 'SLURM_JOB_ID' in os.environ, 'Must run under native Slurm'
    job_id=os.environ['SLURM_JOB_ID']
    node=socket.gethostname().split('.')[0]
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8' and run['completed_updates']==0
    detail=subprocess.check_output(['scontrol','show','job','-d',job_id],text=True)
    job=fields(detail)
    from reproduction.common8.shared_slurm import POLICY as SHARED_POLICY
    sharing=run.get('slurm_gpu_sharing_policy')==SHARED_POLICY
    from reproduction.common8.topology import world_size
    size=world_size(run)
    allocated=[] if sharing else allocated_indices(detail,node,size)
    assert job['JobId']==job_id
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    active=subprocess.check_output(['squeue','-h','-w',node,'-t','RUNNING,COMPLETING','-o','%i'],text=True).split()
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True)
    rows=[[x.strip() for x in r] for r in csv.reader(io.StringIO(raw))]
    if size==4 and not sharing:
        xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=30)
        with (base/'logs/native-gpu-inventory.xml').open('x') as f:
            f.write(xml);f.flush();os.fsync(f.fileno())
        rows=add_gpu_minor_numbers(rows,xml)
    raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    applications=[r for r in csv.reader(io.StringIO(raw_apps)) if r]
    assert torch.cuda.device_count()==size
    actual=cuda_uuids(size)
    observation=dict(run_id=run['run_id'],slurm_job_id=job_id,node=node,
        training_world_size=size,actual_allocation_detail=detail,visible_devices=visible,
        slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),actual_allocated_indices=allocated,
        physical_gpu_inventory=rows,actual_cuda_runtime_uuids=actual,
        active_node_jobs=active,foreign_compute_applications=applications,checked_unix=time.time(),
        accepted=False,scope='raw actual resource observation before admission; not acceptance')
    with (base/'logs/native-allocation-observation.json').open('x') as f:
        json.dump(observation,f,indent=2);f.flush();os.fsync(f.fileno())
    cohosts = run.get('shared_gpu_cohosts')
    allowed = None
    if cohosts is not None:
        assert run.get('shared_gpu_policy') == 'preserve_bound_existing_processes'
        from reproduction.a800.owned_processes import same
        allowed = [record['application'] for record in cohosts if same(record['process_identity'])]
    if sharing:
        from reproduction.common8.shared_slurm import validate as validate_shared
        assert allowed is not None
        for cohost,birth in run['shared_native_cohost_births'].items():
            if cohost in active:
                current=fields(subprocess.check_output(['scontrol','show','job','-o',cohost],text=True))
                assert all(current[k]==birth[k] for k in ('JobId','StartTime','NodeList','Command'))
                assert current['JobState'] in ('RUNNING','COMPLETING')
        validate_shared(job,node,visible,active,rows,applications,actual,allowed,run)
    elif size==4:
        assert run.get('execution_site')=='hpc3' and cohosts is None
        validate_native_four(job,node,visible,active,rows,applications,allocated,actual)
    else:
        validate(job,node,visible,active,rows,applications,allocated,actual,allowed)
    memory_raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    memory_rows = [[x.strip() for x in r] for r in csv.reader(io.StringIO(memory_raw))]
    proof=dict(accepted=True,run_id=run['run_id'],slurm_job_id=job_id,node=node,
        actual_job_state=job['JobState'],requested_gres=job.get('TresPerNode'),
        visible_devices=visible,actual_allocated_indices=allocated,actual_cuda_runtime_uuids=actual,
        actual_allocation_detail=detail,physical_gpu_inventory=rows,active_node_jobs=active,
        foreign_compute_applications=applications,shared_gpu_policy=run.get('shared_gpu_policy'),
        bound_cohosts=cohosts,actual_memory_inventory=memory_rows,checked_unix=time.time(),
        training_world_size=size, scope='native allocation and actual UUID mapping before project GPU work; not model/GRPO acceptance')
    if size==4 and not sharing:
        by_uuid={row[1]:int(row[0]) for row in rows}
        proof.update(gpu_index_mapping_policy='slurm_global_minor_to_process_nvml_uuid_v1',
            actual_nvml_indices=[by_uuid[value] for value in actual],
            actual_gpu_minor_inventory=rows,slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'))
    if sharing:
        proof.update(slurm_gpu_sharing_policy=SHARED_POLICY,slurm_reserved_gpu_count=0,
            actual_shared_physical_indices=list(range(8)),
            slurm_shared_authorization_sha256=run['slurm_shared_authorization_sha256'],
            scope='actual native CPU/memory allocation plus authorized shared physical8 CUDA UUIDs; no exclusive or second Slurm GPU reservation claimed')
    target=base/'logs/native-allocation-admission.json'
    with target.open('x') as f:
        json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
    return proof


def local_placement(run):
    """Only the human-authorized A800 placement is accepted by this adapter."""
    from reproduction.common8.topology import world_size, execution_job_id
    assert execution_job_id(run) is None and world_size(run)==4
    training=run['training_physical_indices'];auxiliary=run['auxiliary_physical_indices']
    retriever=run['retriever_physical_indices'];probe=run['probe_physical_index']
    assert training==[0,1,2,3] and auxiliary==[5,6] and retriever==[5,6] and probe==6
    assigned=training+auxiliary
    assert all(type(i) is int for i in assigned) and len(set(assigned))==len(assigned)
    return training,auxiliary,retriever,probe


def admit_local(base,run):
    """Bind actual local process birth and assigned UUIDs without a Slurm fiction.

    GPUs outside the assigned set are inventoried for mapping only. Their jobs
    are neither an admission failure nor eligible for project cleanup.
    """
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.topology import validate_base
    import torch
    validate_base(base,run)
    assert run['profile']=='common8' and run['completed_updates']==0
    training,auxiliary,retriever,probe=local_placement(run)
    assigned=training+auxiliary
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    assert visible==list(map(str,training)), 'Local pipeline must expose only its four training GPUs'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
    rows=[[x.strip() for x in r] for r in csv.reader(io.StringIO(raw))]
    physical={int(r[0]):r[1] for r in rows}
    assert set(assigned)<=set(physical) and len({physical[i] for i in assigned})==len(assigned)
    raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=30)
    assigned_uuids={physical[i] for i in assigned}
    applications=[[x.strip() for x in r] for r in csv.reader(io.StringIO(raw_apps)) if r]
    occupied=[r for r in applications if r[0] in assigned_uuids]
    assert not occupied, 'Authorized local GPUs are occupied: '+repr(occupied)
    # Check idle assignment first, before creating CUDA contexts or any claim.
    assert torch.cuda.device_count()==len(training)
    actual=cuda_uuids(len(training))
    assert actual==[physical[i] for i in training], 'Actual training CUDA UUID mapping differs from assignment'
    producer=identity(os.getpid())
    proof=dict(accepted=True,profile='common8',run_id=run['run_id'],slurm_job_id=None,
        execution_backend='ssh_local',execution_id=run['execution_id'],node=socket.gethostname(),
        pipeline_identity=producer,training_world_size=len(training),
        visible_devices=list(map(str,assigned)),training_visible_devices=visible,
        actual_allocated_indices=assigned,training_physical_indices=training,
        auxiliary_physical_indices=auxiliary,retriever_physical_indices=retriever,
        probe_physical_index=probe,assigned_gpu_uuids={str(i):physical[i] for i in assigned},
        actual_cuda_runtime_uuids=actual,physical_gpu_inventory=rows,
        assigned_compute_applications=occupied,unassigned_compute_applications=[r for r in applications if r[0] not in assigned_uuids],
        checked_unix=time.time(),scope='actual local PID birth/host and idle assigned GPU UUIDs before project launch; not model/GRPO acceptance')
    target=base/'logs/native-allocation-admission.json'
    with target.open('x') as f:
        json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(target.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    print(json.dumps(admit(p.parse_args().base)))
