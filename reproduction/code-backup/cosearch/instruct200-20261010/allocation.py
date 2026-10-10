"""Bind only this Slurm job's four GPUs and hold per-UUID leases."""
import fcntl,json,os,re,subprocess
from pathlib import Path

def acquire(root):
    root=Path(root);job=os.environ['SLURM_JOB_ID']
    description=subprocess.check_output(['scontrol','show','job',job],text=True)
    match=re.search(r'AllocTRES=(\S+)',description)
    if not match or not re.search(r'(?:^|,)gres/gpu=4(?:,|$)',match[1]):raise RuntimeError('Allocation must contain exactly four GPUs')
    if os.environ.get('SLURM_NNODES')!='1':raise RuntimeError('Single-node allocation required')
    mask=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(mask)!=4 or len(set(mask))!=4:raise RuntimeError('Exactly four distinct assigned GPUs required')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    by_index={index.strip():uuid.strip() for index,uuid in (line.split(',') for line in inventory.splitlines())}
    uuids=[value if value.startswith('GPU-') else by_index[value] for value in mask]
    if len(set(uuids))!=4:raise RuntimeError('Duplicate leased UUID')
    lease_dir=root.parent/'cosearch-gpu-leases';lease_dir.mkdir(exist_ok=True)
    leases=[]
    for uuid in sorted(uuids):
        handle=(lease_dir/(uuid+'.lock')).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(handle)
    mapping={uuid:int(index) for index,uuid in by_index.items() if uuid in uuids}
    os.environ['COSEARCH_LEASED_UUIDS']=','.join(uuids)
    os.environ['COSEARCH_LEASED_UUID_TO_INDEX']=json.dumps(mapping)
    (root/'allocation.json').write_text(json.dumps({'job':job,'uuids':uuids,'mapping':mapping,'slurm':description,'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
    return leases
