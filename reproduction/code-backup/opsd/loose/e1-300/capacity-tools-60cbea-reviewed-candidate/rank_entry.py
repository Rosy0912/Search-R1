"""Bind actual worker identity/CUDA UUID before running unchanged pinned probe."""
import ctypes,hashlib,json,os,re,runpy,sys,uuid
from pathlib import Path
repo=Path(os.environ['SEARCH_R1_REPO']);probe=repo/'reproduction/baselines/e1-300/microbatch_probe_e1.py'
assert hashlib.sha256(probe.read_bytes()).hexdigest()=='bbe157e70f959fb654f0d10836d91f32a1c6af0919eacdec8b2caf5c446e9b2b'
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert rank==local and int(os.environ['WORLD_SIZE'])==8
cgroup=Path('/proc/self/cgroup').read_text();assert re.search(r'(?:^|/)job_'+re.escape(os.environ['SLURM_JOB_ID'])+r'(?:/|$)',cgroup,re.M)
import torch
torch.cuda.set_device(local)
torch.cuda.current_stream(local)  # Establish the actual current Torch context.
assert torch.cuda.current_device()==local
lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
device=ctypes.c_int();assert lib.cuCtxGetDevice(ctypes.byref(device))==0
raw=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid_v2(ctypes.byref(raw),device)==0
actual='GPU-'+str(uuid.UUID(bytes=bytes(raw)))
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert actual in {r['uuid'] for r in leases}
record=dict(rank=rank,local_rank=local,world_size=8,pid=os.getpid(),uid=os.getuid(),uids=re.search(r'^Uid:\s+(.*)',Path('/proc/self/status').read_text(),re.M).group(1).split(),start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],job_id=os.environ['SLURM_JOB_ID'],cgroup=cgroup,cuda_device=local,actual_cuda_uuid=actual,source=os.environ['E1_EXPECTED_SOURCE'],probe_path=str(probe),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
with (Path(os.environ['OPSD_EVIDENCE'])/f'worker-identity-{rank}.json').open('x') as f:json.dump(record,f,sort_keys=True)
sys.argv=[str(probe),'--micro','2','--input-width','6144','--response-width','4096','--output-dir',os.environ['OPSD_EVIDENCE']]
runpy.run_path(str(probe),run_name='__main__')
