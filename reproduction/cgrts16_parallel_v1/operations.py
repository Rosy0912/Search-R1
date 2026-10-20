"""Scoped task layout and deterministic launch command; no implicit submission."""
from pathlib import Path
import shlex
import copy
from .contract import read_plan,validate_tasks

def bind_layout(cluster):
 assert not getattr(cluster,'_cgrts16_layout_bound',False)
 original=cluster._build_tasks
 def build(*args,**kwargs):
  # The SDK shares the same resource dict across roles. Copy each task
  # independently so changing worker RAM cannot also change the master.
  tasks=[copy.deepcopy(t) for t in original(*args,**kwargs)]
  assert [(t['role'],t['replicas']) for t in tasks]==[('PYTORCH_MASTER',1),('PYTORCH_WORKER',1)]
  for i,t in enumerate(tasks):
   t['resource_spec']['cpu_count']=64 if i==0 else 48
   t['resource_spec']['memory_gib']=900 if i==0 else 512
  validate_tasks(tasks);return tasks
 cluster._build_tasks=build;cluster._cgrts16_layout_bound=True

def parameters(p,pp):
 q=shlex.quote;root=Path(p['root']);python=p['runtime'];source=p['source'];name=p['job']
 command='\n'.join(['set -euo pipefail','set -o noclobber',
  'export PATH='+q(str(Path(python).parent))+':"$PATH"',
  'export PYTHONPATH='+q(source)+' CGRTS16_PARALLEL_PLAN='+q(str(pp))+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
  'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
  'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1',
  'export TRITON_PTXAS_PATH='+q(p['compiler']['path'])+' TRITON_CACHE_DIR='+q('/tmp/'+name+'-triton'),
  'export HF_HOME='+q(str(root/'cache/hf'))+' XDG_CACHE_HOME='+q(str(root/'cache')),
  'export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn',
  'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS SLURM_JOB_ID SLURM_JOB_NAME CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES',
  'cd '+q(source),'ulimit -n 65535',
  q(python)+' -m reproduction.cgrts16_parallel_v1.node launch '+q(str(pp))+' > '+q(str(root))+'/node-launch-"${HOSTNAME}".log 2>&1'])
 return dict(job_name=name,num_nodes=2,tasks_per_node=1,gpus_per_task=8,cpus_per_task=64,memory_per_task=900,shm_size_gib=32,cluster='cluster-p',queue='queue-p-reserved-llmarchitecture',machine_type='n2ls.ie.i60b',rdma_name='ib-cluster-01',no_env=True,retry=0,priority=1,enable_privileged=False,cmd=command,tmpdir=str(root/'submit'),mount=['PV_OCEANSTOR:019f4bc0-3d47-7cd9-914b-593ab2d69e4c:/plm-dhw'])
