"""Allocation policy only; no arithmetic/backend/precision change."""
import os,sys
NAME='PYTORCH_CUDA_ALLOC_CONF'
VALUE='backend:native,max_split_size_mb:128'
POLICY={'environment_variable':NAME,'value':VALUE,'backend':'native'}
def bootstrap():
 if 'torch' in sys.modules:raise RuntimeError('Allocator must be configured before importing Torch')
 if NAME in os.environ and os.environ[NAME]!=VALUE:raise RuntimeError('Unexpected CUDA allocator environment')
 os.environ[NAME]=VALUE
 return VALUE
def check_config(c):
 if c.get('cuda_allocator')!=POLICY:raise ValueError('Exact immutable native split allocator policy required')
def check_environment():
 if os.environ.get(NAME)!=VALUE:raise RuntimeError('Actual process CUDA allocator environment differs')
 return os.environ[NAME]
def worker_environment(environment):
 if environment.get(NAME)!=VALUE:raise RuntimeError('Actual Ray allocator environment absent or different')
 keys=('LD_PRELOAD','LD_LIBRARY_PATH','CC','CXX','CUDA_HOME','CPATH','PYTHONPATH','R1_RESOURCE_AUDIT_DIRECTORY','SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','SEARCH_R1_GPU_LEASES','SEARCH_R1_GPU_PREFLIGHT','VLLM_CACHE_ROOT','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','VLLM_WORKER_MULTIPROC_METHOD','HF_HOME','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR',NAME)
 return {k:v for k,v in environment.items() if k in keys or k.startswith(('HF_','TRANSFORMERS_'))}

def evidence(torch_module):
 # Original native allocator snapshot only; no new tensor allocation.
 snapshot=torch_module.cuda.memory._snapshot()
 return {'environment':os.environ.get(NAME),'backend':torch_module.cuda.memory.get_allocator_backend(),'original_snapshot':snapshot,'segment_count':len(snapshot['segments'])}
def check_evidence(value):
 if value.get('environment')!=VALUE or value.get('backend')!='native':raise ValueError('Actual worker native allocator environment/backend differs')
 snapshot=value['original_snapshot'];segments=snapshot['segments']
 if snapshot.get('allocator_settings',{}).get('max_split_size')!=134217728:raise ValueError('Original allocator parsed max_split_size differs')
 if not segments or value['segment_count']!=len(segments):raise ValueError('Actual allocator snapshot empty/count differs')
 return value
