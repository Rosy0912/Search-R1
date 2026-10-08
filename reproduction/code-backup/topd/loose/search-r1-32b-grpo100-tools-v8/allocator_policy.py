"""Allocation policy only; no arithmetic/backend/precision change."""
import os,sys
NAME='PYTORCH_CUDA_ALLOC_CONF'
VALUE='expandable_segments:True'
POLICY={'environment_variable':NAME,'value':VALUE,'backend':'native'}
def bootstrap():
 if 'torch' in sys.modules:raise RuntimeError('Allocator must be configured before importing Torch')
 if NAME in os.environ and os.environ[NAME]!=VALUE:raise RuntimeError('Unexpected CUDA allocator environment')
 os.environ[NAME]=VALUE
 return VALUE
def check_config(c):
 if c.get('cuda_allocator')!=POLICY:raise ValueError('Exact immutable expandable allocator policy required')
def check_environment():
 if os.environ.get(NAME)!=VALUE:raise RuntimeError('Actual process CUDA allocator environment differs')
 return os.environ[NAME]
def worker_environment(environment):
 if environment.get(NAME)!=VALUE:raise RuntimeError('Actual Ray allocator environment absent or different')
 keys=('CPATH','PYTHONPATH','R1_RESOURCE_AUDIT_DIRECTORY','SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','SEARCH_R1_GPU_LEASES','SEARCH_R1_GPU_PREFLIGHT','VLLM_CACHE_ROOT','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','VLLM_WORKER_MULTIPROC_METHOD','HF_HOME','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR',NAME)
 return {k:v for k,v in environment.items() if k in keys or k.startswith(('HF_','TRANSFORMERS_'))}

def evidence(torch_module):
 # Original allocator metadata only: no tensor allocation or numerical operation.
 snapshot=torch_module.cuda.memory._snapshot();segments=snapshot['segments']
 backend=torch_module.cuda.memory.get_allocator_backend()
 supported=bool(segments) and all(type(s.get('is_expandable')) is bool for s in segments)
 return {'environment':os.environ.get(NAME),'backend':backend,'original_snapshot':snapshot,'snapshot_segments':segments,'segment_count':len(segments),'is_expandable_field_supported':supported,'expandable_segment_count':sum(s.get('is_expandable') is True for s in segments)}
def check_evidence(value):
 if value.get('environment')!=VALUE or value.get('backend')!='native' or value.get('is_expandable_field_supported') is not True:raise ValueError('Actual worker expandable allocator metadata unsupported/different')
 segments=value['snapshot_segments'];snapshot=value['original_snapshot']
 if snapshot.get('segments')!=segments or snapshot.get('allocator_settings',{}).get('expandable_segments') is not True:raise ValueError('Original allocator parsed expandable setting not true')
 if not segments or value['segment_count']!=len(segments) or not all(type(s.get('is_expandable')) is bool for s in segments):raise ValueError('Actual allocator snapshot field/count absent')
 if value['expandable_segment_count']!=sum(s['is_expandable'] for s in segments) or value['expandable_segment_count']<=0:raise ValueError('No actual expandable allocator segment observed')
 return value
