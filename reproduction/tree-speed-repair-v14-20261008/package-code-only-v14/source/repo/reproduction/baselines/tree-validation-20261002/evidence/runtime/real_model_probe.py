import importlib.util,json,os,pathlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');assets=root/'assets-root'
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')=='h100-9'
s=importlib.util.spec_from_file_location('download_assets',root/'repo/reproduction/h20/download_assets.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
model_assets=[a for a in json.loads((assets/'assets.json').read_text()) if a['path'].startswith('qwen2.5-3b/')]
assert all(h.valid(assets/'assets'/a['path'],a) for a in model_assets),'Immutable model not complete/verified'
import torch
from vllm import LLM,SamplingParams
assert torch.cuda.device_count()==8
t=time.perf_counter();model=LLM(model=str(assets/'assets/qwen2.5-3b'),tensor_parallel_size=1,gpu_memory_utilization=.12,max_model_len=1024,enforce_eager=True)
out=model.generate(['The capital of France is'],SamplingParams(temperature=0,max_tokens=16));tokens=out[0].outputs[0].token_ids
assert tokens
result={'slurm_job_id':os.environ['SLURM_JOB_ID'],'allocated_gpu_count':8,'tp':1,'model_manifest_verified':True,'model_path':str(assets/'assets/qwen2.5-3b'),'generated_token_count':len(tokens),'generated_text':out[0].outputs[0].text,'seconds':time.perf_counter()-t,'production_search_training':False,'torch_gpu0_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'torch_gpu0_peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
(root/'logs/model-acceptance.json').write_text(json.dumps(result,indent=2));print('REAL_MODEL_VLLM_CUDA_OK',json.dumps(result))
