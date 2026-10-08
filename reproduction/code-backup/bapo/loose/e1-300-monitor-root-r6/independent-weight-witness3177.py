import pathlib,json,torch,hashlib
from safetensors import safe_open
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004');base=r/'assets/qwen2.5-3b-instruct-aa8e725';ck=r/'runs/bapo-instruct300-formal300-3177/checkpoints/actor/global_step_5';torch.set_num_threads(8);a=json.loads((base/'model.safetensors.index.json').read_text())['weight_map'];b=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map'];ak='model.embed_tokens.weight';bk='lm_head.weight'
with safe_open(base/a[ak],framework='pt',device='cpu')as f:x=f.get_slice(ak)[0:1024]
with safe_open(ck/b[bk],framework='pt',device='cpu')as f:y=f.get_slice(bk)[0:1024]
d=y.double()-x.to(y.dtype).double();assert x.device.type==y.device.type=='cpu'and torch.isfinite(d).all()and d.abs().max()>0
print(json.dumps({'status':'INDEPENDENT_CPU_BASE_CAST_WEIGHT_CHUNK_CHANGED','base_storage_key':ak,'checkpoint_logical_key':bk,'slice':[0,1024],'base_dtype':str(x.dtype),'checkpoint_dtype':str(y.dtype),'max_abs_delta':d.abs().max().item(),'l1_delta':d.abs().sum().item(),'source_tensor_finite':bool(torch.isfinite(x).all()and torch.isfinite(y).all()),'CUDA_VISIBLE_DEVICES':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'sys_prefix':__import__('sys').prefix}),flush=True)
