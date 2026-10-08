import json,os,pathlib,torch
from flash_attn import flash_attn_func
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')=='h100-9'
assert torch.cuda.device_count()==8
torch.manual_seed(42);torch.cuda.set_device(0)
q,k,v=[torch.randn(2,128,2,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
y=flash_attn_func(q,k,v,dropout_p=0.,causal=False)
ref=((q.transpose(1,2).float()@k.transpose(1,2).float().transpose(-2,-1))/8).softmax(-1)@v.transpose(1,2).float();ref=ref.transpose(1,2)
error=(y.float()-ref).abs().max().item()
assert torch.isfinite(y).all() and error<.03,(y.shape,error)
record={'slurm_job_id':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURM_JOB_NODELIST'],'gpu_count':8,'torch':torch.__version__,'flash_attention_bf16_max_error_vs_fp32_math':error,'flash_attention_passed':True,'nccl_bf16_evidence':'hardware-cuda.log','production_model_inference':False,'gpu_names':[torch.cuda.get_device_name(i) for i in range(8)]}
pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/hardware-acceptance.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
