"""Real single-GPU ABI/kernel/encoder/vLLM smoke, with no actor optimization.

This isolated debug output cannot substitute for teacher search trajectories,
eight-rank admission or any of the 200 successful formal training outers.
"""
import argparse
import gc
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    from reproduction.hpc3_qprobe.control import own_base
    base=own_base(a.base);job=os.environ['SLURM_JOB_ID']
    root=base/'preparation'/('debug-'+job);root.mkdir(exist_ok=False)
    from search_g1.qprobe_warmup import write_once
    from reproduction.common8.native_allocation import fields,cuda_uuids
    detail=subprocess.check_output(['scontrol','show','job','-d',job],text=True)
    actual=fields(detail);node=socket.gethostname().split('.')[0]
    if actual['JobState']!='RUNNING' or actual['NodeList']!=node or 'gres/gpu=1' not in actual.get('TRES','').split(','):
        raise ValueError('DEBUG_REQUIRES_ACTUAL_SINGLE_GPU_ALLOCATION')
    line=next((x for x in detail.splitlines() if 'Nodes='+node+' ' in x),'')
    indices=re.search(r'GRES=gpu:1\(IDX:(\d+)\)',line)
    if not indices:raise ValueError('DEBUG_ACTUAL_PHYSICAL_GPU_INDEX_MISSING')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    physical={x.split(',')[0].strip():x.split(',')[1].strip() for x in rows.splitlines()}
    expected=physical[indices.group(1)]
    if (physical.get(visible,visible))!=expected:raise ValueError('DEBUG_VISIBLE_ALLOCATION_MAPPING_DIFFERS')
    import torch
    if torch.cuda.device_count()!=1:raise ValueError('DEBUG_CUDA_VISIBILITY_DIFFERS')
    actual_uuids=cuda_uuids(1)
    if actual_uuids!=[expected]:raise ValueError('DEBUG_ACTUAL_CUDA_UUID_DIFFERS')
    proof=dict(slurm_job_id=job,node=node,allocation_detail=detail,cuda_uuids=actual_uuids,
        formal_completed_outers=0,teacher_search_trajectories=0,checked_unix=time.time())
    def phase(name,**value):
        write_once(root/(name+'.json'),dict(proof,phase=name,**value))
        print(json.dumps(dict(stage='qprobe_single_gpu_debug',phase=name,**value)),flush=True)
    phase('allocation')
    z=torch.arange(1024,device='cuda',dtype=torch.float32)/1000
    def calculation(x):return x.sin()+x.cos()
    compiled=torch.compile(calculation,fullgraph=True)(z)
    torch.cuda.synchronize()
    if not torch.allclose(compiled,calculation(z),rtol=1e-5,atol=1e-5):raise ValueError('CUDA_JIT_DIFFERS')
    from flash_attn import flash_attn_func
    q=torch.randn(2,32,4,64,device='cuda',dtype=torch.bfloat16)
    attention=flash_attn_func(q,q,q,causal=True)
    if not torch.isfinite(attention).all():raise ValueError('FLASH_ATTENTION_NONFINITE')
    phase('kernels',torch_version=torch.__version__,cuda_version=torch.version.cuda,cuda_jit=True,flash_attention=True)
    del q,z,compiled,attention
    model_path=Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
    from reproduction.hpc3_qprobe.model_manifest import create
    create(model_path,root/'official-model-manifest.json')
    from transformers import AutoModel,AutoTokenizer
    from search_g1.qprobe_features import extract_question_features
    model=AutoModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    questions=['What is the capital of France?','Who wrote Pride and Prejudice?']
    features=extract_question_features(model,tokenizer,questions,feature_layer=27,device='cuda')
    if tuple(features.shape)!=(2,2048) or not torch.isfinite(features).all():raise ValueError('ACTUAL_QWEN_FEATURES_INVALID')
    torch.save(features,root/'debug-question-features.pt')
    phase('encoder',actual_questions=2,feature_shape=list(features.shape),feature_finite=True,
          feature_scope='isolated smoke questions, not canonical training feature cache')
    del model,features,tokenizer;gc.collect();torch.cuda.empty_cache()
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(model_path),tokenizer=str(model_path),dtype='bfloat16',
        tensor_parallel_size=1,gpu_memory_utilization=0.25,max_model_len=512,
        enforce_eager=True,disable_custom_all_reduce=True)
    results=llm.generate(questions,SamplingParams(temperature=1.0,top_p=1.0,max_tokens=16,logprobs=1))
    if len(results)!=2 or any(len(x.outputs)!=1 or not x.outputs[0].token_ids for x in results):
        raise ValueError('ACTUAL_VLLM_SMOKE_INCOMPLETE')
    phase('vllm',actual_finished_sequences=len(results),actual_token_counts=[len(x.outputs[0].token_ids) for x in results])
    phase('complete',status='SINGLE_GPU_KERNEL_ENCODER_VLLM_SMOKE_PASS',
          scope='runtime compatibility only; no full feature/warmup/PPO/evaluation acceptance')


if __name__=='__main__':main()
