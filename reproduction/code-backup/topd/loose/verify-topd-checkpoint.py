#!/usr/bin/env python3
"""Independent CPU HF load/forward/semantic comparison; never trains."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','checkpoint','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Must hide CUDA for independent CPU verification'
    path=Path(__file__).with_name('verify-topd-teacher.py')
    spec=importlib.util.spec_from_file_location('topd_teacher_gate',path)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    base,checkpoint=Path(args.base),Path(args.checkpoint)
    semantic=gate.compare_weights(base,checkpoint)
    tokenizer_result=gate.tokenizers(base,checkpoint,semantic['output_vocab_size'])
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().requires_grad_(False).to('cpu')
    ids=tokenizer('What is the capital of France?',return_tensors='pt')
    with torch.no_grad():
        logits=model(**ids,use_cache=False).logits
    assert torch.isfinite(logits).all() and logits.shape[-1]==semantic['output_vocab_size']
    result=dict(status='PASS',scope='CPU HF full load + finite forward + base semantic update; independent search inference still required',
        checkpoint=str(checkpoint.resolve()),semantic_weights=semantic,tokenizer=tokenizer_result,
        hf_files={str(p.relative_to(checkpoint)):gate.digest_file(p) for p in checkpoint.rglob('*') if p.is_file()},
        cpu_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),cpu_threads=torch.get_num_threads())
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('TOPD_CHECKPOINT_CPU_LOAD_FORWARD_SEMANTIC_PASS')


if __name__=='__main__':
    main()
