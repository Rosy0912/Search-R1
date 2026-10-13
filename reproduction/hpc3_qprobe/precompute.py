"""One allocated GPU prepares fixed question features; no actor optimization."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args();base=Path(args.base).resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-'):
        raise ValueError('HPC3_QPROBE_REQUIRES_ITS_OWN_REMOTE_NAMESPACE')
    if 'SLURM_JOB_ID' not in os.environ or len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')) != 1:
        raise ValueError('FEATURE_PRECOMPUTE_REQUIRES_ONE_ACTUALLY_ALLOCATED_GPU')
    import torch
    if torch.cuda.device_count()!=1:
        raise ValueError('FEATURE_PRECOMPUTE_VISIBLE_CUDA_COUNT_DIFFERS')
    from reproduction.hpc3_qprobe.model_manifest import create
    from search_g1.qprobe_prepare import precompute
    from search_g1.qprobe_warmup import write_once
    manifest=base/'preparation/official-model-manifest.json'
    model=Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
    create(model, manifest)
    result=precompute(SimpleNamespace(data=str(base/'data'),model=str(model),
        official_manifest=str(manifest),cache_dir=str(base/'question-features'),device='cuda:0',
        batch_size=args.batch_size,feature_layer=27,max_prompt_tokens=4096,prompt_key='prompt'))
    result.update(slurm_job_id=os.environ['SLURM_JOB_ID'],actual_visible_cuda_count=1,
                  completed_training_outers=0,target_training_outers=200,extra_teacher_warmup_outers=1)
    write_once(base/'preparation/features-ready.json', result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
