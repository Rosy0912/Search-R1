"""Explicit once-only CPU preparation and held submission; no automatic retry."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from .contract import ROOT,PARENTS,PARENT_SOURCE,job_name,overrides,sha,validate_plan,validate_override_delta


def once(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def exact_zero(size):
    name=job_name(size)
    rows=subprocess.check_output(['sacct','-n','-P','-X','-S','2026-10-18','-u','jmzhang',
                                  '--format=JobID,JobName%90,State'],text=True).splitlines()
    parsed=[r.split('|') for r in rows if r.strip()]
    assert any(r[0]==PARENTS[size][0] for r in parsed), 'Missing scheduler positive control'
    assert not any(r[1]==name for r in parsed), 'Registered name already exists in accounting'
    live=subprocess.check_output(['squeue','-h','-u','jmzhang','-o','%i|%j|%T'],text=True).splitlines()
    assert not any(r.split('|')[1]==name for r in live)
    return dict(at=time.time(),exact_name=name,sacct_exact=0,squeue_exact=0,positive_control=PARENTS[size][0])


def prepare(size,deployment):
    import torch
    from hydra import compose,initialize_config_module
    from omegaconf import OmegaConf
    from reproduction.official_branpo.entry import build_trainer
    from .entry import verify
    assert not torch.cuda.is_initialized()
    base=Path(ROOT)/'receipts/branpo150-v2'
    proof=base/(size+'-native140')/'native140-proof.json'
    local_gate=base/(size+'-native140-local-sha.json')
    gate=json.loads(local_gate.read_text())
    assert gate['local_raw_sha_verified'] is True and gate['proof_sha256']==sha(proof)
    parent_path=Path(deployment['parent_plans'][size]);parent=json.loads(parent_path.read_text())
    assert sha(parent_path)==PARENTS[size][2]
    root=Path(ROOT)/'runs'/(size+'-150-v1')
    assert not root.exists()
    zero=exact_zero(size)
    intent=base/(size+'-prepare-once.json');once(intent,dict(at=time.time(),zero=zero,no_auto_retry=True))
    plan=dict(size=size,job=job_name(size),root=str(root),parent_job_id=PARENTS[size][0],
              parent_plan=str(parent_path),parent_plan_sha256=PARENTS[size][2],parent_adapter_source=PARENT_SOURCE,
              source=deployment['source'],adapter=deployment['adapter'],upstream=deployment['upstream'],
              adapter_files=deployment['adapter_files'],upstream_files=deployment['upstream_files'],
              resume_label=240,final_label=250,actual_updates_target=150,new_updates=10,
              allocation=dict(gpus=8,cpus=64,memory_gib=600),no_auto_retry=True,del_local_ckpt_after_load=False,
              native_proof=str(proof),native_proof_sha256=sha(proof),local_native_gate=str(local_gate),
              local_native_gate_sha256=sha(local_gate),overrides=overrides(parent,size),
              no_claim_of_bitwise_uninterrupted_rollout_trajectory=True)
    validate_plan(plan,parent)
    draft=base/(size+'-plan-draft.json');once(draft,plan)
    verify(draft,check_cpu=False)
    if not OmegaConf.has_resolver('mul'):OmegaConf.register_new_resolver('mul',lambda x,y:int(x)*int(y))
    with initialize_config_module(config_module='rllm.trainer.config',version_base=None):
        cfg=compose(config_name='agent_ppo_trainer',overrides=plan['overrides'])
        previous=compose(config_name='agent_ppo_trainer',overrides=parent['overrides'])
    trainer=build_trainer(parent,cfg);previous_trainer=build_trainer(parent,previous)
    def leaves(value,prefix=''):
        if isinstance(value,dict):
            result={}
            for k,v in value.items():result.update(leaves(v,prefix+'.'+k if prefix else k))
            return result
        return {prefix:value}
    old=leaves(OmegaConf.to_container(previous,resolve=True));new=leaves(OmegaConf.to_container(cfg,resolve=True))
    from .contract import CHANGES
    delta={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    assert delta<=CHANGES, delta
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type=='constant'
    assert cfg.actor_rollout_ref.actor.optim.lr==1e-6
    assert cfg.actor_rollout_ref.actor.checkpoint.load_contents==['model','optimizer','extra']
    assert not torch.cuda.is_initialized()
    cpu=base/(size+'-cpu-preflight.json')
    once(cpu,dict(at=time.time(),status='PASSED',actual_gpu_training=False,overrides=plan['overrides'],
                  resolved_delta=sorted(delta),full_parent_inputs_verified=True,full_native140_files_verified=True,
                  real_official_driver_constructor=True,no_cuda_initialized=True))
    plan.update(cpu_preflight=str(cpu),cpu_preflight_sha256=sha(cpu))
    path=base/(size+'-plan.json');once(path,plan);verify(path)
    result=dict(at=time.time(),size=size,job=plan['job'],plan=str(path),plan_sha256=sha(path),root=str(root),
                source=deployment['source'],native_proof_sha256=sha(proof),status='CPU_PREPARED_NO_GPU_SUBMISSION')
    once(base/(size+'-prepared.json'),result)
    print('PREPARED='+json.dumps(result),flush=True)
    return result


def submit(size,deployment):
    from .entry import verify
    base=Path(ROOT)/'receipts/branpo150-v2';prepared=json.loads((base/(size+'-prepared.json')).read_text())
    plan,parent,proof=verify(prepared['plan']);assert sha(prepared['plan'])==prepared['plan_sha256']
    assert plan['source']==deployment['source'] and plan['size']==size
    zero=exact_zero(size);root=Path(plan['root']);assert not root.exists()
    root.mkdir(exist_ok=False)
    for partition in ('ADA6000','L40S'):
        assert 'MaxTime=7-00:00:00' in subprocess.check_output(['scontrol','show','partition',partition],text=True)
    q=shlex.quote
    lines=['#!/bin/bash','set -euo pipefail',
           'export PATH='+q(ROOT+'/env/bin')+':"$PATH"',
           'export PYTHONPATH='+q(plan['adapter']+':'+plan['upstream'])+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
           'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
           'export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000 VLLM_USE_FLASHINFER_SAMPLER=0',
           'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False',
           'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled',
           'export HF_HOME='+q(ROOT+'/cache/hf')+' XDG_CACHE_HOME='+q(ROOT+'/cache')+' TRITON_CACHE_DIR=/tmp/branpo150-${SLURM_JOB_ID}-triton',
           'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS',
           'cd '+q(plan['upstream']),'ulimit -n 65535','date +%s > '+q(str(root/'start.epoch')),'set +e',
           q(ROOT+'/env/bin/python')+' -m reproduction.branpo150.entry '+q(prepared['plan'])+' > '+q(str(root/'train.log'))+' 2>&1',
           'rc=$?','echo "$rc" > '+q(str(root/'exit.code')),'exit "$rc"']
    script=root/'job.sh';script.open('x').write('\n'.join(lines)+'\n')
    args=['sbatch','--hold','--parsable','--no-requeue','--job-name='+plan['job'],
          '--partition=ADA6000,L40S','--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=64',
          '--mem=600G','--time=7-00:00:00','--output='+str(root/'slurm-%j.out'),
          '--error='+str(root/'slurm-%j.err'),str(script)]
    once(root/'submission-intent.json',dict(at=time.time(),job=plan['job'],plan_sha256=prepared['plan_sha256'],
                                           args=args,zero=zero,no_auto_retry=True))
    raw=subprocess.check_output(args,text=True).strip();jid=raw.split(';')[0];assert jid.isdigit()
    result=dict(at=time.time(),size=size,job=plan['job'],job_id=jid,root=str(root),plan=prepared['plan'],
                plan_sha256=prepared['plan_sha256'],source=deployment['source'],parent_job_id=plan['parent_job_id'],
                inherited_actual_updates=140,new_updates=10,target_actual_updates=150,gpus=8,no_auto_retry=True)
    once(root/'submission.json',result);once(base/(size+'-submission.json'),result)
    row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    assert 'JobName='+plan['job']+' ' in row and 'JobState=PENDING' in row and 'Reason=JobHeldUser' in row
    subprocess.check_call(['scontrol','release',jid])
    print('SUBMISSION='+json.dumps(result),flush=True)
    return result
