"""Read-only readiness by default; --execute submits the fixed three slots once."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from reproduction.official_eval_v2.driver import once
from .contract import ROOT,BASE,RUNTIME,UID,TRAIN_SOURCE,TRAIN_PLAN,sha,verify,job_name
from .scheduler import cluster,exact,zero


def command(plan,path):
    root=Path(plan['root']);q=shlex.quote
    runtime=str(Path(RUNTIME).parent.parent)
    return '\n'.join([
      'set -euo pipefail',
      'export PATH='+q(str(Path(RUNTIME).parent))+':"$PATH"',
      'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
      'export SEARCH_R1_EXECUTION_BACKEND=pj_pt SEARCH_R1_JOB_ID='+q(plan['job'])+' RECURSIVE_SOURCE_COMMIT='+q(plan['source_commit'])+' PYTHONPATH='+q(plan['source']),
      'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0',
      'export VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0',
      'export TRITON_PTXAS_PATH='+q(runtime+'/cuda12/bin/ptxas')+' TRITON_CACHE_DIR=/tmp/'+plan['job']+'-triton',
      'export HF_HOME='+q(str(BASE/'cache/hf'))+' XDG_CACHE_HOME='+q(str(BASE/'cache')),
      'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS SLURM_JOB_ID CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES',
      'cd '+q(plan['source']),'date +%s > '+q(str(root/'start.epoch')),'set +e',
      q(RUNTIME)+' -m reproduction.trained_tree7b_v2.launch --plan '+q(str(path))+' > '+q(str(root/'worker.log'))+' 2>&1',
      'rc=$?','echo "$rc" > '+q(str(root/'exit.code')),'exit "$rc"'])


def params(plan,path):
    return dict(job_name=plan['job'],cluster='cluster-p',queue='queue-p-reserved-llmarchitecture',machine_type='n2ls.ie.i60b',
                num_nodes=1,gpus_per_task=8,cpus_per_task=64,memory_per_task=600,shm_size_gib=32,rdma_name='ib-cluster-01',
                priority=1,retry=0,no_env=True,enable_privileged=False,cmd=command(plan,path),tmpdir=str(ROOT/('submit-r'+str(plan['repeat']))))


def run(execute=False):
    if not (ROOT/'prepared.json').exists():
        print(json.dumps(dict(status='WAIT_FINAL_NATIVE_HF_RELOAD_AND_FULL51713_CPU_PREPARED',no_gpu_action=True)));return
    prepared=json.loads((ROOT/'prepared.json').read_text())
    assert prepared['status']=='TRAINED7B_TREE_COMMONV2_THREE_PLANS_CPU_PREPARED_NO_SUBMISSION' and prepared['uid']==UID
    assert len(prepared['plans'])==3
    c=cluster();control=exact(c.client,'zhangjunming-ttt-b20-150-r1-54c0')
    assert len(control)==1 and control[0]['uid']=='33ac6bbf-7e6a-4a66-bcc6-261b66371975'
    training=c.client.get_training_job('zhangjunming-7b-treegrpo-pj-v1')
    assert training['uid']==UID and training['status']['state']=='COMPLETED'
    training_root=BASE/'runs/treegrpo-v1'
    sub=json.loads((training_root/'submission.json').read_text())
    assert sub['source']==TRAIN_SOURCE and sub['plan_sha256']==TRAIN_PLAN
    assert (training_root/'exit.code').read_text().strip()=='0'
    plans=[]
    for index,item in enumerate(prepared['plans'],1):
        path=Path(item['plan']);assert sha(path)==item['plan_sha256']
        plan=json.loads(path.read_text());assert plan['repeat']==index and plan['job']==job_name(index)
        _,root=verify(plan);zero(c.client,plan['job'])
        assert not any(root.iterdir()), 'Slot already contains intent/partial; inspect instead of retry'
        plans.append((plan,path))
    assert not (ROOT/'submit-three-once.json').exists(), 'An earlier submission may be uncertain; inspect exact intents/SDK'
    if not execute:
        print(json.dumps(dict(status='READY_FIXED_THREE_EXACT_ZERO_CPU_VERIFIED',no_gpu_action=True)));return
    # Import only after get_cluster initializes this SDK's account/session.
    from clusterx.launcher.ssp.ssp import SSPRunParams
    once(ROOT/'submit-three-once.json',dict(at=time.time(),prepared_sha256=sha(ROOT/'prepared.json'),status='SUBMISSION_STARTED_NO_AUTO_RETRY'))
    receipts=[]
    for plan,path in plans:
        root=Path(plan['root']);zero(c.client,plan['job']);assert not any(root.iterdir())
        p=params(plan,path)
        assert not p.get('include') and p['retry']==0
        once(root/'submission-intent.json',dict(at=time.time(),job=plan['job'],plan_sha256=sha(path),params=p,repeat_index=plan['repeat']))
        c.run(SSPRunParams(**p))
        job=c.client.get_training_job(plan['job']);assert job['name']==plan['job'] and job['uid']
        receipt=dict(at=time.time(),job=plan['job'],uid=job['uid'],source=plan['source_commit'],plan=str(path),
                     plan_sha256=sha(path),root=str(root),repeat_index=plan['repeat'],model_identity=plan['model_identity'])
        once(root/'submission.json',receipt)
        env=dict(os.environ,PYTHONPATH=plan['source'],PYTHONDONTWRITEBYTECODE='1')
        with (root/'publisher.stdout').open('x') as out,(root/'publisher.stderr').open('x') as err:
            child=subprocess.Popen([sys.executable,'-m','reproduction.trained_tree7b_v2.gate',str(path)],env=env,
                cwd=plan['source'],stdout=out,stderr=err,start_new_session=True)
        once(root/'publisher-launch.json',dict(at=time.time(),pid=child.pid,uid=job['uid'],no_auto_restart=True))
        receipts.append(receipt);print('SUBMISSION='+json.dumps(receipt),flush=True)
    once(ROOT/'submissions.json',dict(at=time.time(),submissions=receipts,no_auto_retry=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();run(a.execute)
