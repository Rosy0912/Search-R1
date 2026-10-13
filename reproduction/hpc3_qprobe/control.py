"""Once-only HPC3 stage submission and read-only reuse of existing assets."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REUSED = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
DATA = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')


def own_base(value):
    base=Path(value).resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-'):
        raise ValueError('HPC3_QPROBE_REQUIRES_ITS_OWN_OUTPUT_NAMESPACE')
    return base


def publish(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def link(target,source):
    if target.is_symlink():
        if target.resolve()!=source.resolve():raise ValueError('Existing asset link differs: '+str(target))
    elif target.exists():raise ValueError('Existing asset path is not a read-only link: '+str(target))
    else:target.symlink_to(source)


def prepare(base):
    base=own_base(base)
    for name in ('logs','preparation','data','tmp','cache/hf','env/bin','env/lib/python3.10'):
        (base/name).mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet',
                 'final.parquet','exclusions-audit.json'):
        source=DATA/name
        if not source.is_file():raise FileNotFoundError(str(source))
        link(base/'data'/name,source)
    site=REUSED/'source/env/lib/python3.10/site-packages'
    binary=REUSED/'runtime/retriever-python310-candidate/bin/python3.10'
    if not site.is_dir() or not binary.is_file():raise ValueError('Verified reusable Python runtime unavailable')
    link(base/'env/lib/python3.10/site-packages',site)
    wrapper='#!/usr/bin/env bash\nset -euo pipefail\n'
    wrapper+='export PYTHONPATH="${QPROBE_SOURCE:-'+str(base/'repo')+'}:'+str(REUSED/'runtime/gpu-faiss-alignment-candidate')+':'+str(site)+'${PYTHONPATH:+:$PYTHONPATH}"\n'
    wrapper+='export LD_LIBRARY_PATH='+shlex.quote(str(REUSED/'runtime/retriever-python310-candidate/lib'))+'\n'
    wrapper+='unset PYTHONHOME\nexec '+shlex.quote(str(binary))+' "$@"\n'
    python=base/'env/bin/python'
    if python.exists():
        if python.read_text()!=wrapper:raise ValueError('Existing runtime wrapper differs')
    else:
        with python.open('x') as f:f.write(wrapper);f.flush();os.fsync(f.fileno())
        python.chmod(0o755)
    return dict(base=str(base),data_source=str(DATA),python=str(python),
        model_source='/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725',
        output_resolves_to_remote=True,existing_experiments_modified=False)


def submit_features(base,commit):
    base=own_base(base);prepare(base)
    source=base/'source'/commit
    verification=json.loads((base/'preparation'/('source-'+commit+'-verification.json')).read_text())
    if verification['commit']!=commit or verification['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':
        raise ValueError('Feature source publication/deployment not accepted')
    receipt=base/'preparation/features-submission.json'
    intent=base/'preparation/features-submit-claim.json'
    if intent.exists() or receipt.exists():
        raise ValueError('Feature submission already attempted; inspect actual job instead of replaying')
    command=['sbatch','--parsable','--partition=acd_u','--nodes=1','--ntasks=1',
        '--gres=gpu:1','--cpus-per-task=8','--mem=64G','--time=06:00:00',
        '--job-name=qprobe-features200','--export=NONE',
        '--output='+str(base/'logs/features-%j.log'),'--error='+str(base/'logs/features-%j.log'),
        str(source/'reproduction/hpc3_qprobe/features.sh'),str(base),str(source)]
    publish(intent,dict(commit=commit,command=command,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    if result.returncode==0:
        job=result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise ValueError('Slurm returned an invalid feature job identity')
        value['job']=job
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Feature submission failed; inspect saved evidence')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit-features'))
    p.add_argument('--base',required=True);p.add_argument('--commit')
    a=p.parse_args()
    result=prepare(a.base) if a.mode=='prepare' else submit_features(a.base,a.commit)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
