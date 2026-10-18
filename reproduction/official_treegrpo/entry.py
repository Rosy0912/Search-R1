"""Unique Slurm entry: verify immutable inputs, then call the upstream trainer."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from .contract import sha,JOB,MANIFEST_SHA

def verify(plan):
    root=Path(plan['root']);src=Path(plan['upstream_source'])
    assert plan['job']==JOB and plan['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    for name,h in plan['source_files'].items():assert sha(src/name)==h,name
    for name,h in plan['wrapper_files'].items():assert sha(Path(plan['wrapper_source'])/name)==h,name
    for name,h in plan['model_files'].items():assert sha(Path(plan['model'])/name)==h,name
    data=Path(plan['data']);assert sha(data/'manifest.json')==MANIFEST_SHA
    manifest=json.loads((data/'manifest.json').read_text())
    for split,count in [('train',169028),('validation',512),('final',51713)]:
        s=manifest['splits'][split];assert s['rows']==count and sha(data/s['path'])==s['sha256']
    import importlib.metadata as m
    for name,version in plan['required_versions'].items():assert m.version(name).split('+')[0]==version,(name,m.version(name))
    return root,src

def run(planfile):
    plan=json.loads(Path(planfile).read_text());root,src=verify(plan)
    sub=json.loads((root/'submission.json').read_text());jid=os.environ['SLURM_JOB_ID']
    assert sub['job_id']==jid and sub['job']==JOB and sub['plan_sha256']==sha(planfile)
    row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    assert 'JobName='+JOB+' ' in row and 'NodeList=node10 ' in row and 'gres/gpu=8' in row
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines()
    assert len(inv)==8 and len({r.split(',')[0] for r in inv})==8 and all('L40S' in r for r in inv),inv
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    assert not processes,'Unexpected GPU processes; do not preempt'
    with (root/'entry-once.json').open('x') as f:json.dump(dict(job_id=jid,at=time.time(),gpus=inv,plan_sha256=sha(planfile)),f)
    os.environ['TREEGRPO_AUDIT_DIR']=str(root/'audit')
    os.chdir(src);sys.path.insert(0,str(src))
    # Ray children inherit PYTHONPATH; neither live study code nor environments are modified.
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(src/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='ppo_trainer',overrides=plan['overrides'])
    (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    import urllib.request
    req=urllib.request.Request(plan['retriever']+'/retrieve',data=json.dumps({'queries':['What is the capital of France?'],'topk':3,'return_scores':True}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:probe=json.load(r)
    assert isinstance(probe,dict) and len(probe['result'])==1 and len(probe['result'][0])==3
    (root/'retriever-probe.json').write_text(json.dumps(probe))
    sys.argv=['main_ppo_format_ts']+plan['overrides']
    runpy.run_module('verl.trainer.main_ppo_format_ts',run_name='__main__')

if __name__=='__main__':run(sys.argv[1])
