"""Verified 7B training/evaluation allocation; unchanged scientific runners."""
import json
import os
import runpy
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from .contract import ROOT, OLD, PROFILE, verify_model, verify_data, validate_plan, make_config, model_path
from reproduction.seven_dataset.contracts import sha, DATA_SHA


def verify(plan):
    root=validate_plan(plan)
    manifest=Path(plan['source_manifest'])
    assert sha(manifest)==plan['source_manifest_sha256']
    for n,h in json.loads(manifest.read_text()).items():assert sha(Path(plan['source'])/n)==h,n
    assert verify_model(plan['model_label'])==plan['model_files']
    verify_data()
    assert sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256']
    return root


def check_allocation(plan,planfile):
    root=Path(plan['root']);jid=os.environ['SLURM_JOB_ID']
    sub=json.loads((root/'submission.json').read_text())
    assert sub['job_id']==jid and sub['plan_sha256']==sha(planfile) and sub['job']==plan['job']
    row=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    assert 'JobName='+plan['job']+' ' in row and 'UserId=jmzhang(1066)' in row and 'gres/gpu=8' in row
    node=os.environ['SLURM_JOB_NODELIST']
    assert node in ('node8','node9','node10','node11') and (plan['node'] is None or node==plan['node'])
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inv)==8 and len({r.split(',')[0] for r in inv})==8
    assert all(float(r.split(',')[-1])>=44000 for r in inv)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    with (root/'entry-once.json').open('x') as f:json.dump(dict(at=time.time(),job_id=jid,gpus=inv,plan_sha256=sha(planfile)),f)


def run(planfile):
    plan=json.loads(Path(planfile).read_text());root=verify(plan)
    check_allocation(plan,planfile)
    if plan['kind']=='treegrpo':
        from reproduction.official_treegrpo.contract import recipe,JOB
        src=Path(plan['upstream_source'])
        for n,h in plan['upstream_files'].items():assert sha(src/n)==h,n
        expected=[v.replace('='+JOB,'='+plan['job']) for v in recipe(
            (src/'train_multihopqa_tree_search.sh').read_text(),model_path('base'),plan['data_root'],root,plan['retriever'])]
        assert expected==plan['overrides']
        import importlib.metadata as m
        for name,v in plan['required_versions'].items():assert m.version(name).split('+')[0]==v
        os.environ['TREEGRPO_AUDIT_DIR']=str(root/'audit')
        os.chdir(src);sys.path.insert(0,str(src))
        # Ray must import upstream verl, not the recursive-method implementation.
        os.environ['PYTHONPATH']=str(src)+':'+plan['source']
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(src/'verl/trainer/config'),version_base=None):
            cfg=compose(config_name='ppo_trainer',overrides=plan['overrides'])
        (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
        sys.argv=['main_ppo_format_ts']+plan['overrides']
        runpy.run_module('verl.trainer.main_ppo_format_ts',run_name='__main__')
        return
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(plan['config_text']);cfg.baselines.local_runtime_host=socket.gethostname().split('.')[0]
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    if plan['kind'] in ('base','searchr1'):
        import pandas as pd
        from search_r1.baselines.holdout import plain
        from reproduction.seven_dataset.contracts import validate_expected_rows
        expected=[]
        for _,r in pd.read_parquet(Path(plan['data_root'])/'final.parquet').iterrows():
            gold=plain(r['reward_model'])['ground_truth']['target']
            expected.append(dict(question_id=plain(r['extra_info'])['index'],source=r['data_source'],gold_answers=[gold] if isinstance(gold,str) else gold))
        validate_expected_rows(expected)
        if plan['kind']=='searchr1':
            from reproduction.seven_dataset.model_view import validate_view
            validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
        from reproduction.seven_dataset.entry import run_frozen
        run_frozen(plan,planfile,cfg,expected,root)
        return
    run_original(plan,root,cfg)


def run_original(plan,root,cfg):
    from omegaconf import OmegaConf
    # Original method: same resolved scientific recipe and observer environment as 3B.
    out=root/'output';out.mkdir(exist_ok=False)
    config=root/'resolved.yaml';config.write_text(OmegaConf.to_yaml(cfg,resolve=True));h=sha(config)
    os.environ.update(SEARCH_R1_BASELINE_RUN_ID=plan['evaluation_run_id'],E1_RUN_ID=plan['evaluation_run_id'],
        SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',SEARCH_R1_BASELINE_COMMIT=plan['evaluation_code_commit'],
        SEARCH_R1_BASELINE_CONFIG_SHA256=h,SEARCH_R1_BASELINE_DATA_IDENTITY=DATA_SHA,
        SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-7B-Instruct@'+plan['model_revision'],
        SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(out/'trajectories'),
        TREE_COMMON_DATA=plan['data_root'],TREE_INSTRUCT_MODEL=plan['model'],RECURSIVE_A_LUMIA_FRESH_PROFILE='1',
        RECURSIVE_A_OMIT_FULL_BETA='1',RECURSIVE_A_GUARD_ROUND_PROGRESS='1',TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',
        RECURSIVE_A_SPARSE_DETAILS='1',RECURSIVE_LUMIA_PAIRED_EVALUATION='1')
    from reproduction.seven_dataset.health import wait_retriever_health
    rr=json.loads(Path(plan['retriever_receipt']).read_text());wait_retriever_health(rr['url'],rr['job'])
    from http.server import ThreadingHTTPServer
    from reproduction.recursive_lumia.proxy import Handler
    server=ThreadingHTTPServer(('127.0.0.1',29146),Handler);server.remote=rr['url']
    threading.Thread(target=server.serve_forever,daemon=True).start()
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    ctx=ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,object_store_memory=8*1024**3,
        _node_ip_address='127.0.0.1',_temp_dir='/tmp/7b-'+os.environ['SLURM_JOB_ID'])
    os.environ['RAY_ADDRESS']=ctx.address_info['gcs_address']
    (root/'train-start.json').write_text(json.dumps(dict(at=time.time(),job=os.environ['SLURM_JOB_ID'],run_id=plan['evaluation_run_id'],config_sha256=h)))
    from search_r1.baselines.main import main
    try:main.__wrapped__(cfg)
    finally:ray.shutdown();server.shutdown()
    (root/'training-returned.json').write_text(json.dumps(dict(at=time.time(),job=os.environ['SLURM_JOB_ID'])))


if __name__=='__main__':run(sys.argv[1])
