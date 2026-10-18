"""Independent80GB PJ replacement of the failed unsaved official7B Tree run."""
import importlib.metadata as metadata
import json,os,runpy,subprocess,sys,time
from pathlib import Path
from reproduction.official_treegrpo.contract import UPSTREAM,MANIFEST_SHA,JOB as TEMPLATE_JOB,recipe,sha
from reproduction.official_treegrpo.pj_entry import VERSIONS,COMPILER
from .contract import models,REVISION
BASE='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-20261019-v1'
COMMON='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
JOB='zhangjunming-7b-treegrpo-pj-v1'
ALLOCATION=dict(gpus=8,cpus=64,memory_gib=600,machine_type='n2ls.ie.i60b',gpu_model='A100',gpu_memory_gib=80)
FAILED_PLAN_SHA='07ba21cc570aa1caedbb3f085aef4764898116abdbf61e52e4f3054650e9f942'

def scientific_overrides(script,plan):
    return [s.replace('='+TEMPLATE_JOB,'='+JOB) for s in recipe(script,plan['model'],plan['data'],plan['root'],plan['retriever'])]

def validate_spec(plan,script,old):
    assert plan['protocol']=='official_treegrpo_7b_pj_replacement_v1' and plan['job']==JOB and len(JOB)<=32
    assert plan['root']==BASE+'/runs/treegrpo-v1' and plan['model']==BASE+'/models/base/'+REVISION
    assert plan['model_revision']==REVISION and plan['model_files']=={f['file']:f['sha256'] for f in models()['base']['files']}
    assert plan['data']==COMMON+'/common-data' and plan['data_manifest_sha256']==MANIFEST_SHA
    assert plan['upstream_source']==COMMON+'/official-treegrpo/deployment-pj-v1/upstream' and plan['upstream_commit']==UPSTREAM
    assert plan['allocation']==ALLOCATION and plan['required_versions']==VERSIONS and plan['compiler']==COMPILER
    assert plan['no_hostname_affinity'] is True and plan['no_auto_retry'] is True and plan['actual_iterations_expected']==179
    failure=plan['old_failure'];assert failure['job']=='139102' and failure['completed_iterations']==9 and failure['native_checkpoint_absent'] and failure['status']=='FAILED_CUDA_OOM_9_COMPLETE_PARTIAL10_NO_NATIVE'
    assert plan['old_plan_sha256']==FAILED_PLAN_SHA
    assert old['kind']=='treegrpo' and old['job']=='zhangjunming-7b-treegrpo-0' and old['model_revision']==REVISION
    assert old['model_files']==plan['model_files'] and old['required_versions']=={k:VERSIONS[k] for k in ['torch','vllm','transformers','ray','flash-attn']}
    assert plan['overrides']==scientific_overrides(script,plan),'Official recipe differs'
    old_expected=[s.replace('='+TEMPLATE_JOB,'='+old['job']) for s in recipe(script,old['model'],old['data_root'],old['root'],old['retriever'])]
    assert old['overrides']==old_expected,'Failed original was not the same official scientific recipe'
    return Path(plan['root'])

def validate_params(plan,params):
    assert params['job_name']==JOB and params['num_nodes']==1 and params['gpus_per_task']==8
    assert params['cpus_per_task']==64 and params['memory_per_task']==600
    assert params['retry']==0 and params['no_env'] is True and not params.get('include')

def validate_gate(plan,submission,gate):
    assert gate['plan_sha256']==submission['plan_sha256']
    a=gate['actual'];assert a['name']==plan['job']==submission['job']==JOB and a['uid']==submission['uid'] and a['state']=='RUNNING' and a['resources']==ALLOCATION
    return a

def verify(plan):
    assert sha(plan['old_plan'])==FAILED_PLAN_SHA;old=json.loads(Path(plan['old_plan']).read_text())
    src=Path(plan['upstream_source']);root=validate_spec(plan,(src/'train_multihopqa_tree_search.sh').read_text(),old)
    assert len(plan['source_files'])==231
    for n,h in plan['source_files'].items():assert sha(src/n)==h,n
    for n,h in plan['wrapper_files'].items():assert sha(Path(plan['wrapper_source'])/n)==h,n
    for n,h in plan['model_files'].items():assert sha(Path(plan['model'])/n)==h,n
    c=json.loads((Path(plan['model'])/'config.json').read_text());assert (c['model_type'],c['hidden_size'],c['num_hidden_layers'],c['num_attention_heads'],c['num_key_value_heads'])==('qwen2',3584,28,28,4)
    data=Path(plan['data']);assert sha(data/'manifest.json')==MANIFEST_SHA;manifest=json.loads((data/'manifest.json').read_text())
    for split,count in [('train',169028),('validation',512),('final',51713)]:
        s=manifest['splits'][split];assert s['rows']==count and sha(data/s['path'])==s['sha256']
    assert sys.version_info[:3]==(3,12,9)
    for n,v in VERSIONS.items():assert metadata.version(n).split('+')[0]==v,n
    assert sha(COMPILER['path'])==COMPILER['sha256'] and 'release 12.4,' in subprocess.check_output([COMPILER['path'],'--version'],text=True)
    for n,h in plan['evidence_files'].items():assert sha(n)==h,n
    failure=json.loads(Path(plan['old_failure_receipt']).read_text())
    for k,v in plan['old_failure'].items():assert failure[k]==v,k
    assert sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256'];assert json.loads(Path(plan['retriever_receipt']).read_text())['url']==plan['retriever']
    proof=json.loads(Path(plan['runtime_preflight']).read_text());assert sha(plan['runtime_preflight'])==plan['runtime_preflight_sha256']
    assert proof['status']=='PASSED' and proof['actual_gpu_training'] is False
    assert proof['observer_parameters_and_adam_bit_exact'] and proof['observer_rng_unchanged'] and proof['overrides']==plan['overrides'] and proof['model_files']==plan['model_files'] and proof['source_files']==plan['source_files'] and proof['wrapper_files']==plan['wrapper_files']
    return root,src

def run(planfile):
    plan=json.loads(Path(planfile).read_text());root,src=verify(plan)
    assert not os.environ.get('SLURM_JOB_ID') and os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='pj_pt' and os.environ.get('SEARCH_R1_JOB_ID')==JOB
    intent=json.loads((root/'submission-intent.json').read_text());assert intent['job']==JOB and intent['plan_sha256']==sha(planfile)
    end=time.monotonic()+150
    while not ((root/'submission.json').exists() and (root/'scheduler-release-once.json').exists()):
        if time.monotonic()>=end:raise TimeoutError('No external exact UID release; science not started')
        time.sleep(.5)
    sub=json.loads((root/'submission.json').read_text());gate=json.loads((root/'scheduler-release-once.json').read_text());actual=validate_gate(plan,sub,gate)
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inv)==8 and len({s.split(',')[1].strip() for s in inv})==8 and all('A100' in s and float(s.split(',')[3])>=80000 for s in inv)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    assert not (root/'checkpoints').exists()
    with (root/'entry-once.json').open('x') as f:json.dump(dict(at=time.time(),scheduler=actual,gpus=inv,plan_sha256=sha(planfile),fresh_initialization_after_failed_unsaved_run=True),f,indent=2)
    os.environ.update(TRITON_PTXAS_PATH=COMPILER['path'],CUDA_VISIBLE_DEVICES=','.join(s.split(',')[0].strip() for s in inv),TREEGRPO_AUDIT_DIR=str(root/'audit'),PYTHONPATH=str(src)+':'+plan['wrapper_source'])
    sys.path.insert(0,str(src));os.chdir(src)
    subprocess.run([sys.executable,'-m','reproduction.official_treegrpo.pj_compiler_smoke',str(root/'compiler-smoke.json')],check=True,timeout=120)
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(src/'verl/trainer/config'),version_base=None):cfg=compose(config_name='ppo_trainer',overrides=plan['overrides'])
    (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    import urllib.request
    ready=json.loads(Path(plan['retriever_receipt']).read_text())
    with urllib.request.urlopen(ready['url']+'/health',timeout=20) as r:assert json.load(r)['job']==ready['job']
    import ray
    assert not ray.is_initialized();ray.init(runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN','TRITON_PTXAS_PATH':COMPILER['path']}})
    sys.argv=['main_ppo_format_ts']+plan['overrides'];runpy.run_module('verl.trainer.main_ppo_format_ts',run_name='__main__')

if __name__=='__main__':run(sys.argv[1])
