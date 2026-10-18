"""Fresh, independently inspected replacement on eight PJ A100 80GB devices."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contract import UPSTREAM, MODEL_REVISION, MANIFEST_SHA, JOB as OLD_JOB, sha, recipe

BASE = '/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
ROOT = BASE + '/official-treegrpo'
JOB = 'zhangjunming-ttt-treegrpo-pj-v3'
MODEL_FILES = {
    'config.json': 'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b',
    'generation_config.json': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'model-00001-of-00002.safetensors': '67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2',
    'model-00002-of-00002.safetensors': 'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d',
    'model.safetensors.index.json': 'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
VERSIONS = {'torch':'2.6.0','vllm':'0.8.5.post1','transformers':'4.51.3',
            'ray':'2.45.0','flash-attn':'2.7.4.post1','tensordict':'0.6.2',
            'datasets':'3.5.0','numpy':'2.2.6','pandas':'3.0.6'}


def validate_spec(plan, script):
    assert plan['protocol'] == 'official_treegrpo_pj_inspected_replacement_v1'
    assert plan['job'] == JOB and len(JOB) <= 32
    assert plan['root'] == ROOT+'/runs/fair-replacement-v3'
    assert plan['upstream_source'] == ROOT+'/deployment-pj-v1/upstream'
    assert plan['upstream_commit'] == UPSTREAM and plan['model_revision'] == MODEL_REVISION
    assert plan['model'] == BASE+'/assets/model' and plan['model_files'] == MODEL_FILES
    assert plan['data'] == BASE+'/common-data' and plan['data_manifest_sha256'] == MANIFEST_SHA
    assert plan['required_versions'] == VERSIONS
    assert plan['allocation'] == dict(gpus=8,cpus=48,memory_gib=512,
                                     machine_type='n2ls.ie.i60b',gpu_model='A100',gpu_memory_gib=80)
    assert plan['old_failure']['job_id'] == '139034'
    assert plan['old_failure']['state'] == 'FAILED' and plan['old_failure']['exit_code'] == '1'
    assert plan['old_failure']['complete_iterations'] == 55 and plan['old_failure']['native_checkpoint_absent']
    assert plan['actual_iterations_expected'] == 179 and plan['no_auto_retry'] is True
    assert plan['intended_node'] is None and plan['no_hostname_affinity'] is True
    previous=plan['placement_recovery']
    assert previous['job']=='zhangjunming-ttt-treegrpo-pj-v1'
    assert previous['uid']=='e078d102-57e4-4ed1-a359-610671e4e42b'
    assert previous['status']=='TERMINATED_NEVER_STARTED_ZERO_GPU_EXECUTION'
    assert previous['allocated_seconds']==0 and previous['entry_absent'] is True
    startup=plan['startup_recovery']
    assert startup['job']=='zhangjunming-ttt-treegrpo-pj-v2'
    assert startup['uid']=='f51ffc58-4fdf-418d-914a-6b8c2220010f'
    assert startup['status']=='FAILED_BEFORE_SCIENTIFIC_TRAINING_DNS_API_CHECK'
    assert startup['scientific_updates']==0 and startup['scientific_worker_not_started'] is True
    expected = [v.replace('='+OLD_JOB,'='+JOB) for v in recipe(
        script,plan['model'],plan['data'],plan['root'],plan['retriever'])]
    assert plan['overrides'] == expected, 'Official scientific recipe changed'
    return Path(plan['root'])


def validate_submission_params(plan, params):
    assert params['job_name']==plan['job']==JOB
    assert not params.get('include'), 'PJ explicit hostname affinity is known not to schedule'
    assert params['num_nodes']==1 and params['gpus_per_task']==8
    assert params['cpus_per_task']==48 and params['memory_per_task']==512
    assert params['retry']==0 and params['no_env'] is True


def validate_gate(plan, submission, gate):
    assert gate['plan_sha256']==submission['plan_sha256']
    actual=gate['actual']
    assert actual['name']==plan['job']==submission['job']==JOB
    assert actual['uid']==submission['uid'] and actual['state']=='RUNNING'
    assert actual['resources']==plan['allocation']
    return actual


def verify(plan):
    src=Path(plan['upstream_source'])
    root=validate_spec(plan,(src/'train_multihopqa_tree_search.sh').read_text())
    assert len(plan['source_files']) == 231
    for n,h in plan['source_files'].items(): assert sha(src/n)==h,n
    for n,h in plan['wrapper_files'].items(): assert sha(Path(plan['wrapper_source'])/n)==h,n
    for n,h in MODEL_FILES.items(): assert sha(Path(plan['model'])/n)==h,n
    data=Path(plan['data']);assert sha(data/'manifest.json')==MANIFEST_SHA
    manifest=json.loads((data/'manifest.json').read_text())
    for split,count in [('train',169028),('validation',512),('final',51713)]:
        item=manifest['splits'][split]
        assert item['rows']==count and sha(data/item['path'])==item['sha256'],split
    assert sys.version_info[:3] == (3,12,9)
    for name,version in VERSIONS.items(): assert metadata.version(name).split('+')[0]==version,name
    for f,h in plan['evidence_files'].items(): assert sha(f)==h,f
    previous=plan['placement_recovery']
    assert sha(previous['receipt'])==previous['receipt_sha256']
    retirement=json.loads(Path(previous['receipt']).read_text())
    for key in ['job','uid','status','allocated_seconds','entry_absent']:
        assert retirement[key]==previous[key],key
    startup=plan['startup_recovery']
    assert sha(startup['receipt'])==startup['receipt_sha256']
    inspected=json.loads(Path(startup['receipt']).read_text())
    for key in ['job','uid','status','scientific_updates','scientific_worker_not_started']:
        assert inspected[key]==startup[key],key
    assert sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256']
    ready=json.loads(Path(plan['retriever_receipt']).read_text());assert ready['url']==plan['retriever']
    assert sha(plan['runtime_preflight'])==plan['runtime_preflight_sha256']
    preflight=json.loads(Path(plan['runtime_preflight']).read_text())
    assert preflight['status']=='PASSED' and preflight['actual_gpu_training'] is False
    assert preflight['observer_parameters_and_adam_bit_exact'] and preflight['observer_rng_unchanged']
    assert preflight['overrides']==plan['overrides'] and preflight['source_files']==plan['source_files']
    assert preflight['model_files']==plan['model_files'] and preflight['wrapper_files']==plan['wrapper_files']
    return root,src


def validate_scheduler(plan,intent,submission,actual):
    assert intent['job']==plan['job']==submission['job']==actual['name']==JOB
    assert intent['plan_sha256']==submission['plan_sha256']
    assert actual['uid']==submission['uid'] and actual['state']=='RUNNING'
    assert actual['resources']==plan['allocation']
    assert not os.environ.get('SLURM_JOB_ID'), 'Never fabricate a Slurm identity on PJ'
    assert os.environ['SEARCH_R1_EXECUTION_BACKEND']=='pj_pt'
    assert os.environ['SEARCH_R1_JOB_ID']==JOB


def run(planfile):
    plan=json.loads(Path(planfile).read_text());root,src=verify(plan)
    intent=json.loads((root/'submission-intent.json').read_text())
    assert intent['plan_sha256']==sha(planfile)
    deadline=time.monotonic()+90
    while not (root/'submission.json').exists():
        if time.monotonic()>=deadline:raise TimeoutError('Submission UID receipt absent; no training started')
        time.sleep(.5)
    submission=json.loads((root/'submission.json').read_text())
    # Compute pods cannot resolve the scheduler API. A single external CPU publisher
    # checks the actual UID/allocation and atomically releases this exact plan.
    gatefile=root/'scheduler-release-once.json';deadline=time.monotonic()+120
    while not gatefile.exists():
        if time.monotonic()>=deadline:raise TimeoutError('External scheduler release absent; no training started')
        time.sleep(.5)
    gate=json.loads(gatefile.read_text());actual=validate_gate(plan,submission,gate)
    validate_scheduler(plan,intent,submission,actual)
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inv)==8 and len({s.split(',')[1].strip() for s in inv})==8
    assert all('A100' in s and float(s.split(',')[3])>=80000 for s in inv)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'Do not preempt other GPU processes'
    assert not (root/'checkpoints').exists()
    with (root/'entry-once.json').open('x') as f:json.dump(dict(at=time.time(),scheduler=actual,gpus=inv,plan_sha256=sha(planfile),fresh_initialization_after_failed_unsaved_run=True),f,indent=2)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(s.split(',')[0].strip() for s in inv)
    os.environ['TREEGRPO_AUDIT_DIR']=str(root/'audit')
    os.environ['PYTHONPATH']=str(src)+':'+plan['wrapper_source']
    sys.path.insert(0,str(src));os.chdir(src)
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(src/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='ppo_trainer',overrides=plan['overrides'])
    (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    import urllib.request
    ready=json.loads(Path(plan['retriever_receipt']).read_text())
    with urllib.request.urlopen(ready['url']+'/health',timeout=20) as r:assert json.load(r)['job']==ready['job']
    import runpy
    sys.argv=['main_ppo_format_ts']+plan['overrides']
    runpy.run_module('verl.trainer.main_ppo_format_ts',run_name='__main__')


if __name__=='__main__':run(sys.argv[1])
