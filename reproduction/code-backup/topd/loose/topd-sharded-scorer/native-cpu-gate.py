#!/usr/bin/env python3
"""Own pinned Linux CPU-only23 fixtures; emits no GPU or teacher-quality approval."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
DIRECTORY=Path(__file__).resolve().parent


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def native_preflight():
    require(Path(os.environ.get('AGENT_ROOT','/invalid')).resolve()==ROOT and DIRECTORY==ROOT/'tools/topd-sharded-scorer','Foreign ownROOT/tool location')
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Foreign node')
    job=os.environ.get('SLURM_JOB_ID');require(job and job.isdigit(),'Need actual own Slurm job')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=8 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=16384,'Need actualCPU/RAM allocation')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'CPU gate must have no GPU allocation/visibility')
    paths=[line.split(':',2)[-1] for line in Path('/proc/self/cgroup').read_text().splitlines()]
    require(any('job_'+job in path.split('/') for path in paths),'Outside own job cgroup')
    require(Path(sys.executable).absolute()==ROOT/'env/bin/python','Wrong native interpreter')
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
        require(os.environ.get(key)=='1','Offline environment missing')
    manifest=json.loads((DIRECTORY/'files.json').read_text());dependencies=json.loads((DIRECTORY/'dependencies.json').read_text())
    require(manifest['scope']=='CPU_CONTRACT_PACKAGE_NOT_GPU_READY' and manifest['expected_cpu_fixtures']==23,'Wrong package scope')
    expected={'scorer.py','test_scorer.py','README.md','dependencies.json','native-cpu-gate.py','native-cpu.sbatch','DEPLOYMENT.md'}
    require(set(manifest['files'])==expected,'Incomplete frozen prototype package')
    for name,digest in manifest['files'].items():
        require(sha(DIRECTORY/name)==digest,'Frozen tool bytes changed: '+name)
        print(digest+'  '+str(DIRECTORY/name),flush=True)
    actual_commit=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
    require(actual_commit==dependencies['code_commit'],'Wrong production ancestor commit')
    source={name:sha(ROOT/'repo/search_r1/baselines'/name) for name in dependencies['source_files']}
    fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    require(source==dependencies['source_files'] and fingerprint==dependencies['source_fingerprint'],'Unchanged original scoring source differs')
    versions=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}')
    for name in ('torch','transformers','accelerate','tensordict'):versions[name]=importlib.import_module(name).__version__
    require(versions==dependencies['versions'],'Native dependency version changed')
    for name,digest in dependencies['installed_sources'].items():
        module=importlib.import_module(name)
        require(Path(module.__file__).resolve().is_relative_to(ROOT/'env'),'Foreign installed library location')
        require(sha(module.__file__)==digest,'Installed dispatch/hook source changed')
        print(digest+'  '+module.__file__,flush=True)
    from search_r1.baselines.distillation import FrozenTeacher
    require(hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest()==dependencies['original_method_sha256'],'Original scoring method differs')
    import torch
    require(not torch.cuda.is_initialized(),'CPU gate started CUDA context')
    return manifest,dependencies,versions,paths


class Collection:
    collected=None
    passed=0
    skipped=0
    failed=0
    def pytest_runtest_logreport(self,report):
        if report.passed and report.when=='call':self.passed+=1
        if report.skipped:self.skipped+=1
        if report.failed:self.failed+=1
    def pytest_collection_finish(self,session):self.collected=len(session.items)


def cpu_hook_control():
    """Actual pinnedAccelerate allCPU hook/remove check; not multiGPU proof."""
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from accelerate import dispatch_model
    from accelerate.hooks import remove_hook_from_module
    from search_r1.baselines.distillation import FrozenTeacher
    from types import SimpleNamespace
    torch.manual_seed(2200);torch.set_num_threads(1)
    config=Qwen2Config(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,pad_token_id=0)
    config._attn_implementation='eager'
    teacher=FrozenTeacher.__new__(FrozenTeacher);teacher.tokenizer=SimpleNamespace(pad_token_id=0)
    teacher.model=Qwen2ForCausalLM(config).float().eval().requires_grad_(False);teacher.device=torch.device('cpu');teacher.scored_tokens=0
    prompts=torch.tensor([[1,2,3],[0,4,5]]);responses=torch.tensor([[6,7,0],[8,9,0]]);mask=torch.tensor([[1,1,0],[1,1,0]])
    before=FrozenTeacher.log_probs(teacher,prompts,responses,mask,temperature=1.,targets=None,chunk=16)
    dispatch_model(teacher.model,{'':'cpu'},main_device='cpu',force_hooks=True)
    require(hasattr(teacher.model,'_hf_hook'),'ActualCPU dispatch did not install hook')
    hooked=FrozenTeacher.log_probs(teacher,prompts,responses,mask,temperature=1.,targets=None,chunk=16)
    require(torch.equal(before,hooked),'CPU dispatch hook changed unchanged original output')
    remove_hook_from_module(teacher.model,recurse=True);teacher.model.to('cpu')
    require(not any(hasattr(module,'_hf_hook') or hasattr(module,'_old_forward') for module in teacher.model.modules()),'Hook remained after actual removal')
    after=FrozenTeacher.log_probs(teacher,prompts,responses,mask,temperature=1.,targets=None,chunk=16)
    require(torch.equal(before,after) and all(p.device.type=='cpu' for p in teacher.model.parameters()) and all(b.device.type=='cpu' for b in teacher.model.buffers()),'Hook removal changed output or retained nonCPU state')
    require(not torch.cuda.is_initialized(),'ActualCPU hook control started CUDA')
    return dict(scope='ACTUAL_PINNED_ACCELERATE_SINGLE_CPU_HOOK_CONTROL_ONLY',installed_and_removed=True,before_hook_after_torch_equal=True,cuda_initialized=False)


def main():
    started=time.time();manifest,dependencies,versions,paths=native_preflight()
    import pytest,torch
    plugin=Collection();exitcode=pytest.main([str(DIRECTORY/'test_scorer.py'),'-q','-ra'],plugins=[plugin])
    require(exitcode==0 and plugin.collected==plugin.passed==23 and plugin.skipped==plugin.failed==0,'Native23CPU suite failed or collection changed')
    hook_control=cpu_hook_control()
    require(not torch.cuda.is_initialized(),'Fixtures/hook control initialized CUDA')
    record=dict(status='NATIVE_CPU_CONTRACTS_PASS_NOT_GPU_READY',scope='CPU_ONLY_NO_TRAINING_NO_TEACHER_QUALITY_APPROVAL',job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),cgroup_paths=paths,
                runtime=versions,collected=plugin.collected,passed=plugin.passed,skipped=plugin.skipped,failed=plugin.failed,pytest_exit_code=int(exitcode),cpu_hook_control=hook_control,
                cuda_initialized=False,code_commit=dependencies['code_commit'],source_fingerprint=dependencies['source_fingerprint'],original_method_sha256=dependencies['original_method_sha256'],
                package_manifest_sha256=sha(DIRECTORY/'files.json'),files=manifest['files'],started_epoch=started,completed_epoch=time.time())
    output=ROOT/'logs'/f"topd-sharded-scorer-native-cpu-{record['job']}.json"
    with output.open('x') as stream:json.dump(record,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(record,sort_keys=True),flush=True)


if __name__=='__main__':main()
