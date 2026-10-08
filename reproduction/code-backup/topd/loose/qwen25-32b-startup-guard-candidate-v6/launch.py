"""Explicit parent wrapper; sealed supervisor and child files stay unchanged."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import startup_guard

ROOT = Path('/dataset/shaoshitong/search-r1-rl/topd')
TOOLS = ROOT / 'tools/qwen25-32b-terminal-fix-refreeze'
PINS = {'supervisor.py':'d79fc9dc99894ebc62d3a68ac1e12db16d100d6de46428ae84ec995229cb9964',
        'eval64.py':'24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee',
        'runtime_safety.py':startup_guard.C7_SHA,
        'eval64.sbatch':'21aa558e8f6ce0ad3e719b56c91e8c32644add9dc28326400e1d3655e00bf0ba'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def configure_own_cache(root, job_id):
    base = root / 'cache' / ('qwen25-32b-eval64-' + str(job_id))
    result = {}
    for key, directory in [('VLLM_CACHE_ROOT', 'vllm'), ('XDG_CACHE_HOME', 'xdg'),
                           ('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('TRITON_CACHE_DIR', 'triton')]:
        path = base / directory
        require = startup_guard.require
        parents = [path] + [p for p in path.parents if p != root and root in p.parents]
        require(root in path.parents and all(not p.is_symlink() for p in parents),
                'Own cache path symlink/escape')
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(path.stat().st_uid == os.getuid(), 'Foreign cache owner')
        os.environ[key] = str(path)
        result[key] = str(path)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--binding',required=True)
    args=parser.parse_args()
    startup_guard.require(os.environ.get('AGENT_ROOT')==str(ROOT) and os.environ.get('SLURM_JOB_ID'), 'Own Slurm required')
    own=Path(__file__).resolve().parent
    startup_guard.require(own==ROOT/'tools/qwen25-32b-startup-guard-candidate-v6','Unexpected wrapper path')
    manifest=json.loads((own/'files.json').read_text())
    for name,digest in manifest.items():
        startup_guard.require(sha(own/name)==digest,'Wrapper bytes changed: '+name)
    for name,digest in PINS.items():
        startup_guard.require(sha(TOOLS/name)==digest,'Sealed input bytes changed: '+name)
    startup_guard.require(os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')=='spawn','Require explicit spawn')
    cache_env=configure_own_cache(ROOT,os.environ['SLURM_JOB_ID'])
    trace=ROOT/'logs'/('qwen25-32b-startup-guard-'+os.environ['SLURM_JOB_ID']+'.jsonl')
    with trace.open('x') as stream:
        stream.write(json.dumps(dict(scope='PARENT_STARTUP_GUARD_NOT_QUALITY',wrapper_files=manifest,
            sealed_files=PINS,python=sys.executable,cwd=os.getcwd(),
            pythonpath=os.environ.get('PYTHONPATH'),worker_method='spawn',cache_environment=cache_env))+'\n')
    # Explicit module selection; no sitecustomize or hidden source shadow.
    sys.path.insert(0,str(TOOLS))
    load_file('eval64',TOOLS/'eval64.py')
    sealed=load_file('runtime_safety',TOOLS/'runtime_safety.py')
    wrapper=startup_guard.StartupGuard(sealed,expected_path=TOOLS/'runtime_safety.py',trace_path=trace)
    wrapper.closed=True
    state=dict(armed=False,readiness=None)
    original_popen=subprocess.Popen
    def guarded_popen(command,*popen_args,**popen_kwargs):
        if isinstance(command,(list,tuple)) and len(command)>1 and str(command[1])==str(TOOLS/'eval64.py'):
            startup_guard.require(not state['armed'],'Evaluator spawn repeated')
            startup_guard.require('--output' in command,'Missing actual evaluator output path')
            output=Path(command[command.index('--output')+1]).resolve()
            startup_guard.require(output.is_relative_to(ROOT/'logs'),'Evaluator output escaped own logs')
            state.update(armed=True,readiness=output/'readiness.json')
            wrapper.closed=False
            wrapper._write(dict(event='ACTUAL_EVALUATOR_SPAWN_ELIGIBILITY_ARM',deadline_monotonic=None,
                command=list(command),accepted=False))
        return original_popen(command,*popen_args,**popen_kwargs)
    wrapper.ready_check=lambda: state['readiness'] is not None and state['readiness'].is_file()
    def parent_gate(runtime,pre_model=False):
        if state['readiness'] is not None and state['readiness'].is_file():
            wrapper.close_startup()
        return wrapper.active_gate(runtime,pre_model=pre_model)
    sealed.active_gate=parent_gate
    subprocess.Popen=guarded_popen
    try:
        supervisor=load_file('explicit_sealed_supervisor',TOOLS/'supervisor.py')
        sys.argv=[str(TOOLS/'supervisor.py'),'--binding',str(Path(args.binding).resolve())]
        supervisor.main()
    finally:
        wrapper.close_startup()
        subprocess.Popen=original_popen
        sealed.active_gate=wrapper.original_gate
        # Original supervisor joins its monitor before returning/raising.
        sealed.nvml_snapshot=wrapper.original_snapshot
        wrapper._write(dict(event='WRAPPER_FINISHED',accepted=False,evaluator_spawned=state['armed']))


if __name__=='__main__':main()
