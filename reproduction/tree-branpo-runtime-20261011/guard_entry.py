"""Run the unchanged foreign-resource guard; deny its unbound scheduler cancel.

A STOP/exception exits this owned child. The supervisor then performs its
existing cgroup-owned cleanup and exits the batch. No foreign PID is touched.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from production_contract import require,read_bound
from slurm_identity import verify_runtime,local_environment,fsync_directory,verify_runtime_cgroup

def cancellation_refusal(command,evidence_path):
    value=dict(status='UNBOUND_SCHEDULER_CANCELLATION_REFUSED_OWN_CLEANUP_REQUIRED',command=list(command),
               scheduler_cancellation_executed=False,foreign_processes_touched=False)
    path=Path(evidence_path)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    fsync_directory(path.parent)
    raise RuntimeError('Guard bare-ID cancellation denied; own supervisor cleanup required')

def guarded_run(original,evidence_path):
    def run(command,*args,**kwargs):
        require(type(command) in (list,tuple) and command,'Guard subprocess must have explicit argv')
        if Path(str(command[0])).name=='scancel':return cancellation_refusal(command,evidence_path)
        verify_runtime_cgroup()
        kwargs['env']=local_environment(kwargs.get('env',os.environ))
        return original(command,*args,**kwargs)
    return run

def install_nvml_cgroup_guards(module):
    names=('nvmlInit','nvmlDeviceGetCount','nvmlDeviceGetHandleByIndex','nvmlDeviceGetUUID',
        'nvmlDeviceGetMemoryInfo','nvmlDeviceGetUtilizationRates','nvmlDeviceGetComputeRunningProcesses',
        'nvmlDeviceGetGraphicsRunningProcesses','nvmlDeviceGetProcessUtilization')
    originals={name:getattr(module,name) for name in names}
    for name,function in originals.items():
        def guarded(*args,_function=function,**kwargs):
            verify_runtime_cgroup()
            return _function(*args,**kwargs)
        setattr(module,name,guarded)
    return originals

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--identity',required=True);parser.add_argument('--identity-sha',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();saved,_,_=read_bound(args.identity,args.identity_sha);current=verify_runtime()
    require(saved['manifest_sha256']==os.environ['TREE_INSTRUCT_PACKAGE_SHA256']
            and saved['review_sha256']==os.environ['TREE_INSTRUCT_REVIEW_SHA256']
            and saved['run_id']=='tree-branpo-fresh200-'+str(current['job_id'])
            and Path(args.identity).parent.name==saved['run_id'],'Guard own run/package binding differs')
    require(all(saved[k]==current[k] for k in ('cluster_name','job_id','node_list','gpus','requested_cpus','memory_gib','wall_hours','no_requeue','restarts')),'Guard allocation lineage changed')
    from build_package_draft import read_regular_bound
    raw=read_regular_bound(args.source,args.source_sha)
    module=type(sys)('_owned_guard_original');module.__file__=args.source
    # Every original subprocess.run call resolves the controlled wrapper below.
    import pynvml
    nvml_originals=install_nvml_cgroup_guards(pynvml)
    original=subprocess.run;subprocess.run=guarded_run(original,Path(args.output).with_suffix('.cancellation-refused.json'))
    try:
        # Original frozen guard is a top-level argparse/while-loop script.
        # Install its original argv BEFORE execution; do not invent a main().
        sys.argv=[args.source,'--output',args.output]
        module.__dict__['__name__']='__main__'
        exec(compile(raw,args.source,'exec'),module.__dict__)
    finally:
        subprocess.run=original
        for name,function in nvml_originals.items():setattr(pynvml,name,function)

if __name__=='__main__':main()
