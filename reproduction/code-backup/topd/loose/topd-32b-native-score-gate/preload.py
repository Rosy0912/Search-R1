"""Actual CPU identity producer only, never a selectedteacher/GPU approval."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from score_gate import ROOT, SOURCE_SHA, METHOD_SHA, PARAMETERS, CANDIDATE_FILES, require, digest, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True)
    parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU preload requires hidden CUDA')
    require(socket.gethostname()=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Actual ownh11 CPU job')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=32 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=256*1024,'CPU preload32threads256GiB minimum')
    out=Path(args.output).resolve();require(ROOT/'logs' in out.parents and not out.exists(),'Exclusive ownlogs preload receipt')
    safety=ROOT/'tools/qwen25-32b-terminal-fix-refreeze';sys.path.insert(0,str(safety))
    from eval64 import asset_gate
    started=time.monotonic();binding,model,_=asset_gate(args.binding,ROOT);source_gate_seconds=time.monotonic()-started
    # This first native CPU-only producer supports official base identity only.
    # Future G1 must get a separately reviewed export-loader binding, not reuse it.
    require(binding['revision']=='1818d35814b8319459f4bd55ed1ac8709630f003','Official base source only')
    import torch
    from transformers import AutoTokenizer
    from search_r1.baselines.distillation import FrozenTeacher
    require(torch.__version__=='2.4.0+cu121','Pinned nativeTorch')
    require(digest(Path(inspect.getfile(inspect.unwrap(FrozenTeacher.log_probs))))==SOURCE_SHA,'Original loadedsource drift')
    require(digest(Path(args.candidate)/'pipeline.py')=='8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde','Snapshot candidate drift')
    original=inspect.unwrap(FrozenTeacher.log_probs)
    require(Path(inspect.getfile(original)).resolve()==ROOT/'repo-terminal-fixed413/search_r1/baselines/distillation.py','Actual own isolated originalsource path required')
    require(hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==METHOD_SHA,'Original scoring method drift')
    repo=ROOT/'repo-terminal-fixed413'
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='413a10c50f32febaafc83db8170e2b005e0e979f','Isolated source commit drift')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Dirty isolatedsource')
    candidate=Path(args.candidate).resolve()
    require(candidate==ROOT/'tools/topd-32b-pipeline-candidate','Snapshot helper must be own pinned candidate')
    for name,expected in CANDIDATE_FILES.items():require(digest(candidate/name)==expected,'CPU helper dependency drift: '+name)
    sys.path.insert(0,str(candidate));from pipeline import snapshot
    torch.set_num_threads(16);load_started=time.monotonic()
    teacher=FrozenTeacher(model,AutoTokenizer.from_pretrained(ROOT/'assets-root/assets/qwen2.5-3b'),device='cpu')
    load_seconds=time.monotonic()-load_started;hash_started=time.monotonic();snap=snapshot(teacher.model)
    require(snap['parameter_count']==PARAMETERS and snap['parameter_bytes']==PARAMETERS*4
            and snap['parameter_devices']==['cpu'] and snap['training_modules']==snap['requires_grad_parameters']==snap['grad_parameters']==0,'Actual frozenFP32 32B CPU model required')
    require(not torch.cuda.is_initialized(),'CPU preload initialized CUDA')
    write_json(out,dict(status='CPU_FP32_IDENTITY_CAPTURE_UNREVIEWED_NOT_TEACHER_QUALITY',
                       scope='OFFICIAL_HF_TO_ACTUAL_FP32_CPU_IDENTITY_ONLY_NO_GPU_NO_SCORING_NO_TRAINING',
                       model_directory=str(model),hf_file_sha256=binding['model_files'],snapshot=snap,
                       binding_sha256=digest(args.binding),source_gate_seconds=source_gate_seconds,
                       cpu_load_seconds=load_seconds,cpu_snapshot_seconds=time.monotonic()-hash_started,
                       torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       pid=os.getpid(),ppid=os.getppid(),hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                       proc_starttime_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                       cgroup=Path('/proc/self/cgroup').read_text(),cuda_initialized=False,
                       selected_teacher_approved=False,quality_approved=False,scoring_executed=False,
                       source_commit='413a10c50f32febaafc83db8170e2b005e0e979f',original_method_sha256=METHOD_SHA,
                       original_unwrapped_file_sha256=SOURCE_SHA,producer_sha256=digest(__file__)))

if __name__=='__main__':main()
