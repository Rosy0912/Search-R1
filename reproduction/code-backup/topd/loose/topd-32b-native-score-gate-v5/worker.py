"""Own direct scorer child. Supervisor must establish lease/baseline first."""
import argparse
import json
import inspect
import hashlib
import os
from pathlib import Path
import sys
import time
from score_gate import ROOT, SOURCE_SHA, METHOD_SHA, check_identity, require, write_json, run_native_score_gate, digest
from caller_contracts import same_cpu_identity, authorization, verify_bundle


def wait(path, seconds=1800):
    end=time.monotonic()+seconds
    while not path.exists():
        require(time.monotonic()<end, 'Supervisor authorization timeout')
        time.sleep(.2)


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--provenance',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--directory',required=True)
    args=parser.parse_args();out=Path(args.directory)
    bundle=verify_bundle(Path(__file__).parent)
    require(out.is_dir() and ROOT/'logs' in out.resolve().parents, 'Own supervisor output only')
    require(os.environ.get('TOPD32B_SCORER_PARENT')==str(os.getppid()), 'Own direct parent required')
    own_identity=dict(pid=os.getpid(),starttime_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    provenance_sha=digest(args.provenance)
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    provenance=json.loads(Path(args.provenance).read_text())
    model=check_identity(provenance,args.candidate)
    sys.path.insert(0,args.candidate)
    from transformers import AutoTokenizer
    from search_r1.baselines.distillation import FrozenTeacher
    source=Path(inspect.getfile(inspect.unwrap(FrozenTeacher.log_probs))).resolve()
    require(source == ROOT/'repo-terminal-fixed413/search_r1/baselines/distillation.py'
            and digest(source)==SOURCE_SHA
            and hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest()==METHOD_SHA,
            'CPU-loaded original teacher/source differs before CUDA')
    from pipeline import PipelineRuntime, snapshot
    from capacity_reference import PARAMETERS_32B
    started=time.monotonic();torch.set_num_threads(16)
    teacher=FrozenTeacher(model,AutoTokenizer.from_pretrained(ROOT/'assets-root/assets/qwen2.5-3b'),device='cpu')
    load_seconds=time.monotonic()-started;hash_started=time.monotonic()
    cpu=snapshot(teacher.model);same_cpu_identity(cpu,provenance)
    require(teacher.model.config.num_hidden_layers==64 and teacher.model.config.vocab_size==152064,
            'Fresh CPU teacher architecture differs')
    require(not torch.cuda.is_initialized(), 'CPU ready allocated CUDA too early')
    write_json(out/'cpu-ready.json',dict(**own_identity,ppid=os.getppid(),cpu_load_seconds=load_seconds,
               cpu_snapshot_seconds=time.monotonic()-hash_started,bundle=bundle,
               provenance_sha256=provenance_sha,snapshot=cpu,parent_cuda_initialized=False,
               scope='SELECTED32B_CPU_READY_NOT_GPU_ACCEPTANCE'))
    wait(out/'allow-context.json')
    authorization(json.loads((out/'allow-context.json').read_text()),'CONTEXT_ONLY',own_identity,provenance_sha,os.environ['SLURM_JOB_ID'])
    validate_inherited_gpu_leases(8)  # Fresh absolute dual15 immediately before any CUDA context.
    records=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    runtime=PipelineRuntime(records,os.environ['TOPD32B_SCORER_CGROUP'],os.environ['SLURM_JOB_ID'],ROOT)
    runtime.flags()
    # Real context on each UUID makes strict NVML identity checks meaningful;
    # no teacher CUDA dispatch occurs before the second parent authorization.
    for device in range(8):
        token=torch.empty((),device='cuda:%d'%device,dtype=torch.float32)
        torch.cuda.synchronize(device);del token
    released=runtime.release_allocators()
    require(all(x['allocated_bytes']==x['reserved_bytes']==0 for x in released), 'Context init left allocations')
    write_json(out/'context-ready.json',dict(pid=os.getpid(),gpu_pid_uuid=runtime.gpu_identity(),allocators=released))
    wait(out/'allow-scoring.json')
    authorization(json.loads((out/'allow-scoring.json').read_text()),'SCORING_ONLY',own_identity,provenance_sha,os.environ['SLURM_JOB_ID'])
    # Public neutral token patterns, no questions, accepted answers or gold.
    vocab=sorted(set(teacher.tokenizer.get_vocab().values())-{teacher.tokenizer.pad_token_id})
    p=torch.empty((16,2048),dtype=torch.long);r=torch.empty((16,4096),dtype=torch.long);a=torch.ones_like(r)
    for row in range(16):
        p[row]=vocab[32+row];r[row]=vocab[80+row]
        if row>=8:p[row,:1792]=teacher.tokenizer.pad_token_id;a[row,500:]=0
    tensors=dict(prompts=p,responses=r,attention=a,row_indices=torch.arange(16))
    scored=run_native_score_gate(teacher,runtime,tensors,args.candidate,provenance,out/'scoring')
    write_json(out/'scoring-done.json',dict(pid=os.getpid(),status=scored['status']))
    # Keep only empty CUDA contexts until the parent drains its identity sampler.
    wait(out/'allow-exit.json',seconds=120)
    authorization(json.loads((out/'allow-exit.json').read_text()),'EXIT_ONLY',own_identity,provenance_sha,os.environ['SLURM_JOB_ID'])

if __name__=='__main__':main()
