#!/usr/bin/env python3
"""Same-base weights CPU↔single-GPU original FP32 scoring; NO_COEXISTENCE_NO_TRAINING."""
import argparse,hashlib,json,os,resource,socket,subprocess,time
from pathlib import Path

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def usage():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall=time.perf_counter(),user=r.ru_utime,system=r.ru_stime,peak_rss_kib_cumulative=r.ru_maxrss,
        proc_memory=[v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith(('VmRSS:','VmHWM:'))])
def gpu_snapshot(uid):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[v for v in raw.splitlines() if uid in v];assert len(rows)==1
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return dict(time=time.time(),gpu=rows[0],compute_apps=[v for v in apps.splitlines() if uid in v])
def cpu_parameter_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        assert p.device.type=='cpu' and p.dtype.__str__()=='torch.float32' and p.is_contiguous()
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        raw=p.detach().view(-1).view(__import__('torch').uint8)
        for offset in range(0,raw.numel(),1024*1024):h.update(raw[offset:offset+1024*1024].numpy().tobytes())
    return h.hexdigest()
def release_owned_scorer(torch,teacher,initial_parameter_sha,write,uid,baseline):
    """Only this standalone scoring process; never call inside actor/Ray workers."""
    torch.cuda.synchronize()
    teacher.model.to('cpu');teacher.device=torch.device('cpu')
    torch.cuda.synchronize()
    def state():
        return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            parameter_devices=sorted({p.device.type for p in teacher.model.parameters()}),
            buffer_devices=sorted({b.device.type for b in teacher.model.buffers()}),
            parameter_dtypes=sorted({str(p.dtype) for p in teacher.model.parameters()}),
            snapshot=gpu_snapshot(uid))
    before=state()
    write(dict(kind='workspace_before_clear',**before))
    api=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    if not callable(api):raise RuntimeError('Pinned Torch2.4 lacks private own-process cuBLAS workspace clear API; fail closed, no library upgrade')
    api()  # Process-local workspace registry, not another process or actor worker.
    torch.cuda.empty_cache();torch.cuda.synchronize()
    after=state()
    roundtrip_parameter_sha=cpu_parameter_sha(teacher.model)
    write(dict(kind='workspace_after_clear',parameter_sha256=roundtrip_parameter_sha,
        parameters_unchanged=roundtrip_parameter_sha==initial_parameter_sha,
        private_api='torch._C._cuda_clearCublasWorkspaces',private_api_scope='standalone own scorer process only',
        baseline=baseline,**after))
    if after['parameter_devices']!=['cpu'] or any(d!='cpu' for d in after['buffer_devices']):raise RuntimeError('Scorer parameters/buffers remain outside CPU')
    if after['parameter_dtypes']!=['torch.float32'] or roundtrip_parameter_sha!=initial_parameter_sha:raise RuntimeError('FP32 scorer parameters changed')
    if after['allocated_bytes']!=0 or after['reserved_bytes']!=0:raise RuntimeError('Own CUDA allocated/reserved memory remains after workspace clear; zero gate retained')
    return after
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path('/dataset/shaoshitong/search-r1-rl/topd')
    assert os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=32
    assert socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11'
    assert int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=64*1024
    assert int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==1 and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==1
    assert os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1'
    selected=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v];assert len(selected)==1
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==1;os.fstat(int(leases[0]['fd']));uid=leases[0]['uuid']
    output=Path(a.output);assert output.resolve().is_relative_to(root);output.parent.mkdir(parents=True,exist_ok=True)
    stream=output.open('x')
    def write(record):stream.write(json.dumps(record,allow_nan=False,sort_keys=True)+'\n');stream.flush()
    baseline=gpu_snapshot(uid);write(dict(kind='baseline',**baseline))
    import torch,transformers,tensordict
    assert torch.__version__.startswith('2.4.0') and transformers.__version__=='4.46.3' and tensordict.__version__=='0.5.0'
    torch.set_num_threads(16);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.is_autocast_enabled()
    assert torch.cuda.device_count()==1
    from transformers import AutoTokenizer
    from search_r1.baselines import distillation
    base=root/'assets-root/assets/qwen2.5-3b';tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    cases=[(256,32),(256,500),(1024,500)]
    write(dict(kind='metadata',scope='HARDWARE_BASE_SURROGATE_ONLY_NO_COEXISTENCE_NO_TRAINING_NOT_ALGORITHM_TEACHER',
        slurm_job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),uuid=uid,cuda_visible_devices=selected,
        base_path=str(base),cases=cases,chunk=16,temperature=1.,targets_none=True,atol=1e-4,rtol=1e-4,
        torch_threads=torch.get_num_threads(),dtype='float32',attn_implementation='eager',allow_tf32=False,autocast=False,
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,tensordict=tensordict.__version__),
        code_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),
        code_sha256=sha_file(distillation.__file__),probe_sha256=sha_file(__file__),
        common_config_sha256={name:sha_file(root/'repo'/name) for name in ['verl/trainer/config/ppo_baselines.yaml','verl/trainer/config/ppo_trainer.yaml']},
        original_cpu_probe_source_sha256=sha_file(root/'tools/topd-teacher-cpu-cost-probe.py'),
        faiss_index_loaded=False,retriever_loaded=False,algorithm_teacher_loaded=False,
        sbatch_sha256=sha_file(root/'tools/topd-teacher-gpu-fp32-probe.sbatch'),
        model_files_sha256={f.name:sha_file(f) for f in sorted(base.iterdir()) if f.is_file() and (f.suffix in ('.json','.safetensors') or f.name=='merges.txt')},
        memory_scope='torch allocator allocated/reserved excludes external baseline; nvidia-smi total includes baseline and ownCUDA context; RSS cumulative whole process'))
    teacher=None;cpu_outputs={};gpu_started=False;numeric_pass=False
    try:
        t=time.perf_counter();teacher=distillation.FrozenTeacher(str(base),tokenizer,'cpu')
        assert teacher.model.config._attn_implementation=='eager'
        assert all(p.dtype==torch.float32 and not p.requires_grad for p in teacher.model.parameters()) and not teacher.model.training
        write(dict(kind='model_load_cpu',wall_seconds=time.perf_counter()-t,usage=usage(),parameters=teacher.parameter_count))
        initial_parameter_sha=cpu_parameter_sha(teacher.model)
        write(dict(kind='cpu_parameter_identity',sha256=initial_parameter_sha,scope='streamed complete CPUFP32 parameters before scoring'))
        token=next(v for v in tokenizer.encode('the answer',add_special_tokens=False) if v not in {tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id})
        def inputs(pl,rl):return torch.full((1,pl),token,dtype=torch.long),torch.full((1,rl),token,dtype=torch.long),torch.ones((1,rl),dtype=torch.long)
        for pl,rl in cases:
            values=inputs(pl,rl);reference=None
            for repetition in range(2):
                before=usage();out=teacher.log_probs(*values,temperature=1.,targets=None,chunk=16);after=usage()
                assert out.shape==(1,rl) and bool(torch.isfinite(out).all())
                if reference is None:reference=out.clone()
                else:torch.testing.assert_close(out,reference,atol=1e-4,rtol=1e-4)
                write(dict(kind='score',device='cpu',prompt_length=pl,response_length=rl,repetition=repetition,
                    global_first_call=(pl,rl,repetition)==(256,32,0),per_shape_first_call=repetition==0,
                    wall_seconds=after['wall']-before['wall'],process_cpu_seconds=after['user']+after['system']-before['user']-before['system'],usage_after=after,
                    finite=True,min_logp=float(out.min()),max_logp=float(out.max())))
            cpu_outputs[(pl,rl)]=reference
        t=time.perf_counter();gpu_started=True;teacher.model.to(torch.device('cuda:0'));teacher.device=torch.device('cuda:0');torch.cuda.synchronize()
        assert all(p.dtype==torch.float32 for p in teacher.model.parameters())
        write(dict(kind='model_move_to_gpu',wall_seconds=time.perf_counter()-t,snapshot=gpu_snapshot(uid)))
        for pl,rl in cases:
            values=inputs(pl,rl)
            for repetition in range(2):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=usage()
                out=teacher.log_probs(*values,temperature=1.,targets=None,chunk=16);torch.cuda.synchronize();after=usage()
                reference=cpu_outputs[(pl,rl)];delta=(out-reference).abs();close=bool(torch.allclose(out,reference,atol=1e-4,rtol=1e-4))
                finite=bool(torch.isfinite(out).all());write(dict(kind='score',device='cuda_fp32',prompt_length=pl,response_length=rl,repetition=repetition,
                    global_first_call=(pl,rl,repetition)==(256,32,0),per_shape_first_call=repetition==0,
                    timing_scope='cuda synchronized full original log_probs includes output transfer toCPU; no excluded warmup',
                    wall_seconds=after['wall']-before['wall'],process_cpu_seconds=after['user']+after['system']-before['user']-before['system'],usage_after=after,
                    finite=finite,numeric_close=close,max_absolute_delta=float(delta.max()),mean_absolute_delta=float(delta.mean()),
                    atol=1e-4,rtol=1e-4,min_logp=float(out.min()),max_logp=float(out.max()),
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),snapshot=gpu_snapshot(uid)))
                assert finite and close,'CPU/GPU FP32 logp comparison failed; thresholds must not be relaxed silently'
        numeric_pass=True
        write(dict(kind='numeric_complete',status='FP32_SCORE_MATCH_PASS_NO_COEXISTENCE_NO_TRAINING',scope='numeric phase only, not overall completion'))
    except BaseException as e:
        write(dict(kind='failure',error=repr(e)));raise
    finally:
        try:
            if teacher is not None and gpu_started:
                release_owned_scorer(torch,teacher,initial_parameter_sha,write,uid,baseline)
                if numeric_pass:
                    write(dict(kind='complete',status='FP32_SCORE_AND_RELEASE_PASS_NO_COEXISTENCE_NO_TRAINING',
                        numeric_close=True,owned_allocator_zero=True,owned_reserved_zero=True,
                        all_parameters_buffers_cpu=True,parameters_unchanged=True))
        except BaseException as e:
            write(dict(kind='release_failure',error=repr(e)));raise
        finally:
            stream.close()
if __name__=='__main__':main()
