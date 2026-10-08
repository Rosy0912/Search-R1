#!/usr/bin/env python3
"""Owned calibration child, original generation/hidden/assembly code blocks."""
import argparse, hashlib, importlib.util, json, os, pathlib, time, types
from original_blocks import DUMP_SHA,execute,trace,restore,block_hashes

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    p=pathlib.Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'x') as f:os.chmod(tmp,0o600);json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def validate_inputs(config,directory):
    args=config['dump_args'];model=pathlib.Path(args['model']);data=pathlib.Path(args['data'])
    if str(model.resolve())!=config['canonical_model'] or str(data.resolve())!=config['canonical_data']:raise ValueError('input alias/path drift')
    if model.is_symlink() or data.is_symlink() or not model.resolve().is_relative_to(ROOT.resolve()) or not data.resolve().is_relative_to(ROOT.resolve()):raise ValueError('non-own input paths')
    if not pathlib.Path(directory).resolve().is_relative_to((ROOT/'logs').resolve()):raise ValueError('non-own output path')
    if sha(data)!=config['data_sha256']:raise ValueError('data bytes changed before original prepare')
    for name,expected in config['model_files'].items():
        p=model/name
        if p.is_symlink() or not p.resolve().is_relative_to(model.resolve()) or sha(p)!=expected:raise ValueError('model bytes/path changed')

def load_original(path,phase):
    # B never imports or instantiates vLLM; original function bodies stay exact.
    import ast
    from original_blocks import source
    tree=source(path)
    if phase=='B':tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!='vllm']
    namespace={'__name__':'fixed_original_dump','__file__':str(path)}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace

def mapping():
    result={n:'cuda:0' for n in ('model.embed_tokens','model.rotary_emb','model.norm','lm_head')};offset=0
    for device,count in enumerate((6,9,9,8,8,8,8,8)):
        for layer in range(offset,offset+count):result[f'model.layers.{layer}']=f'cuda:{device}'
        offset+=count
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--phase',choices=['A','B'],required=True);ap.add_argument('--directory',required=True);a=ap.parse_args()
    config=json.loads(pathlib.Path(a.config).read_text());directory=pathlib.Path(a.directory);os.umask(0o077)
    if len(config['uuids'])!=8 or os.environ.get('SLURM_JOB_ID')!=str(config['job']) or config['cgroup'] not in pathlib.Path('/proc/self/cgroup').read_text():raise RuntimeError('owned allocation identity mismatch')
    records=config['leases']
    import fcntl
    for r in records:
        fd=int(r['fd']);fcntl.fcntl(fd,fcntl.F_GETFD)
        path=pathlib.Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(str(r['uuid'])+'.lock')
        if os.fstat(fd).st_ino!=path.stat().st_ino:raise RuntimeError('inherited lease fd/UUID mismatch')
        other=os.open(path,os.O_RDWR)
        try:
            try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise RuntimeError('inherited lease is not held')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(other)
    validate_inputs(config,directory)
    source_path=pathlib.Path(__file__).parent/'original/ts_probe_dump.py';namespace=load_original(source_path,a.phase)
    torch=namespace['torch'];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=types.SimpleNamespace(**config['dump_args']);args.out=str(directory/'dump');pathlib.Path(args.out).mkdir(exist_ok=True)
    namespace.update(args=args,layers=[21,24,27],device=torch.device('cuda:0'),t0=time.time())
    started=time.time();identity={'phase':a.phase,'pid':os.getpid(),'ppid':os.getppid(),'job':config['job'],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'uuids':config['uuids'],'source_sha256':DUMP_SHA,'block_sha256':block_hashes(source_path),'config_sha256':sha(a.config)}
    if a.phase=='A':
        execute(source_path,'prepare',namespace)
        from vllm import LLM
        engine=LLM(model=args.model,dtype='bfloat16',gpu_memory_utilization=args.gpu_mem,max_model_len=4096,enforce_eager=True,disable_log_stats=True,tensor_parallel_size=8,distributed_executor_backend='mp')
        from phase_control import notify
        notify('READY',a.phase,a.config,directory)
        calls=[]
        class RecordingEngine:
            def generate(self,prompts,params,**kwargs):
                result=engine.generate(prompts,params,**kwargs)
                calls.append({'prompts':prompts,'sampling':{k:getattr(params,k) for k in ('temperature','top_p','max_tokens','stop','include_stop_str_in_output')},'outputs':[{'prompt_token_ids':list(o.prompt_token_ids),'token_ids':list(o.outputs[0].token_ids),'text':o.outputs[0].text} for o in result]})
                return result
        namespace['llm']=RecordingEngine();execute(source_path,'generation',namespace)
        write(directory/'generation-token-trace.json',calls)
        payload=trace(namespace);write(directory/'generation-trace.json',payload)
        # Exact contextual strings plus original tokenizer token IDs and mask/position
        # metadata for the public trace; B re-tokenizes and independently checks.
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.model,padding_side='left',local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        token_trace={}
        for key in ('commit_ctx','ans_ctx','prompts'):
            token_trace[key]=[dict(tok(payload[key][i:i+8],return_tensors='pt',add_special_tokens=False,padding=True,truncation=True,max_length=4096)) for i in range(0,len(payload[key]),8)]
            token_trace[key]=[{k:v.tolist() for k,v in batch.items()} for batch in token_trace[key]]
        write(directory/'hidden-input-token-trace.json',token_trace)
        identity.update(trace_sha256=sha(directory/'generation-trace.json'),token_trace_sha256=sha(directory/'hidden-input-token-trace.json'),rows=len(payload['valid']),parameters_dtype='BF16',tp=8,generation_tokens_sha256=sha(directory/'generation-token-trace.json'),stage_complete=True)
    else:
        barrier=json.loads((directory/'phase-A-owner-barrier.json').read_text())
        if (barrier.get('returncode')!=0 or barrier.get('own_processes_remaining')!=[] or barrier.get('own_gpu_records_remaining')!=[] or not barrier.get('baseline_preserved') or not barrier.get('original_global_gpu_baseline_preserved') or barrier.get('config_sha256')!=sha(a.config) or set(barrier.get('seen_actual_gpu_uuids',[]))!=set(config['uuids'])):raise RuntimeError('phaseA actual owner release missing')
        payload=json.loads((directory/'generation-trace.json').read_text());restore(payload,args,namespace)
        if sha(directory/'generation-trace.json')!=barrier['trace_sha256']:raise RuntimeError('changed phaseA trace')
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tok=AutoTokenizer.from_pretrained(args.model,padding_side='left',local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tokens=json.loads((directory/'hidden-input-token-trace.json').read_text())
        if sha(directory/'hidden-input-token-trace.json')!=barrier['token_trace_sha256']:raise RuntimeError('token trace mismatch')
        for key in tokens:
            actual=[dict(tok(payload[key][i:i+8],return_tensors='pt',add_special_tokens=False,padding=True,truncation=True,max_length=4096)) for i in range(0,len(payload[key]),8)]
            actual=[{k:v.tolist() for k,v in batch.items()} for batch in actual]
            if actual!=tokens[key]:raise RuntimeError('original batched hidden token/mask input changed')
        model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval()
        if model.config.num_hidden_layers!=64 or model.config.vocab_size!=152064:raise RuntimeError('wrong32B architecture')
        for parameter in model.parameters():parameter.requires_grad_(False)
        from accelerate import dispatch_model
        map_=mapping()
        for name,value in list(model.named_parameters())+list(model.named_buffers()):
            if not any(name==key or name.startswith(key+'.') for key in map_):raise RuntimeError('unmapped original model tensor '+name)
        model=dispatch_model(model,device_map=map_,main_device='cuda:0',force_hooks=True)
        from phase_control import notify
        notify('READY',a.phase,a.config,directory)
        namespace.update(hf=model,tok=tok);execute(source_path,'hidden',namespace);execute(source_path,'assembly',namespace)
        import numpy as np
        npz=pathlib.Path(args.out)/f'{args.tag}_{args.offset}.npz';z=np.load(npz,allow_pickle=False)
        if any(not np.isfinite(z[n]).all() for n in z.files if z[n].dtype.kind in 'fci'):raise RuntimeError('nonfinite original dump')
        identity.update(npz_sha256=sha(npz),jsonl_sha256=sha(npz.with_suffix('.jsonl')),rows=len(payload['valid']),dtype='BF16',attn_implementation='sdpa',device_map=map_,micro=8,max_length=4096,stage_complete=True)
    identity.update(elapsed_seconds=time.time()-started,peak_allocated={str(i):torch.cuda.max_memory_allocated(i) for i in range(8)},peak_reserved={str(i):torch.cuda.max_memory_reserved(i) for i in range(8)})
    write(directory/f'phase-{a.phase}-receipt.json',identity)
    from phase_control import notify
    notify('EXIT',a.phase,a.config,directory)
    # The owner waits process exit and actual cgroup/NVML disappearance. This
    # receipt does not claim allocator0 while the stage model is still resident.
if __name__=='__main__':main()
