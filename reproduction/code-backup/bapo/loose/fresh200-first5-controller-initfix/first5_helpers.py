#!/usr/bin/env python3
"""behavior V2 read-only CPU helpers. Generic reader code reused, no historical receipts."""
import hashlib,json,math,os,re,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
PRODUCTION_TOOLS_SHA='fbd0098b946e23a8eb7304912cad01b8f339e249f627c919ddec8313c2337859'
RUN=None # Bind only to a new actually submitted formal200 process.
JOB=None
MODEL=ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
REVIEW_SHA=None # Bind only after independent review of actual new fifth-outer evidence.
def require(value,message):
    if not value:raise ValueError(message)
def finite(value):return type(value)in(int,float)and math.isfinite(value)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def binding(path):return dict(path=str(Path(path)),sha256=digest(path),bytes=Path(path).stat().st_size)
def write_new(path,value):
    data=json.dumps(value,indent=2,allow_nan=False).encode()
    with Path(path).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def owned(path):
    path=Path(path);require(path.is_absolute()and path==path.resolve()and ROOT in path.parents and not path.is_symlink(),'Exact owned regular path required');return path
def require_run_binding():
    require(type(JOB) is int and JOB>0 and isinstance(RUN,Path),"UNBOUND: actual new formal run/job required")
    require(RUN.is_absolute() and RUN.parent==ROOT/"runs" and RUN.name.endswith("-"+str(JOB)),"Exact new owned run/job binding required")
def behavior_proof(proof):
    require(proof.get("probability_contract")=="ACTUAL_NATIVE_BEHAVIOR_V2_HF_DIAGNOSTIC_SEPARATE" and proof.get("hf_native_drift_is_identity_gate") is False and proof.get("behavior_policy_role")=="actual_native_stochastic_sampling_policy","Actual native behavior V2 required; HF drift is diagnostic")
    return proof
def bind_source():
    require_run_binding()
    require(type(SOURCE) is str and re.fullmatch("[0-9a-f]{40}",SOURCE) and type(PRODUCTION_TOOLS_SHA) is str and re.fullmatch("[0-9a-f]{64}",PRODUCTION_TOOLS_SHA),"UNBOUND: frozen source and production tool manifest required")
    import subprocess
    require(not sys.flags.optimize,'Optimized Python forbidden')
    require(sys.platform=='linux'and Path(sys.prefix).resolve()==ROOT/'env','Actual isolated target Linux env required')
    require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Frozen source differs')
    require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Source dirty')
    manifest=ROOT/'tools/tools-manifest.json'
    if not manifest.exists():manifest=ROOT/'logs/tools-manifest.json'
    require(digest(manifest)==PRODUCTION_TOOLS_SHA,'Exact behavior V2 tools manifest required')
    tools=json.loads(manifest.read_text());require(tools['source_commit']==SOURCE and len(tools['files'])==21,'Full21 new source tools required')
    for name,sha in tools['files'].items():require(digest(ROOT/'tools'/name)==sha,'Frozen behavior V2 tool changed '+name)
    sys.path.insert(0,str(ROOT/'repo'))
def official_manifest():
    path=MODEL/'VERIFIED-MODEL-MANIFEST.json';require(digest(path)==MODEL_SHA,'Official manifest identity changed')
    value=json.loads(path.read_text())
    for entry in value['files']:
        p=MODEL/entry['file'];require(p.is_file()and p.stat().st_size==entry['bytes']and digest(p)==entry['sha256'],'Official model bytes differ')
    return binding(path)
def marked(text,marker):return [json.loads(line.split(marker,1)[1])for line in text.splitlines()if marker in line]
def immutable_prefix(source,target):
    raw=Path(source).read_bytes();require(raw.endswith(b'\n'),'Partial terminal log line')
    with Path(target).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(live_path=str(source),snapshot_path=str(target),bytes=len(raw),sha256=digest(target))

def safetensor_map(root):
    from safetensors import safe_open
    found = {}
    for path in sorted(root.glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in found:
                    raise ValueError('Duplicate weight key: '+key)
                found[key] = path
    if not found:
        raise ValueError('No safetensors in '+str(root))
    return found

def resolve_qwen2_tied_alias(base, checkpoint, before, after):
    """Only admit the exact observed redundant Qwen2 tied-head serialization."""
    if set(before) == set(after):
        return {}
    if set(before) - set(after) or set(after) - set(before) != {'lm_head.weight'}:
        raise ValueError('Base/checkpoint tensor key sets differ')
    embed, head = 'model.embed_tokens.weight', 'lm_head.weight'
    if embed not in before or embed not in after or head in before:
        raise ValueError('Invalid Qwen2 tied alias pattern')
    configs = [json.loads((root / 'config.json').read_text()) for root in (base, checkpoint)]
    for config in configs:
        if (config.get('model_type') != 'qwen2' or config.get('tie_word_embeddings') is not True
                or config.get('architectures') != ['Qwen2ForCausalLM']):
            raise ValueError('Extra head requires verified tied Qwen2 configurations')
    shape = [configs[0]['vocab_size'], configs[0]['hidden_size']]
    if shape != [configs[1]['vocab_size'], configs[1]['hidden_size']]:
        raise ValueError('Tied embedding dimensions differ')
    import torch
    from safetensors import safe_open
    checked = 0
    with safe_open(str(after[embed]), framework='pt', device='cpu') as f, safe_open(str(after[head]), framework='pt', device='cpu') as g:
        if (f.get_slice(embed).get_shape() != shape or g.get_slice(head).get_shape() != shape
                or f.get_slice(embed).get_dtype() != g.get_slice(head).get_dtype()):
            raise ValueError('Serialized tied weights shape/dtype differ')
        for start in range(0, shape[0], 1024):
            a = f.get_slice(embed)[start:start+1024]
            b = g.get_slice(head)[start:start+1024]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a, b):
                raise ValueError('Serialized tied weights must be entirely finite and exactly equal')
            checked += a.numel()
    before[head] = before[embed]
    return dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED',
                base_logical_key=head, base_storage_key=embed,
                checkpoint_extra_key=head, checkpoint_equal_key=embed,
                shape=shape, elements_compared=checked, exact_equal=True)

def weight_witness(base, checkpoint):
    """First real changed tensor: float64 subtraction avoids dtype rounding.
    Base is first cast to checkpoint export dtype so rounding alone is not an update.
    Equality after this casting is proved only if every tensor was compared.
    """
    import torch
    from safetensors import safe_open
    before, after = safetensor_map(base), safetensor_map(checkpoint)
    alias = resolve_qwen2_tied_alias(base, checkpoint, before, after)
    def storage_key(key):
        return alias['base_storage_key'] if alias and key == alias['base_logical_key'] else key
    # Validate every shape before an early changed-weight witness can return.
    for key in sorted(before):
        with safe_open(str(before[key]), framework='pt', device='cpu') as f, safe_open(str(after[key]), framework='pt', device='cpu') as g:
            if f.get_slice(storage_key(key)).get_shape() != g.get_slice(key).get_shape():
                raise ValueError('Tensor shape differs: '+key)
    checked = 0
    for key in sorted(before):
        with safe_open(str(before[key]), framework='pt', device='cpu') as f, safe_open(str(after[key]), framework='pt', device='cpu') as g:
            shape_a, shape_b = f.get_slice(storage_key(key)).get_shape(), g.get_slice(key).get_shape()
            if shape_a != shape_b:
                raise ValueError('Tensor shape differs: '+key)
            spans = range(0, shape_a[0], 1024) if shape_a else [0]
            for start in spans:
                end = min(start+1024, shape_a[0]) if shape_a else 0
                a = f.get_slice(storage_key(key))[start:end] if shape_a else f.get_tensor(storage_key(key))
                b = g.get_slice(key)[start:end] if shape_a else g.get_tensor(key)
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite tensor chunk: '+key)
                # Align base to export dtype before subtraction: export rounding is not training.
                raw_delta = (a.to(torch.float64)-b.to(torch.float64)).abs()
                aligned = a.to(b.dtype)
                if not torch.isfinite(aligned).all():
                    raise ValueError('Base cast is nonfinite: '+key)
                delta = (aligned.to(torch.float64)-b.to(torch.float64)).abs()
                if delta.count_nonzero():
                    return dict(changed=True, tensor=key, tied_weight_resolution=alias, first_axis_slice=[start,end],
                                tensors_completed=checked, max_abs_delta=float(delta.max()),
                                chunk_l1_delta=float(delta.sum()), base_dtype=str(a.dtype),checkpoint_dtype=str(b.dtype),
                                raw_max_abs_delta=float(raw_delta.max()),comparison='base_cast_to_checkpoint_dtype_then_float64',
                                note='One changed-weight chunk; not proof all parameters finite')
            checked += 1
    return dict(changed=False, tensors_checked=checked, tied_weight_resolution=alias, note='All weight tensors compared after base cast to checkpoint export dtype')

def tokenizer_proof(left,right):
    lv,rv=left.get_vocab(),right.get_vocab()
    vh=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    require(lv==rv and left.bos_token_id==right.bos_token_id and left.eos_token_id==right.eos_token_id,'Official Instruct tokenizer differs')
    require(left.pad_token_id==right.pad_token_id and left.special_tokens_map==right.special_tokens_map,'Official pad/special tokens differ')
    require(bool(left.chat_template)and left.chat_template==right.chat_template,'Official Instruct chat_template differs')
    def added_metadata(tokenizer):
        return {str(key):dict(content=tok.content,single_word=tok.single_word,lstrip=tok.lstrip,rstrip=tok.rstrip,normalized=tok.normalized,special=tok.special)for key,tok in tokenizer.added_tokens_decoder.items()}
    left_added,right_added=added_metadata(left),added_metadata(right)
    require(left_added==right_added,'AddedToken normalization/special metadata differs')
    conversation=[{'role':'user','content':'What is the capital of France?'}]
    left_chat=left.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True)
    right_chat=right.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True)
    require(isinstance(left_chat,list)and left_chat and left_chat==right_chat,'Real Instruct chat-template tokenization differs')
    token=dict(full_vocabulary_equal=True,bos_equal=True,eos_equal=True,base_vocab_sha256=vh(lv),checkpoint_vocab_sha256=vh(rv),
        base_vocab_size=len(lv),checkpoint_vocab_size=len(rv),bos=left.bos_token_id,eos=left.eos_token_id,pad=left.pad_token_id,
        pad_equal=True,special_tokens_equal=True,special_tokens_map=left.special_tokens_map,added_token_metadata_equal=True,added_token_metadata_sha256=vh(left_added),
        chat_template_equal=True,chat_template_sha256=hashlib.sha256(json.dumps(left.chat_template,sort_keys=True).encode()).hexdigest(),
        real_chat_template_tokenization_equal=True,real_chat_token_ids=left_chat,
        scope='REAL_TOKENIZER_PAIR_EQUALITY_REQUIRES_OFFICIAL_BASE_BINDING')
    return token

def hf_proof(base,checkpoint):
    """Real local CPU HF; failure is an issue, never synthetic success."""
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    require(not torch.cuda.is_initialized(),'Extractor must never initialize CUDA')
    left=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
    right=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    token=tokenizer_proof(left,right)
    token['official_base']='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1'
    delta=weight_witness(base,checkpoint)
    require(delta.get('changed')is True and finite(delta.get('max_abs_delta'))and delta['max_abs_delta']>0,'No actual dtype-aligned weight change')
    alias=delta.get('tied_weight_resolution',{})
    if alias:require(alias.get('shape')==[151936,2048]and alias.get('elements_compared')==311164928 and alias.get('exact_equal')is True,'Incomplete311m tied-alias proof')
    model,info=AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False,
        torch_dtype='auto',attn_implementation='eager',output_loading_info=True)
    require(all(not info.get(k)for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'HF loading mismatch')
    require(all(p.device.type=='cpu'for p in model.parameters()),'HF must run entirely CPU')
    require(model.config.model_type=='qwen2'and model.config.vocab_size==151936 and model.config.hidden_size==2048,'Wrong model dimensions')
    tied=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    require(model.config.tie_word_embeddings is True and tied,'HF tied storage not preserved')
    bad=[n for n,p in model.named_parameters()if not torch.isfinite(p).all()]
    require(not bad,'Nonfinite loaded parameter '+repr(bad));count=sum(p.numel()for p in model.parameters())
    require(count==3085938688,'Wrong full parameter count')
    model.eval();inputs=right('The capital of France is',return_tensors='pt',truncation=True,max_length=16)
    with torch.inference_mode():output=model(**inputs,use_cache=False)
    require(output.logits.device.type=='cpu'and torch.isfinite(output.logits).all(),'CPU forward nonfinite')
    shape=list(output.logits.shape);require(shape==[1,inputs['input_ids'].numel(),151936],'Wrong logits shape')
    proof=dict(status='REAL_CPU_HF_FINITE_FORWARD',device='cpu',parameters=count,nonfinite_parameters=bad,
        tied_input_output_storage=tied,loading_info=info,forward=dict(logits_finite=True,logits_shape=shape,
        device=str(output.logits.device),next_token_id=int(output.logits[0,-1].argmax())))
    del model,output
    return dict(tokenizer=token,parameter_delta=delta,hf=proof)

def process_proof(pid,run):
    p=Path('/proc')/str(pid);stat=(p/'stat').read_text().rsplit(')',1)[1].split();start=stat[19]
    env=dict(part.split(b'=',1)for part in (p/'environ').read_bytes().split(b'\0')if b'='in part)
    require(env.get(b'BAPO_RUN_ID')==str(run).encode(),'Actual driver/worker belongs to another run')
    require((p/'stat').read_text().rsplit(')',1)[1].split()[19]==start,'PID reused during capture')
    return dict(pid=pid,start_ticks=int(start),state=stat[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
        selected_env={k.decode():env[k].decode()for k in (b'SLURM_JOB_ID',b'BAPO_RUN_ID',b'CUDA_VISIBLE_DEVICES',b'SEARCH_R1_GPU_PREFLIGHT',b'SEARCH_R1_GPU_LEASES',b'SEARCH_R1_GPU_RESOURCE_GUARD',b'RAY_DEDUP_LOGS')if k in env})

def scheduler_proof(path,job):
    text=Path(path).read_text()
    require(re.search(r'\bJobId='+str(job)+r'\b',text)and re.search(r'\bJobState=RUNNING(?:\s|$)',text)and re.search(r'\bNodeList=h100-5(?:\s|$)',text),'Actual Slurm job/state/node mismatch')
    found=re.search(r'RunTime=(?:(\d+)-)?(\d+):(\d+):(\d+)',text)
    require(found,'Actual elapsed RunTime absent')
    days,hours,minutes,seconds=found.groups();elapsed=int(days or 0)*86400+int(hours)*3600+int(minutes)*60+int(seconds)
    return dict(job_id=job,state='RUNNING',node='h100-5',elapsed_seconds=elapsed,reserved_gpu_hours_so_far=8*elapsed/3600,
        monetary_gpu_price_assumed=False,scope='Allocation elapsed includes setup/training/stage validation/pause; snapshot predates extra CPUHF work')

def lease_proof(process,uuids,run):
    env=process['selected_env'];require(env.get('SEARCH_R1_GPU_PREFLIGHT')=='1','GPU preflight marker absent')
    leases=json.loads(env.get('SEARCH_R1_GPU_LEASES','[]'));guard=json.loads(env.get('SEARCH_R1_GPU_RESOURCE_GUARD','{}'))
    require(len(leases)==8 and {r['uuid']for r in leases}==set(uuids)and len({r['fd']for r in leases})==8,'UUID lease inventory differs')
    require(guard.get('run')==str(run)and type(guard.get('pid'))is int,'Wrong resource guard')
    actual=process_proof(guard['pid'],run);require(actual['start_ticks']==guard.get('start_ticks'),'Guard process reused')
    locks=Path('/proc/locks').read_text().splitlines();records=[]
    for item in leases:
        named=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(item['uuid']+'.lock')
        a=named.stat();b=(Path('/proc')/str(guard['pid'])/'fd'/str(item['fd'])).stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Lease FD no longer binds actual file')
        target=f'{os.major(a.st_dev):02x}:{os.minor(a.st_dev):02x}:{a.st_ino}'
        matched=[line for line in locks if len(line.split())>=6 and line.split()[5]==target and line.split()[1:4]==['FLOCK','ADVISORY','WRITE']]
        require(all(int(line.split()[4])==guard['pid']for line in matched),'UUID lock owner differs from actual guard')
        require(len(matched)==1,'Exclusive actual UUID flock missing/ambiguous')
        records.append(dict(uuid=item['uuid'],fd=item['fd'],inode=a.st_ino,exclusive_lock_record=matched[0]))
    return dict(guard=actual,leases=records,verification='read_only_proc_fd_inode_and_proc_locks_no_flock_mutation')

def live_retriever_proof(raw,run,uuids):
    lines=(raw/'cpu-process-monitor.jsonl').read_bytes().splitlines(keepends=True)
    records=[json.loads(line)for line in lines if line.endswith(b'\n')]
    require(records,'Actual CPU process monitor absent')
    candidates=[]
    for item in records[-1]['processes']:
        if item.get('role')!='retriever':continue
        try:proof=process_proof(item['pid'],run)
        except (OSError,ValueError):continue
        if 'reproduction/h20/retriever_local.py' in proof['cmdline']:candidates.append(proof)
    require(len(candidates)==1,'One actual local retriever process required')
    proof=candidates[0];cmd=proof['cmdline']
    require('--device cuda' in cmd and '--faiss_gpu' in cmd and '--host 127.0.0.1' in cmd and '--port 18083' in cmd,'Actual retriever flags changed')
    rows=[]
    for line in (raw/'process-monitor.csv').read_text().splitlines():
        fields=[x.strip()for x in line.split(',')]
        if len(fields)>=4 and fields[0]in uuids and fields[1]==str(proof['pid']):rows.append(line)
    require(rows,'Actual retriever GPU process absent from GPU monitor')
    return dict(process=proof,gpu_process_observations=rows,
        verification='Actual marked PID commandline CUDA encoder/GPU FAISS/loopback flags and allocated-UUID GPU memory observations')
