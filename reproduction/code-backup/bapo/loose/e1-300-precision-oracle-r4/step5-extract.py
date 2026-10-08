#!/usr/bin/env python3
"""Actual E1 step5 read-only extraction. Never issues a PASS/release receipt.
Writes only a fresh evidence directory; never trains, opens CUDA, contacts retrieval,
or changes source/run checkpoints. Real CPU HF work is mandatory after raw capture.
"""
import argparse, hashlib, json, math, re, time, resource, subprocess, os, sys
from pathlib import Path
EXPECTED_SOURCE = "733097feeae371b4efd8fd48940c46ff36da839b"
sys.dont_write_bytecode=True
ROOT = Path("/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r4-20261004")
MODEL = ROOT / "assets/qwen2.5-3b-instruct-aa8e725"
MODEL_MANIFEST_SHA = "7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44"
DATA_MANIFEST_SHA = "0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24"
AUDITS = ("bapo/old_logprob_nonfinite", "bapo/new_logprob_nonfinite", "bapo/mask_binary_errors",
          "bapo/mask_on_observation_tokens", "bapo/mask_on_padding_tokens", "bapo/model_tokens_missing_loss")
HF_FILES = {"added_tokens.json", "config.json", "generation_config.json", "merges.txt",
 "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
 "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


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




def require(value, message):
    if not value: raise ValueError(message)

def json_read(path):
    return json.loads(Path(path).read_text())

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def binding(path):
    p=Path(path).resolve()
    require(p.is_file() and p.stat().st_size>0, 'Missing/empty evidence '+str(p))
    return dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size)

def write_new(path, value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False)

def capture_prefix(source, target, terminal=None):
    """Freeze bytes before validation; never overwrite an earlier evidence file."""
    data=Path(source).read_bytes()
    if terminal:
        lines=data.splitlines(keepends=True)
        ends=[i for i,line in enumerate(lines) if terminal.encode() in line]
        require(len(ends)==1, 'Exactly one step5 paused marker required')
        require(lines[ends[0]].endswith(b'\n'), 'Incomplete paused marker line')
        data=b''.join(lines[:ends[0]+1])
    require(bool(data),'Empty snapshot '+str(source))
    with Path(target).open('xb') as f:f.write(data)
    with Path(source).open('rb') as f:require(f.read(len(data))==data,'Input prefix changed during capture')
    return dict(live_path=str(Path(source).resolve()),snapshot_path=str(Path(target).resolve()),
                bytes=len(data),sha256=digest(Path(target)))

def events(text, marker):
    found=[]
    for line in text.splitlines():
        if marker in line:found.append(json.loads(line.split(marker,1)[1]))
    return found

def validate_metrics(train, ranks):
    require([e.get('step')for e in train]==list(range(1,6)), 'Require exact actual outer1..5')
    expected=[];best=-math.inf;stalled=0;early=True
    for step,event in enumerate(train,1):
        m=event['metrics']
        require(type(m.get('bapo/actual_outer_updates'))is int and m['bapo/actual_outer_updates']==step,'Actual outer mismatch')
        for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm','bapo/stage_validation','timing_s/step'):
            require(finite(m.get(key)), 'Missing/nonfinite '+key)
        for key in ('bapo/loss_tokens','bapo/grad_norm_min','bapo/grad_norm_max'):
            require(finite(m.get(key))and m[key]>0,'Nonpositive '+key)
        require(m.get('actor/optimizer_steps')==20,'Expected20 mini Adam calls per outer per rank')
        require(all(finite(m.get(key))and m[key]==0 for key in AUDITS),'Missing/nonzero numeric/mask audit')
        for key in ('bapo/stage_cost/trajectories','bapo/stage_evaluated_count'):
            require(type(m.get(key))is int and m[key]==64,'Stage64 count mismatch')
        require(all(not isinstance(v,(int,float))or math.isfinite(v)for v in m.values()),'Nonfinite metric')
        score=m['bapo/stage_validation'];require(0<=score<=1,'Stage score out of range')
        if score>best:best=score;stalled=0
        else:stalled+=1
        if stalled>=5:early=False
        require(m.get('bapo/early_stage')==int(early)and m.get('bapo/stalled')==stalled,'BAPO stage replay mismatch')
        expected.append(dict(best=best,stalled=stalled,early=early,patience=5))
    require(len(ranks)==40,'Need40 actual per-rank outer records')
    seen=set()
    for event in ranks:
        rank=event.get('rank');outer=event.get('scheduler_last_epoch');key=(rank,outer)
        require(type(rank)is int and rank in range(8)and type(outer)is int and outer in range(1,6)and key not in seen,'Rank/outer duplicate/missing/out of range')
        seen.add(key);require(event.get('optimizer_steps')==20,'Rank must execute20 Adam calls')
        for key,expect in [('lr_before',1e-6*(outer-1)/57),('lr_after',1e-6*outer/57)]:
            require(finite(event.get(key))and math.isclose(event[key],expect,rel_tol=1e-12,abs_tol=1e-18),'Warmup57 outer clock mismatch '+key)
        m=event.get('metrics',{})
        for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm'):
            require(isinstance(m.get(key),list)and len(m[key])==20 and all(finite(value)for value in m[key]),'Rank must have20 finite mini metrics '+key)
        require(all(all(finite(x)for x in v)if isinstance(v,list)else not isinstance(v,(int,float))or math.isfinite(v)for v in m.values()),'Rank nonfinite metric')
    require(seen=={(r,s)for r in range(8)for s in range(1,6)},'Incomplete rank clock')
    return expected[-1]

def validate_config(c,run):
    actor=c['actor_rollout_ref']['actor'];rollout=c['actor_rollout_ref']['rollout'];t=c['trainer'];b=c['baselines']
    require(t['total_training_steps']==301 and t['save_freq']==25 and t['acceptance_step']==5 and t['e1_gate_enabled']is True and t['smoke_only']is False,'Wrong300/5 save protocol')
    require(t['n_gpus_per_node']==8 and t['nnodes']==1 and t['experiment_name']=='bapo-qwen2.5-3b-instruct-e1-300','Wrong topology/run')
    require(Path(t['default_local_dir']).resolve()==run/'checkpoints' and Path(t['e1_gate_receipt']).resolve()==run/'logs/step5-independent-release.json','Wrong checkpoint/release path')
    require(b['method']=='bapo'and b['algorithm_source_commit']==EXPECTED_SOURCE and c['algorithm']['adv_estimator']=='grpo','Wrong source/algorithm')
    require(actor['loss_reduction']=='sequence_mean'and actor['ppo_mini_batch_size']==128 and actor['ppo_micro_batch_size']==16,'Driver global actor batching must be128/16')
    # Worker configs are mutated per-rank copies; driver resolved config retains global128/16.
    require(rollout['n']==1 and rollout['n_agent']==5,'Wrong rollout budget')
    require(actor['optim']['num_warmup_steps']==57 and actor['optim']['lr']==1e-6,'Wrong warmup/lr')
    require(c['retriever']['url']=='http://127.0.0.1:18083/retrieve','Retrieval must be local loopback')
    require(b['holdout']['input_mode']=='prebuilt_manifest'and b['holdout']['manifest_sha256']==DATA_MANIFEST_SHA and b['bapo']['stage_validation_size']==64,'Wrong common split/stage')
    require(Path(c['actor_rollout_ref']['model']['path']).resolve()==MODEL,'Wrong official Instruct model')

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

def verify_members(manifest,run):
    items=manifest['files'];seen=set()
    for item in items:
        p=Path(item['path']);real=p.resolve()
        require(p.is_absolute()and run in real.parents and not p.is_symlink()and str(real)==str(p),'Unsafe checkpoint member')
        require(str(real)not in seen,'Duplicate member');seen.add(str(real))
        require(p.is_file()and p.stat().st_size==item['bytes']and digest(p)==item['sha256'],'Checkpoint member changed '+str(p))
    reports=manifest['rank_roundtrips'];require(len({r.get('pid')for r in reports})==8,'Native rank PIDs must be distinct');require(len(reports)==8 and {r['rank']for r in reports}==set(range(8)),'Incomplete native ranks')
    members={x['path']:x['sha256']for x in items}
    for report in reports:
        p=run/'checkpoints/actor/global_step_5/training_state'/f"rank_{report['rank']}.pt"
        require(report['path']==str(p)and members.get(str(p))==report['sha256'],'Native state SHA/rank mapping mismatch')
        require(json_read(p.with_suffix('.json'))==report and str(p.with_suffix('.json'))in members,'Actual native JSON mismatch')
        require(report['status']=='EXACT_FULL_STATE_RELOAD_OK'and report['world_size']==8 and report['step']==5 and report['scheduler_last_epoch']==5 and report['optimizer_state_entries']>0,'Actual native reload mismatch')
        require(report.get('model_state_dict_type')=='SHARDED_STATE_DICT'and type(report.get('model_dtensor_entries'))is int and report['model_dtensor_entries']>0,'Native state must use actual SHARDED/DTensor schema')
        require(re.fullmatch('[0-9a-f]{64}',report.get('state_digest',''))and type(report.get('pid'))is int and report['pid']>0,'Missing native digest/PID')
    return items

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


def resource_proof(raw,run,job,watch_name='resource-watch.jsonl'):
    data=(raw/watch_name).read_bytes();lines=data.splitlines(keepends=True)
    complete=[line for line in lines if line.endswith(b'\n')]
    records=[json.loads(line)for line in complete]
    require(records,'No actual resource watcher samples')
    uuids=None
    for event in records:
        require(event.get('run')==str(run)and event.get('slurm_job_id')==str(job)and event.get('pause')is False and not event.get('error'),'Resource watcher fault/foreignrun/pause')
        devices=event.get('allocated_gpus',{})
        found={row['uuid']for row in devices.values()}
        require(len(devices)==8 and len(found)==8,'Need8 actual allocated GPU UUIDs')
        if uuids is None:uuids=found
        require(found==uuids,'GPU allocation changed')
        totals=event.get('external_totals',{})
        require(set(totals)==set(devices),'Incomplete external GPU occupancy rows')
        require(all(finite(row.get(key))and 0<=row[key]<=15 for row in totals.values()for key in ('external_sm_upper_bound_percent','external_memory_percent')),'External occupancy threshold violated')
    local=json_read(raw/'retrieval-assets-locality.json')
    require(local['proxy_env']==[] and isinstance(local.get('host'),str)and local['host'].split('.',1)[0]=='h100-5','Retrieval proxy/node locality mismatch')
    assets=(ROOT.parent/'bapo/assets-root/assets').resolve()
    require((ROOT/'assets-root/assets').resolve()==assets,'Shared retrieval asset alias changed')
    expected_paths=[str((assets/name).resolve())for name in ('wiki/e5_Flat.index','wiki/wiki-18.jsonl','e5-base-v2')]
    require(len(local.get('paths',[]))==3 and [item.get('path')for item in local['paths']]==expected_paths,'Exact local encoder/index/corpus paths mismatch')
    require(all(isinstance(item.get('mount'),str)and len(item['mount'].split())==3 and item['mount'].split()[1]in ('ext4','xfs')for item in local['paths']),'Non-local retrieval mount')
    profile=(raw/'runtime-profile.txt').read_text()
    require('CUDA E5' in profile and 'GPU FAISS' in profile and 'loopback127.0.0.1:18083' in profile,'GPU retrieval runtime profile missing')
    return dict(samples=len(records),allocated_uuids=sorted(uuids),last_sample_time=records[-1]['time'],
        partial_final_record_present=len(complete)!=len(lines),retrieval_locality=local,
        lease_scope='Actual inherited lease markers/process lifecycle recorded; independent reviewer additionally confirms launcher exclusive UUID lease FDs')


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job-id',type=int,required=True)
    parser.add_argument('--driver-pid',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--slurm-evidence',type=Path,required=True)
    parser.add_argument('--cpu-threads',type=int,default=8)
    parser.add_argument('--cpu-price-per-hour',type=float,default=None,help='Optional price per process CPU hour; no assumed price')
    args=parser.parse_args();run=args.run.resolve();out=args.output_dir.resolve()
    require(args.job_id>0 and args.driver_pid>0 and 1<=args.cpu_threads<=32,'Positive job/PID required')
    require(args.run==run and not args.run.is_symlink()and run==ROOT/'runs'/f'bapo-instruct300-formal300-{args.job_id}','Exact independent formal run required')
    require(run/'logs' in out.parents and not out.exists(),'Fresh run-local evidence directory required')
    require(args.cpu_price_per_hour is None or finite(args.cpu_price_per_hour)and args.cpu_price_per_hour>=0,'Invalid CPU price')
    require(ROOT in args.slurm_evidence.resolve().parents and args.slurm_evidence.is_file(),'Owned actual Slurm snapshot required')
    # Mask GPU before importing torch/transformers; leave the training processes alone.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    out.mkdir(parents=True);raw=out/'raw';raw.mkdir()
    started=time.monotonic();cpu=time.process_time();result=dict(status='REQUIRES_INDEPENDENT_REVIEW',issues=[],run_root=str(run),job_id=args.job_id,
        expected_driver_pid=args.driver_pid,source_expected=EXPECTED_SOURCE,tool=binding(Path(__file__)),actual_training_accepted=False)
    try:
        prefix=capture_prefix(run/'train.log',raw/'step5-prefix.log','E1_STEP5_PAUSED_JSON=')
        result['train_log_prefix']=prefix
        snapshots={};result['raw_snapshots']=snapshots
        names=('source-commit.txt','resolved-config.yaml','slurm-allocation.txt','gpus-before.csv','processes-before.csv','gpu-monitor.csv',
            'process-monitor.csv','cpu-process-monitor.jsonl','resource-watch.jsonl','retriever.log','retrieval-assets-locality.json',
            'runtime-profile.txt','phase.log','launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt','hardware-receipt-check.log')
        for name in names:snapshots[name]=capture_prefix(run/name,raw/name)
        snapshots['slurm-current.txt']=capture_prefix(args.slurm_evidence,raw/'slurm-current.txt')
        mp=run/'checkpoints/full_state/global_step_5/manifest.json'
        for name,path in [('checkpoint-manifest.json',mp),('hardware-acceptance.json',ROOT/'logs/hardware-acceptance.json'),
            ('common-manifest.json',ROOT/'common-data/manifest.json'),('model-manifest.json',MODEL/'VERIFIED-MODEL-MANIFEST.json'),('source-expected.txt',ROOT/'source-expected.txt')]:
            snapshots[name]=capture_prefix(path,raw/name)
        result['raw_snapshots']=snapshots
        text=(raw/'step5-prefix.log').read_text();paused=events(text,'E1_STEP5_PAUSED_JSON=')
        require(len(paused)==1 and paused[0]['pid']==args.driver_pid and paused[0]['manifest_path']==str(mp) and paused[0]['manifest_sha256']==digest(mp),'Actual paused marker/PID/manifest mismatch')
        require(not events(text,'E1_STEP5_RELEASED_JSON=')and not (run/'logs/step5-independent-release.json').exists(),'Run already released; not initialstep5 extraction')
        require((run/'source-commit.txt').read_text().strip()==EXPECTED_SOURCE and (ROOT/'source-expected.txt').read_text().strip()==EXPECTED_SOURCE,'Wrong run-start source')
        actual=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
        require(actual==EXPECTED_SOURCE and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Source changed/dirty')
        m=json_read(mp);result['checkpoint_manifest']=binding(mp)
        require(m['run_root']==str(run)and m['source_commit']==EXPECTED_SOURCE and m['global_step']==5 and m['actual_outer_updates']==5 and m['horizon']==300 and m['warmup_steps']==57,'Checkpoint identity mismatch')
        config=run/'checkpoints/full_state/global_step_5/resolved-config.json'
        require(digest(config)==m['config_sha256'],'Actual config binding mismatch');validate_config(json_read(config),run)
        items=verify_members(m,run);result['checkpoint_members']=items
        result['resources']=resource_proof(raw,run,args.job_id);result['scheduler_cost']=scheduler_proof(raw/'slurm-current.txt',args.job_id)
        result['actual_processes']=[process_proof(pid,run)for pid in [args.driver_pid]+[r['pid']for r in m['rank_roundtrips']]]
        result['actual_retriever']=live_retriever_proof(raw,run,result['resources']['allocated_uuids'])
        result['actual_uuid_leases']=lease_proof(result['actual_processes'][0],result['resources']['allocated_uuids'],run)
        require(all(p['selected_env'].get('RAY_DEDUP_LOGS')=='0'for p in result['actual_processes']),'RAY_DEDUP_LOGS=0 required for all actual rank logs; numeric JSON may otherwise be folded')
        require(args.driver_pid not in {r['pid']for r in m['rank_roundtrips']},'Driver PID must differ from rank PIDs')
        require(all(p['selected_env'].get('SLURM_JOB_ID')==str(args.job_id)for p in result['actual_processes']),'Live process Slurm identity mismatch')
        train=events(text,'BAPO_TRAIN_METRICS_JSON=');rankevents=events(text,'ACTOR_RANK_UPDATE_JSON=')
        expected_bapo=validate_metrics(train,rankevents);require(all(p['state']not in ('Z','X')for p in result['actual_processes']),'Training PID exited');result.update(train_metrics=train,rank_updates=rankevents,expected_bapo=expected_bapo)
        sys.path.insert(0,str(ROOT/'repo'))
        from search_r1.baselines.full_state import state_digest
        import torch
        torch.set_num_threads(args.cpu_threads)
        directory=run/'checkpoints/full_state/global_step_5';driver=torch.load(directory/'driver.pt',map_location='cpu',weights_only=False)
        dreport=json_read(directory/'driver-reload.json')
        require(driver['driver_pid']==args.driver_pid and driver['global_step']==5 and driver['next_global_step']==6 and driver['actual_outer_updates']==5 and driver['bapo']==expected_bapo,'Driver state/counter/stage mismatch')
        require(dreport['status']=='EXACT_DRIVER_RELOAD_OK'and dreport['pid']==args.driver_pid and dreport['step']==5 and dreport['path']==str(directory/'driver.pt')and dreport['sha256']==digest(directory/'driver.pt')and dreport['state_digest']==state_digest(driver),'Driver reload proof changed')
        require(driver['rng']['cuda']==[],'Driver must not initialize GPU RNG')
        sampler=driver['sampler'];require(sampler['size']>2560 and sampler['cursor']==2560 and sampler['epoch']==0 and len(sampler['order'])==sampler['size']and set(sampler['order'])==set(range(sampler['size'])),'Sampler order/cursor mismatch')
        state=json_read(run/'checkpoints/algorithm/global_step_5/state.json')
        require(state['method']=='bapo'and state['global_step']==5 and state['actual_outer_updates']==5 and state['algorithm_source_commit']==EXPECTED_SOURCE and state['bapo']==expected_bapo,'Algorithm state5 differs from replay')
        result['driver_state']=dict(sha256=digest(directory/'driver.pt'),state_digest=state_digest(driver),driver_pid=driver['driver_pid'],
            global_step=5,next_global_step=6,actual_outer_updates=5,sampler=dict(size=sampler['size'],epoch=sampler['epoch'],cursor=sampler['cursor'],full_state_digest=state_digest(sampler)),bapo=driver['bapo'])
        result['algorithm_state']=state;result['data_protocol']=None
        from omegaconf import OmegaConf
        from search_r1.baselines.holdout import prepare_common_holdout,assert_prebuilt_unchanged
        data_config=OmegaConf.create(json_read(config));data_record=prepare_common_holdout(data_config)
        assert_prebuilt_unchanged(data_config,data_record);result['data_protocol']=data_record
        result['native_state']=dict(reports=m['rank_roundtrips'],offline_rank_pt_unpickle_performed=False,
            proof_scope='Actual native same-rank producer full-state exact reload plus complete artifact hashes; CPU extractor does not fake an eight-rank ShardedTensor process group')
        require(digest(ROOT/'common-data/manifest.json')==DATA_MANIFEST_SHA and digest(MODEL/'VERIFIED-MODEL-MANIFEST.json')==MODEL_MANIFEST_SHA,'Commondata/officialmodel identity changed')
        model_manifest=json_read(MODEL/'VERIFIED-MODEL-MANIFEST.json')
        require(model_manifest['repo']=='Qwen/Qwen2.5-3B-Instruct'and model_manifest['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','Official model source mismatch')
        result['official_base_files']=[]
        for item in model_manifest['files']:
            p=MODEL/item['file'];require(p.parent==MODEL and not p.is_symlink()and digest(p)==item['sha256']and p.stat().st_size==item['bytes'],'Base changed '+str(p));result['official_base_files'].append(binding(p))
        ck=run/'checkpoints/actor/global_step_5';files={p.name for p in ck.iterdir()if p.is_file()}
        require(files==HF_FILES,'Require exactly12 HF export files (native subdirectory excluded)')
        result['hf_files']=[binding(ck/name)for name in sorted(HF_FILES)]
        hstart=time.monotonic();hc=time.process_time()
        try:result['cpu_hf']=hf_proof(MODEL,ck)
        finally:result['cpu_hf_cost']=dict(wall_seconds=time.monotonic()-hstart,process_cpu_seconds=time.process_time()-hc,
            cpu_price_per_hour=args.cpu_price_per_hour,estimated_cpu_cost=None if args.cpu_price_per_hour is None else (time.process_time()-hc)/3600*args.cpu_price_per_hour)
        # Revalidate all frozen/static members after expensive CPU verification.
        verify_members(m,run);assert_prebuilt_unchanged(data_config,data_record);require(digest(mp)==result['checkpoint_manifest']['sha256'],'Manifest changed duringCPUHF')
        result['resource_after_cpu_prefix']=capture_prefix(run/'resource-watch.jsonl',raw/'resource-watch-after-cpu.jsonl')
        result['resources_after_cpu']=resource_proof(raw,run,args.job_id,'resource-watch-after-cpu.jsonl')
        require(result['resources_after_cpu']['allocated_uuids']==result['resources']['allocated_uuids']and 0<=time.time()-result['resources_after_cpu']['last_sample_time']<=45,'Final resource watcher stale/allocation changed')
        result['actual_processes_after_cpu']=[process_proof(p['pid'],run)for p in result['actual_processes']]
        for before,after in zip(result['actual_processes'],result['actual_processes_after_cpu']):
            require(after['pid']==before['pid']and after['start_ticks']==before['start_ticks']and after['state']not in ('Z','X')and after['selected_env'].get('SLURM_JOB_ID')==str(args.job_id)and after['selected_env'].get('BAPO_RUN_ID')==str(run),'Actual driver/rank PID lifecycle changed duringCPUHF')
        result['actual_uuid_leases_after_cpu']=lease_proof(result['actual_processes_after_cpu'][0],result['resources']['allocated_uuids'],run)
        for b in result['official_base_files']:require(digest(Path(b['path']))==b['sha256'],'Official base changed duringCPUHF')
        require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==EXPECTED_SOURCE and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Source changed duringCPUHF')
        require(process_proof(args.driver_pid,run)['start_ticks']==result['actual_processes'][0]['start_ticks'],'Actual paused driver changed')
        with (run/'train.log').open('rb')as f:require(hashlib.sha256(f.read(prefix['bytes'])).hexdigest()==prefix['sha256'],'Live prefix changed')
        tail=(run/'train.log').read_text();require(len(events(tail,'BAPO_TRAIN_METRICS_JSON='))==5 and not events(tail,'E1_STEP5_RELEASED_JSON='),'Run advanced while extracting')
        result['release_identity']={k:m[k]for k in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce')}
        result['release_identity']['checkpoint_manifest_sha256']=digest(mp)
        result['release_evidence_bindings']=[dict(path=item['path'],sha256=item['sha256'])for item in items]+[dict(path=str(mp),sha256=digest(mp))]+[dict(path=str(p.resolve()),sha256=digest(p))for p in sorted(raw.iterdir())]
        result['extraction_complete']=True
    except Exception as exc:
        result['issues'].append(type(exc).__name__+': '+str(exc));result['extraction_complete']=False
    finally:
        result['extraction_cost']=dict(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
            additional_reserved_gpu_hours_while_paused=8*(time.monotonic()-started)/3600,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_rss_units='KiB on Linux; bytes on macOS',platform=sys.platform,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
            monetary_price_assumed=False,cpu_price_per_hour=args.cpu_price_per_hour,torch_cpu_threads=args.cpu_threads)
        write_new(out/'evidence.json',result)
    print(json.dumps(dict(status=result['status'],extraction_complete=result['extraction_complete'],issues=result['issues'],result=str(out/'evidence.json')),allow_nan=False))
    return 0 if result['extraction_complete']else 2

if __name__=='__main__':raise SystemExit(main())
