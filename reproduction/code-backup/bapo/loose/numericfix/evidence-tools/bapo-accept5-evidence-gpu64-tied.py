#!/usr/bin/env python3
"""Read-only rewardfix evidence extractor; independent review still required. Never trains, contacts a service, or changes a run.
Optional HF load uses CPU/eager only; optional comparison reads safetensor weights.
A successful script exit means extraction completed, NOT production acceptance.
"""
import argparse, hashlib, json, math, re, time, resource, subprocess, os
from pathlib import Path


EXPECTED_PHASE = 'accept5'
EXPECTED_OUTERS = 5
EXPECTED_SOURCE = '1cceda2f7e201de3f68c876bb9a3b895563c20ce'
EXPECTED_ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric')

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



EXPECTED_DATASETS = {'2wikimultihopqa':12576, 'bamboogle':125, 'hotpotqa':7405,
                     'musique':2417, 'nq':3610, 'popqa':14267, 'triviaqa':11313}
AUDITS = ('bapo/old_logprob_nonfinite','bapo/new_logprob_nonfinite',
          'bapo/mask_binary_errors','bapo/mask_on_observation_tokens',
          'bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss')

def finite_numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def validate_outer_metrics(metrics, step):
    issues=[]
    if type(metrics.get('bapo/actual_outer_updates')) is not int or metrics['bapo/actual_outer_updates']!=step:
        issues.append('Actual outer counter mismatch')
    for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm','bapo/stage_validation','state_tokens/total','timing_s/step'):
        if not finite_numeric(metrics.get(key)): issues.append('Missing/nonfinite '+key)
    for key in ('actor/optimizer_steps','bapo/loss_tokens','bapo/grad_norm_min','bapo/grad_norm_max'):
        if not finite_numeric(metrics.get(key)) or metrics[key]<=0: issues.append('Missing/nonpositive '+key)
    for key in AUDITS:
        if not finite_numeric(metrics.get(key)) or metrics[key]!=0: issues.append('Missing/nonzero audit '+key)
    for key in ('bapo/stage_cost/trajectories','bapo/stage_evaluated_count'):
        if type(metrics.get(key)) is not int or metrics[key]!=64: issues.append('Stage validation must contain64 '+key)
    for key,value in metrics.items():
        if isinstance(value,(int,float)) and not math.isfinite(value): issues.append('Nonfinite metric '+key)
    if finite_numeric(metrics.get('bapo/stage_validation')) and not 0<=metrics['bapo/stage_validation']<=1:
        issues.append('Stage score outside[0,1]')
    return issues

def validate_formal_events(events, checks):
    issues=[]
    if any('error' in e for e in events): issues.append('Malformed structured event')
    train=[e.get('value') for e in events if e['marker'] in
           ('BAPO_TRAIN_METRICS_JSON','BASELINE_TRAIN_METRICS_JSON')]
    if any(not isinstance(x,dict) or type(x.get('step')) is not int or
           not isinstance(x.get('metrics'),dict) for x in train):
        issues.append('Malformed training event or noninteger step')
    for x in train:
        if isinstance(x,dict) and type(x.get('step')) is int and isinstance(x.get('metrics'),dict):
            issues.extend('Step '+str(x['step'])+': '+issue for issue in validate_outer_metrics(x['metrics'],x['step']))
    steps=[x['step'] for x in train if isinstance(x,dict) and type(x.get('step')) is int]
    if steps != list(range(1,EXPECTED_OUTERS+1)): issues.append('Training steps must be ordered exactly 1..'+str(EXPECTED_OUTERS)+' with no duplicates/missing/out-of-range')
    if len(checks)!=EXPECTED_OUTERS or any(x.get('status')!='NUMERIC_SUMMARY_PRESENT' for x in checks.values()):
        issues.append('One or more outer numeric/audit/stage checks incomplete or failed')
    if EXPECTED_OUTERS != 500: return issues
    final=[e.get('value') for e in events if e['marker']=='SEARCH_R1_FINAL_METRICS_JSON']
    if len(final)!=1 or not isinstance(final[0],dict):
        issues.append('Need exactly one complete final metrics event'); return issues
    m=final[0]
    for k,v in m.items():
        if isinstance(v,(int,float)) and not math.isfinite(v): issues.append('Nonfinite final metric '+k)
    if m.get('val/qa_count/all')!=51713 or m.get('val/cost/trajectories')!=51713:
        issues.append('Final all QA and trajectory counts must be 51713')
    for prefix in ('val/qa_count/','val/evaluated_count/'):
        got={k[len(prefix):]:v for k,v in m.items() if k.startswith(prefix) and k!=prefix+'all'}
        if got!=EXPECTED_DATASETS: issues.append('Final dataset counts mismatch '+prefix)
        if any(type(v) is not int for v in got.values()) or sum(got.values())!=51713:
            issues.append('Final dataset counts not complete integer sum51713 '+prefix)
    for key in ('val/em/all','val/f1/all'):
        if not finite_numeric(m.get(key)) or not 0<=m[key]<=1: issues.append('Missing/invalid final metric '+key)
    return issues

def validate_expected_state(states, source):
    selected=[s for s in states if Path(s['file']).parent.name=='global_step_'+str(EXPECTED_OUTERS)]
    if len(selected)!=1: return ['Need exactly one stateEXPECTED_OUTERS']
    state=selected[0]['value']; issues=[]
    for k,v in {'method':'bapo','global_step':EXPECTED_OUTERS,'actual_outer_updates':EXPECTED_OUTERS,
                'algorithm_source_commit':source,'resume_supported':False}.items():
        if state.get(k)!=v: issues.append('stateEXPECTED_OUTERS mismatch '+k)
    control=state.get('bapo',{})
    if control.get('patience')!=5 or control.get('early') is not (EXPECTED_OUTERS<500):
        issues.append('Expected state stage controller phase/patience5 mismatch')
    if not finite_numeric(control.get('best')) or not 0<=control['best']<=1:
        issues.append('stateEXPECTED_OUTERS best invalid')
    if type(control.get('stalled')) is not int or not 0<=control['stalled']<EXPECTED_OUTERS or (control.get('early') is True and control['stalled']>=5):
        issues.append('stateEXPECTED_OUTERS stalled invalid')
    return issues


def validate_controller_replay(states, train):
    selected=[s for s in states if Path(s['file']).parent.name=='global_step_'+str(EXPECTED_OUTERS)]
    if len(selected)!=1: return ['Cannot replay without unique final state']
    best=float('-inf');stalled=0;early=True
    if len(train)!=EXPECTED_OUTERS: return ['Cannot replay incomplete/duplicate training events']
    for step,event in enumerate(train,1):
        value=event.get('value',{})
        metrics=value.get('metrics',{}) if isinstance(value,dict) else {}
        score=metrics.get('bapo/stage_validation')
        if value.get('step')!=step or not finite_numeric(score): return ['Cannot replay unordered/nonfinite stage events']
        if score>best:best=float(score);stalled=0
        else:stalled+=1
        if stalled>=5:early=False
    got=selected[0]['value'].get('bapo',{})
    expected=dict(best=best,stalled=stalled,early=early,patience=5)
    return ['State controller disagrees with full stage-score replay'] if any(got.get(k)!=v for k,v in expected.items()) else []


def validate_checkpoint(ck):
    issues=[]; files=ck.get('files',[])
    expected_names={'added_tokens.json','config.json','generation_config.json','merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}
    if {x['path'] for x in files}!=expected_names: issues.append('Checkpoint expected12 filenames mismatch')
    if len(files)!=12 or len({x['path'] for x in files})!=12:
        issues.append('Need exactly12 distinct checkpoint files')
    for f in files:
        if not isinstance(f.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',f['sha256']) or f.get('bytes',0)<=0:
            issues.append('Checkpoint file hash/size missing '+f.get('path','?'))
    tokenizer=ck.get('tokenizer_compatibility',{})
    if tokenizer.get('status')!='TOKENIZER_MATCH': issues.append('Tokenizer mismatch or missing proof')
    if not all(tokenizer.get(k) is True for k in ('full_vocabulary_equal','bos_equal','eos_equal')):
        issues.append('Concrete tokenizer equality proof missing/contradictory')
    vocab_hash=tokenizer.get('base_vocab_sha256')
    if not isinstance(vocab_hash,str) or not re.fullmatch('[0-9a-f]{64}',vocab_hash) or vocab_hash!=tokenizer.get('checkpoint_vocab_sha256'):
        issues.append('Tokenizer vocabulary hashes missing/different')
    if tokenizer.get('base_vocab_size')!=151665 or tokenizer.get('checkpoint_vocab_size')!=151665:
        issues.append('Tokenizer vocabulary size mismatch')
    if tokenizer.get('base_bos') is not None or tokenizer.get('checkpoint_bos') is not None or tokenizer.get('base_eos')!=151643 or tokenizer.get('checkpoint_eos')!=151643:
        issues.append('Tokenizer concrete BOS/EOS mismatch')
    hf=ck.get('hf_load',{})
    if hf.get('status')!='HF_LOAD_AND_FORWARD_OK' or hf.get('device')!='cpu' or hf.get('parameters')!=3085938688:
        issues.append('Missing/failed real CPU HF finite scan/forward or wrong parameter count')
    if hf.get('nonfinite_parameters')!=[] or hf.get('tied_input_output_storage') is not True:
        issues.append('HF finite parameters/tied storage missing or failed')
    if any(hf.get('loading_info',{}).get(k)!=[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        issues.append('HF loading info missing or invalid')
    forward=hf.get('forward',{})
    tokens=forward.get('input_tokens')
    if forward.get('logits_finite') is not True or forward.get('device')!='cpu':
        issues.append('Concrete finite CPU forward proof missing/contradictory')
    if type(tokens) is not int or not 1<=tokens<=16 or forward.get('logits_shape')!=[1,tokens,151936]:
        issues.append('Forward logits shape/input tokens mismatch')
    if type(forward.get('next_token_id')) is not int or not 0<=forward['next_token_id']<151936:
        issues.append('Finite forward next token witness missing/invalid')
    delta=ck.get('parameter_delta',{})
    if delta.get('changed') is not True or not finite_numeric(delta.get('max_abs_delta')) or delta['max_abs_delta']<=0:
        issues.append('No real positive dtype-aligned weight change witness')
    if delta.get('comparison')!='base_cast_to_checkpoint_dtype_then_float64': issues.append('Weight change precision mismatch')
    alias=delta.get('tied_weight_resolution',{})
    expected_alias=dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED',base_logical_key='lm_head.weight',
        base_storage_key='model.embed_tokens.weight',checkpoint_extra_key='lm_head.weight',
        checkpoint_equal_key='model.embed_tokens.weight',shape=[151936,2048],elements_compared=311164928,exact_equal=True)
    if any(alias.get(k)!=v for k,v in expected_alias.items()) or alias.get('exact_equal') is not True:
        issues.append('Missing/contradictory complete finite exact Qwen2 tied alias proof')
    return issues

def parse_scheduler_cost(text, expected_job_id):
    fields=dict(re.findall(r'(\w+)=([^\s]+)',text)); issues=[]
    if fields.get('JobId')!=str(expected_job_id) or fields.get('JobState')!='COMPLETED' or fields.get('ExitCode')!='0:0':
        issues.append('Need matching actual Slurm job COMPLETED ExitCode0:0')
    runtime=fields.get('RunTime',''); seconds=None
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',runtime)
    if match:
        days,hours,mins,secs=match.groups();seconds=int(days or 0)*86400+int(hours)*3600+int(mins)*60+int(secs)
    else: issues.append('Missing Slurm RunTime')
    return dict(fields=fields, issues=issues, wall_seconds=seconds,
                reserved_gpu_hours=seconds*8/3600 if seconds is not None else None,
                cost_scope='8-GPU reservation, not measured GPU active time')



def read_run_source_record(root):
    path=root/'source-commit.txt'
    if not path.is_file() or path.resolve().parent!=root.resolve():
        raise ValueError('Run source-commit.txt missing or outside run')
    value=path.read_text().strip()
    if value!=EXPECTED_SOURCE:
        raise ValueError('Run source-commit.txt must record exact corrected 1cc source')
    return dict(file=str(path),sha256=digest(path),source_commit=value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--log', action='append', type=Path, default=[])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--base-model', type=Path)
    p.add_argument('--compare-weights', action='store_true')
    p.add_argument('--hf-load', action='store_true')
    p.add_argument('--hash-checkpoint', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-revision', default=EXPECTED_SOURCE)
    p.add_argument('--job-id', required=True, type=int)
    p.add_argument('--slurm-record', required=True, type=Path)
    args = p.parse_args()
    started=time.monotonic(); cpu_started=time.process_time()
    root, dest = args.run_root.resolve(), args.output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    if args.source_revision != EXPECTED_SOURCE or args.job_id<=0:
        p.error('Require fixed corrected source and positive job identity')
    if root.parent != EXPECTED_ROOT/'runs':
        p.error('Run root must be a direct child of fixed corrected ROOT/runs')
    if root.name != 'bapo-'+EXPECTED_PHASE+'-'+str(args.job_id):
        p.error('Run root phase/job identity mismatch')
    try: run_source_record=read_run_source_record(root)
    except ValueError as exc: p.error(str(exc))
    repo=EXPECTED_ROOT/'repo'
    actual_source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty_source=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    if actual_source!=EXPECTED_SOURCE or dirty_source:
        p.error('Fixed corrected source must be exact and clean')
    if not (args.hf_load and args.compare_weights and args.hash_checkpoint and args.base_model):
        p.error('Formal extraction requires --hf-load --compare-weights --hash-checkpoint --base-model')
    if not (root/'TRAIN_RETURNED_SUCCESSFULLY').is_file():
        p.error('Corrected accept5 extraction requires completed successful run marker')
    if dest == root or root in dest.parents:
        p.error('Output must be outside run-root to preserve the run')
    logs = args.log or sorted(set(root.rglob('*.log')) | set(root.rglob('*.out')) | set(root.rglob('*.jsonl')))
    if any(root not in path.resolve().parents for path in logs):
        p.error('All evidence logs must belong to same corrected run')
    events, console = [], []
    markers = ('BAPO_TRAIN_METRICS_JSON=', 'BASELINE_TRAIN_METRICS_JSON=', 'BASELINE_EVALUATION_METRICS_JSON=',
               'BASELINE_SPLIT_MANIFEST_JSON=', 'BASELINE_GPU_OCCUPANCY_JSON=',
               'BAPO_PRODUCTION_AUDIT_JSON=', 'SEARCH_R1_FINAL_METRICS_JSON=')
    for log in logs:
        with log.open(errors='replace') as f:
            for n, line in enumerate(f, 1):
                for marker in markers:
                    if marker in line:
                        raw = line.split(marker, 1)[1].strip()
                        try:
                            value = json.JSONDecoder().raw_decode(raw)[0]
                            events.append(dict(marker=marker[:-1], file=str(log), line=n, value=value))
                        except (ValueError, TypeError):
                            events.append(dict(marker=marker[:-1], file=str(log), line=n, error='Invalid JSON'))
                hit = re.search(r'\bstep:(\d+) - ', line)
                if hit:
                    m = {}
                    for key, val in re.findall(r'(?:^| - )([\w/]+):([-+\deE.]+|nan|inf|-inf)(?= - |\s*$)', line[hit.start():].strip()):
                        try: m[key] = float(val)
                        except ValueError: pass
                    console.append(dict(step=int(hit.group(1)), metrics=m, file=str(log), line=n,
                                        precision='rounded_to_3_decimals; not authoritative finite/nonzero proof'))
    train = [e for e in events if e['marker'] in ('BAPO_TRAIN_METRICS_JSON','BASELINE_TRAIN_METRICS_JSON') and 'value' in e]
    indexed = {}
    for event in train:
        obj = event['value']
        if isinstance(obj,dict) and 'step' in obj and isinstance(obj.get('metrics'),dict):
            indexed.setdefault(obj['step'], []).append(event)
    checks = {}; observations = []
    for step in range(1,EXPECTED_OUTERS+1):
        es = indexed.get(step, [])
        if len(es)!=1:
            checks[str(step)] = dict(status='INCOMPLETE', reason='Need one full-precision training metric event', found=len(es))
            continue
        metrics=es[0]['value']['metrics']; issues=validate_outer_metrics(metrics, step)
        if 'bapo/stage_validation' in metrics: observations.append(step)
        checks[str(step)] = dict(status='FAIL' if issues else 'NUMERIC_SUMMARY_PRESENT', issues=issues,
                                evidence=dict(file=es[0]['file'],line=es[0]['line']))
    states=[]
    for path in sorted(root.rglob('state.json')):
        try:
            value=json.loads(path.read_text())
            if value.get('method')=='bapo': states.append(dict(file=str(path),sha256=digest(path),value=value))
        except (ValueError,AttributeError): pass
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [q for q in root.rglob('global_step_'+str(EXPECTED_OUTERS)) if (q/'config.json').is_file()]
        if len(candidates)==1: checkpoint=candidates[0]
    checkpoint_started=time.monotonic(); checkpoint_cpu_started=time.process_time()
    ck = dict(status='INCOMPLETE')
    if checkpoint:
        checkpoint=checkpoint.resolve()
        if checkpoint.name != 'global_step_'+str(EXPECTED_OUTERS) or root not in checkpoint.parents:
            p.error('Checkpoint must be this run global_step_EXPECTED_OUTERS')
        ck=dict(path=str(checkpoint),files=[])
        for q in sorted(checkpoint.rglob('*')):
            if q.is_file():
                item=dict(path=str(q.relative_to(checkpoint)),bytes=q.stat().st_size)
                if args.hash_checkpoint: item['sha256']=digest(q)
                ck['files'].append(item)
        if args.base_model:
            from transformers import AutoTokenizer
            left_tokenizer=AutoTokenizer.from_pretrained(str(args.base_model.resolve()),local_files_only=True,trust_remote_code=False)
            right_tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
            left_vocab,right_vocab=left_tokenizer.get_vocab(),right_tokenizer.get_vocab()
            def vocab_sha(vocab):
                return hashlib.sha256(json.dumps(vocab,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            token_check=dict(full_vocabulary_equal=left_vocab==right_vocab,base_vocab_sha256=vocab_sha(left_vocab),
                checkpoint_vocab_sha256=vocab_sha(right_vocab),base_vocab_size=len(left_vocab),checkpoint_vocab_size=len(right_vocab),
                bos_equal=left_tokenizer.bos_token_id==right_tokenizer.bos_token_id,
                eos_equal=left_tokenizer.eos_token_id==right_tokenizer.eos_token_id,
                base_bos=left_tokenizer.bos_token_id,checkpoint_bos=right_tokenizer.bos_token_id,
                base_eos=left_tokenizer.eos_token_id,checkpoint_eos=right_tokenizer.eos_token_id)
            token_check['status']='TOKENIZER_MATCH' if all(token_check[k] for k in ('full_vocabulary_equal','bos_equal','eos_equal')) else 'TOKENIZER_MISMATCH'
            ck['tokenizer_compatibility']=token_check
        if args.compare_weights:
            if not args.base_model: p.error('--compare-weights requires --base-model')
            ck['parameter_delta']=weight_witness(args.base_model.resolve(),checkpoint)
        if args.hf_load:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
            model, loading_info=AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,
                trust_remote_code=False,torch_dtype='auto',attn_implementation='eager',output_loading_info=True)
            if any(loading_info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise ValueError('HF checkpoint loading reported missing/unexpected/mismatched/error weights')
            tied_storage = (model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr())
            if model.config.tie_word_embeddings and not tied_storage:
                raise ValueError('HF model does not preserve declared tied weight storage')
            model.eval()
            bad=[name for name,param in model.named_parameters() if not torch.isfinite(param).all()]
            encoded=tokenizer('The capital of France is',return_tensors='pt',truncation=True,max_length=16)
            with torch.inference_mode():
                forward=model(**encoded,use_cache=False)
                forward_finite=bool(torch.isfinite(forward.logits).all())
                forward_check=dict(device=str(forward.logits.device),input_tokens=int(encoded['input_ids'].numel()),
                    logits_shape=list(forward.logits.shape),logits_finite=forward_finite,
                    next_token_id=int(forward.logits[0,-1].argmax()) if forward_finite else None)
            del forward
            ck['hf_load']=dict(status='HF_LOAD_AND_FORWARD_OK' if not bad and forward_finite else 'HF_LOAD_OR_FORWARD_NONFINITE',
                               forward=forward_check,device='cpu', loading_info=loading_info, tied_input_output_storage=tied_storage, parameters=sum(p.numel() for p in model.parameters()),
                               nonfinite_parameters=bad, tokenizer_size=len(tokenizer),
                               eos_token_id=tokenizer.eos_token_id, model_type=model.config.model_type)
            del model
    checkpoint_validation_cost=dict(wall_seconds=time.monotonic()-checkpoint_started,process_cpu_seconds=time.process_time()-checkpoint_cpu_started)
    evidence_issues=validate_formal_events(events, checks)
    evidence_issues += validate_expected_state(states, args.source_revision)
    evidence_issues += validate_controller_replay(states, train)
    evidence_issues += validate_checkpoint(ck)
    scheduler_text=args.slurm_record.read_text()
    scheduler=parse_scheduler_cost(scheduler_text,args.job_id)
    if scheduler['issues']: evidence_issues += scheduler['issues']
    result=dict(source_revision=args.source_revision,run_source_record=run_source_record,source_checkout=dict(path=str(repo),head=actual_source,dirty=dirty_source),job_id=args.job_id,
                efficacy_scope='CORRECTED_REWARD_ARTIFACT_REQUIRES_INDEPENDENT_REVIEW; NO_EFFICACY_ACCEPTANCE',
                evidence_issues=evidence_issues,
                checkpoint_validation_cost=checkpoint_validation_cost,
                training_logged_step_seconds=sum(e['value']['metrics'].get('timing_s/step',0) for e in train if isinstance(e.get('value'),dict) and isinstance(e['value'].get('metrics'),dict)),
                extraction_cost=dict(wall_seconds=time.monotonic()-started,
                    process_cpu_seconds=time.process_time()-cpu_started,
                    max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    max_rss_units='KiB on Linux; bytes on Darwin', device='cpu',
                    cost_scope='extractor process only; scheduler reservation separate'),
                scheduler_record=dict(file=str(args.slurm_record),sha256=digest(args.slurm_record),**scheduler),
                status='REQUIRES_INDEPENDENT_REVIEW',expected_outer_updates=EXPECTED_OUTERS,run_root=str(root),
                inputs=[dict(file=str(q),sha256=digest(q)) for q in logs],
                events=events,rounded_console_metrics=console,outer_step_checks=checks,
                stage_observed_steps=observations,algorithm_states=states,checkpoint=ck,
                mandatory_manual_checks=['same run/code/assets/split/GPU/Slurm/environment identity',
                    'all-rank full-precision logprob/loss/grad and mask audit',
                    'no mocked model/retrieval, real production inputs',
                    'HF load and changed-weight witness',
                    'retriever service CPU/RAM and GPU peak/cost'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    def serializable(value):
        if isinstance(value,float) and not math.isfinite(value): return 'NONFINITE:'+str(value)
        if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
        if isinstance(value,list): return [serializable(v) for v in value]
        return value
    dest.write_text(json.dumps(serializable(result),indent=2,allow_nan=False))
    print(str(dest))

if __name__=='__main__': main()
