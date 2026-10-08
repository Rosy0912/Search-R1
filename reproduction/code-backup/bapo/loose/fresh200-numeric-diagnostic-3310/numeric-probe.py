#!/usr/bin/env python3
"""Three-point zero-write numerical localization; never a production admission."""
import argparse, ast, hashlib, importlib.util, inspect, json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
RAW_SOURCE='aec5e057093b4c491bfb0a47abd7605d98981b56'
POINTS=((500,40),(119,106),(1595,575))
HELPER_SHA='ada012340292269b7aef3f6c3bd322067328b3dd92ba3ded18bb9d13585a3ce5'
REFERENCE_SHA='058e52ab43a05f82f29e14849d8f03eaf0f0910b466f8ddd6aaaa403569e88ec'
WITNESS_SHA='12ed8011eb1d94f0076551ce1e420db1b824cce0637ab8d097c022d8da1c6af1'
helper_path=Path(__file__).with_name('hf-probe-helpers.py')
if hashlib.sha256(helper_path.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('Frozen probe helper bytes changed')
spec=importlib.util.spec_from_file_location('numeric_helpers',helper_path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def witness(tensors,journals,row,position):
    ordinal,offset,turn=h.locate_turn(journals[row],position)
    prompt_width=tensors['input_ids'].shape[1]-tensors['responses'].shape[1]
    query=prompt_width+position-1
    full=tensors['input_ids'][row,:query+1][tensors['attention_mask'][row,:query+1].bool()].tolist()
    per_turn=turn['actual_request_token_ids']+turn['raw_generated_token_ids'][:offset]
    target=int(tensors['responses'][row,position])
    h.require(full==per_turn,'Actual causal predictor token prefixes differ')
    h.require(target==turn['raw_generated_token_ids'][offset],'Actual selected label differs')
    h.require(bool(tensors['sampling_mask'][row,position]),'Target is not sampled token')
    positions=tensors['position_ids'][row,:query+1][tensors['attention_mask'][row,:query+1].bool()].tolist()
    h.require(positions==list(range(len(full))),'Actual predictor positions differ')
    return dict(row=row,position=position,turn=ordinal,offset=offset,target_token=target,
        full_padded_query_index=query,query_position=len(full)-1,request_tokens=len(turn['actual_request_token_ids']),
        predictor_tokens=full,predictor_positions=positions,prefix_sha256=hashlib.sha256(json.dumps(full).encode()).hexdigest(),
        old_hf=float(tensors['old_log_probs'][row,position]),native=float(tensors['sampling_log_probs'][row,position]))


def case_batches(tensors,journals,point,vocab):
    import torch
    row,position=point['row'],point['position'];_,offset,turn=h.locate_turn(journals[row],position)
    single={k:tensors[k][[row]].clone()for k in h.KEYS}
    cases=[('FULL',single,0,position),('EXACT_REQUEST',h.turn_batch(turn),0,offset),
           ('TRUNCATED',h.turn_batch(turn,offset),0,offset)]
    total=tensors['responses'].shape[0]
    adjacent=list(range(row//4*4,min(row//4*4+4,total)));slot=adjacent.index(row)
    changed=next(rows for mode,rows in h.groups([dict(row=row)],total)if mode=='NEW_PACKED_PROBE_CHANGED_NEIGHBORS')
    for name,rows in [('FIXED_SLOT_ADJACENT',adjacent),('FIXED_SLOT_CHANGED_LENGTHS',changed)]:
        cases.append((name,{k:tensors[k][rows].clone()for k in h.KEYS},slot,position))
    token_control={k:tensors[k][adjacent].clone()for k in h.KEYS}
    for index in range(len(adjacent)):
        if index==slot:continue
        active=token_control['attention_mask'][index].bool()
        token_control['input_ids'][index,active]=(token_control['input_ids'][index,active]+1)%vocab
        active_response=token_control['attention_mask'][index,-token_control['responses'].shape[1]:].bool()
        token_control['responses'][index,active_response]=(token_control['responses'][index,active_response]+1)%vocab
    cases.append(('FIXED_SLOT_SAME_SHAPE_OTHER_TOKENS',token_control,slot,position))
    return cases


def predictor_index(batch,slot,position,remove_padding):
    query=batch['input_ids'].shape[1]-batch['responses'].shape[1]+position-1
    if not remove_padding:return slot,query
    preceding=int(batch['attention_mask'][:slot].sum())
    return 0,preceding+int(batch['attention_mask'][slot,:query+1].sum())-1


def vector_summary(vector,label):
    import torch
    v=vector.detach().cpu().float().contiguous();values,indices=torch.topk(v,min(10,v.numel()))
    return dict(dtype=str(vector.dtype),elements=v.numel(),float32_bytes_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest(),
        selected_logit=float(v[label]),minimum=float(v.min()),maximum=float(v.max()),
        cpu_fp64_logprob=float(v.double()[label]-torch.logsumexp(v.double(),0)),
        top10_tokens=indices.tolist(),top10_logits=values.tolist())


def validate_sidecar(sidecar):
    """Independent consumers can recompute CE from all original query logits."""
    import torch
    h.require(sidecar['schema_version']==1 and sidecar['source_commit']==SOURCE and sidecar['raw_sha256']==h.RAW_SHA,'Sidecar source/raw schema')
    records=sidecar['records'];h.require(len(records)==36,'Expected exact36 original logit vectors')
    identities=set()
    for record in records:
        key=(record['precision'],record['mode'],record['row'],record['position'])
        h.require(key not in identities,'Duplicate sidecar query');identities.add(key)
        vector=record['original_logits_float32']
        h.require(vector.device.type=='cpu'and vector.dtype==torch.float32 and vector.ndim==1 and
            vector.numel()==sidecar['vocab_size']and bool(torch.isfinite(vector).all()) and
            type(record['label'])is int and 0<=record['label']<vector.numel(),'Invalid original query logits')
        h.require(record['original_dtype']in('torch.bfloat16','torch.float32'),'Original logits dtype')
        h.require((record['row'],record['position'])in POINTS,'Unapproved sidecar point')
    expected={(precision,mode,row,pos)for precision in ('BF16_ORIGINAL','FP32_EAGER_CONFOUNDED')
        for mode in ('FULL','EXACT_REQUEST','TRUNCATED','FIXED_SLOT_ADJACENT','FIXED_SLOT_CHANGED_LENGTHS','FIXED_SLOT_SAME_SHAPE_OTHER_TOKENS')for row,pos in POINTS}
    h.require(identities==expected,'Incomplete sidecar matrix')
    return [dict(precision=r['precision'],mode=r['mode'],row=r['row'],position=r['position'],label=r['label'],
        original_dtype=r['original_dtype'],stored_dtype=str(r['original_logits_float32'].dtype),elements=r['original_logits_float32'].numel(),
        cpu_fp64_logprob=float(r['original_logits_float32'].double()[r['label']]-torch.logsumexp(r['original_logits_float32'].double(),0)))for r in records]


def persist_sidecar(path,sidecar):
    import torch
    replay=validate_sidecar(sidecar)
    with Path(path).open('xb')as stream:torch.save(sidecar,stream);stream.flush();os.fsync(stream.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(Path(path).resolve()),sha256=h.digest(path),bytes=Path(path).stat().st_size,
        count=len(replay),stored_dtype='torch.float32',scope='Original selected predictor logits only; no reprojected vectors',queries=replay)


def approval(args):
    a=json.loads(Path(args.approval).read_text())
    h.require(a.get('protocol')=='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1','Dedicated numeric diagnostic approval required')
    h.require(a.get('source_commit')==SOURCE and a.get('raw_sha256')==h.RAW_SHA and
        a.get('numeric_probe_sha256')==h.digest(__file__)and a.get('reference_result_sha256')==h.digest(args.reference_result)and
        a.get('prefix_witness_sha256')==WITNESS_SHA and a.get('helper_sha256')==HELPER_SHA,'Approval source/code/raw/reference binding')
    for key,value in [('repo',args.repo),('model',args.model),('raw_batch',args.raw_batch)]:h.require(Path(a[key]).resolve()==Path(value).resolve(),'Approval path mismatch')
    h.require(a.get('native_acceptance')is False and a.get('production_release')is False and
        all(type(a.get(k))is int and a[k]==0 for k in ('optimizer_writes','formal_outer_updates','scheduler_updates')),'Zero-write scope')
    h.require(a.get('allocation_gpu_count')==8 and a.get('probe_gpu_count')==1,'Eight reserved/one compute scope')
    receipt_path=Path(a['job_receipt_path']).resolve();receipt=json.loads(receipt_path.read_text())
    job=receipt['job_id'];phase='numeric-zero-adam-replay'
    h.require(type(job)is int and job>0 and receipt.get('source')==SOURCE and receipt.get('phase')==phase and receipt.get('nonce')==a['nonce'],'Exact numeric job receipt')
    h.require(receipt.get('comment')=='BAPO_ZERO_ADAM:'+phase+':'+SOURCE+':'+a['nonce'],'Numeric receipt comment')
    run=str(Path(a['root'])/'runs'/f'{phase}-{a["nonce"]}-{job}')
    h.require(os.environ.get('SLURM_JOB_ID')==str(job)and os.environ.get('BAPO_RUN_ID')==run and
        os.environ.get('BAPO_DIAGNOSTIC_RECEIPT')==str(receipt_path)and os.environ.get('FRESH200_ZERO_ADAM_GUARD_READY')=='1','Live guarded numeric run required')
    h.require(Path(run).resolve()in Path(args.output).resolve().parents,'Output must remain own diagnostic run')
    return dict(path=str(Path(args.approval).resolve()),sha256=h.digest(args.approval),receipt_sha256=h.digest(receipt_path),job_id=job)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','raw-batch','model','output','reference-result','prefix-witness'):p.add_argument('--'+key,required=True)
    p.add_argument('--execute-gpu',action='store_true');p.add_argument('--approval');args=p.parse_args()
    h.require(not Path(args.output).exists(),'Existing diagnostic output cannot be overwritten')
    if args.execute_gpu:h.require(not Path(args.output).with_name(Path(args.output).name+'.selected-logits.pt').exists(),'Existing sidecar cannot be overwritten')
    import torch
    repo=Path(args.repo).resolve();h.require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Wrong frozen source')
    h.require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Dirty frozen repo')
    h.require(h.digest(args.raw_batch)==h.RAW_SHA,'Original raw SHA changed')
    h.require(h.digest(args.reference_result)==REFERENCE_SHA,'Actual3310 reference bytes changed')
    h.require(h.digest(args.prefix_witness)==WITNESS_SHA,'ActualCPU V2 prefix witness changed')
    payload=torch.load(args.raw_batch,map_location='cpu',weights_only=False);reference=json.loads(Path(args.reference_result).read_text())
    h.require(payload['source_commit']==RAW_SOURCE and reference['source_commit']==RAW_SOURCE and reference['raw_sha256']==h.RAW_SHA and reference['probe_sha256']==HELPER_SHA,'Original raw/reference source mismatch')
    actor_text=(repo/'verl/workers/actor/dp_actor.py').read_text()
    actor_class=next(n for n in ast.parse(actor_text).body if isinstance(n,ast.ClassDef)and n.name=='DataParallelPPOActor')
    method=next(n for n in actor_class.body if isinstance(n,ast.FunctionDef)and n.name=='_forward_micro_batch')
    source=''.join(actor_text.splitlines(keepends=True)[method.lineno-1:method.end_lineno])
    h.require(hashlib.sha256(source.encode()).hexdigest()==reference['runtime']['actor_forward_sha256'],'Frozen behavior-source forward not identical to actual aec5')
    sys.path.insert(0,str(repo))
    from search_r1.baselines.fresh200_contract import verify_official_model_assets,critical_source_hashes
    from search_r1.llm_agent.sampling_tokens import audit_batch
    from verl import DataProto
    data=DataProto.from_dict(tensors=payload['tensors'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info']);audit=audit_batch(data)
    h.require(audit==reference['context_audit'],'Reference actual context scope changed')
    h.require(float(payload['meta_info']['temperature'])==1.,'Temperature mismatch')
    official=verify_official_model_assets(args.model)
    points=[witness(payload['tensors'],payload['non_tensor_batch']['fresh200_turns'],row,pos)for row,pos in POINTS]
    saved_witness=json.loads(Path(args.prefix_witness).read_text())
    h.require(saved_witness['all22_exact']is True and saved_witness['raw_sha256']==h.RAW_SHA,'Actual CPU witness scope')
    for point in points:
        previous=next(x for x in saved_witness['points']if (x['row'],x['position'])==(point['row'],point['position']))
        h.require(previous['label_token']==point['target_token']and previous['full_query_position_id']==point['query_position']and
            previous['full_physical_query_index']==point['full_padded_query_index'],'New witness differs from actual CPU V2')
    report=dict(status='CPU_ACTUAL_PREFIX_WITNESS_NOT_GPU',source_commit=SOURCE,raw_source_commit=RAW_SOURCE,
        raw_sha256=h.RAW_SHA,reference_result_sha256=h.digest(args.reference_result),numeric_probe_sha256=h.digest(__file__),
        actual_prefix_witness_sha256=WITNESS_SHA,
        helper_sha256=HELPER_SHA,critical_source_sha256=critical_source_hashes(),official_model_manifest=official,
        context_audit=audit,points=points,native_acceptance=False,production_release=False,optimizer_writes=0,
        scope='Three point ordinaryHF numerical localization; BF16FA2 vs FP32eager changes bothprecision and attention backend')
    if args.execute_gpu:
        h.require(args.approval is not None,'Explicit numeric approval missing');report['approval']=approval(args)
        h.require(sys.platform=='linux'and torch.cuda.is_available()and torch.cuda.device_count()in(1,8),'Guarded Linux GPU required');torch.cuda.set_device(0)
        from transformers import AutoModelForCausalLM
        import transformers
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        import verl.utils.torch_functional as functional
        forward_hash=hashlib.sha256(inspect.getsource(DataParallelPPOActor._forward_micro_batch).encode()).hexdigest()
        h.require(forward_hash==reference['runtime']['actor_forward_sha256'],'Behavior source actor forward differs from aec5 reference')
        report['actor_forward_sha256']=forward_hash;report['results']=[];report['model_states']=[];sidecar_records=[]
        report['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,executable=sys.executable,
            prefix=sys.prefix,cuda=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),
            flash_cross_entropy_available=functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
            reserved_gpus=8,compute_gpus=1,logical_compute_device=0)
        started=time.monotonic()
        for precision,backend in [('BF16_ORIGINAL','flash_attention_2'),('FP32_EAGER_CONFOUNDED','eager')]:
            dtype=torch.bfloat16 if precision=='BF16_ORIGINAL'else torch.float32
            model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,torch_dtype=dtype,attn_implementation=backend).cuda().eval()
            h.require(model.config._attn_implementation==backend and all(p.dtype==dtype and p.device.index==0 for p in model.parameters()),'Model backend/dtype/device mismatch')
            vocab_size=model.config.vocab_size
            for parameter in model.parameters():parameter.requires_grad_(False)
            before=h.parameter_digest(model)
            attention=model.model.layers[0].self_attn
            helper=getattr(inspect.getmodule(type(attention)),'_flash_attention_forward',None)
            backend_evidence=dict(precision=precision,attention_backend=backend,official_config=model.config.to_dict(),
                attention_class=type(attention).__name__,attention_forward_source=inspect.getsource(type(attention).forward),
                flash_helper_source=None if helper is None else inspect.getsource(helper))
            actor=DataParallelPPOActor.__new__(DataParallelPPOActor);actor.actor_module=model;actor.actor_optimizer=None
            actor.ulysses_sequence_parallel_size=1;actor.use_ulysses_sp=False;actor.compute_entropy_from_logits=functional.entropy_from_logits
            for point in points:
                for mode,batch,slot,position in case_batches(payload['tensors'],payload['non_tensor_batch']['fresh200_turns'],point,model.config.vocab_size):
                    remove_padding=precision=='BF16_ORIGINAL'and mode.startswith('FIXED_SLOT')
                    if precision=='BF16_ORIGINAL'and mode=='FULL':remove_padding=True
                    query_slot,query_index=predictor_index(batch,slot,position,remove_padding)
                    # Query labels are identical across modes; whole-row neighbor
                    # controls preserve target's full tensor including padding.
                    h.require(int(batch['responses'][slot,position])==point['target_token'],'Case selected label drift')
                    captured={}
                    def before_head(module,inputs):captured['hidden']=inputs[0][query_slot,query_index].detach().clone()
                    def after_head(module,inputs,output):captured['logits']=output[query_slot,query_index].detach().clone()
                    hook1=model.lm_head.register_forward_pre_hook(before_head);hook2=model.lm_head.register_forward_hook(after_head)
                    cuda_batch={k:v.cuda()for k,v in batch.items()}
                    try:
                        with torch.inference_mode():
                            if precision=='BF16_ORIGINAL':
                                actor.use_remove_padding=remove_padding;_,lp=actor._forward_micro_batch(cuda_batch,1.);observed=float(lp[slot,position])
                            else:
                                # Eager cannot accept flat packed samples without a block-diagonal mask.
                                # Preserve independent padded batch rows; select only predictor hidden.
                                output=model.model(input_ids=cuda_batch['input_ids'],attention_mask=cuda_batch['attention_mask'],position_ids=cuda_batch['position_ids'],use_cache=False)
                                hidden=output.last_hidden_state[slot,query_index];captured['hidden']=hidden.detach().clone()
                                logits=torch.nn.functional.linear(hidden,model.lm_head.weight,model.lm_head.bias);captured['logits']=logits.detach().clone()
                                observed=float(torch.log_softmax(logits,dim=-1)[point['target_token']])
                    finally:hook1.remove();hook2.remove()
                    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                        reproject=torch.nn.functional.linear(captured['hidden'].float(),model.lm_head.weight.float(),None if model.lm_head.bias is None else model.lm_head.bias.float())
                    single_bf16_summary=None
                    if precision=='BF16_ORIGINAL':
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            single_bf16=torch.nn.functional.linear(captured['hidden'],model.lm_head.weight,model.lm_head.bias)
                        single_bf16_summary=vector_summary(single_bf16,point['target_token'])
                        del single_bf16
                    sidecar_records.append(dict(precision=precision,mode=mode,row=point['row'],position=point['position'],label=point['target_token'],
                        original_dtype=str(captured['logits'].dtype),original_logits_float32=captured['logits'].detach().cpu().float().contiguous()))
                    hidden_cpu=captured['hidden'].float().cpu().contiguous()
                    report['results'].append(dict(precision=precision,attention_backend=backend,mode=mode,row=point['row'],position=point['position'],batch_slot=slot,
                        remove_padding=remove_padding,input_shape=list(batch['input_ids'].shape),row_valid_tokens=batch['attention_mask'].sum(-1).tolist(),
                        input_tensor_sha256=hashlib.sha256(batch['input_ids'].contiguous().numpy().tobytes()).hexdigest(),
                        predictor_index=query_index,predictor_slot=query_slot,actor_or_eager_lp=observed,
                        hf_minus_native=observed-point['native'],hidden_float32_sha256=hashlib.sha256(hidden_cpu.numpy().tobytes()).hexdigest(),
                        hidden_original_dtype=str(captured['hidden'].dtype),
                        hidden_float32_values=hidden_cpu.tolist(),original_logits=vector_summary(captured['logits'],point['target_token']),
                        selected_hidden_fp32_head=vector_summary(reproject,point['target_token']),
                        same_hidden_bf16_single_query_head=single_bf16_summary,
                        projection_confound='Single-query head changes GEMM shape; FP32 changes arithmetic on same BF16-loaded weights, not a pureprecision control'))
            after=h.parameter_digest(model);h.require(before==after and not model.training and actor.actor_optimizer is None and all(p.grad is None for p in model.parameters()),'Zero-write invariant failed')
            report['model_states'].append(dict(precision=precision,attention_backend=backend,before=before,after=after,zero_adam=True,
                model_eval=True,gradients_absent=True,optimizer_constructed=False,scheduler_constructed=False,backend_evidence=backend_evidence))
            del model,actor,captured,cuda_batch,reproject
            if precision=='FP32_EAGER_CONFOUNDED':del output
            torch.cuda.empty_cache()
        report.update(status='ACTUAL_ZERO_ADAM_NUMERIC_DIAGNOSTIC_NOT_ADMISSION',wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    if args.execute_gpu:
        report['selected_logit_sidecar']=persist_sidecar(target.with_name(target.name+'.selected-logits.pt'),
            dict(schema_version=1,source_commit=SOURCE,raw_sha256=h.RAW_SHA,numeric_probe_sha256=h.digest(__file__),vocab_size=vocab_size,records=sidecar_records))
        report['model_precision_confound']='FP32 official reload may restore weightbits lost in BF16 load; attention also changes FA2 to eager. Not production acceptance.'
    with target.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(target.resolve()),sha256=h.digest(target),status=report['status'])))

if __name__=='__main__':main()
