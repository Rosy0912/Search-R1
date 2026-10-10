"""Deferred fixed HF weights; batch generation, original per-question executor."""
import importlib,json,os,random,time
from collections import Counter
from pathlib import Path

BATCH_SIZE=32

def chunks(data,rank):
    indices=list(range(rank,512,8))
    if len(data)!=512 or rank not in range(8):raise ValueError('Fixed512/eight logical ranks required')
    return [(indices[start:start+BATCH_SIZE],[data[i] for i in indices[start:start+BATCH_SIZE]])
        for start in range(0,64,BATCH_SIZE)]

def enrich(q,result,score):
    return dict(q,**result,sibling=0,reward=score['protocol_em'],protocol=score)

def run_worker(request,directory,rank):
    from evaluate import read,proof,write,require,modules,rows,PROTOCOL
    require(os.getuid()==1001 and rank in range(8)
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(4+rank%4)
        and os.environ.get('E1_RUN_ID')==request['evaluation_run_id'],'Worker UID/GPU/run differs')
    config=read(request['config']);gates,sampling=modules(config)
    require(request['completed_outer'] in (0,15,25)
        and directory.resolve().is_relative_to(Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/hf-batched-endpoint-evaluations').resolve()),
        'Endpoint time or remote output root differs')
    import evaluation,evaluation_worker
    identity_record=read(proof(directory/'model-identity.json'))
    require(identity_record['accepted'] and identity_record['protocol']==PROTOCOL
        and identity_record['no_weight_merge'],'Unmerged raw HF identity required')
    identity=identity_record['identity'];evaluation.verify_metadata(identity['source_files']+identity['adapter_files'])
    require(identity['validation']['sha256']==request['data']['sha256']
        and proof(request['data']['path'])['sha256']==request['data']['sha256'],'Fixed512 identity differs')
    adapter_cfg=read(identity['adapter_config'])
    require(evaluation.validate_adapter_config(adapter_cfg,config)==identity['adapter_config_contract'],
        'Adapter forward configuration differs')
    from dependencies import import_core
    run_questions,greedy_batch=import_core(request['batch_core'])
    # All model/Torch imports are leaf-only, after the CPU parent's release gate.
    started=time.time()
    import torch,numpy,requests
    from importlib import metadata
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,get_peft_model_state_dict
    require(torch.__version__.split('+')[0]=='2.4.0' and metadata.version('transformers')=='4.46.3'
        and metadata.version('peft')=='0.13.2','Original frozen HF environment differs')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One authorized physical GPU required')
    values=load_file(str(Path(identity['adapter']['path'])/'adapter_model.safetensors'),device='cpu')
    saved=evaluation_worker.tensor_checks(values,identity['expected_adapter_tensor_sha256'],request['completed_outer']==0);del values
    tokenizer=AutoTokenizer.from_pretrained(config['paths']['model'],use_fast=True)
    base=AutoModelForCausalLM.from_pretrained(config['paths']['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    # Frozen Qwen2/HF auto-selects the last-position logits during generate.
    require(base._supports_num_logits_to_keep(),'Generation would materialize full prefill vocabulary logits')
    version='source100' if request['completed_outer']==0 else config['variant']
    model=PeftModel.from_pretrained(base,identity['adapter']['path'],adapter_name=version,is_trainable=False)
    values=get_peft_model_state_dict(model,adapter_name=version)
    loaded=evaluation_worker.tensor_checks(values,identity['expected_adapter_tensor_sha256'],request['completed_outer']==0);del values
    forward=evaluation_worker.validate_loaded_lora(model,version)
    require(Counter(str(p.dtype) for n,p in model.named_parameters() if '.'+version+'.' in n and 'lora_' in n)=={'torch.float32':504},
        'LoRA precision differs from original unmerged FP32')
    require(base.get_output_embeddings().weight.dtype==torch.bfloat16,'BF16 head required')
    require(base.generation_config.repetition_penalty==1.05 and base.generation_config.eos_token_id==[151645,151643]
        and base.generation_config.num_beams==1,'Native generation settings differ')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    model.set_adapter(version)
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    random.seed(42);numpy.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    core=importlib.import_module('search_e1.core');qa=importlib.import_module('search_r1.qa_protocol')
    from training_metrics import protocol
    questions=rows(request['data']['path']);owner=gates.process(os.getpid())
    require(owner['uid']==1001 and owner['run_id']==request['evaluation_run_id'],'Actual worker ownership differs')
    write(directory/('worker-%d-started.json'%rank),dict(accepted=True,status='RUNNING',rank=rank,rank_seed=42,
        process=owner,physical_gpu=4+rank%4,gpu_uuid=config['owner']['gpu_uuids'][rank%4],
        input_identity=identity,actual_saved_adapter=saved,actual_loaded_adapter=loaded,actual_loaded_forward=forward,
        generation_config=base.generation_config.to_dict(),protocol=PROTOCOL,batch_size=BATCH_SIZE,
        no_training=True,model_load_seconds=time.time()-started,batch_rounding_not_claimed_bitwise_single=True))
    context={};generation_seconds=0.;primary=[]
    with (directory/('generation-%d.jsonl'%rank)).open('x') as generated,(directory/('http-%d.jsonl'%rank)).open('x') as http:
        proxy=sampling.RequestsTraceProxy(requests,http,rank,loaded['tensor_sha256'],lambda:dict(context))
        def record(event):
            generated.write(json.dumps(dict(event,context=dict(context)),ensure_ascii=False,allow_nan=False)+'\n')
        def decode(texts,logged=True):
            nonlocal generation_seconds
            before=time.time();result=greedy_batch(model,tokenizer,texts,500,record if logged else None)
            generation_seconds+=time.time()-before
            return result
        def execute(qs,logged=True):
            def events(event):
                context['question_slot']=event['question_slot'];context['turn']=event['turn']
                record(event)
            return run_questions([q['question'] for q in qs],tokenizer,core.prompt,
                lambda texts:decode(texts,logged),proxy,config['paths']['retriever_url'],500,8192,events if logged else None)
        # One native/one batched-size1/one no-adapter question, zero only on rank0.
        # They are controls, excluded from every formal512 denominator.
        if rank==0 and request['completed_outer']==0:
            context.update(control=True,validation_indices=[0],kind='single_native_null_effect')
            rng=evaluation_worker.rng_snapshot(torch)
            rollout=importlib.import_module('search_e1.rollout')
            agent=rollout.SearchAgent(model,tokenizer,config['paths']['retriever_url'],500,8192)
            original_http=rollout.requests;rollout.requests=proxy
            try:native=agent.run(questions[0]['question'],temperature=0)
            finally:rollout.requests=original_http
            evaluation_worker.restore_rng(torch,rng)
            single=execute(questions[:1],False)[0];evaluation_worker.restore_rng(torch,rng)
            with model.disable_adapter():zero=execute(questions[:1],False)[0]
            evaluation_worker.restore_rng(torch,rng)
            require(native==single==zero,'Batch-size1 executor or zero-LoRA differs from native')
            write(directory/'zero-and-native-control.json',dict(accepted=True,native=native,batch_size1=single,
                disabled_adapter=zero,extra_questions=3,excluded_from_primary_denominator=True))
        context.clear()
        with (directory/('results-%d.jsonl'%rank)).open('x') as sink:
            for batch,(indices,qs) in enumerate(chunks(questions,rank)):
                context.clear();context.update(control=False,batch=batch,validation_indices=indices)
                before=time.time();rr=execute(qs)
                require(len(rr)==len(qs),'Batch result count differs')
                for i,q,r in zip(indices,qs,rr):
                    decoded=protocol(core,qa,r,q['answers']);row=enrich(q,r,decoded)
                    row.update(validation_index=i,logical_rank=rank,rank_seed=42,generation_protocol=PROTOCOL)
                    sink.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');primary.append(row)
                sink.flush();generated.flush();http.flush()
                write(directory/('worker-%d-batch-%d-complete.json'%(rank,batch)),dict(accepted=True,rank=rank,
                    completed_questions=len(primary),batch_questions=len(qs),seconds=time.time()-before,protocol=PROTOCOL))
        require(not proxy.logging_errors,'Full retriever trace recording failed')
    require(len(primary)==64 and {r['validation_index'] for r in primary}==set(range(rank,512,8)),'Shard64 coverage differs')
    evaluation.verify_metadata(identity['source_files']+identity['adapter_files'])
    write(directory/('worker-%d-complete.json'%rank),dict(accepted=True,rank=rank,results=proof(directory/('results-%d.jsonl'%rank)),
        request=request,input_identity=identity,process=owner,actual_loaded_adapter=loaded,actual_loaded_forward=forward,
        generation=proof(directory/('generation-%d.jsonl'%rank)),http=proof(directory/('http-%d.jsonl'%rank)),
        protocol=PROTOCOL,rank_seed=42,batch_size=BATCH_SIZE,primary_rows=64,
        total_seconds=time.time()-started,generation_seconds=generation_seconds,
        formal_weights='raw_unmerged_FP32_LoRA',no_training=True))
