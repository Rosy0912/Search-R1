"""Future admitted GPU worker. Importing this file does not import Torch/models."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib
import os
from pathlib import Path
import random
import time
try:
    from .utils import *
    from . import prepare,gates
    from .hooks import capture_native
except ImportError:
    from utils import *
    import prepare,gates
    from hooks import capture_native

def tensor_digest(values):
    # Exact original immutable trainer.tensor_digest algorithm/key naming.
    import torch
    h=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        t=tensor.detach().cpu().contiguous();h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def tensor_checks(values,version,expected):
    import torch
    require(len(values)==504 and all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in values.values()),'Adapter504 FP32/finite differs')
    digest=tensor_digest(values);require(digest==expected,'Actual adapter tensor content differs from committed checkpoint')
    B=[t for n,t in values.items() if 'lora_B' in n]
    require(len(B)==252,'LoRA B count differs')
    zero=all(torch.count_nonzero(t).item()==0 for t in B)
    if version.endswith('zero'):require(zero,'Common initial adapter is not exactly zero-effect B')
    return dict(tensors=504,dtype='torch.float32',all_finite=True,tensor_sha256=digest,lora_B_count=252,lora_B_all_zero=zero)

def rng_snapshot(torch):return (random.getstate(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
def restore_rng(torch,state):random.setstate(state[0]);torch.set_rng_state(state[1]);torch.cuda.set_rng_state(state[2])
def rng_digest(state):return dict(python=hashlib.sha256(repr(state[0]).encode()).hexdigest(),torch_cpu=hashlib.sha256(bytes(state[1].tolist())).hexdigest(),torch_cuda=hashlib.sha256(bytes(state[2].tolist())).hexdigest())

def protocol(core,qa,result,answers):
    spans=result['policy_spans'];last=result['response'][spans[-1][0]:spans[-1][1]] if spans else ''
    answer=qa.extract_final_answer([last] if spans else [])
    score=core.reward(result['response'],answers,spans);require(score in (0.,1.),'Native strict reward not binary')
    return dict(protocol_em=int(score),protocol_valid_final_action=answer is not None,
        failure_class=None if score else ('final_invalid' if answer is None else 'valid_gold_mismatch'),
        extracted_final_answer=answer,policy_tag_balanced_diagnostic={tag:sum(result['response'][a:b].count('<'+tag+'>') for a,b in spans)==sum(result['response'][a:b].count('</'+tag+'>') for a,b in spans) for tag in ('think','search','answer')},
        balanced_tags_are_not_reward=True)

def validate_ticket(config,config_ref,ticket,version,rank,output):
    require(ticket['accepted'] is True and ticket['run_id']==config['run_id'] and ticket['config_sha256']==config_ref['sha256']
        and ticket['version']==version and ticket['worker_count']==8 and ticket['logical_rank_map']==list(range(8)),'Wrong immutable launch ticket')
    require(Path(output).resolve()==Path(ticket['output']).resolve(),'Worker output ticket differs')
    require(os.environ.get('E1_RUN_ID')==config['run_id'] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(rank),'Worker run/device environment differs')
    actual=gates.process(config,ticket['controller']['pid'])
    require(gates.same(actual,ticket['controller']) and actual['run_id']==config['run_id']
        and str(Path(config['paths']['helper_root'])/'controller.py') in actual['argv']
        and '--config' in actual['argv'] and actual['argv'][actual['argv'].index('--config')+1]==config_ref['path']
        and '--config-sha256' in actual['argv'] and actual['argv'][actual['argv'].index('--config-sha256')+1]==config_ref['sha256'],
        'Exact admitted evaluator controller no longer owns launch')

def run(config,config_ref,ticket,version,rank,output):
    prepare.validate_config(config);require(os.getuid()==config['owner']['uid'],'Wrong worker UID');gates.publication(config)
    validate_ticket(config,config_ref,ticket,version,rank,output);gates.oracle(config)
    data,validation=cohort(config);weights=source_files(config);core,qa,science_proof=science(config)
    rollout=importlib.import_module('search_e1.rollout')
    adapter=prepare.adapter(config,version);directory=Path(output)/f'worker-{rank}';directory.mkdir(exist_ok=False)
    import torch
    from importlib import metadata as package_metadata
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,get_peft_model_state_dict
    require(torch.__version__.split('+')[0]=='2.4.0' and package_metadata.version('transformers')=='4.46.3'
        and package_metadata.version('peft')=='0.13.2','Actual frozen environment differs')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see one physical GPU')
    values=load_file(str(Path(adapter['path'])/'adapter_model.safetensors'),device='cpu')
    actual_tensor=tensor_checks(values,version,adapter['tensor_sha256'])
    train,_=original_config(config);initial,_=checked_json(train['common_initial_adapter_manifest'])
    for f in initial['payload_files']:fileproof(f['path'],f['sha256'])
    initial_values=load_file(initial['adapter_model']['path'],device='cpu')
    tensor_checks(initial_values,'bf16_zero',train['common_initial_adapter_tensor_sha256'])
    require(set(values)==set(initial_values),'Current/initial tensor key inventory differs')
    changed=sum(not torch.equal(t,initial_values[n]) for n,t in values.items())
    require(changed==0 if version.endswith('zero') else changed>0,'Raw boundary not changed from common initial / zero template differs')
    actual_tensor.update(changed_tensors_vs_common_initial=changed,common_initial_tensor_sha256=train['common_initial_adapter_tensor_sha256'])
    del values,initial_values
    model_path=train['paths']['model'];tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=True)
    deploy_proof=None;loaded_identity=None
    require(version=='raw_bf16','Only requested native BF16 slot is supported')
    base=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,adapter['path'],adapter_name=version,is_trainable=False)
    loaded=get_peft_model_state_dict(model,adapter_name=version)
    loaded_identity=tensor_checks(loaded,version,adapter['tensor_sha256']);del loaded
    require(Counter(str(p.dtype) for n,p in model.named_parameters() if '.'+version+'.' in n and 'lora_' in n)=={'torch.float32':504},'Actual unmerged FP32 adapter differs')
    require(all(p.dtype==torch.bfloat16 for n,p in model.named_parameters() if 'lora_' not in n),'Actual main512 base not entirely BF16')
    model.set_adapter(version);model_kind='BF16_base_FP32_unmerged_LoRA'
    require(model.generation_config.repetition_penalty==1.05 and model.generation_config.eos_token_id==[151645,151643],'Actual native repetition/EOS differs')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval();check_metadata(weights['files']+adapter['files'])
    agent=rollout.SearchAgent(model,tokenizer,config['paths']['retriever_url'],max_new_tokens=500,max_context=8192)
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    identity=gates.process(config,os.getpid());status=dict(schema='opsd-pr100-native512-worker-v1',status='RUNNING',version=version,rank=rank,
        process=identity,run_id=config['run_id'],config=config_ref,adapter=adapter,source=weights,science=science_proof,
        model_kind=model_kind,deployment=deploy_proof,variant=config['variant'],study_id=config['study_id'],
        actual_adapter=actual_tensor,actual_loaded_adapter=loaded_identity,actual_generation_config=model.generation_config.to_dict(),
        physical_gpu=rank,gpu_uuid=config['owner']['gpu_uuids'][rank],rank_seed=42,primary_rows=0,privileged_reference=False,no_training=True)
    atomic_json(directory/'status.json',status)
    with (directory/'results.jsonl').open('x') as sink:
        for slot,index in enumerate(range(rank,512,8)):
            gates.oracle(config);row=data[index];case=directory/f'slot-{slot:02d}';case.mkdir()
            before=rng_snapshot(torch);plain=None;plain_after=None;zero_proof=None
            with torch.no_grad():
                if slot==0:
                    plain=agent.run(row['question'],temperature=0)
                    plain_after=rng_snapshot(torch);restore_rng(torch,before)
                    atomic_json(case/'plain-control.json',dict(result=plain,rng_before=rng_digest(before),rng_after=rng_digest(plain_after),excluded_from_primary_denominator=True))
                    if version=='bf16_zero':
                        with model.disable_adapter():zero=agent.run(row['question'],temperature=0)
                        zero_after=rng_snapshot(torch)
                        zero_proof=dict(accepted=object_sha(zero)==object_sha(plain) and rng_digest(zero_after)==rng_digest(plain_after),
                            raw_base_result_sha256=object_sha(zero),cp0_result_sha256=object_sha(plain),actual_cp0=actual_tensor,excluded_from_primary_denominator=True)
                        atomic_json(case/'zero-effect-control.json',dict(**zero_proof,cp0_result=zero));require(zero_proof['accepted'],'Actual rawbase/CP0 zero-effect control differs');restore_rng(torch,before)
                try:
                    with capture_native(agent,rollout) as trace:result=agent.run(row['question'],temperature=0)
                except BaseException:
                    atomic_json(case/'native-capture-failed.json',dict(trace=trace,question_id=row['question_id'],version=version,validation_index=index,excluded_from_completed_primary_slots=True))
                    raise
            after=rng_snapshot(torch);require(not trace['logging_errors'],'Native logging incomplete; do not accept row')
            control=dict(required=slot==0,accepted=None)
            if slot==0:
                control=dict(required=True,accepted=object_sha(result)==object_sha(plain) and rng_digest(after)==rng_digest(plain_after),
                    plain_result_sha256=object_sha(plain),hooked_result_sha256=object_sha(result),plain_rng_after=rng_digest(plain_after),hooked_rng_after=rng_digest(after),excluded_from_primary_denominator=True)
                atomic_json(case/'hook-control.json',control);require(control['accepted'],'Actual hook result/RNG control differs')
            decoded=protocol(core,qa,result,row['answers'])
            trace.update(question_id=row['question_id'],question=row['question'],answers=row['answers'],version=version,
                validation_index=index,rank=rank,slot=slot,native_result=result,protocol=decoded,rng_before=rng_digest(before),rng_after=rng_digest(after),
                hook_control=control,zero_effect_control=zero_proof,privileged_reference=False,primary_ordinary_prompt=True)
            trace_ref=atomic_json(case/'native-capture.json',trace,exclusive=True)
            record=dict(status='complete',run_id=config['run_id'],version=version,rank=rank,slot=slot,validation_index=index,
                question_id=row['question_id'],dataset=row['data_source'],row_semantic_sha256=validation['semantic_sha256'][index],
                protocol_em=decoded['protocol_em'],protocol_valid_final_action=decoded['protocol_valid_final_action'],failure_class=decoded['failure_class'],
                termination=result['termination'],search_calls=result['search_calls'],native_result_sha256=object_sha(result),trace=trace_ref,
                hook_control=control,zero_effect_control=zero_proof,primary=True)
            append_json(sink,record);status.update(primary_rows=slot+1,updated_unix=time.time());atomic_json(directory/'status.json',status)
    check_metadata(weights['files']+adapter['files']);gates.publication(config);gates.oracle(config);cohort(config)
    require(status['primary_rows']==64,'Incomplete rank64')
    status.update(status='COMPLETE',ended_unix=time.time(),results=fileproof(directory/'results.jsonl'))
    atomic_json(directory/'complete.json',dict(status,accepted=True),exclusive=True);atomic_json(directory/'status.json',status)

def main():
    p=argparse.ArgumentParser()
    for name in ('config','config-sha256','ticket','ticket-sha256','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--version',choices=VERSIONS,required=True);p.add_argument('--rank',type=int,choices=range(8),required=True);a=p.parse_args()
    config,fp=checked_json(dict(path=a.config,sha256=a.config_sha256));ticket,_=checked_json(dict(path=a.ticket,sha256=a.ticket_sha256))
    run(config,fp,ticket,a.version,a.rank,a.output)

if __name__=='__main__':main()
