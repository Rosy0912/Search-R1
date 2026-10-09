"""Native512 GPU leaf. Models are imported only after immutable CPU admission."""
import argparse
from collections import Counter
import importlib
import os
from pathlib import Path
import random
try:
    from .checkpoint import require, checked_json, checked_file, fileproof, exclusive_json, object_sha
    from .training_metrics import protocol, rows_from_ref
    from . import evaluation, prepare, gates
except ImportError:
    from checkpoint import require, checked_json, checked_file, fileproof, exclusive_json, object_sha
    from training_metrics import protocol, rows_from_ref
    import evaluation, prepare, gates

"""Scoped null-effect native logging; never patch the global requests module."""
from contextlib import contextmanager
import hashlib
import json
import threading
import time
_LOCK=threading.Lock()

def snapshot(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(k):snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [snapshot(v) for v in value]
    if hasattr(value,'detach'):return value.detach().cpu().tolist()
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'to_dict'):return snapshot(value.to_dict())
    return dict(python_type=type(value).__module__+'.'+type(value).__qualname__,
        attributes={k:snapshot(getattr(value,k)) for k in ('start','max_length','eos_token_id') if hasattr(value,k)})

def restore(obj,name,existed,old):
    if existed:setattr(obj,name,old)
    else:delattr(obj,name)

class TokenizerProxy:
    def __init__(self,original,record):self._original=original;self._record=record
    def __getattr__(self,name):return getattr(self._original,name)
    def __call__(self,*args,**kwargs):
        output=self._original(*args,**kwargs)
        self._record(dict(kind='tokenizer_call',args=snapshot(args),kwargs=snapshot(kwargs),input_ids=snapshot(output.input_ids)))
        return output

@contextmanager
def capture_native(agent,rollout):
    if not _LOCK.acquire(False):raise RuntimeError('Native capture overlap is forbidden')
    events=[];errors=[]
    trace=dict(schema='online-results-transparent-native-capture-v1',events=events,logging_errors=errors,
       started_unix=time.time(),generation_options_modified=False,return_objects_modified=False,http_retries_added=0,
       teacher_forced=False,processed_per_token_scores_recorded=False)
    def record(value):
        try:events.append(value)
        except Exception as error:errors.append(type(error).__name__)
    def safely(function):
        try:function()
        except Exception as error:errors.append(type(error).__name__)
    model=agent.model;tokenizer=agent.tokenizer;requests=rollout.requests
    own_generate='generate' in vars(model);old_gen=vars(model).get('generate');generate=model.generate
    owner=model.get_base_model() if hasattr(model,'get_base_model') else model
    has_processor=hasattr(owner,'_get_logits_processor');own_processor='_get_logits_processor' in vars(owner)
    old_processor=vars(owner).get('_get_logits_processor');processor=getattr(owner,'_get_logits_processor',None)
    def generating(*args,**kwargs):
        event=dict(kind='model_generate',started_unix=time.time())
        safely(lambda:event.update(args=snapshot(args),kwargs=snapshot(kwargs)));record(event)
        try:output=generate(*args,**kwargs)
        except BaseException as error:event.update(exception_type=type(error).__name__,ended_unix=time.time());raise
        safely(lambda:event.update(output_ids=snapshot(output.sequences if hasattr(output,'sequences') else output),
                                 output_type=type(output).__module__+'.'+type(output).__qualname__,ended_unix=time.time()))
        return output
    class RequestsProxy:
        def __getattr__(self,name):return getattr(requests,name)
        def post(self,*args,**kwargs):
            event=dict(kind='retriever_post',started_unix=time.time())
            safely(lambda:event.update(args=snapshot(args),kwargs=snapshot(kwargs)));record(event)
            try:response=requests.post(*args,**kwargs)
            except BaseException as error:event.update(exception_type=type(error).__name__,ended_unix=time.time());raise
            def capture():
                body=response.content
                event.update(status_code=response.status_code,response_bytes=len(body),response_sha256=hashlib.sha256(body).hexdigest(),ended_unix=time.time())
                try:event['response_json']=json.loads(body)
                except (ValueError,UnicodeError):event['response_text']=body.decode(errors='replace')
            safely(capture)
            return response
    def processing(*args,**kwargs):
        result=processor(*args,**kwargs)
        safely(lambda:record(dict(kind='actual_hf_logits_processors',args=snapshot(args),kwargs=snapshot(kwargs),
            processors=[dict(python_type=type(item).__module__+'.'+type(item).__qualname__,attributes=snapshot(vars(item))) for item in result],
            returned_original_list=True)))
        return result
    try:
        model.generate=generating;agent.tokenizer=TokenizerProxy(tokenizer,record);rollout.requests=RequestsProxy()
        if has_processor:owner._get_logits_processor=processing
        yield trace
    except BaseException as error:
        trace['exception_type']=type(error).__name__;raise
    finally:
        rollout.requests=requests;agent.tokenizer=tokenizer;restore(model,'generate',own_generate,old_gen)
        if has_processor:restore(owner,'_get_logits_processor',own_processor,old_processor)
        trace.update(ended_unix=time.time(),hooks_restored=True);_LOCK.release()



def tensor_digest(values):
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_checks(values,expected,zero):
    import torch
    require(len(values)==504 and all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in values.values()),"Adapter must be504 finite FP32 tensors")
    digest=tensor_digest(values);require(digest==expected,"Actual adapter digest differs from raw committed checkpoint")
    B=[t for n,t in values.items() if 'lora_B' in n]
    require(len(B)==252,"Expected252 B tensors")
    allzero=all(torch.count_nonzero(t).item()==0 for t in B)
    require(not zero or allzero,"Common initial B is not exactly zero-effect")
    return dict(tensors=504,dtype='torch.float32',all_finite=True,tensor_sha256=digest,lora_B_count=252,lora_B_all_zero=allzero)


def validate_loaded_lora(model,adapter_name):
    """Check the actual active layer forward scaling, not only saved A/B bytes."""
    layers=[]
    for name,module in model.named_modules():
        if not hasattr(module,'lora_A') or adapter_name not in module.lora_A:
            continue
        require(adapter_name in module.lora_B and module.r[adapter_name]==16
                and module.lora_alpha[adapter_name]==32
                and type(module.scaling[adapter_name]) in (int,float)
                and module.scaling[adapter_name]==2.,'Loaded LoRA layer rank/alpha/scaling differs: '+name)
        dropout=module.lora_dropout[adapter_name]
        require(getattr(dropout,'p',None)==0. or type(dropout).__name__=='Identity',
                'Loaded LoRA dropout changes eval forward: '+name)
        A=module.lora_A[adapter_name].weight;B=module.lora_B[adapter_name].weight
        require(str(A.dtype)==str(B.dtype)=='torch.float32' and len(A.shape)==len(B.shape)==2
                and A.shape[0]==B.shape[1]==16,'Loaded layer LoRA tensor dtype/shape differs: '+name)
        layers.append(dict(name=name,rank=16,alpha=32,scaling=2.,dropout=0.,A_dtype=str(A.dtype),B_dtype=str(B.dtype)))
    require(len(layers)==252,'Actual selected adapter must have252 checked LoRA layers')
    return dict(accepted=True,adapter_name=adapter_name,layers=252,scaling=2.,rank=16,alpha=32,dropout=0.,
                A_B_dtype='torch.float32',layer_checks=layers)


def rng_snapshot(torch):
    import numpy
    return random.getstate(),numpy.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()


def restore_rng(torch,state):
    import numpy
    random.setstate(state[0]);numpy.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3])


def rng_digest(state):
    import numpy
    return dict(python=hashlib.sha256(repr(state[0]).encode()).hexdigest(),
        numpy=hashlib.sha256(state[1][0].encode()+state[1][1].tobytes()+repr(state[1][2:]).encode()).hexdigest(),
        torch_cpu=hashlib.sha256(bytes(state[2].tolist())).hexdigest(),
        torch_cuda=hashlib.sha256(bytes(state[3].tolist())).hexdigest())


def validate_request(config,config_ref,request,rank):
    require(request['schema']=='opsd-signal-native512-request-v1' and request['protocol']==evaluation.PROTOCOL,
            "Wrong immutable native512 request")
    require(request['config']['sha256']==config_ref['sha256'] and request['run_id']==config['run_id']
            and request['variant']==config['variant'] and request['completed_outer'] in (0,15,25),
            "Request config/run/endpoint differs")
    slot=config['sampling']['logical_rank_to_slot'][rank]
    physical=config['owner']['physical_gpus'][slot]
    require(os.getuid()==config['owner']['uid'] and os.environ.get('E1_RUN_ID')==config['run_id']
            and os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical),"Worker physical mapping/UID/run differs")
    actual=gates.process(request['controller']['pid'])
    require(gates.same_process(actual,request['controller']) and actual['uid']==config['owner']['uid']
            and actual['run_id']==config['run_id']
            and str(Path(config['paths']['helper_root'])/'evaluation.py') in actual['argv'],
            "Immutable evaluation CPU parent identity no longer owns worker")
    require(Path(request['output']).resolve().is_relative_to(Path(config['paths']['output_root']).resolve()),"Output escaped remote group root")
    return slot,physical


def run(request_ref,rank):
    request,rp=checked_json(request_ref);config,cp=checked_json(request['config']);config['_config_sha256']=cp['sha256']
    prepare.validate_config(config,launch=True);slot,physical=validate_request(config,cp,request,rank)
    gates.publication_admission(config,gates.bound(config['gates']['publication']))
    gates.retriever_admission(config,gates.bound(config['gates']['retriever_admission']))
    science=prepare.verify_runtime(config)
    data,validation=rows_from_ref(config['validation_identity'])
    require(validation['sha256']==request['input_identity']['validation']['sha256'] and len(data)==512,"Validation identity changed")
    identity=request['input_identity'];evaluation.verify_metadata(identity['source_files']+identity['adapter_files'])
    actual_adapter_config,_=checked_json(identity['adapter_config'])
    require(evaluation.validate_adapter_config(actual_adapter_config,config)==identity['adapter_config_contract'],
            'Actual adapter alpha/forward contract changed before model load')
    directory=Path(request['output'])/('worker-'+str(rank));directory.mkdir(exist_ok=False)
    import torch
    import numpy
    from importlib import metadata as package_metadata
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,get_peft_model_state_dict
    require(torch.__version__.split('+')[0]=='2.4.0' and package_metadata.version('transformers')=='4.46.3'
            and package_metadata.version('peft')=='0.13.2',"Frozen native HF environment differs")
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"Worker must see exactly one physical GPU")
    values=load_file(str(Path(identity['adapter']['path'])/'adapter_model.safetensors'),device='cpu')
    actual=tensor_checks(values,identity['expected_adapter_tensor_sha256'],request['completed_outer']==0);del values
    tokenizer=AutoTokenizer.from_pretrained(config['paths']['model'],use_fast=True)
    base=AutoModelForCausalLM.from_pretrained(config['paths']['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    require(base.generation_config.repetition_penalty==1.05 and base.generation_config.eos_token_id==[151645,151643],"Actual native generation_config changed")
    version=request['version'];model=PeftModel.from_pretrained(base,identity['adapter']['path'],adapter_name=version,is_trainable=False)
    loaded=get_peft_model_state_dict(model,adapter_name=version)
    loaded_proof=tensor_checks(loaded,identity['expected_adapter_tensor_sha256'],request['completed_outer']==0);del loaded
    loaded_forward=validate_loaded_lora(model,version)
    require(Counter(str(p.dtype) for n,p in model.named_parameters() if '.'+version+'.' in n and 'lora_' in n)=={'torch.float32':504},"Loaded LoRA dtype/count changed")
    require(model.get_base_model().get_output_embeddings().weight.dtype==torch.bfloat16,"Actual head differs from BF16")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    model.set_adapter(version)
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval()
    core=importlib.import_module('search_e1.core');qa=importlib.import_module('search_r1.qa_protocol');rollout=importlib.import_module('search_e1.rollout')
    agent=rollout.SearchAgent(model,tokenizer,config['paths']['retriever_url'],max_new_tokens=500,max_context=8192)
    random.seed(42);numpy.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    owner=gates.process(os.getpid())
    status=dict(schema='opsd-signal-native512-worker-v1',status='RUNNING',evaluation_id=request['evaluation_id'],
        run_id=config['run_id'],variant=config['variant'],version=version,completed_outer=request['completed_outer'],rank=rank,
        physical_gpu=physical,gpu_uuid=config['owner']['gpu_uuids'][slot],rank_seed=42,config=cp,request=rp,process=owner,
        actual_adapter=actual,actual_loaded_adapter=loaded_proof,actual_loaded_forward=loaded_forward,source=science,primary_rows=0,
        actual_generation_config=model.generation_config.to_dict(),no_training=True,privileged_reference=False)
    exclusive_json(directory/'started.json',status)
    with (directory/'results.jsonl').open('x') as sink:
        for local,index in enumerate(range(rank,512,8)):
            gates.retriever_admission(config,gates.bound(config['gates']['retriever_admission']))
            question=data[index];case=directory/('slot-'+str(local));case.mkdir()
            before=rng_snapshot(torch);plain=None;plain_after=None;zero_proof=None
            with torch.no_grad():
                if local==0:
                    plain=agent.run(question['question'],temperature=0);plain_after=rng_snapshot(torch);restore_rng(torch,before)
                    exclusive_json(case/'plain-control.json',dict(result=plain,rng_before=rng_digest(before),rng_after=rng_digest(plain_after),excluded_from_primary_denominator=True))
                    if request['completed_outer']==0:
                        with model.disable_adapter():zero=agent.run(question['question'],temperature=0)
                        after_zero=rng_snapshot(torch)
                        zero_proof=dict(accepted=object_sha(zero)==object_sha(plain) and rng_digest(after_zero)==rng_digest(plain_after),
                            raw_base_result_sha256=object_sha(zero),zero_adapter_result_sha256=object_sha(plain),excluded_from_primary_denominator=True)
                        exclusive_json(case/'zero-control.json',dict(**zero_proof,result=zero))
                        require(zero_proof['accepted'],"Actual zero adapter/raw base null-effect differs");restore_rng(torch,before)
                try:
                    with capture_native(agent,rollout) as trace:result=agent.run(question['question'],temperature=0)
                except BaseException:
                    exclusive_json(case/'capture-failed.json',dict(trace=trace,question_id=question['question_id'],validation_index=index))
                    raise
            after=rng_snapshot(torch)
            require(not trace['logging_errors'],"Actual native trace logging failed")
            control=dict(required=local==0,accepted=None)
            if local==0:
                control=dict(required=True,accepted=object_sha(result)==object_sha(plain) and rng_digest(after)==rng_digest(plain_after),excluded_from_primary_denominator=True)
                exclusive_json(case/'hook-control.json',control);require(control['accepted'],"Transparent hook changed result/RNG")
            decoded=protocol(core,qa,result,question['answers'])
            trace.update(evaluation_id=request['evaluation_id'],question_id=question['question_id'],question=question['question'],answers=question['answers'],
                version=version,completed_outer=request['completed_outer'],validation_index=index,rank=rank,slot=local,
                native_result=result,protocol=decoded,rng_before=rng_digest(before),rng_after=rng_digest(after),
                hook_control=control,zero_effect_control=zero_proof,privileged_reference=False)
            trace_ref=exclusive_json(case/'native-capture.json',trace)
            row=dict(status='complete',primary=True,evaluation_id=request['evaluation_id'],run_id=config['run_id'],variant=config['variant'],
                version=version,completed_outer=request['completed_outer'],rank=rank,slot=local,validation_index=index,
                question_id=question['question_id'],dataset=question['data_source'],row_semantic_sha256=object_sha(question),
                protocol_em=decoded['protocol_em'],protocol_valid_final_action=decoded['protocol_valid_final_action'],failure_class=decoded['failure_class'],
                termination=result['termination'],search_calls=result['search_calls'],native_result_sha256=object_sha(result),
                trace=trace_ref,hook_control=control,zero_effect_control=zero_proof)
            sink.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');sink.flush();os.fsync(sink.fileno());status['primary_rows']+=1
    require(status['primary_rows']==64,"Logical rank64 incomplete")
    evaluation.verify_metadata(identity['source_files']+identity['adapter_files'])
    prepare.verify_runtime(config);gates.retriever_admission(config,gates.bound(config['gates']['retriever_admission']))
    status.update(status='COMPLETE',accepted=True,results=fileproof(directory/'results.jsonl'),ended_unix=time.time())
    exclusive_json(directory/'complete.json',status)


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--rank',type=int,choices=range(8),required=True);a=p.parse_args()
    run(dict(path=a.request,sha256=a.request_sha256),a.rank)


if __name__=='__main__':main()
