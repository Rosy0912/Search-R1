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
