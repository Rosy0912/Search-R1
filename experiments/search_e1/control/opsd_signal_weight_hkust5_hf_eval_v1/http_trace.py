"""Bounded concurrent HTTP with atomic IDs/writes; unchanged requests/Response."""
import json,threading,time

class ConcurrentHTTP:
    def __init__(self,original,handle,rank,model_identity,context):
        self.original,self.handle,self.rank,self.model_identity,self.context=original,handle,rank,model_identity,context
        self.lock=threading.Lock();self.next_id=0;self.logging_errors=[]
    def for_case(self,slot,turn):
        parent=self;case=dict(question_slot=slot,turn=turn)
        class Bound:
            def post(self,*args,**kwargs):return parent._post(case,args,kwargs)
        return Bound()
    def post(self,*args,**kwargs):return self._post(None,args,kwargs)
    def _record(self,event):
        with self.lock:
            try:self.handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');self.handle.flush()
            except Exception as e:self.logging_errors.append(dict(type=type(e).__name__,message=str(e)))
    def _post(self,case,args,kwargs):
        with self.lock:
            request_id=self.next_id;self.next_id+=1
        # Context is stable for the whole executor turn/barrier. Case is per task.
        event=dict(schema='opsd-online-private-http-request-v1',request_id=request_id,rank=self.rank,
            started_at_unix=time.time(),sampling_model_identity=self.model_identity,generation_call=dict(self.context()),
            executor_case=case,positional_args=list(args),keyword_args=kwargs,concurrent_request_order=True)
        try:response=self.original.post(*args,**kwargs)
        except Exception as e:
            event.update(ended_at_unix=time.time(),request_raised=dict(type=type(e).__name__,message=str(e)));self._record(event);raise
        event['ended_at_unix']=time.time()
        try:
            event.update(status_code=response.status_code,response_body_text=response.text)
            try:event['response_json']=response.json()
            except Exception as e:event['response_json_parse_error']=dict(type=type(e).__name__,message=str(e))
        except Exception as e:
            with self.lock:self.logging_errors.append(dict(type=type(e).__name__,message=str(e)))
        self._record(event);return response
