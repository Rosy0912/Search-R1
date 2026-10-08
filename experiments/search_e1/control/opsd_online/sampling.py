"""Current-student sampling with unchanged frozen generation/mining operators."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

try:
    from .checkpoint import require,checked_file,checked_json,fileproof,object_sha,exclusive_json,stat_identity
except ImportError:
    from checkpoint import require,checked_file,checked_json,fileproof,object_sha,exclusive_json,stat_identity


def read_rows(path):
    data=Path(path).read_bytes()
    require(not data or data.endswith(b'\n'),'Partial JSONL line: '+str(path))
    return [json.loads(line) for line in data.splitlines()]


def write_rows(path,rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    return fileproof(path)


def bind_model_files(config,identity):
    """One CPU full-byte verification per sampling attempt, not eight repeated scans."""
    if identity['kind']=='raw-source100':
        model,_=checked_json(config['source_model_manifest'])
        files=model.get('files',model.get('source_files'))
        require(isinstance(files,list) and len(files)==12,'Source100 requires 12 bound model files')
    else:
        files=identity['files']
    root=Path(identity['path']).resolve();checked=[checked_file(f) for f in files]
    require(all(Path(f['path']).is_relative_to(root) for f in checked),'Model inventory outside model root')
    identity=dict(identity,files=checked,load_identity_scope='Fresh CPU full SHA256, metadata before/after each actual LLM load and after generation')
    return identity


def verify_model_metadata(identity):
    require(identity.get('files'),'Model source bytes have not been freshly verified')
    for reference in identity['files']:
        actual=stat_identity(reference['path'])
        require(actual=={k:reference[k] for k in actual},'Model changed since full CPU SHA binding')
        require(Path(reference['path']).resolve().is_relative_to(Path(identity['path']).resolve()),'Model path escaped bound root')
    return True


class RequestsTraceProxy:
    """Only the native module's requests binding changes; responses stay identical."""
    def __init__(self,original,handle,rank,model_identity,call_reader=lambda:None):
        self.original=original;self.handle=handle;self.rank=rank;self.model_identity=model_identity
        self.call_reader=call_reader;self.next_id=0;self.logging_errors=[]

    def __getattr__(self,name):return getattr(self.original,name)

    def _record(self,record):
        try:
            self.handle.write(json.dumps(record,ensure_ascii=False)+'\n');self.handle.flush()
        except Exception as exc:
            # An I/O failure cannot replace the original requests exception/Response.
            # No worker-complete receipt is accepted if even one record was lost.
            self.logging_errors.append(dict(type=type(exc).__name__,message=str(exc)))

    def post(self,*args,**kwargs):
        record=dict(schema='opsd-online-private-http-request-v1',request_id=self.next_id,rank=self.rank,
            started_at_unix=time.time(),sampling_model_identity=self.model_identity,generation_call=self.call_reader(),
            positional_args=list(args),keyword_args=kwargs)
        self.next_id+=1
        try:response=self.original.post(*args,**kwargs)
        except Exception as exc:
            record.update(ended_at_unix=time.time(),request_raised=dict(type=type(exc).__name__,message=str(exc)))
            self._record(record)
            raise  # same original exception object, never replaced by logging failure
        record['ended_at_unix']=time.time()
        try:
            record.update(status_code=response.status_code,response_body_text=response.text)
            try:record['response_json']=response.json()
            except Exception as exc:record['response_json_parse_error']=dict(type=type(exc).__name__,message=str(exc))
        except Exception as exc:
            self.logging_errors.append(dict(type=type(exc).__name__,message=str(exc)))
        self._record(record)
        return response  # caller still performs its original raise_for_status / JSON checks


def verify_http_trace(reference,rank,model_identity,expected_count):
    proof=checked_file(reference);records=read_rows(proof['path'])
    require(type(expected_count) is int and expected_count>=0 and len(records)==expected_count,'Worker HTTP trace count mismatch')
    for i,record in enumerate(records):
        require(record['schema']=='opsd-online-private-http-request-v1' and record['rank']==rank
                and record['request_id']==i and record['sampling_model_identity']==model_identity,'Worker HTTP trace provenance mismatch')
        require('request_raised' not in record and 'response_json_parse_error' not in record
                and isinstance(record.get('response_json'),dict),'Successful worker lost full HTTP response JSON')
        require(record['ended_at_unix']>=record['started_at_unix'] and isinstance(record['response_body_text'],str),
                'Incomplete successful HTTP trace')
    return proof


def _tag_balance(text):
    stack=[];wrong_closures=0;counts=Counter()
    for match in re.finditer(r'<(/?)(think|search|answer|information)>',text):
        close,tag=match.groups();counts[('/' if close else '')+tag]+=1
        if close:
            if not stack or stack[-1]!=tag:wrong_closures+=1
            else:stack.pop()
        else:stack.append(tag)
    return dict(balanced=not stack and wrong_closures==0,unclosed=list(stack),wrong_closures=wrong_closures,counts=dict(counts))


def rollout_diagnostics(row):
    """No joined text scoring and no new reward/selection criterion."""
    from search_r1.qa_protocol import parse_model_action
    spans=[row['response'][start:end] for start,end in row['policy_spans']]
    action=parse_model_action(spans[-1]) if spans else None
    balances=[_tag_balance(span) for span in spans]
    return dict(schema='opsd-online-rollout-diagnostics-v1',strict_reward=row['reward'],
        strict_last_action_valid=action is not None,strict_final_answer_valid=action is not None and action[0]=='answer',
        strict_final_action_kind=action[0] if action else None,termination=row['termination'],
        executed_search_calls=row['search_calls'],no_executed_search=row['search_calls']==0,
        all_policy_spans_tag_balanced=all(b['balanced'] for b in balances),policy_span_tag_balance=balances,
        joined_policy_text_tag_balance=_tag_balance(''.join(spans)),
        tag_balance_scope='Diagnostic only; separate spans and joined literal tag balance do not redefine native reward or executor validity')


def diagnostic_counts(rows):
    values=[rollout_diagnostics(row) for row in rows]
    return dict(denominator=len(values),strict_reward=dict(Counter(str(v['strict_reward']) for v in values)),
        strict_last_action_valid=sum(v['strict_last_action_valid'] for v in values),
        strict_final_answer_valid=sum(v['strict_final_answer_valid'] for v in values),
        strict_final_action_kind=dict(Counter(str(v['strict_final_action_kind']) for v in values)),
        termination=dict(Counter(v['termination'] for v in values)),
        executed_search_calls=dict(Counter(str(v['executed_search_calls']) for v in values)),
        no_executed_search=sum(v['no_executed_search'] for v in values),
        all_policy_spans_tag_balanced=sum(v['all_policy_spans_tag_balanced'] for v in values),
        joined_policy_text_tag_balanced=sum(v['joined_policy_text_tag_balance']['balanced'] for v in values),
        balance_is_diagnostic_only=True)


def cohort(config,cursor):
    """Only bound TRAIN rows. Cursor includes consumed boundary duplicates."""
    train=checked_file(config['train_identity']);validation=checked_file(config['validation_identity'])
    require(Path(train['path']).resolve()==Path(config['paths']['train']).resolve(),'TRAIN path not bound')
    require(Path(validation['path']).resolve()==Path(config['paths']['validation']).resolve(),'Validation path not bound')
    rows=read_rows(train['path']);heldout=read_rows(validation['path'])
    require(len(rows)==169103 and len({r['question_id'] for r in rows})==len(rows),'Wrong/duplicate TRAIN population')
    require(len(heldout)==512,'Wrong heldout population')
    forbidden_ids={r['question_id'] for r in heldout};forbidden_text={object_sha(r['question']) for r in heldout}
    cycle,offset=cursor['cycle'],cursor['offset'];n=len(rows);chosen=[];seen=set();skipped=[];permutation=None
    require(type(cycle) is int and cycle>=0 and type(offset) is int and 0<=offset<n,'Invalid TRAIN cursor')
    begin=dict(cursor)
    while len(chosen)<512:
        if permutation is None:
            permutation=list(range(n));random.Random(42+cycle).shuffle(permutation)
        index=permutation[offset];offset+=1
        if offset==n:offset=0;cycle+=1;permutation=None
        row=rows[index];qid=row['question_id']
        if qid in seen:skipped.append(dict(train_index=index,reason='boundary_duplicate_consumed'));continue
        require(qid not in forbidden_ids and object_sha(row['question']) not in forbidden_text,'Selected TRAIN overlaps heldout validation; fail rather than silently resample')
        require(set(row)>={'question_id','question','answers'},'Invalid TRAIN schema')
        seen.add(qid);chosen.append(dict(row))
    require(stat_identity(train['path'])=={k:train[k] for k in stat_identity(train['path'])},'TRAIN changed during selection')
    return chosen,dict(before=begin,after=dict(cycle=cycle,offset=offset),boundary_skips=skipped,
        train_identity=train,validation_identity=validation,question_ids_sha256=object_sha([r['question_id'] for r in chosen]),
        row_semantic_sha256=[object_sha(r) for r in chosen],selection_kind='TRAIN only; seed42 per-cycle deterministic permutation')


def merge_latest(config,checkpoint,output,checkpoint_ref):
    """Native CPU BF16 PEFT merge, latest raw adapter; deliberately no best reload."""
    require(not Path(output).exists(),'Merged cache path already exists')
    source=bind_model_files(config,dict(kind='raw-source100',path=config['paths']['model']))
    adapter=dict(path=checkpoint['adapter']['path'],files=[checked_file(f) for f in checkpoint['adapter']['files']])
    verify_model_metadata(source);verify_model_metadata(adapter)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    require(not torch.cuda.is_initialized(),'CPU merge must not initialize CUDA')
    base=AutoModelForCausalLM.from_pretrained(config['paths']['model'],torch_dtype=torch.bfloat16,device_map=None)
    require(all(p.device.type=='cpu' for p in base.parameters()),'Merge model left CPU')
    model=PeftModel.from_pretrained(base,checkpoint['adapter']['path'])
    model.merge_and_unload().save_pretrained(output,safe_serialization=True)
    AutoTokenizer.from_pretrained(config['paths']['model'],use_fast=True).save_pretrained(output)
    verify_model_metadata(source);verify_model_metadata(adapter)
    files=[fileproof(p) for p in sorted(Path(output).rglob('*')) if p.is_file()]
    require(files and any(Path(f['path']).suffix=='.safetensors' for f in files),'Merged model missing tensors')
    return dict(kind='original-PEFT-CPU-BF16-merge-latest',source_model_identity=config['source_model_manifest']['sha256'],
        checkpoint_optimizer_updates=checkpoint['optimizer_updates'],optimizer_updates=checkpoint['optimizer_updates'],
        checkpoint=checkpoint_ref,adapter_tensor_sha256=checkpoint['all_rank_adapter_sha256'][0],
        adapter=checkpoint['adapter'],source_load_proof=source,adapter_load_proof=adapter,path=str(Path(output).resolve()),
        files=files,sha256=object_sha(files),numerical_scope='BF16 merged forward need not be bitwise equal to unmerged LoRA; original E1 merge method retained')


def verify_rollouts(rows,questions,core):
    expected={q['question_id']:q for q in questions};seen=set();groups={q:[] for q in expected}
    require(len(questions)==512 and len(expected)==512,'Require 512 unique TRAIN questions')
    for row in rows:
        qid=row['question_id'];sibling=row['sibling'];key=qid,sibling
        require(qid in expected and type(sibling) is int and 0<=sibling<5 and key not in seen,'Missing/duplicate/unexpected rollout slot')
        require(row['question']==expected[qid]['question'] and row['answers']==expected[qid]['answers'],'Question/answer identity mismatch')
        require(row['reward']==core.reward(row['response'],row['answers'],row['policy_spans']),'Native strict score mismatch')
        require(type(row['search_calls']) is int and 0<=row['search_calls']<=4,'Impossible search budget')
        if 'diagnostics' in row:
            require(row['diagnostics']==rollout_diagnostics(row),'Stored diagnostics differ from actual authored spans')
        seen.add(key);groups[qid].append(row)
    require(len(seen)==2560 and all(len(g)==5 for g in groups.values()),'Incomplete 512x5 sampling')
    pairs=core.mine_pairs(rows)
    classes=Counter('all_wrong' if sum(r['reward'] for r in g)==0 else 'all_correct' if sum(r['reward'] for r in g)==5 else 'mixed' for g in groups.values())
    require(len(pairs)==512-classes['all_wrong'],'Native per-question mining P mismatch')
    return pairs,dict(questions=512,generated_trajectories=2560,P=len(pairs),classes=dict(classes),
        all_wrong_skipped=classes['all_wrong'],all_correct_success_fallback=classes['all_correct'],
        diagnostics=diagnostic_counts(rows),
        native_mining='min search/token reference, fail-preferred max full-response normalized rapidfuzz Levenshtein; first ties preserved')


def worker(config_ref,questions_path,model_identity_ref,rank,output,trace,http_trace):
    config,cp=checked_json(config_ref);identity,_=checked_json(model_identity_ref)
    require(0<=rank<8 and os.getuid()==config['owner']['uid'],'Wrong worker rank/UID')
    require(os.environ['CUDA_VISIBLE_DEVICES']==str(config['owner']['physical_gpus'][rank]),'Wrong worker GPU')
    require(os.environ['VLLM_ATTENTION_BACKEND']=='XFORMERS','Attention backend changed')
    require(os.environ['E1_INFERENCE_UTIL']==str(config['sampling']['gpu_memory_utilization']),'vLLM utilization changed')
    rows=read_rows(questions_path);require(len(rows)==64,'Worker must receive exactly64 TRAIN rows')
    require(identity['source_model_identity']==config['source_model_manifest']['sha256'],'Worker teacher/source identity mismatch')
    verify_model_metadata(identity)
    sys.path.insert(0,config['paths']['runtime'])
    try:from .scoring import verify_scoring
    except ImportError:from scoring import verify_scoring
    scoring_proof=verify_scoring(config)
    import torch
    from search_e1 import vllm_rollout as native
    from search_e1.core import reward
    torch.manual_seed(42+rank)
    original=native.LLM;original_requests=native.requests;call_state=dict(last_completed_generate=None)
    # Logging wrapper only: original LLM kwargs, generate inputs/params, returned objects unchanged.
    with Path(trace).open('x') as trace_handle,Path(http_trace).open('x') as http_handle:
        http_proxy=RequestsTraceProxy(original_requests,http_handle,rank,identity['sha256'],lambda:call_state['last_completed_generate'])
        class LoggedLLM:
            def __init__(self,*args,**kwargs):
                verify_model_metadata(identity)
                self.inner=original(*args,**kwargs);self.call=0
                verify_model_metadata(identity)
            def __getattr__(self,key):return getattr(self.inner,key)
            def generate(self,*args,**kwargs):
                start=time.time();result=self.inner.generate(*args,**kwargs);end=time.time()
                inputs=args[0] if args else kwargs.get('prompts',[])
                for index,out in enumerate(result):
                    generated=out.outputs[0]
                    trace_handle.write(json.dumps(dict(call=self.call,index=index,rank=rank,started=start,ended=end,
                        input_text_sha256=object_sha(inputs[index]),prompt_token_ids=list(out.prompt_token_ids),
                        generated_token_ids=list(generated.token_ids),native_output_text=generated.text,
                        finish_reason=generated.finish_reason,stop_reason=generated.stop_reason,
                        sampling_model_identity=identity['sha256']),ensure_ascii=False)+'\n')
                trace_handle.flush();call_state['last_completed_generate']=self.call;self.call+=1
                return result
        native.LLM=LoggedLLM;native.requests=http_proxy
        try:
            results=native.batched_rollouts(identity['path'],rows,5,config['paths']['retriever_url'],1.,42+rank,64)
            with Path(output).open('x') as target:
                for result in results:
                    result['reward']=reward(result['response'],result['answers'],result['policy_spans'])
                    result['diagnostics']=rollout_diagnostics(result)
                    target.write(json.dumps(result,ensure_ascii=False)+'\n');target.flush()
                os.fsync(target.fileno())
        finally:native.LLM=original;native.requests=original_requests
        trace_handle.flush();os.fsync(trace_handle.fileno())
        http_handle.flush();os.fsync(http_handle.fileno())
        require(not http_proxy.logging_errors,'HTTP logging lost records; completion rejected')
    verify_model_metadata(identity)
    exclusive_json(str(output)+'.worker-complete.json',dict(schema='opsd-online-sampling-worker-complete-v1',accepted=True,
        rank=rank,pid=os.getpid(),uid=os.getuid(),run_id=os.environ['E1_RUN_ID'],config_sha256=cp['sha256'],
        sampling_model_identity=identity['sha256'],rows=320,rollouts=fileproof(output),generation_trace=fileproof(trace),
        http_trace=fileproof(http_trace),http_requests=http_proxy.next_id,scoring_identity=scoring_proof,
        diagnostic_counts=diagnostic_counts(read_rows(output))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--questions',required=True);p.add_argument('--model-identity',required=True);p.add_argument('--model-identity-sha256',required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--trace',required=True);p.add_argument('--http-trace',required=True)
    a=p.parse_args();worker(dict(path=a.config,sha256=a.config_sha256),a.questions,
        dict(path=a.model_identity,sha256=a.model_identity_sha256),a.rank,a.output,a.trace,a.http_trace)


if __name__=='__main__':main()
