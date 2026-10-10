"""Separate privileged teacher self-generation. Never a training-q replay."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

try:
    from .checkpoint import require,checked_json,checked_file,fileproof,object_sha,exclusive_json
    from .sampling import read_rows,verify_model_metadata,RequestsTraceProxy,rollout_diagnostics
    from .scoring import verify_scoring
    from .prepare import validate_config
except ImportError:
    from checkpoint import require,checked_json,checked_file,fileproof,object_sha,exclusive_json
    from sampling import read_rows,verify_model_metadata,RequestsTraceProxy,rollout_diagnostics
    from scoring import verify_scoring
    from prepare import validate_config

VERSION='opsd-ablation-privileged-teacher-autoregressive-v1'
STRATA=('self_pair','CW','CC')


def select_cases(pairs,outer_index):
    require(type(outer_index) is int and 0<=outer_index<10,'Diagnostic outer0..9 required')
    pairs=list(pairs);require(len(pairs)<=512 and len({p['question_id'] for p in pairs})==len(pairs),'Unique eligible pairs required')
    pools={key:[] for key in STRATA}
    for index,pair in enumerate(pairs):
        require(pair['reference_reward']==1,'Autonomous diagnostic cannot invent successful B')
        m=pair['mining'];kind='self_pair' if m['self_pair_by_sibling'] else m['pair_class']
        require(kind in pools,'Unknown mutually exclusive teacher diagnostic stratum')
        digest=object_sha([VERSION,42,outer_index,pair['question_id']])
        pools[kind].append(dict(pair_index=index,question_id=pair['question_id'],stratum=kind,
            selection_sha256=digest,pair_sha256=object_sha(pair)))
    for pool in pools.values():pool.sort(key=lambda row:(row['selection_sha256'],row['question_id']))
    selected=[];offset={key:0 for key in STRATA}
    while len(selected)<min(8,len(pairs)):
        for kind in STRATA:
            if offset[kind]<len(pools[kind]) and len(selected)<8:
                selected.append(pools[kind][offset[kind]]);offset[kind]+=1
    return dict(schema=VERSION,seed=42,outer_index=outer_index,max_cases=8,eligible_pairs=len(pairs),
        selected=selected,count=len(selected),available_strata={key:len(pools[key]) for key in STRATA},
        selected_strata=dict(Counter(row['stratum'] for row in selected)),missing_strata=[key for key in STRATA if not pools[key]],
        selection_rule='SHA256 canonical [version,seed42,outer,qid], stable lexicographic hash/qid within stratum; round-robin self_pair,CW,CC',
        not_applicable=len(pairs)==0,not_applicable_reason='No eligible successful B; all-wrong questions are not supplemented' if not pairs else None,
        diagnostic_teacher_autoregressive=True,original_training_objective=False,diagnostic_precision_common_bf16=True)


def snapshot(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(k):snapshot(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [snapshot(v) for v in value]
    if hasattr(value,'detach'):return value.detach().cpu().tolist()
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'to_dict'):return snapshot(value.to_dict())
    return dict(type=type(value).__module__+'.'+type(value).__qualname__)


@contextmanager
def capture(agent,native,case,http_handle):
    """Transparent records scoped to one original SearchAgent call."""
    model,tokenizer=agent.model,agent.tokenizer
    events=[];errors=[];call=[0]
    def record(make_item):
        try:events.append(make_item())
        except Exception as exc:errors.append(type(exc).__name__)
    class Tokenizer:
        def __getattr__(self,name):return getattr(tokenizer,name)
        def __call__(self,*a,**kw):
            out=tokenizer(*a,**kw);record(lambda:dict(kind='tokenize_context',args=snapshot(a),kwargs=snapshot(kw),ids=snapshot(out.input_ids)));return out
        def encode(self,*a,**kw):
            out=tokenizer.encode(*a,**kw);record(lambda:dict(kind='encode_before_observation_truncation',args=snapshot(a),kwargs=snapshot(kw),ids=snapshot(out)));return out
        def decode(self,*a,**kw):
            out=tokenizer.decode(*a,**kw);record(lambda:dict(kind='decode_actual_selected_ids',args=snapshot(a),kwargs=snapshot(kw),text=out));return out
    generated=model.generate;own_generate='generate' in vars(model);old_generate=vars(model).get('generate')
    original_prompt=native.prompt;original_requests=native.requests
    proxy=RequestsTraceProxy(original_requests,http_handle,0,'source100-teacher-diagnostic',lambda:call[0]-1 if call[0] else None)
    def privileged(question):
        require(question==case['question'],'Privileged reference leaked across diagnostic cases')
        return original_prompt(question,reference=case['reference'])
    def generate(*a,**kw):
        start=time.time();out=generated(*a,**kw)
        record(lambda:dict(kind='native_generate',call=call[0],started_unix=start,ended_unix=time.time(),
            args=snapshot(a),kwargs=snapshot(kw),raw_output_ids=snapshot(out.sequences if hasattr(out,'sequences') else out)))
        call[0]+=1;return out
    try:
        native.prompt=privileged;native.requests=proxy;model.generate=generate;agent.tokenizer=Tokenizer()
        yield dict(schema=VERSION,events=events,logging_errors=errors,http_proxy=proxy,
            teacher_forced=False,original_training_objective=False,diagnostic_teacher_autoregressive=True,
            diagnostic_precision_common_bf16=True,return_objects_modified=False,generation_options_modified=False)
    finally:
        native.prompt=original_prompt;native.requests=original_requests;agent.tokenizer=tokenizer
        if own_generate:model.generate=old_generate
        else:delattr(model,'generate')


def run(request_ref,receipt_path):
    request,request_proof=checked_json(request_ref);config,config_proof=checked_json(request['config'])
    validate_config(config,launch=True)
    require(config['run_id']==request['run_id']==os.environ.get('E1_RUN_ID') and os.getuid()==config['owner']['uid'],
            'Teacher diagnostic run/UID changed')
    physical=config['owner']['physical_gpus'][0]
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical),'Teacher diagnostic must use its bound GPU0 child')
    require(config['diagnostics']['teacher_precision']=='bf16' and config['diagnostics']['teacher_max_cases_per_outer']==8,
            'Common diagnostic protocol changed')
    scoring=verify_scoring(config);native=importlib.import_module('search_e1.rollout');core=importlib.import_module('search_e1.core')
    pairs_ref=checked_file(request['pairs']);pairs=read_rows(pairs_ref['path'])
    selection=select_cases(pairs,request['outer_index'])
    require(selection==checked_json(request['selection'])[0] and selection['count']>0,'Diagnostic selection differs or no B')
    source=request['source_identity'];require(source['source_model_identity']==config['source_model_manifest']['sha256']
        and source['kind']=='raw-source100' and Path(source['path']).resolve()==Path(config['paths']['model']).resolve(),
            'Diagnostic teacher/source identity changed')
    verify_model_metadata(source)
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM,AutoTokenizer
    # This process is independent: no parent/student training RNG is read or restored.
    import random
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    before_rng=dict(torch_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    tokenizer=AutoTokenizer.from_pretrained(config['paths']['model'],use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(config['paths']['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    require(all(p.dtype==torch.bfloat16 for p in model.parameters()),'Common diagnostic base is not actualBF16')
    verify_model_metadata(source)
    agent=native.SearchAgent(model,tokenizer,config['paths']['retriever_url'],max_new_tokens=500,max_context=8192)
    root=Path(request['output_root']);require(not root.exists(),'Teacher output already exists; partial preserved')
    root.mkdir(parents=True);cases=[]
    for slot,item in enumerate(selection['selected']):
        pair=pairs[item['pair_index']];require(object_sha(pair)==item['pair_sha256'],'Selected actual B/D changed')
        http_path=root/f'case-{slot:02d}-http.jsonl'
        with http_path.open('x') as http:
            with capture(agent,native,pair,http) as trace:
                result=agent.run(pair['question'],temperature=0.)
            http.flush();os.fsync(http.fileno())
            proxy=trace.pop('http_proxy');require(not proxy.logging_errors and not trace['logging_errors'],'Diagnostic transparent logs incomplete')
        result=dict(result,question_id=pair['question_id'],answers=pair['answers'])
        result['reward']=core.reward(result['response'],pair['answers'],result['policy_spans'])
        result['diagnostics']=rollout_diagnostics(result)
        private=dict(schema=VERSION,diagnostic_teacher_autoregressive=True,original_training_objective=False,
            diagnostic_precision_common_bf16=True,run_id=config['run_id'],variant=config['variant'],outer_index=request['outer_index'],
            selected=item,source_model_identity=config['source_model_manifest']['sha256'],original_pairs=pairs_ref,
            original_B=pair['reference'],original_D=pair['student'],result=result,trace=trace,http=fileproof(http_path),
            original_teacher_training_q_scope='Teacher own history is not the original D history training q',
            actual_generation_config=model.generation_config.to_dict())
        cases.append(exclusive_json(root/f'case-{slot:02d}.json',private))
    verify_model_metadata(source)
    exclusive_json(receipt_path,dict(schema='opsd-ablation-teacher-diagnostic-child-v1',accepted=True,
        request=request_proof,selection=request['selection'],pairs=pairs_ref,run_id=config['run_id'],variant=config['variant'],
        config_sha256=config_proof['sha256'],pid=os.getpid(),uid=os.getuid(),count=len(cases),cases=cases,
        source_identity=source,scoring_identity=scoring,diagnostic_teacher_autoregressive=True,original_training_objective=False,
        diagnostic_precision_common_bf16=True,training_or_validation_denominator_contribution=0,
        rng_isolation='Independent child, fixedseed42; does not consume sampling/trainer RNG',initial_rng=before_rng))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--receipt',required=True);a=p.parse_args();run(dict(path=a.request,sha256=a.request_sha256),a.receipt)

if __name__=='__main__':main()
