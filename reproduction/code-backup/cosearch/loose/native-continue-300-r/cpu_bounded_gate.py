"""Actual native CPU DataProto retry call-chain; no CUDA/model/PPO updates."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from verl import DataProto
from bounded_sampling import generate_supported,input_digest
from phase_sampling import require_training_support

def run(supports,validation=False,fail=None):
    batch=DataProto.from_dict(tensors={'input_ids':torch.arange(16).reshape(8,2)},
        non_tensors={'uid':np.array(['q'+str(i) for i in range(8)],dtype=object)},
        meta_info={'global_steps':10,'validate':validation,'reranker_sampling_params':{'temperature':1.,'top_p':1.}})
    before=input_digest(batch);support=[];calls=[];events=[];raw=[];outputs=[]
    rng=torch.get_rng_state().clone()
    def original(forwarded):
        assert input_digest(forwarded)==before
        calls.append(forwarded);torch.rand(1)
        support.append(supports[len(calls)-1])
        if fail=='generation':raise RuntimeError('original failure')
        main=DataProto.from_dict(tensors={'responses':torch.ones(8,2,dtype=torch.long),'attention_mask':torch.ones(8,4,dtype=torch.long)})
        ranker=DataProto.from_dict(tensors={'responses':torch.ones(4,2,dtype=torch.long),'attention_mask':torch.ones(4,4,dtype=torch.long),'rm_scores':torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])},non_tensors={'uid':np.array(['same']*4,dtype=object)})
        result=main if validation else (main,ranker)
        if fail=='malformed':result=object()
        outputs.append(result)
        forwarded.meta_info['mutated_by_manager']=True
        return result
    def record(kind,value):events.append((kind,value))
    def preserve(kind,value):
        if fail=='evidence':raise OSError('evidence failure')
        raw.append(value)
    def reject(*a):
        if fail=='context':raise RuntimeError('context rejection')
    failed=False
    try:
        result=generate_supported(original,batch,support,require_training_support,record,preserve,reject,10,validation)
        assert result is outputs[-1]
    except (AssertionError,RuntimeError,OSError):failed=True
    assert input_digest(batch)==before
    final_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    for _ in calls:torch.rand(1)
    assert torch.equal(final_rng,torch.get_rng_state())
    expected_calls=1 if validation or fail or supports[0]>0 else 2
    assert len(calls)==expected_calls
    assert failed==bool(fail or (not validation and supports[-1]==0))
    for item in raw:
        assert item['input_sha256']==before and item['attempt'] in (1,2)
    return dict(supports=supports,validation=validation,failure=fail,calls=len(calls),rejected=failed,
                unchanged_original_input=True,original_RNG_advanced_naturally=True,
                PPO_updates=0,optimizer_updates=0,scheduler_updates=0)
results=[run([1]),run([0,1]),run([0,0]),run([0],validation=True)]
results += [run([0],fail=x) for x in ['generation','evidence','context','malformed']]
a=DataProto.from_dict(tensors={'input_ids':torch.ones(1,1)},meta_info={'x':[1,23]})
b=DataProto.from_dict(tensors={'input_ids':torch.ones(1,1)},meta_info={'x':[12,3]})
assert input_digest(a)!=input_digest(b)
# Execute the actual submitted nested generation wrapper, not a reimplementation.
import ast,types
source=ast.parse(Path(__file__).with_name('launch_observed.py').read_text())
node=next(x for x in ast.walk(source) if isinstance(x,ast.FunctionDef) and x.name=='generate')
batch=DataProto.from_dict(tensors={'input_ids':torch.ones(8,2,dtype=torch.long)},non_tensors={'uid':np.array(['q']*8,dtype=object)},meta_info={'global_steps':10})
support=[];events=[];attempts=[]
def original(b):
    attempts.append(input_digest(b));support.append(0 if len(attempts)==1 else 1)
    main=DataProto.from_dict(tensors={'responses':torch.ones(8,2,dtype=torch.long),'attention_mask':torch.ones(8,4,dtype=torch.long)})
    ranker=DataProto.from_dict(tensors={'responses':torch.ones(4,2,dtype=torch.long),'attention_mask':torch.ones(4,4,dtype=torch.long),'rm_scores':torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])},non_tensors={'uid':np.array(['same']*4,dtype=object)})
    return main,ranker
import sys
previous=sys.modules.get('observe_worker')
sys.modules['observe_worker']=types.SimpleNamespace(preserve_raw_payload=lambda *a:None)
namespace=dict(write=lambda k,v:events.append((k,v)),original_generate=original,qualified_support=support,phase_sampling=types.SimpleNamespace(require_training_support=require_training_support),reject_context_before_ppo=lambda *a:None)
try:
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-R-generate','exec'),namespace)
    namespace['generate'](batch)
finally:
    if previous is None:sys.modules.pop('observe_worker')
    else:sys.modules['observe_worker']=previous
assert len(attempts)==2 and attempts[0]==attempts[1]
assert sum(k=='generation_start' for k,v in events)==sum(k=='generation_end' for k,v in events)==1
out=dict(passed=True,actual_submitted_wrapper_executed=True,framed_input_digest_collision_case_passed=True,cases=results,CUDA_initialized=torch.cuda.is_initialized(),RL_updates=0,
          retry_limit=1,original_zero_support_gate_preserved=True,budget_and_selection_changed=True)
assert not out['CUDA_initialized']
Path(__file__).with_name('cpu-bounded-gate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
