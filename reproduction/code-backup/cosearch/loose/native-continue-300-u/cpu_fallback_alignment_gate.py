"""Actual Ray modified-class fallback/author alignment; CPU only, 0 PPO."""
import copy, hashlib, json, random
from pathlib import Path
import torch
from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import _InternalAgentLoopOutput, AgentLoopMetrics
import fallback_alignment

worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
instance=worker.__new__(worker)
original=worker._fallback_keep_outputs_to_align
align=worker._align_to_multiple
records=[]; payloads=[]
fallback_alignment.install(worker,lambda k,v:records.append((k,v)),lambda k,v:payloads.append((k,v)))
installed=worker._fallback_keep_outputs_to_align
fallback_alignment.install(worker,lambda *a:None,lambda *a:None)
assert worker._fallback_keep_outputs_to_align is installed

def row(i):
    return _InternalAgentLoopOutput(prompt_ids=torch.tensor([10,i]),response_ids=torch.tensor([20,i]),
        input_ids=torch.tensor([10,i,20,i]),position_ids=torch.arange(4),
        response_mask=torch.tensor([1,0]),attention_mask=torch.ones(4,dtype=torch.long),
        reward_score=float(i%2),metrics=AgentLoopMetrics(),extra_fields={'uid':str(i),'nested':[i]})
def plain(x):
    if isinstance(x,torch.Tensor):return {'dtype':str(x.dtype),'shape':list(x.shape),'data':x.tolist()}
    if hasattr(x,'model_dump'):return plain(x.model_dump())
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

results=[]
for n in range(6):
    for empty_group in (False,True):
        groups={}
        if empty_group:groups['empty']=[]
        for i in range(n):groups.setdefault(str(i%2),[]).append(row(i))
        before=plain(groups); rng=random.getstate(); r0=len(records); p0=len(payloads)
        if n==0 and empty_group:
            for method in (original,installed):
                try:method(instance,groups,4)
                except AssertionError as e:assert 'got 0, expected 4' in str(e)
                else:raise AssertionError('original all-empty assertion concealed')
            assert plain(groups)==before and random.getstate()==rng
            results.append({'rows':0,'empty_group':True,'original_error_preserved':True})
            continue
        expected=(align(instance,[g for g in groups.values() if g],4)[0]
                  if 0<n<4 else original(instance,groups,4))
        expected_rng=random.getstate();random.setstate(rng)
        try:actual=installed(instance,groups,4)
        except AssertionError:
            # Nonempty dict of exclusively empty groups still fails original fallback.
            assert n==0 and empty_group
            raise
        assert plain(actual)==plain(expected) and random.getstate()==expected_rng
        assert plain(groups)==before
        flattened=[r for g in groups.values() for r in g]
        assert all(actual[i] is flattened[i] for i in range(min(n,4)))
        if 0<n<4:
            assert len(actual)==4 and len(payloads)-p0==2
            record=records[-1][1]
            assert record['original_rows']==n and record['added_rows']==4-n
            assert record['qualified_ranker_learning'] is False
            for duplicate in actual[n:]:
                assert all(duplicate is not r for r in flattened)
                assert all(duplicate.response_ids.data_ptr()!=r.response_ids.data_ptr() for r in flattened)
                duplicate.extra_fields['nested'].append(99)
            assert plain(groups)==before
        else:assert len(records)==r0 and len(payloads)==p0
        results.append({'rows':n,'empty_group':empty_group,'original_helper_or_fallback_exact':True})

# Failures must not silently become fallback outputs.
for failure in ('preserve_before','preserve_after','record','helper','mismatch'):
    class Isolated(worker):pass
    Isolated._fallback_keep_outputs_to_align=original
    Isolated._native_underfull_fallback_aligned=False
    obj=Isolated.__new__(Isolated); counts=[]
    def preserve(k,v):
        counts.append(k)
        if failure=='preserve_before' or (failure=='preserve_after' and len(counts)==2):raise OSError(failure)
    def record(k,v):
        if failure=='record' and k!='underfull_fallback_alignment_installed':raise OSError(failure)
    fallback_alignment.install(Isolated,record,preserve)
    if failure=='helper':
        def bad(*a):raise OSError('helper')
        Isolated._align_to_multiple=bad
    if failure=='mismatch':Isolated._align_to_multiple=lambda *a: ([],0)
    try:obj._fallback_keep_outputs_to_align({'a':[row(1)]},4)
    except OSError as e:assert str(e)==failure
    except AssertionError:assert failure=='mismatch'
    else:raise AssertionError('failure concealed')
assert not torch.cuda.is_initialized()
output={'passed':True,'cases':results,'failure_propagation':5,'actual_ray_modified_class':True,
        'source_sha256':hashlib.sha256(Path(__import__('sys').modules[worker.__module__].__file__).read_bytes()).hexdigest(),
        'fix_sha256':hashlib.sha256(Path(fallback_alignment.__file__).read_bytes()).hexdigest(),
        'CUDA_initialized':False,'PPO_updates':0,'qualification_not_fabricated':True}
print(json.dumps(output),flush=True)
Path(__file__).with_name('cpu-fallback-alignment-gate.json').write_text(json.dumps(output,indent=2)+'\n')
