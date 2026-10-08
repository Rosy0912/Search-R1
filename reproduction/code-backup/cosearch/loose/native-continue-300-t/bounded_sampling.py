"""Disclosed one-retry whole-batch rejection sampling, never fallback PPO.

Changes rollout budget and conditions batch acceptance. Original model kwargs,
filter/reward/min4, PPO/Adam and RNG are untouched. No exception is retried.
"""
import copy, hashlib, json, time

def input_digest(batch):
    import numpy as np
    import torch
    digest=hashlib.sha256()
    def feed(data):
        digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    def visit(value):
        if torch.is_tensor(value):
            assert value.device.type=='cpu', 'Rollout input fingerprint must stay on CPU'
            feed(json.dumps(['tensor',str(value.dtype),list(value.shape)]).encode())
            feed(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,np.ndarray):
            feed(json.dumps(['array',str(value.dtype),list(value.shape)]).encode())
            if value.dtype.hasobject:
                for element in value.flat:visit(element)
            else:feed(value.tobytes())
        elif isinstance(value,np.generic):visit(value.item())
        elif isinstance(value,dict):
            feed(json.dumps(['dict',len(value)]).encode())
            for key in sorted(value):visit(key);visit(value[key])
        elif isinstance(value,(tuple,list)):
            feed(json.dumps([type(value).__name__,len(value)]).encode())
            for element in value:visit(element)
        elif value is None or isinstance(value,(str,int,float,bool)):
            feed(json.dumps(value,allow_nan=False).encode())
        else:raise TypeError('Unsupported rollout input fingerprint type '+type(value).__name__)
    visit(dict(tensors=dict(batch.batch.items()) if batch.batch is not None else None,
               non_tensors=batch.non_tensor_batch,metadata=batch.meta_info))
    return digest.hexdigest()

def validate_result(result,rows,validation):
    import torch
    from verl import DataProto
    if validation:
        assert isinstance(result,DataProto) and len(result)==rows
        return
    assert isinstance(result,tuple) and len(result)==2
    main,ranker=result
    assert isinstance(main,DataProto) and len(main)==rows
    assert isinstance(ranker,DataProto) and len(ranker)>0
    for value in (main,ranker):
        assert value.batch is not None
        for key in ('responses','attention_mask'):
            assert key in value.batch and value.batch[key].ndim==2
            assert value.batch[key].shape[0]==len(value)
    assert 'rm_scores' in ranker.batch and 'uid' in ranker.non_tensor_batch
    reward=ranker.batch['rm_scores']
    assert reward.ndim==2 and reward.shape[0]==len(ranker)
    assert len(ranker.non_tensor_batch['uid'])==len(ranker)
    assert torch.isfinite(reward).all().item(), 'Nonfinite original ranker rewards'

def generate_supported(original,batch,support,require,record,preserve,reject,step,validation):
    # Pristine copies isolate manager mutation without changing model-facing data.
    pristine=copy.deepcopy(batch)
    expected=input_digest(pristine)
    count=1 if validation else 2
    for attempt in range(1,count+1):
        forwarded=copy.deepcopy(pristine)
        assert input_digest(forwarded)==expected
        support.clear()
        started=time.monotonic_ns()
        record('rollout_attempt_start',dict(step=step,validation=validation,attempt=attempt,
               input_sha256=expected,input_rows=len(forwarded),max_attempts=count))
        result=original(forwarded)
        preserve('controller-generation-return',dict(step=step,validation=validation,
                 attempt=attempt,input_sha256=expected,result=result))
        reject(step,validation)
        validate_result(result,len(forwarded),validation)
        assert validation or len(support)==1, 'Complete original support evidence missing'
        qualified=None if validation else support[0]
        assert validation or qualified>=0
        record('rollout_attempt_return',dict(step=step,validation=validation,attempt=attempt,
               input_sha256=expected,qualified_groups=qualified,
               elapsed_seconds=(time.monotonic_ns()-started)/1e9,
               discarded_before_both_PPO=not validation and qualified==0 and attempt<count))
        if validation:return result
        if qualified>0:
            require(support)
            return result
        if attempt==count:
            require(support)  # Existing exact zero-support hard rejection.
            record('zero_support_original_fallback_forwarded',dict(step=step,attempt=attempt,input_sha256=expected,qualified_ranker_learning=False,diagnostic_only=True))
            return result
        del result,forwarded
        record('zero_support_whole_batch_discarded',dict(step=step,attempt=attempt,
               next_attempt=attempt+1,input_sha256=expected,extra_rollout_budget_disclosed=True,
               both_PPO_calls=0,optimizer_calls=0,scheduler_calls=0))
    raise AssertionError('Unreachable')
