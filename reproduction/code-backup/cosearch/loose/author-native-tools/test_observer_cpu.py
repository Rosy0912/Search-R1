"""Synthetic CPU parity diagnostics; never counted as real training acceptance."""
import json, os, tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
import observe_worker

def actor(lr):
    parameter=torch.nn.Parameter(torch.arange(12,dtype=torch.float32)/20)
    return SimpleNamespace(parameter=parameter,actor_optimizer=torch.optim.AdamW([parameter],lr=lr),
                           _native_observed_role='main')

def original(a,data):
    a.actor_optimizer.zero_grad()
    loss=(a.parameter*torch.arange(1,13,dtype=torch.float32)).square().sum()
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([a.parameter],1.).item()
    a.actor_optimizer.step()
    a.actor_optimizer.zero_grad()
    return {'actor/grad_norm':[norm],'actor/pg_loss':[loss.item()]}

results=[]
with tempfile.TemporaryDirectory() as folder:
    os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']=folder
    for lr in [0.,1e-3]:
        base=actor(lr); observed=actor(lr)
        data=SimpleNamespace(non_tensor_batch={'uid':['synthetic-uid']},batch={'advantages':torch.tensor([[1.,-1.,0.]]),
                                   'response_mask':torch.tensor([[1,1,0]])})
        untouched={k:v.clone() for k,v in data.batch.items()}
        rng_before=torch.get_rng_state().clone()
        control=original(base,data)
        with patch('torch.distributed.get_rank',return_value=0),patch('torch.distributed.get_world_size',return_value=1):
            result=observe_worker.observe_update(original,observed,data)
        assert control==result
        assert torch.equal(base.parameter,observed.parameter)
        a=base.actor_optimizer.state[base.parameter];b=observed.actor_optimizer.state[observed.parameter]
        assert all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
        assert torch.equal(rng_before,torch.get_rng_state())
        assert all(torch.equal(v,data.batch[k]) for k,v in untouched.items())
        record=json.loads((next((__import__('pathlib').Path(folder)/'workers').glob('*.jsonl'))).read_text().splitlines()[-1])
        assert record['actual_adam_delta']==1 and record['local_masked_nonzero_advantage_tokens']==2
        assert (record['parameter_probe_delta_l1']>0)==(lr>0)
        results.append({'lr':lr,'exact_parameters_and_adam_state':True,'rng_unchanged':True,
                        'data_unchanged':True,'actual_adam_delta':1,'parameter_delta':record['parameter_probe_delta_l1']})
print(json.dumps({'synthetic_cpu_only':True,'real_rl_updates':0,'cases':results,'passed':True}))
