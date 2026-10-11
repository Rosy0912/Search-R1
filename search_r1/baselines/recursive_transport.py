"""RPC transport slots are not generated or selected training trajectories."""
import copy
import numpy as np
import torch
from verl.protocol import DataProto, unpad_dataproto

PROTOCOL='recursive-zero-policy-transport-v1'
FIELD='recursive_transport_padding'

def inference_call(worker, data, world):
    """Duplicate only temporary forward slots, never generation, then discard."""
    if len(data)==0:raise ValueError('No real rows for inference')
    missing=(-len(data))%world
    if not missing:return worker(data)
    rows=[data]+[data[i%len(data):i%len(data)+1] for i in range(missing)]
    padded=DataProto.concat(rows)
    return unpad_dataproto(worker(padded),missing)

def actor_slots(data, capacity=2560):
    """Keep the exact empirical question objective under the fixed row divisor.

    Weights already equal real_rows/(questions*n_q*T_qi). Multiplying by
    capacity/real_rows cancels the actor's capacity row denominator. Dummy
    slots inherit a real native forward solely for valid attention/provenance;
    their policy loss mask, advantage and objective weight are exactly zero.
    """
    n=len(data)
    if not 0<n<=capacity:raise ValueError('Invalid real recursive row count')
    if FIELD in data.non_tensor_batch:raise ValueError('Transport slots already installed')
    if n==capacity:return data
    if not {'policy_weights','loss_mask','advantages'}.issubset(data.batch.keys()):
        raise ValueError('Transport padding requires completed real loss inputs')
    result=copy.deepcopy(data)
    result.batch['policy_weights']*=capacity/n
    result.non_tensor_batch[FIELD]=np.array([False]*n,dtype=bool)
    dummy=copy.deepcopy(result[:1])
    dummy.non_tensor_batch[FIELD]=np.array([True],dtype=bool)
    for key in ('policy_weights','loss_mask','advantages','returns','token_level_rewards','token_level_scores'):
        if key in dummy.batch:dummy.batch[key].zero_()
    result=DataProto.concat([result]+[copy.deepcopy(dummy) for _ in range(capacity-n)])
    result.meta_info=dict(result.meta_info,recursive_transport_protocol=PROTOCOL,
                         recursive_real_rows=n,recursive_transport_slots=capacity-n,
                         recursive_capacity=capacity)
    return result

def is_zero_policy_slot(data,row):
    if FIELD not in data.non_tensor_batch:return False
    flags=data.non_tensor_batch[FIELD]
    if not isinstance(flags,np.ndarray) or flags.dtype!=np.bool_ or len(flags)!=len(data):
        raise ValueError('Malformed transport slot flags')
    flag=bool(flags[row])
    if not flag:return False
    if (data.meta_info.get('recursive_transport_protocol')!=PROTOCOL
        or data.meta_info.get('recursive_capacity')!=2560
        or not 0<data.meta_info.get('recursive_real_rows',0)<2560
        or data.meta_info.get('recursive_transport_slots')!=2560-data.meta_info['recursive_real_rows']):
        raise ValueError('Unbound dummy transport slot')
    for key in ('policy_weights','loss_mask','advantages'):
        if key not in data.batch or torch.count_nonzero(data.batch[key][row]).item()!=0:
            raise ValueError('Transport slot has nonzero policy contribution: '+key)
    return True
