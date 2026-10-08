"""Unwired TP1 Qwen2 same-window weight observer, not a release/repair gate.

Inputs must be actual same-window FSDP logical tensors and vLLM live attributes;
this module cannot prove that provenance. It changes no model values or buffers.
A future runner must bind source/config/runtime identity and synchronize refusals.
"""
from dataclasses import dataclass
import hashlib
import torch


@dataclass(frozen=True)
class Mapping:
    target: str
    sources: tuple


def qwen2_tp1_mapping(layers):
    if type(layers) is not int or not 0 < layers <= 100:
        raise ValueError('Invalid explicit Qwen2 layer count')
    result = [Mapping('model.embed_tokens.weight', ('model.embed_tokens.weight',)),
              Mapping('lm_head.weight', ('lm_head.weight',)),
              Mapping('model.norm.weight', ('model.norm.weight',))]
    for i in range(layers):
        p = 'model.layers.%d.' % i
        for suffix in ('input_layernorm.weight', 'post_attention_layernorm.weight',
                       'self_attn.o_proj.weight', 'mlp.down_proj.weight'):
            result.append(Mapping(p+suffix, (p+suffix,)))
        for suffix in ('weight', 'bias'):
            result.append(Mapping(p+'self_attn.qkv_proj.'+suffix,
                          tuple(p+'self_attn.'+part+'_proj.'+suffix for part in ('q', 'k', 'v'))))
        result.append(Mapping(p+'mlp.gate_up_proj.weight',
                             tuple(p+'mlp.'+part+'_proj.weight' for part in ('gate', 'up'))))
    return tuple(result)


def tensor_identity(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError('Expected an actual tensor attribute')
    if value.layout != torch.strided or not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite or unsupported live tensor')
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0')
    data=value.detach().contiguous().reshape(-1).view(torch.uint8)
    for part in data.split(8*1024**2):
        h.update(part.cpu().numpy().tobytes())
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=h.hexdigest())


def compare_tp1_weights(source, actual, *, layers, tensor_parallel_size, expected_dtype,
                        tie_word_embeddings, vocab_size, hidden_size, intermediate_size,
                        q_rows, kv_rows, expected_device):
    """Exhaustive logical names/shapes and actual packed values, with bounded chunks.

    `source` contains already materialized same-window logical FSDP tensors. A
    runner must schedule DTensor full_tensor collectives identically on all ranks
    before calling this function. No hidden gather, .data swap, optimizer or mode
    change occurs here. Shape validation prevents broadcasting and transposition.
    All fragments are checked in actual Q/K/V and gate/up row order, without
    materializing the full concatenated model or treating equal values as aliases.
    """
    if type(tensor_parallel_size) is not int or tensor_parallel_size != 1:
        raise ValueError('This mapping only accepts the original TP1 layout')
    if expected_dtype not in (torch.bfloat16, torch.float32) or type(tie_word_embeddings) is not bool:
        raise ValueError('Explicit native dtype/tied configuration required')
    dimensions=(vocab_size,hidden_size,intermediate_size,q_rows,kv_rows)
    if any(type(x) is not int or x<=0 for x in dimensions) or q_rows!=hidden_size:
        raise ValueError('Invalid fixed Qwen2 dimensions')
    if not isinstance(source,dict) or not isinstance(actual,dict):
        raise ValueError('Explicit logical and live attribute dictionaries required')
    device=torch.device(expected_device)
    plan=qwen2_tp1_mapping(layers)
    expected_names={name for item in plan for name in item.sources}
    if set(source)!=expected_names or set(actual)!={item.target for item in plan}:
        raise ValueError('Logical source or actual live parameter coverage differs')
    if any(not isinstance(x,torch.Tensor) or x.device!=device for x in actual.values()):
        raise ValueError('Actual live parameter device differs from explicit window device')
    # Aliasing is a separate simultaneous live-reference proof.
    embedding, head=actual['model.embed_tokens.weight'],actual['lm_head.weight']
    if not isinstance(embedding,torch.Tensor) or not isinstance(head,torch.Tensor):
        raise ValueError('Tied identity requires actual live tensor references')
    live_alias=bool(torch._C._is_alias_of(embedding,head))
    tied_alias=live_alias and tuple(embedding.shape)==tuple(head.shape) and embedding.stride()==head.stride() and embedding.storage_offset()==head.storage_offset()
    if tied_alias != tie_word_embeddings:
        raise ValueError('Actual simultaneous tied alias differs from explicit configuration')
    if tie_word_embeddings and not torch.equal(source['model.embed_tokens.weight'], source['lm_head.weight']):
        raise ValueError('Tied logical source tensors contain different values')
    shapes={}
    for name in expected_names:
        if name in ('model.embed_tokens.weight','lm_head.weight'): shape=(vocab_size,hidden_size)
        elif name=='model.norm.weight' or name.endswith('layernorm.weight'): shape=(hidden_size,)
        elif '.self_attn.' in name:
            rows=q_rows if '.q_proj.' in name or '.o_proj.' in name else kv_rows
            shape=(rows,) if name.endswith('.bias') else (rows,hidden_size)
        elif name.endswith('mlp.down_proj.weight'): shape=(hidden_size,intermediate_size)
        else: shape=(intermediate_size,hidden_size)
        x=source[name]
        if not isinstance(x,torch.Tensor) or tuple(x.shape)!=shape or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('Source shape/master dtype/finite mismatch: '+name)
        shapes[name]=shape
    records=[]
    for item in plan:
        target=actual[item.target]
        expected_shape=shapes[item.sources[0]] if len(item.sources)==1 else (sum(shapes[name][0] for name in item.sources),)+shapes[item.sources[0]][1:]
        if not isinstance(target,torch.Tensor) or tuple(target.shape)!=expected_shape or target.dtype!=expected_dtype:
            raise ValueError('Actual packed shape/dtype mismatch: '+item.target)
        identity=tensor_identity(target)
        offset=0;different=0;maximum=0.0;fragments=[]
        for name in item.sources:
            original=source[name];count=original.shape[0]
            view=target[offset:offset+count];offset+=count
            expected=original.to(device=view.device,dtype=expected_dtype)
            unequal=0;max_delta=0.0
            for left,right in zip(view.reshape(-1).split(1024*1024),expected.reshape(-1).split(1024*1024)):
                unequal+=int((left!=right).sum())
                max_delta=max(max_delta,float((left.double()-right.double()).abs().max()))
            different+=unequal;maximum=max(maximum,max_delta)
            fragments.append(dict(source=name,source_master_identity=tensor_identity(original),expected_cast_identity=tensor_identity(expected),actual_fragment_identity=tensor_identity(view),different_elements=unequal,max_abs_delta=max_delta))
            del expected
        records.append(dict(target=item.target,actual_identity=identity,fragments=fragments,different_elements=different,max_abs_delta=maximum,values_match=different==0))
    return dict(status='SAME_WINDOW_PARAMETER_VALUE_OBSERVATION_NO_ACCEPTANCE',
                tensor_parallel_size=1,actual_parameter_device=str(device),actual_device_is_cuda=device.type=='cuda',
                serialized_source_keys=len(source),serialized_target_keys=len(actual),
                all_parameter_values_match=all(row['values_match'] for row in records),
                different_elements=sum(row['different_elements'] for row in records),
                max_abs_delta=max(row['max_abs_delta'] for row in records),
                same_window_tied_alias=tied_alias,records=records,
                buffers_compared=False,live_window_provenance_proven_by_this_function=False,
                historical_3268_identity_proven=False,probability_accepted=False,training_accepted=False)
