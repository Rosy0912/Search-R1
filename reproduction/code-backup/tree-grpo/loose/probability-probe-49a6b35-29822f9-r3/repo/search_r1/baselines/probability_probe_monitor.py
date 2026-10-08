"""Frozen monitor/raw exact-context range evidence; never average duplicates."""
import hashlib
import struct

import torch

from .probability_probe_capture import require

PREFIX_VERSION=b'TreeSamplingContext/v1\x00'
COMPANION_KEYS=('matched_count','sampled_min','sampled_max','exact_context_key','effective_mask','matched_mask')


def exact_monitor_keys(data):
    """Exact same v1 prefix definition, includes loss-masked observation tokens."""
    ids=data['input_ids'];response=data['responses'];start=ids.shape[1]-response.shape[1]
    output=torch.zeros((*response.shape,48),dtype=torch.uint8)
    for row in range(ids.shape[0]):
        prefix=hashlib.sha256(PREFIX_VERSION)
        for offset,(token,position,attention) in enumerate(zip(ids[row].tolist(),data['position_ids'][row].tolist(),data['attention_mask'][row].tolist())):
            if not attention:continue
            pair=struct.pack('<qq',int(token),int(position));response_offset=offset-start
            if response_offset>=0 and bool(data['loss_mask'][row,response_offset]):
                output[row,response_offset]=torch.tensor(list(prefix.digest()+pair),dtype=torch.uint8)
            prefix.update(pair)
    return output


def bind_monitor_companion(payload,data,expected_identity,monitor_asset,monitor_sha):
    require(payload.get('identity')==expected_identity,'Frozen sampled companion identity differs')
    require(payload.get('monitor_asset')==monitor_asset and payload.get('monitor_asset_sha256')==monitor_sha,
        'Frozen sampled companion refers to another monitor asset')
    require(payload.get('prefix_version')==PREFIX_VERSION.decode().replace('\x00','\\0'),'Causal-prefix version differs')
    shape=data['responses'].shape
    expected_dtypes=dict(matched_count=torch.int64,sampled_min=torch.float64,sampled_max=torch.float64,
        exact_context_key=torch.uint8,effective_mask=torch.bool,matched_mask=torch.bool)
    for key in COMPANION_KEYS:
        value=payload.get(key)
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==expected_dtypes[key],
            'Frozen sampled companion tensor dtype/device differs: '+key)
        require(value.shape==(*shape,48) if key=='exact_context_key' else value.shape==shape,
            'Frozen sampled companion full response shape differs: '+key)
        require(bool(torch.isfinite(value).all()),'Nonfinite sampled companion: '+key)
    effective=data['attention_mask'][:,-shape[1]:].bool()&data['loss_mask'].bool()
    counts=payload['matched_count'];matched=payload['matched_mask']
    require(bool((counts>=0).all()) and torch.equal(payload['effective_mask'],effective),'Companion effective/count mask differs')
    require(torch.equal(matched,(counts>0)&effective) and not bool(((counts>0)&~effective).any()),'Companion matched mask differs')
    require(bool((payload['sampled_min'][matched]<=payload['sampled_max'][matched]).all()),'Sampled duplicate range is reversed')
    require(torch.equal(payload['exact_context_key'],exact_monitor_keys(data)),
        'Companion recorded keys differ from complete original monitor causal prefixes/positions')
    # Keep original unknown placeholders intact. Every probability comparison
    # below indexes the explicit matched mask before subtraction.
    return {('companion_'+key):payload[key] for key in COMPANION_KEYS}


def matched_monitor_ranges(data,rows,positions,current,weights,statistics):
    effective_match=data['companion_matched_mask'][rows,positions].cpu()
    count=data['companion_matched_count'][rows,positions].cpu()
    low=data['companion_sampled_min'][rows,positions].cpu()
    high=data['companion_sampled_max'][rows,positions].cpu()
    keys=data['companion_exact_context_key'][rows,positions].cpu()
    result=dict(effective_occurrences=int(effective_match.numel()),matched_occurrences=int(effective_match.sum()),
        unknown_occurrences=int((~effective_match).sum()),
        matched_raw_occurrences_denominator=int(count[effective_match].sum()),
        matched_contexts_with_duplicate_raw_occurrences=int((count[effective_match]>1).sum()),
        unique_effective_complete_causal_keys=len({bytes(key.tolist()) for key in keys}),
        unique_matched_complete_causal_keys=len({bytes(key.tolist()) for key in keys[effective_match]}),
        duplicate_probabilities_averaged=False,particular_raw_generation_provenance_proven=False,
        unmatched_placeholders_excluded=True,acceptance=False,variants={})
    values=dict(matched_mask=effective_match,matched_count=count,sampled_min=low,sampled_max=high,exact_context_key=keys)
    for name,value in current.items():
        valid=value[effective_match].double();lo=low[effective_match];hi=high[effective_match]
        # Endpoint deltas span all saved raw occurrences. Closest/worst distances
        # summarize that range, and do not designate an average/single sampler.
        lower_delta=valid-hi;upper_delta=valid-lo
        closest=torch.where((valid>=lo)&(valid<=hi),0.,torch.minimum(lower_delta.abs(),upper_delta.abs()))
        worst=torch.maximum(lower_delta.abs(),upper_delta.abs())
        masked_weights=weights[effective_match]
        result['variants'][name]=dict(current_minus_sampled_max=statistics(lower_delta,masked_weights),
            current_minus_sampled_min=statistics(upper_delta,masked_weights),
            closest_abs_distance_to_saved_range=statistics(closest,masked_weights),
            worst_abs_distance_to_saved_range=statistics(worst,masked_weights))
        values[name+'_minus_sampled_max']=lower_delta
        values[name+'_minus_sampled_min']=upper_delta
    return result,values
