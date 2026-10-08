"""Unwired zero-update numerical probe core; not a GPU runner or acceptance gate.

The original actor entry runs unchanged under a temporary forward hook. CPU tests
exercise mapping and same-logits comparisons. A future audited runner must supply
actual effective-weight receipts, eight-rank synchronization and state guards.
"""
from contextlib import nullcontext
import hashlib
import inspect
import math
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def bind_sources(entries):
    """Expected hashes must come from an independently frozen probe manifest."""
    require(bool(entries),'Missing frozen source contract')
    records=[]
    for entry in entries:
        path=Path(entry['path']).resolve()
        require(file_sha(path)==entry['sha256'],'Changed probe source: '+str(path))
        records.append(dict(path=str(path),sha256=entry['sha256']))
    return records


def tensor_digest(tensor):
    """Full shape/dtype/byte digest; only bounded CPU chunks, never asset export."""
    require(isinstance(tensor,torch.Tensor),'Actual forward attribute is not a tensor')
    require(bool(torch.isfinite(tensor).all()),'Nonfinite effective forward tensor')
    digest=hashlib.sha256()
    digest.update(str(tensor.dtype).encode()+b'\0'+str(tuple(tensor.shape)).encode()+b'\0')
    flat=tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    for chunk in flat.split(8*1024**2):digest.update(chunk.cpu().numpy().tobytes())
    return dict(dtype=str(tensor.dtype),shape=list(tensor.shape),full_shape_dtype_bytes_sha256=digest.hexdigest())


class EffectiveWeightTrace:
    """Observe real module attributes inside the first original native forward.

    Runner retains module references BEFORE FSDP wrapping, together with exhaustive
    logical HF key/attribute specifications and full cast-expected digests. Reading
    actual .weight/.bias attributes avoids treating a pre-gather FSDP flat shard or
    named_parameters registry as the effective mathematical weight. Coverage and
    expected address observations are compared. Addresses observed at separate
    times can be reused by FSDP; this is NOT a tied-alias identity proof. A runner
    needs simultaneous tensor references in one gather window for that proof.
    """
    def __init__(self,targets):
        require(bool(targets),'Missing exhaustive effective-weight targets')
        self.targets=list(targets);self.records={};self.handles=[];self.storage={};self.errors=[];self.observed=set()
        self.completed_successfully=False;self.active=False;self.address_observations={}
        require(len({target['logical_name'] for target in targets})==len(targets),'Duplicate logical effective-weight keys')

    def __enter__(self):
        require(not self.active,'Nested effective-weight trace contexts are unsupported')
        self.records={};self.handles=[];self.storage={};self.errors=[];self.observed=set();self.address_observations={}
        self.completed_successfully=False;self.active=True
        try:
            for target in self.targets:
                def hook(module,args,target=target):
                    name=target['logical_name']
                    if name in self.observed:return
                    self.observed.add(name)
                    # A rank-local hook refusal midway through FSDP would strand
                    # peers in later collectives. Defer metadata refusals until
                    # the whole original forward has returned on all ranks.
                    try:
                        actual=getattr(module,target['attribute'])
                        identity=tensor_digest(actual)
                        require(identity==target['expected_cast_digest'],'Effective forward weight differs from full HF cast identity: '+name)
                        self.records[name]=dict(identity,device=str(actual.device),scope='First observed original native forward attribute')
                        self.storage[name]=(str(actual.device),actual.untyped_storage().data_ptr(),actual.storage_offset())
                    except Exception as error:self.errors.append(type(error).__name__+': '+str(error))
                self.handles.append(target['module'].register_forward_pre_hook(hook))
        except BaseException:
            for handle in self.handles:handle.remove()
            self.active=False
            raise
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        for handle in self.handles:handle.remove()
        self.active=False;self.completed_successfully=False
        if exc_type is None:
            require(not self.errors,'; '.join(self.errors))
            require(set(self.records)=={target['logical_name'] for target in self.targets},'Incomplete effective forward logical-key coverage')
            aliases={}
            for target in self.targets:
                group=target.get('expected_storage_alias_group')
                if group is not None:aliases.setdefault(group,[]).append((target['logical_name'],self.storage[target['logical_name']]))
            self.address_observations={str(group):dict(logical_names=[name for name,_ in values],
                same_observed_address_and_offset=len({address for _,address in values})==1,
                scope='observed_address_only; different forward/gather times may reuse addresses',
                tied_alias_identity_proven=False) for group,values in aliases.items()}
            require(all(record['same_observed_address_and_offset'] for record in self.address_observations.values()),
                    'Expected observed storage alias differs; address observation only, not tied identity proof')
            self.completed_successfully=True
        return False

    def receipt(self):
        require(self.completed_successfully and not self.active,
                'Cannot publish effective-weight evidence without successful complete trace exit')
        require(not self.errors,'Cannot publish effective-weight evidence with recorded hook failures')
        require(set(self.records)=={target['logical_name'] for target in self.targets},'Cannot publish incomplete effective-weight evidence')
        return dict(status='EFFECTIVE_FORWARD_WEIGHT_DIGEST_OBSERVED_NO_ACCEPTANCE',records=dict(self.records),
            logical_tensor_count=len(self.records),exhaustiveness_is_runner_manifest_requirement=True,
            trace_completed_successfully=True,address_observations=dict(self.address_observations),
            tied_alias_identity_proven=False,
            original_3268_effective_weight_identity_proven=False,acceptance=False)


def micro_plan(groups):
    """Original complete micro4 buffers; subdivide in original order, never shuffle."""
    require(bool(groups),'At least one original group is required')
    require(all(0<len(group)<=4 and len(set(group))==len(group) for group in groups),'Invalid original micro4 group')
    return {micro:[dict(group=number,rows=list(group[start:start+micro]),micro=micro,exclude_duplicate=False)
        for number,group in enumerate(groups) for start in range(0,len(group),micro)] for micro in (4,2,1)}


def synchronize_plans(plans):
    """Plan CPU metadata only; padded duplicate forwards are explicitly uncounted."""
    require(len(plans)==8,'All eight ranks must supply plans')
    aligned=[]
    for plan in plans:
        require(set(plan)=={4,2,1} and all(plan[micro] for micro in (4,2,1)),'Missing per-rank micro comparison')
        aligned.append({micro:[dict(call) for call in plan[micro]] for micro in (4,2,1)})
    for micro in (4,2,1):
        count=max(len(plan[micro]) for plan in aligned)
        for plan in aligned:
            while len(plan[micro])<count:
                duplicate=dict(plan[micro][0],exclude_duplicate=True,reason='collective_count_alignment')
                plan[micro].append(duplicate)
    return aligned


def response_prediction_map(data, remove_padding):
    """Reproduce original pad-back [-R-1:-1] prediction locations and rolled labels.

    Masked observations remain in attention. An effective response whose previous
    physical position is padding has no original predictor and is rejected rather
    than matched to a different valid token.
    """
    ids,positions,attention,responses,loss=(data[key] for key in
        ('input_ids','position_ids','attention_mask','responses','loss_mask'))
    require(ids.ndim==2 and positions.shape==attention.shape==ids.shape,'Input shapes differ')
    require(responses.ndim==2 and loss.shape==responses.shape and responses.shape[0]==ids.shape[0],'Response shapes differ')
    require(bool(((attention==0)|(attention==1)).all()) and bool(((loss==0)|(loss==1)).all()),'Nonbinary masks')
    width,response_width=ids.shape[1],responses.shape[1]
    require(0<response_width<width and torch.equal(ids[:,-response_width:],responses),'Response is not exact input suffix')
    effective=attention[:,-response_width:].bool() & loss.bool()
    row,col=effective.nonzero(as_tuple=True)
    predicted_col=width-response_width-1+col
    require(bool(attention[row,predicted_col].bool().all()),'Effective response has an attention-masked native prediction position')
    physical_index=row*width+predicted_col
    if remove_padding:
        valid=attention.flatten().bool()
        source_index=valid.long().cumsum(0)-1
        indices=source_index[physical_index]
        native_labels=torch.roll(ids.flatten()[valid],shifts=-1)[indices]
        expected_logit_rows=int(valid.sum())
    else:
        indices=physical_index
        native_labels=responses[row,col]
        expected_logit_rows=ids.numel()
    require(torch.equal(native_labels,responses[row,col]),'Original rolled labels do not match response tokens')
    return dict(rows=row,response_positions=col,logit_indices=indices,labels=native_labels,
        effective_mask=effective,expected_logit_rows=expected_logit_rows,remove_padding=bool(remove_padding),
        prediction_mapping='Original physical predecessor and pad-back slice; no prefix trimming or retokenization')


def capture_original_call(actor, data, temperature):
    """Call the original actor forward once; no mode change, optimizer or scheduler.

    Retains a reference to that call's logits until reductions finish. A hook does
    not certify original weights; full state/RNG/weight guards belong to the runner.
    """
    require(math.isfinite(float(temperature)) and float(temperature)>0,'Finite positive temperature required')
    mapping=response_prediction_map(data,actor.use_remove_padding)
    captured=[]
    def hook(module,args,kwargs,output):
        logits=output.logits
        require(isinstance(logits,torch.Tensor) and logits.ndim==3,'Unexpected original model logits')
        flat=logits.reshape(-1,logits.shape[-1])
        require(flat.untyped_storage().data_ptr()==logits.untyped_storage().data_ptr(),
                'Logits reshape would copy the original forward tensor')
        require(flat.shape[0]==mapping['expected_logit_rows'],'Original logits rows do not match packed/dense layout')
        require(not captured,'Original actor entry called model more than once')
        selected=torch.tensor([0,flat.numel()-1],device=flat.device)
        captured.append(dict(logits=flat,original_logits=logits,before_temperature_dtype=str(logits.dtype),
            samples_before=flat.flatten()[selected].detach().clone(),sample_indices=selected,
            cuda_autocast_enabled=torch.is_autocast_enabled(),
            cuda_autocast_dtype=torch.get_autocast_gpu_dtype(),
            model_training=bool(module.training),forward_kwargs=tuple(sorted(kwargs)),
            input_shapes={key:list(value.shape) for key,value in kwargs.items() if isinstance(value,torch.Tensor)},
            input_dtypes={key:str(value.dtype) for key,value in kwargs.items() if isinstance(value,torch.Tensor)}))
    handle=actor.actor_module.register_forward_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad():
            entropy,native=actor._forward_micro_batch(data,temperature)
    finally:
        handle.remove()
    require(len(captured)==1,'Original actor model call was not observed')
    value=captured[0]
    # Only bounded samples are copied before temperature; the retained logits are
    # the original in-place scaled tensor, not logits from another forward.
    require(torch.equal(value['logits'].flatten()[value['sample_indices']],value['samples_before']/temperature),
            'Original actor did not scale retained logits in place as the pinned source specifies')
    require(native.shape==data['responses'].shape,'Original actor probability shape changed')
    value.update(mapping=mapping,native_response_log_probs=native.detach(),entropy=entropy.detach(),
        after_temperature_dtype=str(value['logits'].dtype),temperature=float(temperature),
        original_actor_calls=1,weights_independently_verified=False,acceptance=False)
    return value


def compare_same_logits(capture, native_function, branch_receipt, block_rows=16):
    """One retained scaled tensor, identical labels, bounded row normalization.

    The native reduction is repeated on selected rows. Its comparison with the
    original actor output is reported explicitly: a discrepancy prevents isolated
    normalization attribution. No threshold or acceptance is inferred here.
    """
    require(isinstance(block_rows,int) and block_rows>0,'Positive normalization block size required')
    source_path=inspect.getsourcefile(native_function)
    require(source_path is not None and file_sha(source_path)==branch_receipt['source_sha256'],'Native logprob source binding changed')
    require(native_function.__module__==branch_receipt['module'],'Native logprob module changed')
    require(branch_receipt['flash_ce_available'] in (True,False),'Actual import availability must be supplied')
    require(native_function.__globals__.get('FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE') is branch_receipt['flash_ce_available'],
            'Supplied native branch does not match the actual imported availability flag')
    mapping=capture['mapping'];indices=mapping['logit_indices'];labels=mapping['labels']
    values={key:[] for key in ('native_selected_rows','original_dtype_logsoftmax','fp32_logsoftmax','fp64_logsoftmax')}
    output_dtypes={key:set() for key in values}
    context=(torch.autocast('cuda',dtype=capture['cuda_autocast_dtype'])
        if capture['cuda_autocast_enabled'] else nullcontext())
    with torch.no_grad(),context:
        for start in range(0,indices.numel(),block_rows):
            logits=capture['logits'].index_select(0,indices[start:start+block_rows])
            target=labels[start:start+block_rows]
            native=native_function(logits=logits,labels=target)
            output_dtypes['native_selected_rows'].add(str(native.dtype))
            values['native_selected_rows'].append(native.detach().cpu().double())
            for key,dtype in (('original_dtype_logsoftmax',logits.dtype),('fp32_logsoftmax',torch.float32),('fp64_logsoftmax',torch.float64)):
                result=torch.log_softmax(logits.to(dtype),dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
                output_dtypes[key].add(str(result.dtype))
                values[key].append(result.detach().cpu().double())
    values={key:torch.cat(parts) if parts else torch.empty(0,dtype=torch.float64) for key,parts in values.items()}
    original=capture['native_response_log_probs'][mapping['rows'],mapping['response_positions']].detach().cpu().double()
    values['original_actor']=original
    require(all(bool(torch.isfinite(value).all()) for value in values.values()),'Nonfinite same-logits evidence')
    native_delta=values['native_selected_rows']-original
    return dict(token_values=values,metadata=dict(status='SAME_LOGITS_DIAGNOSTIC_ONLY_NO_ACCEPTANCE',
        effective_response_token_occurrences=indices.numel(),original_actor_calls=capture['original_actor_calls'],
        original_forward_logits_dtype=capture['before_temperature_dtype'],scaled_logits_dtype=capture['after_temperature_dtype'],
        original_actor_log_probs_dtype=str(capture['native_response_log_probs'].dtype),
        normalization_actual_output_dtypes={key:sorted(value) for key,value in output_dtypes.items()},
        cuda_autocast_enabled=capture['cuda_autocast_enabled'],cuda_autocast_dtype=str(capture['cuda_autocast_dtype']),
        native_logprob_source=dict(branch_receipt),retained_logits_same_tensor=True,
        selected_native_vs_original_max_abs_delta=float(native_delta.abs().max()) if native_delta.numel() else None,
        selected_native_exactly_matches_original=bool(torch.equal(values['native_selected_rows'],original)),
        weights_independently_verified=False,acceptance=False,
        limits='No second model forward. Selected-row native reduction may differ in kernel layout; attribute normalization only after original-output agreement. Effective weights, RNG, modes, buffers, all-rank schedule and resources need separate audited runner evidence.'))
