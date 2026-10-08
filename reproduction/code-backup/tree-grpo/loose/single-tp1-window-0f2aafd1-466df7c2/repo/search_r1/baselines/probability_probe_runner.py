"""Source-only, eight-rank original-FSDP probability diagnostic, no optimizer.

Execution requires a separately reviewed resource adapter, frozen actual CPU
selection, bound assets and release receipt. This source preparation supplies none
of those approvals. Native CUDA/FSDP failures may time out peers in collectives;
metadata/IO failures are explicitly synchronized before another native call.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import importlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

import torch
import torch.distributed as dist

from .probability_probe_capture import (EffectiveWeightTrace,capture_original_call,
    compare_same_logits,file_sha,micro_plan,require,response_prediction_map,synchronize_plans,tensor_digest)
from .probability_probe_loader import (SameWindowTiedAliasTrace,exhaustive_hf_cast_inventory,
    load_original_hf_prefix,wrap_original_fsdp)
from .probability_probe_monitor import bind_monitor_companion,matched_monitor_ranges
from .probability_probe_io import load_tensor_verified


class GlobalDiagnosticRefusal(RuntimeError):
    pass


class RankCoordinator:
    """All pure local errors are deferred to the same globally matched MAX call."""
    def __init__(self,device):self.device=device;self.sequence=0

    def local_stage(self,label,operation):
        result=None;error=None
        try:result=operation()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.sequence+=1
        failed=torch.tensor(int(error is not None),device=self.device,dtype=torch.long)
        dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if int(failed):
            raise GlobalDiagnosticRefusal('Global diagnostic refusal at '+label+'; local evidence: '+repr(error))
        return result

    def signature(self,values):
        value=torch.tensor(values,device=self.device,dtype=torch.float64)
        require(bool(torch.isfinite(value).all()),'Nonfinite collective signature')
        low,high=value.clone(),value.clone()
        dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
        if not torch.equal(low,high):raise GlobalDiagnosticRefusal('All-rank forward/configuration counts differ')


def module_state(module,include_parameters=True):
    """Whole local-shard/buffer bytes; no export of full weights or gradients."""
    values=dict(modes={name:bool(child.training) for name,child in module.named_modules()},
        buffers={name:dict(tensor_digest(value),device=str(value.device)) for name,value in module.named_buffers(remove_duplicate=False)},
        gradients={name:None if value.grad is None else tensor_digest(value.grad)
            for name,value in module.named_parameters(remove_duplicate=False)},
        dropout={name:float(child.p) for name,child in module.named_modules() if isinstance(child,torch.nn.Dropout)})
    require(all(value==0 for value in values['dropout'].values()),'Nonzero dropout in original model')
    require(all(value is None for value in values['gradients'].values()),'Gradient created in zero-update diagnostic')
    config=getattr(module,'config',None)
    require(config is None or float(getattr(config,'attention_dropout',0))==0,'Nonzero functional attention dropout')
    if include_parameters:
        values['parameters']={name:dict(tensor_digest(value),device=str(value.device),
            private_local_shard=(dict(tensor_digest(value._local_shard),device=str(value._local_shard.device))
                if isinstance(getattr(value,'_local_shard',None),torch.Tensor) else None))
            for name,value in module.named_parameters(remove_duplicate=False)}
    return values


def rng_state(device=None):
    value=dict(cpu=tensor_digest(torch.get_rng_state()))
    if device is not None:value['cuda']=tensor_digest(torch.cuda.get_rng_state(device))
    return value


def verify_lazy_private_shards(before,after):
    """Allow actual Torch2.4 first-forward creation, never master byte changes.

    init_flat_param_attributes establishes _local_shard from flat_param.data
    during _lazy_init, after construction. Postwarm full state remains strict.
    """
    require(set(before)==set(after),'Warmup changed logical master parameter keys')
    changes={}
    for name,old in before.items():
        new=after[name]
        old_public={key:value for key,value in old.items() if key!='private_local_shard'}
        new_public={key:value for key,value in new.items() if key!='private_local_shard'}
        require(old_public==new_public,'Warmup changed full public master bytes/dtype/device: '+name)
        old_private,new_private=old['private_local_shard'],new['private_local_shard']
        if old_private is None and new_private is not None:
            require(new_private==new_public,'Lazy private shard differs from complete public master bytes/dtype/device: '+name)
            changes[name]=dict(before=None,after=new_private,scope='Actual Torch2.4 first native forward lazy private shard creation; identical to unchanged master')
        else:require(old_private==new_private,'Warmup changed/deleted an already-existing private local shard: '+name)
    return changes


def input_identity(data):
    return {key:tensor_digest(value) for key,value in data.items() if isinstance(value,torch.Tensor)}


def immutable_native_call(actor,data,temperature,device,weight_targets=None,alias_targets=None):
    """Run original entry once; metadata exceptions escape only after it returns.

    First-call full effective-weight and same-window alias proofs are optional
    inputs here but mandatory in run_probe's first native warmup call. Neither
    positive cross-window pointer observations nor constructor key counts suffice.
    """
    inputs=input_identity(data);rng=rng_state(device)
    weights=EffectiveWeightTrace(weight_targets) if weight_targets is not None else nullcontext()
    aliases=SameWindowTiedAliasTrace(*alias_targets) if alias_targets is not None else nullcontext()
    with torch.random.fork_rng(devices=[] if device is None else [device]),weights as trace,aliases as alias:
        captured=capture_original_call(actor,data,temperature)
    require(input_identity(data)==inputs,'Native original forward changed input tensor bytes')
    require(rng_state(device)==rng,'Native original forward leaked RNG changes outside fork_rng')
    receipt=dict(input_identity=inputs,rng_unchanged=True,optimizer_instantiated=False,
        scheduler_instantiated=False,backward_called=False,original_actor_calls=1,
        observed_forward_kwargs=list(captured['forward_kwargs']),observed_forward_input_shapes=captured['input_shapes'],
        observed_forward_input_dtypes=captured['input_dtypes'],observed_model_training=captured['model_training'])
    if weight_targets is not None:receipt['effective_weights']=trace.receipt()
    if alias_targets is not None:receipt['same_window_tie']=alias.receipt()
    return captured,receipt


def canonical_response_data(payload,kind):
    """Preserve full buffers. Raw masks are an explicit diagnostic all-ones copy."""
    required=('input_ids','attention_mask','position_ids','responses','old_log_probs')
    require(all(key in payload for key in required),'Missing original response/context tensors')
    data={key:payload[key] for key in required}
    for key in required:
        require(isinstance(data[key],torch.Tensor) and data[key].device.type=='cpu','Saved CPU tensor required: '+key)
    ids,responses=data['input_ids'],data['responses'];width=responses.shape[1]
    require(ids.ndim==2 and responses.ndim==2 and width<ids.shape[1], 'Invalid original response buffer')
    require(data['old_log_probs'].shape==responses.shape,'Old logprob response shape changed')
    if kind=='raw':
        require('sampled_log_probs' in payload and payload['sampled_log_probs'].shape==responses.shape,'Raw sampled probabilities missing')
        data['sampled_log_probs']=payload['sampled_log_probs']
        data['loss_mask']=torch.ones_like(responses,dtype=torch.bool)
        data['policy_weights']=torch.ones_like(responses,dtype=torch.float64)
    else:
        require(kind=='monitor' and 'loss_mask' in payload and 'policy_weights' in payload,'Monitor Tree weights/mask missing')
        for key in ('loss_mask','policy_weights'):
            value=payload[key]
            if value.shape==ids.shape:value=value[:,-width:]
            require(value.shape==responses.shape,'Response-aligned monitor tensor shape changed: '+key)
            data[key]=value
    require(torch.equal(ids[:,-width:],responses),'Response tokens are not full input suffix')
    return data


def selected_rows(data,rows,device):
    index=torch.tensor(rows,dtype=torch.long)
    require(len(rows)==len(set(rows)) and rows and all(0<=row<len(data['input_ids']) for row in rows),'Invalid selected physical rows')
    return {key:value.index_select(0,index).to(device) for key,value in data.items()}


def probability_statistics(delta,weights=None):
    require(delta.ndim==1 and bool(torch.isfinite(delta).all()),'Nonfinite token comparison')
    if weights is None:weights=torch.ones_like(delta)
    weights=weights.double().cpu();delta=delta.double().cpu()
    require(weights.shape==delta.shape and bool(torch.isfinite(weights).all()) and bool((weights>=0).all()),'Invalid comparison Tree weights')
    count=delta.numel();mass=float(weights.sum())
    return dict(physical_effective_token_occurrences=count,tree_weight_denominator=mass,
        max_abs=float(delta.abs().max()) if count else None,
        mean_abs=float(delta.abs().mean()) if count else None,
        rms=float(delta.square().mean().sqrt()) if count else None,
        abs_quantiles={str(q):float(torch.quantile(delta.abs(),q)) if count else None for q in (.5,.9,.99,1.)},
        above_trial_001_count=int((delta.abs()>.01).sum()),
        tree_weighted_abs_numerator=float((weights*delta.abs()).sum()),
        tree_weighted_mean_abs=float((weights*delta.abs()).sum())/mass if mass else None,
        trial_001_is_diagnostic_only=True,acceptance=False)


def tensor_evidence_bytes(value):
    stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()


def prepare_native_inputs(data,rows,device):
    selected=selected_rows(data,rows,device)
    # Validate predictor mapping and all complete input bytes before any rank
    # enters native FSDP. The original entry repeats this calculation unchanged.
    response_prediction_map(selected,remove_padding=True)
    input_identity(selected)
    return selected


def build_comparison_evidence(comparison,capture,data,call,group,micro_baselines,micro,rank,call_index):
    mapping=capture['mapping'];rows=mapping['rows'];positions=mapping['response_positions']
    old=data['old_log_probs'][rows,positions].cpu().double()
    weights=data['policy_weights'][rows,positions].cpu().double()
    tensors=dict(comparison['token_values'],original_old_log_probs=old,tree_policy_weights=weights,
        physical_rows=torch.tensor(call['rows'])[rows.cpu()],response_positions=positions.cpu(),
        full_input_buffers={key:value.detach().cpu() for key,value in data.items()},
        logits_prediction_indices=mapping['logit_indices'].cpu(),labels=mapping['labels'].cpu())
    metrics={key+'_vs_saved_old':probability_statistics(value-old,weights) for key,value in comparison['token_values'].items()}
    if group['kind']=='raw':
        sampled=data['sampled_log_probs'][rows,positions].cpu().double();tensors['original_sampled_log_probs']=sampled
        metrics.update({key+'_vs_actual_sampled':probability_statistics(value-sampled,weights)
            for key,value in comparison['token_values'].items()})
    else:
        require('companion_matched_mask' in data,'UNIMPLEMENTED: frozen full monitor exact-context sampled companion required')
        ranges,range_tensors=matched_monitor_ranges(data,rows,positions,comparison['token_values'],weights,probability_statistics)
        metrics['monitor_vs_sampled_ranges']=ranges;tensors['original_monitor_sampled_ranges']=range_tensors
    for local_row,physical_row in enumerate(call['rows']):
        key=(group['asset'],group['source_group'],physical_row)
        valid=(rows.cpu()==local_row);current=comparison['token_values']['original_actor'][valid]
        if micro==4 and not call['exclude_duplicate']:micro_baselines[key]=(positions.cpu()[valid],current)
        if micro!=4 and not call['exclude_duplicate']:
            reference_positions,reference=micro_baselines[key]
            require(torch.equal(reference_positions,positions.cpu()[valid]),'Micro change shifted effective prediction labels')
            metrics['row-%d-micro%d-vs-original-micro4'%(physical_row,micro)]=probability_statistics(current-reference,weights[valid])
    evidence=dict(rank=rank,micro=micro,call_index=call_index,selection=group,call=call,
        input_identity=input_identity(data),same_logits=comparison['metadata'],metrics=metrics,
        duplicate_excluded_from_occurrence_totals=bool(call['exclude_duplicate']),acceptance=False)
    return tensors,evidence


def verified_bundle_config(bundle):
    """Consume the exact UTF8 config bytes already bound by the owner adapter.

    Reading the config pathname again would reopen the v1 TOCTOU window. This
    function also checks the frozen original SHA, without changing config values.
    """
    import hashlib
    from omegaconf import OmegaConf
    text=bundle['config_text']
    require(type(text) is str,'Exact original config text is required')
    require(hashlib.sha256(text.encode('utf-8')).hexdigest()==bundle['identity']['config_sha256'],
        'Bundle config text differs from frozen original bytes')
    return OmegaConf.create(text)


def run_probe(bundle,coordinator,device,resource_adapter):
    """Called only after separately frozen launch/selection and source validation.

    resource_adapter is owner-provided, source-bound and independently reviewed.
    check() must observe current live guard/leases/background PID identities, not
    merely trust a launch-time JSON. Existing supervisor owns cleanup/cost files.
    """
    from .probability_probe_contracts import atomic_write_json_no_clobber,atomic_write_bytes_no_clobber,validate_fixed_config
    from omegaconf import OmegaConf
    rank=dist.get_rank();start=time.monotonic();cpu_start=time.process_time()
    output=Path(bundle['output'])/('rank-%02d'%rank)
    coordinator.local_stage('new rank output directory',lambda:output.mkdir(parents=False,exist_ok=False))
    config=coordinator.local_stage('exact fixed original actor protocol',lambda:verified_bundle_config(bundle))
    coordinator.local_stage('fixed global protocol and no independent batch defaults',lambda:
        validate_fixed_config(OmegaConf.to_container(config,resolve=True)))
    cfg=config.actor_rollout_ref
    from torch.distributed.device_mesh import init_device_mesh
    mesh=init_device_mesh('cuda',mesh_shape=(8,),mesh_dim_names=['fsdp'])
    namespace,suffix,construction=coordinator.local_stage('original HF model CPU prefix',lambda:load_original_hf_prefix(
        bundle['original_worker_path'],bundle['original_worker_sha256'],bundle['hf_path'],cfg,mesh,rank))
    model=namespace['actor_module'];alias_targets=(model,model.get_submodule('model.embed_tokens'),model.get_submodule('lm_head'))
    targets,hf_receipt=coordinator.local_stage('complete serialized435/unique434 HF cast comparison',lambda:
        exhaustive_hf_cast_inventory(model,bundle['hf_path'],bundle['hf_index'],bundle['hf_identity']['files']))
    # Original wrapping includes its barrier and native FSDP collectives. A native
    # constructor failure is not a recoverable local metadata error; PG timeout
    # and the owner supervisor handle it, retaining failure evidence.
    coordinator.local_stage('before original FSDP live resources',lambda:resource_adapter.check('before_original_fsdp'))
    fsdp=wrap_original_fsdp(namespace,suffix,bundle['original_worker_path'])
    coordinator.local_stage('original actual MixedPrecision contract',lambda:require(namespace['torch_dtype']==torch.float32 and
        namespace['mixed_precision'].param_dtype==torch.bfloat16 and namespace['mixed_precision'].reduce_dtype==torch.float32 and
        namespace['mixed_precision'].buffer_dtype==torch.float32,'Original actor master/MP dtype contract changed'))
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.fsdp_utils import offload_fsdp_param_and_grad,load_fsdp_param_and_grad
    from verl.utils import torch_functional as native
    cfg.actor.ppo_mini_batch_size//=8;cfg.actor.ppo_micro_batch_size//=8
    cfg.rollout.log_prob_micro_batch_size//=8
    cfg.actor.use_remove_padding=cfg.model.use_remove_padding
    actor=coordinator.local_stage('original actor constructor',lambda:DataParallelPPOActor(cfg.actor,fsdp,None))
    def service_checks():
        require(actor.actor_optimizer is None,'Diagnostic must not instantiate an optimizer')
        services=dict(passive_ray_import='ray' in sys.modules,ray_initialized=False,vllm_module_imported='vllm' in sys.modules)
        if 'ray' in sys.modules:services['ray_initialized']=sys.modules['ray'].is_initialized()
        require(not services['ray_initialized'] and not services['vllm_module_imported'],'Unexpected Ray/vLLM service/engine path')
        require(bundle['probe_mode']=='eval_like_original_compute_log_prob','Probe mode must be explicit')
        return services
    imported_services=coordinator.local_stage('zero optimizer and original passive imports',service_checks)
    # An explicit probe mode matches original compute_log_prob's eval call. It
    # does not certify the historic monitor's actual module mode.
    original_modes={name:child.training for name,child in fsdp.named_modules()}
    fsdp.eval()
    def offload():
        offload_fsdp_param_and_grad(fsdp,offload_grad=True);torch.cuda.synchronize(device)
    def load():
        load_fsdp_param_and_grad(fsdp,device,load_grad=True);torch.cuda.synchronize(device)
    coordinator.local_stage('original offload after construction',offload)
    prewarm=coordinator.local_stage('prewarm full local shard/buffer state',lambda:module_state(fsdp))
    selection=bundle['selection']['selections'][rank]
    coordinator.local_stage('canonical local selection rank',lambda:require(selection['rank']==rank,'Selection must be canonical rank ordered'))
    local_data={}
    def load_assets():
        for entry in selection['raw_groups']+[selection['monitor']]:
            key=entry['asset']
            if key in local_data:continue
            asset=bundle['inventory'][key]
            value=load_tensor_verified(asset['path'],asset['sha256'])
            local_data[key]=canonical_response_data(value,asset['kind'])
            if asset['kind']=='monitor':
                companions=bundle['selection']['monitor_companions']
                matches=[entry for entry in companions if entry['rank']==rank and entry['monitor_asset']==key]
                require(len(matches)==1,'Missing/ambiguous actual CPU sampled companion')
                companion=matches[0]
                saved=load_tensor_verified(companion['path'],companion['sha256'])
                local_data[key].update(bind_monitor_companion(saved,local_data[key],bundle['identity'],key,asset['sha256']))
    coordinator.local_stage('selected same-node original tensor inputs',load_assets)
    groups=[]
    for raw in selection['raw_groups']:groups.append(dict(raw,kind='raw'))
    monitor=selection['monitor']
    # Full monitor micro2 is the original safety measure layout. The grouping
    # below gives its same row buffers to additional diagnostic micro4/1 calls.
    for start_row in range(0,32,4):groups.append(dict(asset=monitor['asset'],source_group='monitor:%d'%start_row,
        row_indices=monitor['row_indices'][start_row:start_row+4],kind='monitor'))
    plan=micro_plan([group['row_indices'] for group in groups])
    all_plans=[None]*8;dist.all_gather_object(all_plans,plan)
    aligned=synchronize_plans(all_plans)[rank]
    coordinator.signature([len(aligned[micro]) for micro in (4,2,1)]+[4,2,1,1.])
    # One collective-aligned warmup uses each rank's longest selected original
    # micro4 group. The complete effective-weight/tie proof is from this call.
    warmup=max(groups,key=lambda group:int(local_data[group['asset']]['position_ids'][group['row_indices']].masked_fill(
        ~local_data[group['asset']]['attention_mask'][group['row_indices']].bool(),-1).max()))
    data=coordinator.local_stage('warmup input preparation',lambda:prepare_native_inputs(local_data[warmup['asset']],warmup['row_indices'],device))
    coordinator.local_stage('warmup live resource check',lambda:resource_adapter.check('warmup'))
    coordinator.local_stage('warmup original load parameter/grad path',load)
    warm_capture,warm_receipt=coordinator.local_stage('first original entry effective weights and same-window alias',lambda:
        immutable_native_call(actor,data,1.,device,targets,alias_targets))
    del warm_capture,data
    coordinator.local_stage('warmup original offload parameter/grad path',offload)
    baseline=coordinator.local_stage('post lazy-buffer warmup complete baseline',lambda:module_state(fsdp))
    lazy_private=coordinator.local_stage('warmup preserved all master bytes with explicit first-forward private metadata creation',lambda:
        verify_lazy_private_shards(prewarm['parameters'],baseline['parameters']))
    postwarm_rng=rng_state(device)
    warm_receipt.update(prewarm_state=prewarm,postwarm_baseline=baseline,
        lazy_private_local_shard_initialization=lazy_private,
        lazy_buffer_changes={key:dict(before=prewarm['buffers'].get(key),after=baseline['buffers'].get(key))
            for key in set(prewarm['buffers'])|set(baseline['buffers']) if prewarm['buffers'].get(key)!=baseline['buffers'].get(key)},
        initialization_rng_scope='Initialization RNG consumption excluded; each probe forward uses RNG fork',
        actual_master_dtype=str(namespace['torch_dtype']),actual_mixed_precision=dict(param_dtype=str(namespace['mixed_precision'].param_dtype),
            reduce_dtype=str(namespace['mixed_precision'].reduce_dtype),buffer_dtype=str(namespace['mixed_precision'].buffer_dtype)),
        normalized_actor_mini_batch=cfg.actor.ppo_mini_batch_size,normalized_actor_micro_batch=cfg.actor.ppo_micro_batch_size,
        normalized_raw_old_micro_batch=cfg.rollout.log_prob_micro_batch_size,
        imported_services=imported_services,probe_mode=bundle['probe_mode'],historical_monitor_mode_proven=False)
    coordinator.local_stage('warmup identity no-clobber evidence',lambda:atomic_write_json_no_clobber(output/'load-and-warmup.json',
        dict(construction=construction,hf=hf_receipt,actual_native_warmup=warm_receipt)))
    branch=dict(source_sha256=bundle['native_logprob_sha256'],module=native.logprobs_from_logits.__module__,
        flash_ce_available=native.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE)
    results=[];micro_baselines={};duplicate_calls=0
    for micro in (4,2,1):
        for call_index,call in enumerate(aligned[micro]):
            group=groups[call['group']];label='micro%d-call%03d'%(micro,call_index)
            coordinator.local_stage(label+' live resources',lambda:resource_adapter.check(label))
            data=coordinator.local_stage(label+' prepare immutable full buffers',lambda:
                prepare_native_inputs(local_data[group['asset']],call['rows'],device))
            coordinator.local_stage(label+' original load parameter/grad path',load)
            capture,receipt=coordinator.local_stage(label+' original native forward',lambda:immutable_native_call(actor,data,1.,device))
            comparison=coordinator.local_stage(label+' same retained logits reductions',lambda:
                compare_same_logits(capture,native.logprobs_from_logits,branch))
            tensors,evidence=coordinator.local_stage(label+' original probability comparison/denominators',lambda:
                build_comparison_evidence(comparison,capture,data,call,group,micro_baselines,micro,rank,call_index))
            evidence['native_forward_receipt']=receipt
            def extreme_logit_evidence():
                current=comparison['token_values']['original_actor']
                reference=tensors['original_sampled_log_probs'] if group['kind']=='raw' else tensors['original_old_log_probs']
                if not current.numel():return dict(status='EMPTY_EFFECTIVE_OCCURRENCES',selected_rows=0)
                extreme=int((current-reference).abs().argmax())
                prediction_index=capture['mapping']['logit_indices'][extreme]
                logits=capture['logits'][prediction_index].detach()
                identity=tensor_digest(logits)
                tensors['same_call_extreme_full_vocab_logits']=logits.cpu()
                return dict(identity=identity,effective_occurrence_index=extreme,
                    original_physical_row=int(tensors['physical_rows'][extreme]),
                    original_response_position=int(tensors['response_positions'][extreme]),
                    native_prediction_logit_index=int(prediction_index),
                    choice='Maximum native-vs-actual-sampled delta for raw; maximum native-vs-saved-old for monitor within already frozen selected inputs',
                    temperature_scaled_same_call_logits=True,second_model_forward=False)
            evidence['same_call_extreme_logits']=coordinator.local_stage(label+' full-vocab selected original vector identity',extreme_logit_evidence)
            if call['exclude_duplicate']:duplicate_calls+=1
            def publish():
                tensor_receipt=atomic_write_bytes_no_clobber(output/(label+'.pt'),tensor_evidence_bytes(tensors))
                evidence['tensor_evidence_sha256']=tensor_receipt['sha256']
                evidence['tensor_evidence_path']=str(output/(label+'.pt'))
                atomic_write_json_no_clobber(output/(label+'.json'),evidence)
            coordinator.local_stage(label+' all original probabilities no-clobber publication',publish)
            results.append(dict(micro=micro,label=label,kind=group['kind'],excluded=bool(call['exclude_duplicate'])))
            del capture,comparison,tensors,data
            coordinator.local_stage(label+' original offload parameter/grad path',offload)
            coordinator.local_stage(label+' modes/buffers/gradients unchanged after baseline',lambda:
                require(module_state(fsdp,include_parameters=False)=={key:value for key,value in baseline.items() if key!='parameters'},
                    'Native calls changed warmed module modes/buffers/gradients'))
    end_state=coordinator.local_stage('final complete local-shard bytes and resources',lambda:module_state(fsdp))
    coordinator.local_stage('final full byte/RNG identity',lambda:require(end_state==baseline and rng_state(device)==postwarm_rng,
        'Complete warmed actor/RNG state changed during zero-update diagnostic'))
    def restore_modes():
        for name,child in fsdp.named_modules():child.training=original_modes[name]
        require(all(child.training==original_modes[name] for name,child in fsdp.named_modules()),'Original construction modes were not restored')
    coordinator.local_stage('original construction modes restored',restore_modes)
    coordinator.local_stage('final live owner resource check',lambda:resource_adapter.check('final'))
    final=dict(status='SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE',
        rank=rank,world_size=8,optimizer_instantiated=False,optimizer_committed=0,scheduler_instantiated=False,
        scheduler_advanced=False,backward_called=False,original_forward_calls=1+len(results),
        duplicate_alignment_calls=duplicate_calls,source_and_input_bundle_identity=bundle['identity'],
        full_state_unchanged_since_postwarm_baseline=True,original_module_modes_restored=True,
        own_wall_seconds=time.monotonic()-start,own_cpu_seconds=time.process_time()-cpu_start,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        results=results,selection_scope='Up to20 raw physical rows/rank and all32 monitor rows/rank; not all raw/final leaf inputs',
        original_3268_effective_actor_identity_proven=False,probability_consistency_acceptance=False,
        training_acceptance=False,gpu_training_capacity_proven=False)
    coordinator.local_stage('final rank evidence no-clobber',lambda:atomic_write_json_no_clobber(output/'ready.json',final))
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',required=True)
    args=parser.parse_args()
    # No default package, selection, release, assets or device guess is allowed.
    from .probability_probe_contracts import strict_json_load
    bundle=strict_json_load(args.bundle)
    require(bundle.get('status')=='GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED','UNIMPLEMENTED: reviewed GPU bundle/release and resource adapter required')
    require(socket.gethostname().split('.')[0]=='h100-9' and os.environ.get('SLURM_JOB_NODELIST')=='h100-9','Same-node h100-9 Slurm required')
    require(os.environ.get('MASTER_ADDR')=='127.0.0.1','All-rank process group must use loopback')
    require(int(os.environ.get('WORLD_SIZE','0'))==8,'Eight actual ranks required')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','Offline model loading required')
    # Resource adapter owns complete manifest/source/asset validation, Slurm8GPU
    # and UUID leases,5-second live guard, original8 PID/starttime preservation,
    # cgroup cleanup and Slurm/GPU/allocatedCPU cost. Its contract is deliberately
    # UNIMPLEMENTED in this source-only delivery; no made-up launch proof exists.
    adapter_entry=bundle.get('reviewed_resource_adapter')
    require(adapter_entry is not None,'UNIMPLEMENTED: independent owned-runtime resource adapter')
    require(file_sha(adapter_entry['path'])==adapter_entry['sha256'],'Resource adapter changed')
    spec=importlib.util.spec_from_file_location('tree_probe_reviewed_resource_adapter',adapter_entry['path'])
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    coordinator=RankCoordinator(torch.device('cuda',local_rank))
    try:
        coordinator.local_stage('owner complete bundle/source/selection validation',lambda:
            require(adapter.prepare_and_validate_bundle(bundle) is True,'Owner bundle/source/selection validation failed'))
        run_probe(bundle,coordinator,local_rank,adapter)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
