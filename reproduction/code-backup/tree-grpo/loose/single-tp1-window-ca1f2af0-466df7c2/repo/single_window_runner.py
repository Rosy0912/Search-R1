"""Candidate original-TP1 single-call5 window. No release, launch or training.

All tensor assets and any future execution remain on h100-9. A separately
reviewed owned-runtime adapter must implement this new profile; the older
FSDP-only adapter cannot silently authorize a vLLM engine or new generation.
"""
import argparse
from datetime import timedelta
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from search_r1.baselines.probability_probe_capture import require, file_sha, tensor_digest
from search_r1.baselines.probability_probe_contracts import atomic_write_bytes_no_clobber, atomic_write_json_no_clobber
from search_r1.baselines.probability_probe_io import load_tensor_verified
from search_r1.baselines.probability_probe_loader import load_original_hf_prefix, wrap_original_fsdp, exhaustive_hf_cast_inventory
from search_r1.baselines.probability_probe_runner import (RankCoordinator, module_state, rng_state,
    immutable_native_call, verify_lazy_private_shards, verified_bundle_config, canonical_response_data,
    selected_rows, probability_statistics)
from window_contracts import RELEASED, PROFILE, ORIGINAL_CONFIG_SHA, validate_inputs, validate_window_config, bind_installed_sources, json_payload_sha
from window_runtime import (extract_original_rollout, load_bound_module, LiveWindowWeights,
    NativeWindowWrappers, bind_method, class_source, plain_object, sampling_params_metadata)
from window_observer import WindowObserver


def true_generated_alignment(observer_receipt, output, pad_token_id):
    """Bind new raw returned tensor rows to registered, unpadded actual outputs."""
    records = observer_receipt['generated']; batch = output.batch
    responses = batch['responses']; width = responses.shape[1]
    require(width == 500 and len(records) == len(responses), 'Original response cap/full request rows changed')
    records = sorted(records, key=lambda item:item['input_row'])
    require([r['input_row'] for r in records] == list(range(len(responses))), 'Actual request output rows incomplete')
    emitted = torch.zeros_like(responses, dtype=torch.bool); sampled = torch.zeros_like(responses, dtype=torch.float64)
    for row, record in enumerate(records):
        ids = record['token_ids']; values = [entry[str(token)]['logprob'] for entry,token in zip(record['logprobs'],ids)]
        require(record['true_response_length'] == len(ids) == len(values) and 0 < len(ids) <= width,
                'Actual emitted length/logprob metadata invalid')
        n = len(ids)
        require(responses[row,:n].tolist() == ids and bool((responses[row,n:] == pad_token_id).all()),
                'Facade changed original emitted token IDs or padded suffix')
        emitted[row,:n] = True; sampled[row,:n] = torch.tensor(values, dtype=torch.float64, device=sampled.device)
    require(torch.equal(batch['attention_mask'][:,-width:].bool(), emitted),
            'Original EOS-derived response mask differs from true emitted length; no guessing/repair')
    returned = batch.get('__sampling_log_probs')
    require(returned is not None and returned.shape == responses.shape and
            torch.equal(returned[emitted].double(), sampled[emitted]), 'Returned sampled scalar extraction changed')
    require(bool(torch.isfinite(sampled[emitted]).all()), 'Actual sampled token probability nonfinite')
    return emitted, sampled


def compare_window_anchors(observer_receipt, observed_vectors, actor_records, actor_vectors):
    """Direct bounded full-vocabulary observations; numeric differences are data.

    An anchor is paired only after both actual causal contexts/positions/targets
    match. No historical raw probability, threshold or unknown scalar is used.
    """
    actor = {item['anchor_id']:item for item in actor_records if 'anchor_id' in item}
    require(len(actor) == sum('anchor_id' in item for item in actor_records), 'Repeated actor anchor')
    native = observer_receipt['anchors']
    require(set(actor) == {item['id'] for item in native}, 'Backend anchor sets differ')
    result = []
    def stats(value):
        value = value.double()
        require(value.numel() > 0 and bool(torch.isfinite(value).all()), 'Nonfinite anchor difference')
        return dict(count=value.numel(),minimum=float(value.min()),maximum=float(value.max()),
                    mean=float(value.mean()),mean_abs=float(value.abs().mean()),max_abs=float(value.abs().max()))
    for original in native:
        anchor_id = original['id']; current = actor[anchor_id]
        require((current['input_row'],current['response_offset']) ==
                (original['input_row'],original['response_offset']), 'Anchor physical coordinate differs')
        if original['status'] == 'absent_native_early_finish':
            require(current['status'] == 'ABSENT_EARLY_FINISH' and anchor_id not in actor_vectors and
                    anchor_id not in observed_vectors, 'Early native finish anchor was fabricated')
            result.append(dict(anchor_id=anchor_id,status='absent_native_early_finish',
                request_id=original['request_id'],input_row=original['input_row'],response_offset=original['response_offset'],
                true_response_length=original['true_response_length'],finish_reason=original['finish_reason'],
                probability_accepted=False))
            continue
        require(original['status'] == 'observed' and current['status'] == 'ORIGINAL_ACTOR_SAME_CALL_LOGITS_CAPTURED',
                'Incomplete backend anchor')
        require(current['context_token_ids'] == original['context_token_ids'] and
                current['context_positions'] == original['context_positions'] and
                current['target_token'] == original['target_token_id'], 'Actual backend causal context/target differs')
        vectors = observed_vectors[anchor_id]
        require(set(vectors) == {'head_logits','sampler_input_logits'}, 'Incomplete native anchor vectors')
        head,sampler,policy = vectors['head_logits'].cpu(),vectors['sampler_input_logits'].cpu(),actor_vectors[anchor_id].cpu()
        require(head.ndim == sampler.ndim == policy.ndim == 1 and head.shape == sampler.shape == policy.shape and
                all(bool(torch.isfinite(x).all()) for x in (head,sampler,policy)), 'Anchor vocabulary shape/finite differs')
        target = current['target_token']; require(0 <= target < len(head), 'Anchor target outside actual vocabulary')
        delta = policy.double()-head.double(); center = float(delta.mean())
        normalized = {name:float(torch.log_softmax(value.float(),dim=-1)[target])
                      for name,value in (('vllm_head',head),('vllm_sampler_input',sampler),('actor',policy))}
        sampled,native_probability = original['sampled_logprob'],current['native_logprob']
        require(__import__('math').isfinite(sampled) and __import__('math').isfinite(native_probability),
                'Actual anchor scalar nonfinite')
        result.append(dict(anchor_id=anchor_id,status='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED',
            request_id=original['request_id'],input_row=original['input_row'],response_offset=original['response_offset'],
            vllm_context_token_ids=original['context_token_ids'],actor_context_token_ids=current['context_token_ids'],
            vllm_context_positions=original['context_positions'],actor_context_positions=current['context_positions'],
            vllm_target_token_id=original['target_token_id'],actor_target_token_id=target,vocab_shape=list(head.shape),
            full_vocab_actor_minus_vllm_head=stats(delta),removed_logit_constant=center,
            full_vocab_actor_minus_vllm_head_after_constant=stats(delta-center),
            full_vocab_sampler_input_minus_head=stats(sampler.double()-head.double()),
            fp32_normalized_target_logprobs=normalized,actual_selected_sampler_logprob=sampled,
            actual_original_actor_native_logprob=native_probability,
            fp32_actor_minus_native=normalized['actor']-native_probability,
            fp32_head_minus_actual_sampled=normalized['vllm_head']-sampled,
            fp32_sampler_input_minus_actual_sampled=normalized['vllm_sampler_input']-sampled,
            actual_actor_minus_sampled=native_probability-sampled,
            tensor_identities=dict(head=tensor_digest(head),sampler_input=tensor_digest(sampler),actor=tensor_digest(policy)),
            probability_accepted=False,training_accepted=False))
    return result


class ActorWindowRecorder:
    """Wrap the original compute_log_prob micro4 calls, one native each."""
    def __init__(self, actor, raw_batch, coordinator, device, anchors, weight_targets, alias_targets, resource_adapter):
        self.actor,self.batch,self.coordinator,self.device = actor,raw_batch,coordinator,device
        self.anchors = anchors; self.targets,self.aliases = weight_targets,alias_targets
        self.resource_adapter = resource_adapter; self.calls = 0; self.records = []; self.vectors = {}
        self.original = None; self.local = False; self.old_local = None

    def __enter__(self):
        self.original = self.actor._forward_micro_batch
        self.local = '_forward_micro_batch' in vars(self.actor); self.old_local = vars(self.actor).get('_forward_micro_batch')
        def wrapped(micro_batch, temperature):
            call = self.calls; self.calls += 1; start = call*4
            expected = min(4,len(self.batch['input_ids'])-start)
            def prepare():
                require(expected > 0 and len(micro_batch['input_ids']) == expected and temperature == 1.,
                        'Original micro4 count/temperature changed')
                for name in ('input_ids','attention_mask','position_ids','responses'):
                    require(torch.equal(micro_batch[name], self.batch[name][start:start+expected]),
                            'Actor original split reordered/changed new raw inputs: '+name)
                self.resource_adapter.check('actor-micro4-%03d'%call)
            self.coordinator.local_stage('actor micro4 prepare %03d'%call, prepare)
            proxy = SimpleNamespace(actor_module=self.actor.actor_module, use_remove_padding=self.actor.use_remove_padding,
                                    _forward_micro_batch=self.original)
            capture, receipt = self.coordinator.local_stage('actor original native %03d'%call, lambda:
                immutable_native_call(proxy,micro_batch,temperature,self.device,
                    self.targets if call == 0 else None,self.aliases if call == 0 else None))
            def save_anchors():
                mapping = capture['mapping']
                for anchor in self.anchors:
                    local = anchor['input_row']-start; offset = anchor['response_offset']
                    if not 0 <= local < expected: continue
                    hits = torch.nonzero((mapping['rows'] == local) & (mapping['response_positions'] == offset), as_tuple=False).flatten()
                    if not len(hits):
                        self.records.append(dict(anchor_id=anchor['id'],status='ABSENT_EARLY_FINISH',input_row=anchor['input_row'],response_offset=offset))
                        continue
                    require(len(hits) == 1, 'Ambiguous actor prediction anchor')
                    prediction = int(mapping['logit_indices'][int(hits[0])])
                    vector = capture['logits'][prediction].detach().clone().cpu()
                    valid = micro_batch['attention_mask'][local].bool()
                    prompt_width = micro_batch['input_ids'].shape[1]-micro_batch['responses'].shape[1]
                    prefix_end = prompt_width+offset
                    prefix_mask = valid[:prefix_end]
                    context = micro_batch['input_ids'][local,:prefix_end][prefix_mask].tolist()
                    positions = micro_batch['position_ids'][local,:prefix_end][prefix_mask].tolist()
                    self.vectors[anchor['id']] = vector
                    self.records.append(dict(anchor_id=anchor['id'],status='ORIGINAL_ACTOR_SAME_CALL_LOGITS_CAPTURED',
                        input_row=anchor['input_row'],response_offset=offset,target_token=int(micro_batch['responses'][local,offset]),
                        context_token_ids=context,context_positions=positions,logits_identity=tensor_digest(vector),
                        native_logprob=float(capture['native_response_log_probs'][local,offset])))
                receipt.update(call_index=call,input_start=start,rows=expected)
                self.records.append(dict(native_forward_receipt=receipt))
            self.coordinator.local_stage('actor same-call anchors %03d'%call, save_anchors)
            return capture['entropy'], capture['native_response_log_probs']
        self.actor._forward_micro_batch = wrapped
        return self

    def __exit__(self, kind, value, traceback):
        if self.local:self.actor._forward_micro_batch = self.old_local
        else:delattr(self.actor,'_forward_micro_batch')
        return False


def actual_engine_metadata(rollout):
    engine = rollout.inference_engine.llm_engine; runner = engine.model_executor.worker.model_runner
    properties = {
        'model_config':('dtype','max_model_len','enforce_eager','quantization','seed'),
        'parallel_config':('tensor_parallel_size','pipeline_parallel_size','world_size','disable_custom_all_reduce'),
        'cache_config':('cache_dtype','block_size','gpu_memory_utilization','sliding_window','enable_prefix_caching'),
        'scheduler_config':('max_num_batched_tokens','max_num_seqs','enable_chunked_prefill','num_scheduler_steps'),
    }
    configs = {}
    for name, fields in properties.items():
        obj = getattr(engine,name)
        configs[name] = {field:plain_object(getattr(obj,field)) for field in fields}
    return dict(configs=configs, sampling_params=sampling_params_metadata(rollout.sampling_params),
        classes={name:class_source(obj) for name,obj in [('engine',engine),('executor',engine.model_executor),
            ('worker',engine.model_executor.worker),('model_runner',runner),('model',runner.model),
            ('logits_processor',runner.model.logits_processor),('sampler',runner.model.sampler)]},
        original_constructor_extra_kwargs_added=False)


def run_window(bundle, coordinator, device, resource_adapter):
    """No resource default: this new profile must be validated by owner adapter."""
    start = time.monotonic(); rank = dist.get_rank()
    coordinator.local_stage('new owned runtime profile', lambda:require(bundle['window_profile'] == PROFILE and
        resource_adapter.supports_runtime_profile(PROFILE) is True,
        'UNIMPLEMENTED: reviewed adapter for new generation/vLLM profile required'))
    inputs = coordinator.local_stage('all-rank frozen call5 CPU contract', lambda:validate_inputs(bundle['window_inputs']))
    coordinator.local_stage('explicit installed native source set', lambda:bind_installed_sources(bundle['vllm_installed_sources']))
    config = coordinator.local_stage('exact original config bytes', lambda:verified_bundle_config(bundle))
    from omegaconf import OmegaConf
    coordinator.local_stage('original single-window protocol', lambda:validate_window_config(OmegaConf.to_container(config,resolve=True)))
    require(bundle['identity']['config_sha256'] == ORIGINAL_CONFIG_SHA, 'Original config SHA changed')
    cfg = config.actor_rollout_ref; local = inputs['ranks'][rank]
    coordinator.signature([local['local_rows'],local['expected_actor_micro4_calls'],1,500,4])
    output = Path(bundle['output'])/('rank-%02d'%rank)
    coordinator.local_stage('new rank artifact directory', lambda:output.mkdir(parents=False,exist_ok=False))
    prepared = bundle.pop('_actual_cpu_rank_inputs')
    coordinator.local_stage('prepared prompt bytes/metadata contract', lambda:require(
        prepared['metadata']['rank'] == rank and prepared['metadata']['local_rows'] == local['local_rows'] and
        prepared['metadata']['prompt_width'] == local['prompt_width'], 'Prepared prompt tensors changed'))
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh('cuda',mesh_shape=(8,),mesh_dim_names=['fsdp'])
    namespace,suffix,construction = coordinator.local_stage('original HF CPU prefix', lambda:load_original_hf_prefix(
        bundle['original_worker_path'],bundle['original_worker_sha256'],bundle['hf_path'],cfg,mesh,rank))
    model = namespace['actor_module']; tokenizer = namespace['self'].tokenizer
    alias_targets = (model,model.get_submodule('model.embed_tokens'),model.get_submodule('lm_head'))
    targets,hf_receipt = coordinator.local_stage('full HF435/unique434 effective cast targets', lambda:
        exhaustive_hf_cast_inventory(model,bundle['hf_path'],bundle['hf_index'],bundle['hf_identity']['files']))
    coordinator.local_stage('actual tokenizer IDs match CPU preparation', lambda:require(
        {k:getattr(tokenizer,k) for k in ('pad_token_id','eos_token_id','bos_token_id')} == prepared['tokenizer_metadata']['ids'],
        'Original HF tokenizer differs from explicit CPU-derived IDs'))
    coordinator.local_stage('before original FSDP live guard', lambda:resource_adapter.check('before-fsdp'))
    fsdp = wrap_original_fsdp(namespace,suffix,bundle['original_worker_path'])
    coordinator.local_stage('unchanged original master/MP', lambda:require(namespace['torch_dtype'] == torch.float32 and
        namespace['mixed_precision'].param_dtype == torch.bfloat16 and namespace['mixed_precision'].buffer_dtype == torch.float32,
        'Original FSDP effective dtype changed'))
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.fsdp_utils import offload_fsdp_param_and_grad,load_fsdp_param_and_grad
    cfg.actor.ppo_mini_batch_size//=8; cfg.actor.ppo_micro_batch_size//=8; cfg.rollout.log_prob_micro_batch_size//=8
    cfg.actor.use_remove_padding = cfg.model.use_remove_padding
    actor = DataParallelPPOActor(cfg.actor,fsdp,None); original_modes = {name:child.training for name,child in fsdp.named_modules()}
    fsdp.eval()
    def offload():offload_fsdp_param_and_grad(fsdp,offload_grad=True); torch.cuda.synchronize(device)
    def load():load_fsdp_param_and_grad(fsdp,device,load_grad=True); torch.cuda.synchronize(device)
    coordinator.local_stage('original initial offload',offload)
    prewarm = coordinator.local_stage('prewarm original local state',lambda:module_state(fsdp))
    # One declared actual-source warmup only. These historical raw responses
    # never become new outputs or enter this window's sampler comparisons.
    old = coordinator.local_stage('bound source call5 warmup asset',lambda:canonical_response_data(
        load_tensor_verified(local['source_snapshot']['path'],local['source_snapshot']['sha256']),'raw'))
    warm = coordinator.local_stage('source-only warmup first original adjacent4',lambda:selected_rows(old,list(range(4)),device))
    coordinator.local_stage('warmup original parameter load',load)
    _,warm_receipt = coordinator.local_stage('warmup actual FSDP native/effectiveweights',lambda:
        immutable_native_call(actor,warm,1.,device,targets,alias_targets))
    del old,warm
    coordinator.local_stage('warmup original offload',offload)
    baseline = coordinator.local_stage('postwarm baseline',lambda:module_state(fsdp))
    lazy = coordinator.local_stage('original master preserved through lazy warmup',lambda:
        verify_lazy_private_shards(prewarm['parameters'],baseline['parameters']))
    coordinator.local_stage('before original rollout resources',lambda:resource_adapter.check('before-rollout'))
    coordinator.local_stage('original parameter load for rollout constructor',load)
    self = namespace['self']; self.world_size = 8; self.actor_module_fsdp = fsdp; self.actor_model_config = namespace['actor_model_config']
    build_rollout,rollout_source = coordinator.local_stage('extract unchanged original rollout method',lambda:
        extract_original_rollout(bundle['original_worker_path'],bundle['original_worker_sha256']))
    rollout,manager = build_rollout(self)  # Native original constructor; no extra YAML kwargs.
    runtime = coordinator.local_stage('actual native engine settings/source metadata',lambda:actual_engine_metadata(rollout))
    def passive_services():
        require(actor.actor_optimizer is None and not ('ray' in sys.modules and sys.modules['ray'].is_initialized()),
                'Unexpected optimizer or initialized Ray service')
        return dict(ray_initialized=False,vllm_engine_required=True,optimizer_instantiated=False)
    services = coordinator.local_stage('zero optimizer/no Ray service',passive_services)
    core = coordinator.local_stage('bound accepted weight mapping core',lambda:load_bound_module(bundle['weights_core'],'tree_window_bound_weights'))
    native_model = rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model
    weights = LiveWindowWeights(core,native_model,namespace['actor_model_config'],device,coordinator)
    observer = WindowObserver(native_model,bundle['vllm_installed_sources'],inputs['anchors'],rank)
    def bound_active_methods():
        sources = bundle['vllm_installed_sources']
        decorators = [sources['utils.py']]+list(bundle['torch_wrapper_sources'].values())
        return {name:bind_method(method,sources[path],decorators) for name,method,path in (
            ('model_forward',native_model.forward,'model_executor/models/qwen2.py'),
            ('model_compute_logits',native_model.compute_logits,'model_executor/models/qwen2.py'),
            ('model_sample',native_model.sample,'model_executor/models/qwen2.py'),
            ('logits_processor_forward',native_model.logits_processor.forward,'model_executor/layers/logits_processor.py'),
            ('logits_processor_head',native_model.logits_processor._get_logits,'model_executor/layers/logits_processor.py'),
            ('sampler_forward',native_model.sampler.forward,'model_executor/layers/sampler.py'))}
    actual_active_methods = coordinator.local_stage('actual native active method source chain',bound_active_methods)
    expected_prompts = [prepared['batch']['input_ids'][r][prepared['batch']['attention_mask'][r].bool()].tolist() for r in range(local['local_rows'])]
    observer.observe_prompts(expected_prompts)
    coordinator.local_stage('observer original prompt registration',lambda:require(not observer.errors,'Prompt mapping failed'))
    from verl import DataProto
    prompts = DataProto.from_dict(tensors={k:v.to(device) for k,v in prepared['batch'].items()},
        meta_info=dict(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,do_sample=True))
    native_sources = dict(bundle['native_method_sources'])
    native_sources['decorators'] = [bundle['vllm_installed_sources']['utils.py']]+list(bundle['torch_wrapper_sources'].values())
    wrappers = NativeWindowWrappers(rollout.inference_engine,observer,weights,native_sources)
    before_rng = rng_state(device); inside_rng_before = inside_rng_after = None
    # Preparatory hook binding must succeed on every rank before state_dict sync.
    coordinator.local_stage('passive native hook source binding',lambda:wrappers.__enter__())
    coordinator.local_stage('passive logits/sampler hooks',lambda:observer.__enter__())
    try:
        with manager:
            coordinator.local_stage('completed sync live effectiveweights',lambda:require(not wrappers.errors and weights.pre is not None,
                'Synchronized live weight proof failed: '+repr(wrappers.errors)))
            inside_rng_before = rng_state(device)
            before_params = sampling_params_metadata(rollout.sampling_params)
            prompts = manager.preprocess_data(prompts)
            raw = rollout.generate_sequences(prompts=prompts)
            raw = manager.postprocess_data(raw)
            inside_rng_after = rng_state(device)
            coordinator.local_stage('live generated state before offload',weights.before_offload)
            after_params = sampling_params_metadata(rollout.sampling_params)
            coordinator.local_stage('original actual sampling parameters unchanged',lambda:require(
                before_params == after_params,'Native window changed original SamplingParams'))
    finally:
        observer.__exit__(*sys.exc_info()); wrappers.__exit__(*sys.exc_info())
    def observed_receipt():
        value = observer.receipt()
        require(value['observer_evidence_complete'], 'Actual observer evidence refused: '+repr(value['errors']))
        return value
    observer_receipt = coordinator.local_stage('actual original unpadded output/context mapping',observed_receipt)
    wrapper_receipt = coordinator.local_stage('passive original method once/restoration',wrappers.receipt)
    weight_receipt = coordinator.local_stage('both live weight boundaries',weights.receipt)
    emitted,sampled = coordinator.local_stage('new raw true-emitted-length/scalar alignment',lambda:
        true_generated_alignment(observer_receipt,raw,tokenizer.pad_token_id))
    # Original actor compute_log_prob calls eval and splits local4; no modified
    # denominator/objective or generation from old raw output is introduced.
    raw.meta_info.update(micro_batch_size=4,max_token_len=cfg.rollout.log_prob_max_token_len_per_gpu,
        use_dynamic_bsz=False,temperature=1.)
    recorder = ActorWindowRecorder(actor,raw.batch,coordinator,device,[a for a in inputs['anchors'] if a['rank'] == rank],
                                   targets,alias_targets,resource_adapter)
    with recorder:
        actor_logprobs = actor.compute_log_prob(raw)
    coordinator.local_stage('complete matched original actor local4 forward counts',lambda:require(
        recorder.calls == local['expected_actor_micro4_calls'],'Original compute_log_prob forward-count changed'))
    comparisons = coordinator.local_stage('actual new sampler-vs-native scalar observations',lambda:
        probability_statistics(actor_logprobs[emitted].double()-sampled[emitted]))
    anchor_comparisons = coordinator.local_stage('actual same-context full-vocabulary anchor observations',lambda:
        compare_window_anchors(observer_receipt,observer.tensor_evidence(),recorder.records,recorder.vectors))
    coordinator.local_stage('original final offload',offload)
    coordinator.local_stage('master/buffers/gradients unchanged after original eval normalization',lambda:
        require(module_state(fsdp) == baseline,'Warmed actor master/gradients/buffers/modes changed'))
    def restore():
        for name,child in fsdp.named_modules(): child.training = original_modes[name]
    coordinator.local_stage('original construction modes restored',restore)
    coordinator.local_stage('final live resource/background check',lambda:resource_adapter.check('final'))
    evidence = dict(observer=observer.tensor_evidence(),actor_anchor_logits=recorder.vectors,
        new_raw_inputs={k:v.detach().cpu() for k,v in raw.batch.items()},actual_emitted_mask=emitted.cpu(),
        actual_sampled_logprobs=sampled.cpu(),actual_actor_logprobs=actor_logprobs.cpu())
    def publish():
        stream = io.BytesIO(); torch.save(evidence,stream)
        receipt = atomic_write_bytes_no_clobber(output/'window.pt',stream.getvalue())
        result = dict(status='SINGLE_ORIGINAL_TP1_WINDOW_OBSERVED_NO_PROBABILITY_OR_TRAINING_ACCEPTANCE',
            rank=rank,world_size=8,window_profile=PROFILE,bundle_sha256=os.environ['TREE_PROBE_BUNDLE_SHA256'],
            window_inputs_sha256=json_payload_sha(inputs),generation_invocations=1,
            original_generation_call=5,original_prompt_rows=local['local_rows'],
            maximum_new_emitted_tokens=500*local['local_rows'],actual_new_emitted_tokens=int(emitted.sum()),
            actor_native_forwards=1+recorder.calls,actor_scoring_native_forwards=recorder.calls,warmup_native_forwards=1,
            cpu_preparation=prepared['cpu_preparation'],actual_loaded_binary_identity_proven=False,
            optimizer_instantiated=False,optimizer_committed=0,
            scheduler_instantiated=False,scheduler_advanced=False,backward_called=False,
            original_master_state_unchanged=True,original_modes_restored=True,lazy_warmup_metadata=lazy,
            warmup_historical_input_not_new_generation=True,construction=construction,hf=hf_receipt,
            warmup=warm_receipt,original_rollout_constructor=rollout_source,actual_runtime=runtime,
            actual_active_method_sources=actual_active_methods,
            services=services,live_window_weights=weight_receipt,observer=observer_receipt,wrappers=wrapper_receipt,
            actor_calls=recorder.records,raw_probability_observations=comparisons,
            anchor_observations=anchor_comparisons,
            sampling_params_before=before_params,sampling_params_after=after_params,
            rng_scope=dict(outer_before=before_rng,outer_after=rng_state(device),
                generation_inside_before=inside_rng_before,generation_inside_after=inside_rng_after,
                sampler_rng_unchanged_claim=False,new_generation_not_historical_replay=True),
            tensor_evidence=receipt,source_and_input_bundle_identity=bundle['identity'],
            own_wall_seconds=time.monotonic()-start,probability_accepted=False,training_accepted=False,
            GPU_capacity_proven=False)
        atomic_write_json_no_clobber(output/'ready.json',result)
        return result
    return coordinator.local_stage('new all-rank no-clobber observation evidence',publish)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--bundle',required=True); p.add_argument('--bundle-sha',required=True)
    args = p.parse_args()
    from window_contracts import bound_json
    bundle = bound_json(args.bundle,args.bundle_sha)
    require(os.environ.get('TREE_PROBE_BUNDLE_SHA256') == args.bundle_sha,
            'Owned launch environment and exact bundle bytes differ')
    require(bundle.get('status') == RELEASED,'UNIMPLEMENTED: single-window independent source/body/dispatch release required')
    require(socket.gethostname().split('.')[0] == 'h100-9' and os.environ.get('SLURM_JOB_NODELIST') == 'h100-9' and
            os.environ.get('WORLD_SIZE') == '8' and os.environ.get('MASTER_ADDR') == '127.0.0.1', 'Same-node eight-rank loopback Slurm required')
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1','Offline same-node weights required')
    entry = bundle['reviewed_resource_adapter']
    adapter = load_bound_module(entry,'tree_window_reviewed_resource_adapter')
    require(bundle['window_profile'] == PROFILE,'New single-window profile is not bound')
    # CPU Gloo metadata/tensor loading first; local failures MAX-synchronize
    # before any rank selects a CUDA device or constructs a GPU process group.
    require(not torch.cuda.is_initialized(),'Pre-input CPU phase cannot begin with CUDA initialized')
    dist.init_process_group('gloo',timeout=timedelta(seconds=180)); cpu_coordinator = RankCoordinator(torch.device('cpu'))
    try:
        cpu_coordinator.local_stage('independent new bundle/live owned-runtime validation',lambda:require(
            adapter.prepare_and_validate_bundle(bundle) is True,'Owned-runtime single-window validation failed'))
        from prepare_window_inputs import materialize_rank_inputs
        rank_inputs = cpu_coordinator.local_stage('actual complete call5 CPU tensors before CUDA',lambda:
            materialize_rank_inputs(bundle,dist.get_rank()))
        cpu_coordinator.local_stage('CPU actual shape/hash phase stayed CUDA-free',lambda:require(
            not torch.cuda.is_initialized(),'Actual prompt preparation initialized CUDA'))
    finally: dist.destroy_process_group()
    bundle['_actual_cpu_rank_inputs'] = rank_inputs
    local_rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local_rank); device = torch.device('cuda',local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180)); coordinator = RankCoordinator(device)
    try:
        run_window(bundle,coordinator,device,adapter)
    finally: dist.destroy_process_group()


if __name__ == '__main__': main()
