"""Diagnostic worker subclass: observe real engine probabilities without changing outputs."""
import copy
import json
import os
from pathlib import Path
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from search_r1.baselines.recovery import (atomic_torch_save, tensor_digest, rng_state,
                                        cpu_snapshot, parameter_layout, checkpoint_pause_identity)
from search_r1.baselines.instruct300 import equal_state
from calibration_state_v1 import state_digest
from raw_hf_observation_v1 import capture_existing_flat_calls, clone_cpu, require


def observe_backend_identity(engine, rank):
    """Read selected runner and every attention implementation; no tensor computation."""
    runner = engine.llm_engine.model_executor.worker.model_runner
    backend = runner.attn_backend
    rows = []
    for name, module in runner.model.named_modules():
        if type(module).__module__ == 'vllm.attention.layer' and type(module).__name__ == 'Attention':
            implementation = module.impl
            rows.append(dict(name=name,
                implementation_class=type(implementation).__module__ + '.' + type(implementation).__name__,
                head_size=implementation.head_size, num_heads=implementation.num_heads,
                num_kv_heads=implementation.num_kv_heads,
                sliding_window=list(implementation.sliding_window)
                    if isinstance(implementation.sliding_window, (tuple, list)) else implementation.sliding_window))
    return dict(scope='ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY', rank=rank, pid=os.getpid(),
        source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
        requested_backend=os.environ.get('VLLM_ATTENTION_BACKEND'),
        runner_backend_class=backend.__module__ + '.' + backend.__name__,
        runner_backend_name=backend.get_name(), model_dtype=str(runner.model_config.dtype),
        cache_dtype=runner.cache_config.cache_dtype, block_size=runner.cache_config.block_size,
        model_sliding_window=runner.model_config.get_sliding_window(), attention_layers=rows,
        observation_forward_calls=0, observation_tensor_value_reads=0,
        actual_vllm_runner_positions_observed=False, probability_accepted=False)


class CalibrationWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_backend_identity(self):
        assert self._is_actor
        return observe_backend_identity(self.rollout.inference_engine, self.rank)

    def _complete_pause_state(self):
        """Observe full local actor state after sampling, without a restore or optimizer write."""
        assert self._is_actor and self.config.actor.get('complete_internal_state', False)
        module, optimizer, scheduler = self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler
        manager = self.rollout_sharding_manager
        identity = checkpoint_pause_identity(module, optimizer, scheduler, self.rank, manager, True)
        names = {id(parameter): name for name, parameter in module.named_parameters()}
        def placement(tensor):
            return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                        stride=list(tensor.stride()), storage_offset=tensor.storage_offset())
        metadata = dict(rng=cpu_snapshot(rng_state()), scheduler=cpu_snapshot(scheduler.state_dict()),
            sharding_rng={key: cpu_snapshot(getattr(manager, key, None))
                          for key in ('gen_random_states', 'torch_random_states')},
            optimizer_groups=[dict(names=[names[id(parameter)] for parameter in group['params']],
                                  controls=cpu_snapshot({key: value for key, value in group.items() if key != 'params'}))
                              for group in optimizer.param_groups],
            optimizer_nontensors={name: cpu_snapshot({key: value for key, value in optimizer.state.get(parameter, {}).items()
                                   if not isinstance(value, torch.Tensor)}) for name, parameter in module.named_parameters()},
            parameter_layout=parameter_layout(module, optimizer, self.rank, self.world_size),
            module_training_modes={name: part.training for name, part in module.named_modules()},
            tensor_placements=dict(model={name: placement(parameter) for name, parameter in module.named_parameters()},
                local_shards={name: placement(parameter._local_shard) for name, parameter in module.named_parameters()
                              if hasattr(parameter, '_local_shard')},
                buffers={name: placement(buffer) for name, buffer in module.named_buffers()},
                optimizer={name: {key: placement(value) for key, value in optimizer.state.get(parameter, {}).items()
                                  if isinstance(value, torch.Tensor)} for name, parameter in module.named_parameters()},
                gradients={name: placement(parameter.grad) for name, parameter in module.named_parameters()
                           if parameter.grad is not None}),
            gradients_sha256=tensor_digest((name, parameter.grad) for name, parameter in module.named_parameters()
                                          if parameter.grad is not None))
        identity['complete_state_sha256'] = state_digest(dict(identity=identity, metadata=metadata))
        return dict(identity=identity, metadata=metadata)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_complete_pause_state(self):
        return self._complete_pause_state()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_phase(self, phase):
        assert phase in ('baseline512', 'baseline64', 'train', 'updated512', 'updated64')
        self._calibration_phase = phase
        self._calibration_call = 0
        return dict(rank=self.rank, pid=os.getpid(), phase=phase)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_reference_identity(self):
        assert self._is_ref
        values = list(self.ref_module_fsdp.parameters()) + list(self.ref_module_fsdp.buffers())
        finite = all(bool(torch.isfinite(v.detach()).all()) for v in values)
        return dict(rank=self.rank, pid=os.getpid(), finite=finite,
                    source=self.config.ref.model_path,
                    shards_sha256=tensor_digest(self.ref_module_fsdp.named_parameters()),
                    buffers_sha256=tensor_digest(self.ref_module_fsdp.named_buffers()))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        if self._calibration_phase != 'train':
            return super().generate_sequences(prompts)
        # The engine returns actual chosen-token log-probs; frozen rollout drops them.
        # Tap only this instance for the duration of the existing method call.
        engine = self.rollout.inference_engine
        original = engine.generate
        had_generate_override = 'generate' in vars(engine)
        captured = []

        def observe(*args, **kwargs):
            from raw_runner_observation_v1 import observe_actual_runner
            with observe_actual_runner(engine) as runner_observation:
                value = original(*args, **kwargs)
            parameters = kwargs['sampling_params']
            lengths = engine.last_completion_token_lengths
            require(torch.is_tensor(lengths) and lengths.dtype == torch.long, 'No actual CompletionOutput length evidence')
            captured.append((value[0].detach().cpu().clone(), value[1].detach().cpu().clone(), lengths.detach().cpu().clone(),
                copy.deepcopy(kwargs['prompt_token_ids']), runner_observation, {key: getattr(parameters, key) for key in
                    ('temperature', 'top_p', 'top_k', 'n', 'logprobs', 'max_tokens', 'repetition_penalty', 'presence_penalty', 'frequency_penalty')}))
            return value

        engine.generate = observe
        try:
            with capture_existing_flat_calls(self.actor) as perturn_calls:
                output = super().generate_sequences(prompts)
        finally:
            if had_generate_override:
                engine.generate = original
            else:
                del engine.generate
        assert len(captured) == 1 and 'old_log_probs' in output.batch
        sampled_ids, sampled_logp, sampled_lengths, engine_prompts, runner_observation, sampling_parameters = captured[0]
        require(torch.equal(output.batch['response_sampled_lengths'].cpu(), sampled_lengths), 'Engine/output real sample lengths differ')
        sampled_mask = output.batch['response_sample_mask'].detach().cpu().clone()
        require(torch.equal(sampled_mask.bool(), torch.arange(sampled_mask.shape[1])[None, :] < sampled_lengths[:, None]), 'Actual real sample mask differs')
        assert sampled_ids.shape == sampled_logp.shape
        # Re-run the real hybrid actor method on exactly the returned prefix/tokens.
        # It retains the same temperature/micro settings and offload path.
        before_state = self._complete_pause_state()
        with capture_existing_flat_calls(self.actor) as repeated_perturn_calls:
            repeated = super().compute_log_prob(copy.deepcopy(output)).batch['old_log_probs']
        after_state = self._complete_pause_state()
        rng_unchanged = equal_state(before_state['metadata']['rng'], after_state['metadata']['rng'])
        complete_unchanged = equal_state(before_state, after_state)
        # A changed probe is recorded and rejected on the driver. Do not restore
        # RNG (or anything else) to make a state-changing observation look neutral.
        width = sampled_ids.shape[1]
        valid = sampled_mask[:, :width].bool()  # Length-defined support, including actual sampled PAD.
        old = output.batch['old_log_probs'][:, :width]
        repeated = repeated[:, :width]
        bad = (~torch.isfinite(old) | ~torch.isfinite(repeated) | ~torch.isfinite(sampled_logp)) & valid
        count = int(valid.sum())
        # Evidence always survives a bad comparison; admission happens on driver.
        safe = valid & ~bad
        max_old = float((old.double() - repeated.double())[safe].abs().max()) if bool(safe.any()) else 0.
        max_vllm = float((old.double() - sampled_logp.double())[safe].abs().max()) if bool(safe.any()) else 0.
        phase = self._calibration_phase
        serial = self._calibration_call
        self._calibration_call += 1
        directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / phase
        directory.mkdir(parents=True, exist_ok=True)
        raw = dict(phase=phase, serial=serial, rank=self.rank, pid=os.getpid(),
                   source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                   binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                   tensors={key: output.batch[key].detach().cpu().clone() for key in
                            ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs',
                             'response_sampled_lengths', 'response_sample_mask')},
                   sampled_ids=sampled_ids, sampled_log_probs=sampled_logp,
                   sampled_lengths=sampled_lengths, sampled_mask=sampled_mask,
                   perturn_hf_flat_calls=perturn_calls, repeated_perturn_hf_flat_calls=repeated_perturn_calls,
                   actual_pad_token_id=self.tokenizer.pad_token_id, actual_eos_token_id=self.tokenizer.eos_token_id,
                   actor_repeated_log_probs=repeated, valid_comparison_mask=valid,
                   actual_engine_prompt_token_ids=engine_prompts, actual_runner_observation=runner_observation,
                   actual_sampling_parameters=sampling_parameters,
                   repeated_probe_state_before=before_state, repeated_probe_state_after=after_state)
        receipt = atomic_torch_save(raw, directory / f'call-{serial}-rank-{self.rank}.pt')
        record = dict(phase=phase, serial=serial, rank=self.rank, pid=os.getpid(),
                      source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                      comparison_scope='SAME_PREFIX_SAME_TOKENS', valid_tokens=count,
                      nonfinite_effective_count=int(bad.sum()), actor_old_max_abs_delta=max_old,
                      actor_vllm_max_abs_delta=max_vllm, temperature=sampling_parameters['temperature'],
                      top_p=sampling_parameters['top_p'], top_k=sampling_parameters['top_k'], n=sampling_parameters['n'],
                      rows=len(output), raw=receipt)
        record['repeated_probability_probe_rng_unchanged'] = rng_unchanged
        record['repeated_probability_probe_complete_state_unchanged'] = complete_unchanged
        record['repeated_probe_identity_before'] = before_state['identity']
        record['repeated_probe_identity_after'] = after_state['identity']
        (directory / f'call-{serial}-rank-{self.rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        # Do not alter tensor keys, masks, probabilities, generation or returned metadata.
        return output

    def _capture_raw_complete_hf(self, role, data, original):
        """Observe the existing worker API and its actual independent-action forwards."""
        require(data.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1', 'Raw HF capture needs raw source')
        require(data.meta_info.get('raw_calibration_observation') is True, 'Unarmed diagnostic observer')
        actor = self.ref_policy if role == 'reference' else self.actor
        from search_r1.baselines.action_prefix_forward import ACTION_PREFIX_KEYS
        keys = list(ACTION_PREFIX_KEYS) + ['responses', 'input_ids', 'attention_mask', 'position_ids']
        keys += [key for key in ('info_mask', 'loss_mask', 'step_ids') if key in data.batch]
        inputs = clone_cpu({key: data.batch[key] for key in keys})
        sources = [int(value) for value in data.non_tensor_batch['audit_trajectory_row']]
        question_ids = clone_cpu(data.non_tensor_batch['audit_question_index'])
        provenance = copy.deepcopy(data.non_tensor_batch['action_prefix_records'].tolist())
        with capture_existing_flat_calls(actor) as calls:
            output = original()
        name = 'ref_log_prob' if role == 'reference' else 'old_log_probs'
        errors = []
        if any(call['observer_errors'] for call in calls):
            errors.append('Actual HF input observation has errors')
        raw = dict(schema='RAW_ACTION_HF_FORWARD_OBSERVATION_V1', role=role, rank=self.rank, pid=os.getpid(),
            code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
            actor_token_identity=data.meta_info['actor_token_identity'], actual_pad_token_id=self.tokenizer.pad_token_id,
            actual_eos_token_id=self.tokenizer.eos_token_id, temperature=data.meta_info['temperature'],
            micro_batch_size=data.meta_info['micro_batch_size'], input_tensors=inputs,
            trajectory_sources=sources, trajectory_question_ids=question_ids,
            action_prefix_records=provenance, flat_calls=calls, output_log_probs=clone_cpu(output.batch[name]),
            observer_errors=errors, sampler_runner_positions_observed=False,
            model_source=self.config.ref.model_path if role == 'reference' else self.config.model.path)
        directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / 'complete-actions'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'{role}-rank-{self.rank}.pt'
        require(not target.exists(), 'Raw HF role observer must not overwrite previous evidence')
        receipt = atomic_torch_save(raw, target)
        (directory / f'{role}-rank-{self.rank}.json').write_text(json.dumps(dict(
            schema=raw['schema'], role=role, rank=self.rank, pid=os.getpid(), code_commit=raw['code_commit'],
            binding_sha256=raw['binding_sha256'], raw=receipt, observer_errors=errors,
            actual_forward_calls=len(calls), observation_only=True, actual_adam_writes=0), indent=2, allow_nan=False))
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if data.meta_info.get('raw_calibration_observation') is not True:
            return super().compute_log_prob(data)
        return self._capture_raw_complete_hf('old', data,
            lambda: super(CalibrationWorker, self).compute_log_prob(data))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def calibration_raw_current_log_prob(self, data):
        """One real same-state repeat, recorded separately and required to be neutral."""
        before = self._complete_pause_state()
        output = self._capture_raw_complete_hf('current', data,
            lambda: super(CalibrationWorker, self).compute_log_prob(data))
        after = self._complete_pause_state()
        directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / 'complete-actions'
        (directory / f'current-neutral-rank-{self.rank}.json').write_text(json.dumps(dict(
            rank=self.rank, pid=os.getpid(), before=before['identity'], after=after['identity'],
            complete_state_unchanged=equal_state(before, after), observation_only=True,
            actual_adam_writes=0), indent=2, allow_nan=False))
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        if data.meta_info.get('raw_calibration_observation') is not True:
            return super().compute_ref_log_prob(data)
        before = self.calibration_reference_identity()
        output = self._capture_raw_complete_hf('reference', data,
            lambda: super(CalibrationWorker, self).compute_ref_log_prob(data))
        after = self.calibration_reference_identity()
        directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / 'complete-actions'
        (directory / f'reference-neutral-rank-{self.rank}.json').write_text(json.dumps(dict(
            rank=self.rank, pid=os.getpid(), before=before, after=after, fixed_reference_unchanged=before == after,
            observation_only=True, actual_adam_writes=0), indent=2, allow_nan=False))
        return output
