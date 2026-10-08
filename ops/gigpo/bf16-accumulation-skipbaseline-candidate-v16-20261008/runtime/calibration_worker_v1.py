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
from calibration_state_v1 import (BF16_ACCUMULATION_SCOPE, state_digest,
                                  validate_bf16_accumulation_identity)
from raw_hf_observation_v1 import capture_existing_flat_calls, clone_cpu, require
from observer_safety_v2 import attempt, note, patches, observer_failures
from calibration_boundary_v2 import matched_stage


def observe_backend_identity(engine, rank, bf16_accumulation):
    """Read selected runner and every attention implementation; no tensor computation."""
    executor = engine.llm_engine.model_executor
    runner = executor.worker.model_runner
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
        bf16_accumulation=bf16_accumulation,
        executor_class=type(executor).__module__ + '.' + type(executor).__name__,
        executor_worker_class=type(executor.worker).__module__ + '.' + type(executor.worker).__name__,
        runner_backend_class=backend.__module__ + '.' + backend.__name__,
        runner_backend_name=backend.get_name(), model_dtype=str(runner.model_config.dtype),
        cache_dtype=runner.cache_config.cache_dtype, block_size=runner.cache_config.block_size,
        model_sliding_window=runner.model_config.get_sliding_window(), attention_layers=rows,
        observation_forward_calls=0, observation_tensor_value_reads=0,
        actual_vllm_runner_positions_observed=False, probability_accepted=False)


class CalibrationWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        # The only setter: new role construction precedes all source worker,
        # model/FSDP initialization and vLLM profiling or generation forwards.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self._bf16_accumulation_constructor_pid = os.getpid()
        self._bf16_accumulation_constructor_role = role
        require(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False,
                'BF16 reduction configuration was not applied before source initialization')
        super().__init__(config=config, role=role)
        self._bf16_accumulation_identity()

    def _bf16_accumulation_identity(self):
        """Read the live process flag; a changed flag refuses instead of being reset."""
        row = dict(pid=os.getpid(), bf16_accumulation=dict(
            scope=BF16_ACCUMULATION_SCOPE,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            configured_before_base_worker_initialization=True,
            constructor_pid=self._bf16_accumulation_constructor_pid,
            worker_role=self._bf16_accumulation_constructor_role))
        return validate_bf16_accumulation_identity(row, role=self.role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_bf16_accumulation_identity(self):
        errors, row = [], None
        try:
            row = dict(rank=self.rank, pid=os.getpid(),
                bf16_accumulation=self._bf16_accumulation_identity())
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('bf16_accumulation/local_process_readback', errors)
        return row

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_initialize_actor_fsdp_before_zero(self):
        """Establish native local shards before the immutable zero-state identity."""
        errors, row = [], None
        try:
            self._bf16_accumulation_identity()
            require(self._is_actor and self.config.actor.get('complete_internal_state', False),
                    'Initialize only the complete-state actual actor role')
            require(self.actor_lr_scheduler.last_epoch == 0
                    and all(group['lr'] == 0.0 for group in self.actor_optimizer.param_groups)
                    and not any(self.actor_optimizer.state.values()),
                    'FSDP initialization must precede all Adam/scheduler updates')
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('official_zero/fsdp_initialization_precondition', errors)
        errors = []
        try:
            from verl.utils.fsdp_utils import initialize_fsdp_before_full_restore
            result = initialize_fsdp_before_full_restore(self.actor_module_fsdp)
            require(all(hasattr(parameter, '_local_shard')
                        for parameter in self.actor_module_fsdp.parameters()),
                    'Official zero requires complete native local shards')
            row = dict(rank=self.rank, pid=os.getpid(), initialization=result,
                bf16_accumulation=self._bf16_accumulation_identity(),
                source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                forward_calls=0, Adam_writes=0, scheduler_advances=0)
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('official_zero/native_fsdp_initialization', errors)
        return row

    def _live_admission_identity(self):
        from search_r1.baselines.fresh200_live import worker_admission_state
        return dict(rank=self.rank, pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            role='actor' if self._is_actor else 'reference', state=worker_admission_state(self))

    def _admission_protected_state(self):
        """Full actor state; fixed ref values, placement/modes and all process RNG."""
        if self._is_actor:
            value = copy.deepcopy(self._complete_pause_state())
            value['identity'].pop('complete_state_sha256')
            value['metadata']['source_runtime_metadata'].pop('fresh200_live_admission')
            return value
        require(self._is_ref, 'Only actual actor/reference admission roles')
        module = self.ref_module_fsdp
        placement = lambda values: {name: dict(device=str(tensor.device), dtype=str(tensor.dtype),
            shape=list(tensor.shape), stride=list(tensor.stride()), storage_offset=tensor.storage_offset())
            for name, tensor in values}
        return dict(identity=self.calibration_reference_identity(), rng=cpu_snapshot(rng_state()),
            module_training_modes={name: part.training for name, part in module.named_modules()},
            parameters=placement(module.named_parameters()), buffers=placement(module.named_buffers()),
            local_shards=placement((name, parameter._local_shard) for name, parameter in module.named_parameters()
                if hasattr(parameter, '_local_shard')))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_live_admission_identity(self):
        errors, row = [], None
        try:
            row = self._live_admission_identity()
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('live_admission/local_identity', errors)
        return row

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_install_live_admission(self, payload):
        from search_r1.baselines.fresh200_live import validate_admission, install_worker_admission
        errors, before = [], None
        try:
            validate_admission(payload)
            require(payload['config_epoch'] == 1, 'Install only approved production C1')
            if self._is_actor:
                ledger = self.fresh200_worker_ledger
                require(ledger['completed_outer_updates'] == 5 and self.actor_lr_scheduler.last_epoch == 5,
                        'Worker admission must precede sixth update at actual completed5')
            before = self._admission_protected_state()
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('live_admission/prevalidate_idle_completed5', errors)
        errors, row = [], None
        try:
            install_worker_admission(self, payload)
            after = self._admission_protected_state()
            require(equal_state(before, after), 'Policy admission changed model/Adam/RNG/placement/ledger')
            row = self._live_admission_identity()
            row.update(protected_state_before_sha256=state_digest(before),
                       protected_state_after_sha256=state_digest(after))
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('live_admission/install_and_exact_readback', errors)
        return row

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_restore_live_admission(self, rows):
        from search_r1.baselines.fresh200_live import restore_worker_admission_state
        errors, result = [], None
        try:
            require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)), 'Missing saved role all8 state')
            expected = next(row for row in rows if row['rank'] == self.rank)
            live = self._live_admission_identity()
            require(all(expected[key] == live[key] for key in ('rank', 'pid', 'start_ticks', 'role')),
                    'Admission restoration cannot change PID/start/role')
            before = self._admission_protected_state()
            restore_worker_admission_state(self, expected['state'])
            require(equal_state(before, self._admission_protected_state()), 'Admission reload changed actual training state')
            result = self._live_admission_identity()
            require(result == expected, 'Saved actor/reference config/env/admission actual readback differs')
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('live_admission/restore_and_exact_readback', errors)
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        # Local admission metadata at a known allrank idle boundary. The original
        # distributed update below is deliberately outside the local-error catch.
        from search_r1.baselines.fresh200_live import validate_live_update_boundary
        errors = []
        try:
            self._bf16_accumulation_identity()
            validate_live_update_boundary(self, data)
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('live_admission/pre_update_boundary', errors)
        return super().update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_backend_identity(self):
        assert self._is_actor
        return observe_backend_identity(self.rollout.inference_engine, self.rank,
                                        self._bf16_accumulation_identity())

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
        metadata = dict(source_runtime_metadata=self.checkpoint_runtime_metadata(),
            bf16_accumulation=self._bf16_accumulation_identity(),
            rng=cpu_snapshot(rng_state()), scheduler=cpu_snapshot(scheduler.state_dict()),
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
        errors, state = [], None
        try:
            state = self._complete_pause_state()
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('complete_pause/local_snapshot', errors)
        return state

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_preserve_zero_adam(self):
        """Known matched refusal only: local no-collective save; explicit missing coverage."""
        from search_r1.baselines.recovery import worker_training_state
        result = dict(rank=self.rank, pid=os.getpid(), saved=False, errors=[],
                      exact_resume_accepted=False, validated_resume_checkpoint=False)
        try:
            result['scheduler_step'] = self.actor_lr_scheduler.last_epoch
            result['zero_adam_verified'] = (result['scheduler_step'] == 0
                and not any(bool(slot) for slot in self.actor_optimizer.state.values()))
            # Even corrupted/refused observation evidence cannot block saving actual state.
            state = worker_training_state(self.actor_module_fsdp, self.actor_optimizer,
                self.actor_lr_scheduler, self.rank, self.world_size, 0,
                self.rollout_sharding_manager, complete_internal=True)
            state['diagnostic_failure_scope'] = 'FIRST_SAMPLING_REFUSED_ZERO_ADAM_NOT_RESUME_ACCEPTED'
            state['validated_resume_checkpoint'] = False
            state['automatic_resume_allowed'] = False
            try:
                state['runtime_metadata'] = self.checkpoint_runtime_metadata()
            except BaseException as metadata_error:
                # Full original local shards/RNG still save when metadata observation is damaged.
                result['errors'].append('runtime_metadata: ' + repr(metadata_error))
                state['runtime_metadata_missing'] = True
            path = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'probability-rejection-state' / f'rank_{self.rank}.pt'
            result['raw'] = atomic_torch_save(state, path)
            result['saved'] = True
        except BaseException as error:
            result['errors'].append(repr(error))
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_phase(self, phase):
        assert phase in ('baseline512', 'train', 'train2', 'updated512')
        self._calibration_phase = phase
        self._calibration_call = 0
        return dict(rank=self.rank, pid=os.getpid(), phase=phase)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibration_reference_identity(self):
        assert self._is_ref
        values = list(self.ref_module_fsdp.parameters()) + list(self.ref_module_fsdp.buffers())
        finite = all(bool(torch.isfinite(v.detach()).all()) for v in values)
        return dict(rank=self.rank, pid=os.getpid(), finite=finite,
                    bf16_accumulation=self._bf16_accumulation_identity(),
                    source=self.config.ref.model_path,
                    shards_sha256=tensor_digest(self.ref_module_fsdp.named_parameters()),
                    buffers_sha256=tensor_digest(self.ref_module_fsdp.named_buffers()))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        if self._calibration_phase != 'train':
            return super().generate_sequences(prompts)
        errors, captured, context_record = [], [], dict(observer_errors=[])
        engine, original = None, None
        try:
            self._bf16_accumulation_identity()
            engine = self.rollout.inference_engine
            original = engine.generate
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('generate/before_original_local_setup', errors)

        def observe(*args, **kwargs):
            from contextlib import nullcontext
            context = nullcontext(dict(observer_errors=['Missing runner observer']))
            try:
                from raw_runner_observation_v1 import observe_actual_runner
                context = observe_actual_runner(engine)
            except BaseException as error:
                note(context_record, 'runner_observer_import', error)
            with context as runner_observation:
                value = original(*args, **kwargs)  # Original sampling exactly once.
            def capture():
                parameters = kwargs['sampling_params']
                lengths = engine.last_completion_token_lengths
                require(torch.is_tensor(lengths) and lengths.dtype == torch.long,
                        'No actual CompletionOutput length evidence')
                captured.append((clone_cpu(value[0]), clone_cpu(value[1]), clone_cpu(lengths),
                    copy.deepcopy(kwargs['prompt_token_ids']), runner_observation,
                    {key: getattr(parameters, key) for key in ('temperature', 'top_p', 'top_k', 'n', 'logprobs',
                        'max_tokens', 'repetition_penalty', 'presence_penalty', 'frequency_penalty')}))
            attempt(context_record, 'engine_sample_capture', capture)
            return value

        # Setup failure falls back to original generation/recomputation; missing capture is rejected below.
        with patches(context_record) as install:
            install(engine, 'generate', observe)
            with capture_existing_flat_calls(self.actor) as perturn_calls:
                output = super().generate_sequences(prompts)  # Unknown model error: no new collective.
        before_state = None
        try:
            require(not context_record['observer_errors'] and not context_record.get('_observer_recording_failed'), 'Generation observer setup/capture/restore failed')
            require(not observer_failures(perturn_calls), 'Initial per-turn HF observation failed')
            require(len(captured) == 1 and 'old_log_probs' in output.batch, 'Incomplete per-turn sampling evidence')
            sampled_ids, sampled_logp, sampled_lengths, engine_prompts, runner_observation, sampling_parameters = captured[0]
            require(not runner_observation['observer_errors'] and not runner_observation.get('_observer_recording_failed')
                    and not observer_failures(runner_observation['calls']),
                    'Runner/sampler observer failed')
            require(torch.equal(output.batch['response_sampled_lengths'].cpu(), sampled_lengths),
                    'Engine/output real sample lengths differ')
            sampled_mask = clone_cpu(output.batch['response_sample_mask'])
            require(torch.equal(sampled_mask.bool(), torch.arange(sampled_mask.shape[1])[None, :] < sampled_lengths[:, None]),
                    'Actual real sample mask differs')
            require(sampled_ids.shape == sampled_logp.shape, 'Sample IDs/logp shape differs')
            before_state = self._complete_pause_state()
            repeat_input = copy.deepcopy(output)
        except BaseException as error:
            errors.append(repr(error))
        # Fixed boundary AFTER original sampling+old HF, BEFORE any repeated FSDP API.
        matched_stage('generate/after_original_before_repeat', errors)
        with capture_existing_flat_calls(self.actor) as repeated_perturn_calls:
            repeated_output = super().compute_log_prob(repeat_input)  # Exactly one real repeat on every rank.
        errors = []
        try:
            repeated = repeated_output.batch['old_log_probs']
            require(not observer_failures(repeated_perturn_calls), 'Repeated per-turn HF observer failed')
            after_state = self._complete_pause_state()
            rng_unchanged = equal_state(before_state['metadata']['rng'], after_state['metadata']['rng'])
            complete_unchanged = equal_state(before_state, after_state)
            width = sampled_ids.shape[1]
            valid = sampled_mask[:, :width].bool()
            old = output.batch['old_log_probs'][:, :width]
            repeated = repeated[:, :width]
            bad = (~torch.isfinite(old) | ~torch.isfinite(repeated) | ~torch.isfinite(sampled_logp)) & valid
            safe = valid & ~bad
            max_old = float((old.double() - repeated.double())[safe].abs().max()) if bool(safe.any()) else 0.
            max_vllm = float((old.double() - sampled_logp.double())[safe].abs().max()) if bool(safe.any()) else 0.
            phase, serial = self._calibration_phase, self._calibration_call
            self._calibration_call += 1
            directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / phase
            directory.mkdir(parents=True, exist_ok=True)
            raw = dict(phase=phase, serial=serial, rank=self.rank, pid=os.getpid(),
                bf16_accumulation=self._bf16_accumulation_identity(),
                source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                tensors={key: clone_cpu(output.batch[key]) for key in ('prompts', 'responses', 'input_ids',
                    'attention_mask', 'position_ids', 'old_log_probs', 'response_sampled_lengths', 'response_sample_mask')},
                sampled_ids=sampled_ids, sampled_log_probs=sampled_logp, sampled_lengths=sampled_lengths,
                sampled_mask=sampled_mask, perturn_hf_flat_calls=perturn_calls,
                repeated_perturn_hf_flat_calls=repeated_perturn_calls,
                actual_pad_token_id=self.tokenizer.pad_token_id, actual_eos_token_id=self.tokenizer.eos_token_id,
                actor_repeated_log_probs=repeated, valid_comparison_mask=valid,
                actual_engine_prompt_token_ids=engine_prompts, actual_runner_observation=runner_observation,
                actual_sampling_parameters=sampling_parameters, repeated_probe_state_before=before_state,
                repeated_probe_state_after=after_state)
            receipt = atomic_torch_save(raw, directory / f'call-{serial}-rank-{self.rank}.pt')
            record = dict(phase=phase, serial=serial, rank=self.rank, pid=os.getpid(),
                bf16_accumulation=self._bf16_accumulation_identity(),
                source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], comparison_scope='SAME_PREFIX_SAME_TOKENS',
                valid_tokens=int(valid.sum()), nonfinite_effective_count=int(bad.sum()), actor_old_max_abs_delta=max_old,
                actor_vllm_max_abs_delta=max_vllm, rows=len(output), raw=receipt,
                **{key: sampling_parameters[key] for key in ('temperature', 'top_p', 'top_k', 'n')},
                repeated_probability_probe_rng_unchanged=rng_unchanged,
                repeated_probability_probe_complete_state_unchanged=complete_unchanged,
                repeated_probe_identity_before=before_state['identity'], repeated_probe_identity_after=after_state['identity'])
            (directory / f'call-{serial}-rank-{self.rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('generate/after_repeat_local_evidence', errors)
        return output

    def _capture_raw_complete_hf(self, role, data, original, *, before_capture=None, after_capture=None):
        """All local checks/copies occur at fixed stages bracketing the original API."""
        errors, before = [], None
        try:
            self._bf16_accumulation_identity()
            require(data.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1', 'Raw HF capture needs raw source')
            require(data.meta_info.get('raw_calibration_observation') is True, 'Unarmed diagnostic observer')
            actor = self.ref_policy if role == 'reference' else self.actor
            model_config = actor.actor_module.config.to_dict()
            dropout_config = {key: value for key, value in model_config.items() if 'dropout' in key}
            dropout_modules = {name: part.p for name, part in actor.actor_module.named_modules()
                               if isinstance(part, torch.nn.Dropout)}
            require(model_config.get('model_type') == 'qwen2' and dropout_config.get('attention_dropout') == 0.
                    and all(type(value) in (int, float) and value == 0. for value in dropout_config.values())
                    and all(value == 0. for value in dropout_modules.values()),
                    'Actual train/eval model dropout neutrality was not observed')
            from search_r1.baselines.action_prefix_forward import ACTION_PREFIX_KEYS
            keys = list(ACTION_PREFIX_KEYS) + ['responses', 'input_ids', 'attention_mask', 'position_ids']
            keys += [key for key in ('info_mask', 'loss_mask', 'step_ids') if key in data.batch]
            inputs = clone_cpu({key: data.batch[key] for key in keys})
            sources = [int(value) for value in data.non_tensor_batch['audit_trajectory_row']]
            question_ids = clone_cpu(data.non_tensor_batch['audit_question_index'])
            provenance = copy.deepcopy(data.non_tensor_batch['action_prefix_records'].tolist())
            if before_capture is not None:
                before = before_capture()
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('complete_hf/' + role + '/before_original_local', errors)
        with capture_existing_flat_calls(actor) as calls:
            output = original()  # Unknown original model/collective failure does not start a matching reduce.
        errors = []
        try:
            require(not observer_failures(calls), 'Actual HF input/normalizer observation failed')
            name = 'ref_log_prob' if role == 'reference' else 'old_log_probs'
            raw = dict(schema='RAW_ACTION_HF_FORWARD_OBSERVATION_V1', role=role, rank=self.rank, pid=os.getpid(),
                bf16_accumulation=self._bf16_accumulation_identity(),
                code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                actor_token_identity=data.meta_info['actor_token_identity'], actual_pad_token_id=self.tokenizer.pad_token_id,
                actual_eos_token_id=self.tokenizer.eos_token_id, temperature=data.meta_info['temperature'],
                micro_batch_size=data.meta_info['micro_batch_size'], input_tensors=inputs,
                actual_dropout_config=dropout_config, actual_dropout_modules=dropout_modules,
                dropout_neutral_train_eval_verified=True,
                trajectory_sources=sources, trajectory_question_ids=question_ids, action_prefix_records=provenance,
                flat_calls=calls, output_log_probs=clone_cpu(output.batch[name]), observer_errors=[],
                sampler_runner_positions_observed=False,
                model_source=self.config.ref.model_path if role == 'reference' else self.config.model.path)
            directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']).parent / 'probabilities' / 'complete-actions'
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f'{role}-rank-{self.rank}.pt'
            require(not target.exists(), 'Raw HF role observer must not overwrite previous evidence')
            receipt = atomic_torch_save(raw, target)
            (directory / f'{role}-rank-{self.rank}.json').write_text(json.dumps(dict(
                schema=raw['schema'], role=role, rank=self.rank, pid=os.getpid(), code_commit=raw['code_commit'],
                bf16_accumulation=self._bf16_accumulation_identity(),
                binding_sha256=raw['binding_sha256'], raw=receipt, observer_errors=[],
                actual_forward_calls=len(calls), observation_only=True, actual_adam_writes=0), indent=2, allow_nan=False))
            if after_capture is not None:
                after = after_capture()
                neutral = dict(rank=self.rank, pid=os.getpid(), before=before['identity'] if role == 'current' else before,
                    bf16_accumulation=self._bf16_accumulation_identity(),
                    after=after['identity'] if role == 'current' else after, observation_only=True, actual_adam_writes=0)
                neutral['complete_state_unchanged' if role == 'current' else 'fixed_reference_unchanged'] = equal_state(before, after)
                (directory / f'{role}-neutral-rank-{self.rank}.json').write_text(json.dumps(neutral, indent=2, allow_nan=False))
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('complete_hf/' + role + '/after_original_local_evidence', errors)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if data.meta_info.get('raw_calibration_observation') is not True:
            return super().compute_log_prob(data)
        return self._capture_raw_complete_hf('old', data,
            lambda: super(CalibrationWorker, self).compute_log_prob(data))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def calibration_raw_current_log_prob(self, data):
        errors, current = [], None
        try:
            current = copy.deepcopy(data)
            micro = self.actor.config.ppo_micro_batch_size
            require(type(micro) is int and micro == 2, 'Current native probe must use actual training rankmicro2')
            current.meta_info['micro_batch_size'] = micro
        except BaseException as error:
            errors.append(repr(error))
        matched_stage('complete_hf/current/actual_training_micro_metadata', errors)
        return self._capture_raw_complete_hf('current', current,
            lambda: super(CalibrationWorker, self).compute_log_prob(current),
            before_capture=self._complete_pause_state, after_capture=self._complete_pause_state)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        if data.meta_info.get('raw_calibration_observation') is not True:
            return super().compute_ref_log_prob(data)
        return self._capture_raw_complete_hf('reference', data,
            lambda: super(CalibrationWorker, self).compute_ref_log_prob(data),
            before_capture=self.calibration_reference_identity, after_capture=self.calibration_reference_identity)


# Explicit driver name for the same role class and constructor; no extra MRO.
CalibrationActorRolloutRefWorker = CalibrationWorker
