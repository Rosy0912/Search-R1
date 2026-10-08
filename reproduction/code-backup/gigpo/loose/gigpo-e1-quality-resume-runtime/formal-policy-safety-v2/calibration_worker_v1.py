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


class CalibrationWorker(ActorRolloutRefWorker):
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
        captured = []

        def observe(*args, **kwargs):
            value = original(*args, **kwargs)
            parameters = kwargs['sampling_params']
            captured.append((value[0].detach().cpu().clone(), value[1].detach().cpu().clone(),
                copy.deepcopy(kwargs['prompt_token_ids']), {key: getattr(parameters, key) for key in
                    ('temperature', 'top_p', 'top_k', 'n', 'logprobs', 'max_tokens', 'repetition_penalty', 'presence_penalty', 'frequency_penalty')}))
            return value

        engine.generate = observe
        try:
            output = super().generate_sequences(prompts)
        finally:
            engine.generate = original
        assert len(captured) == 1 and 'old_log_probs' in output.batch
        sampled_ids, sampled_logp, engine_prompts, sampling_parameters = captured[0]
        assert sampled_ids.shape == sampled_logp.shape
        # Re-run the real hybrid actor method on exactly the returned prefix/tokens.
        # It retains the same temperature/micro settings and offload path.
        before_state = self._complete_pause_state()
        repeated = super().compute_log_prob(copy.deepcopy(output)).batch['old_log_probs']
        after_state = self._complete_pause_state()
        rng_unchanged = equal_state(before_state['metadata']['rng'], after_state['metadata']['rng'])
        complete_unchanged = equal_state(before_state, after_state)
        # A changed probe is recorded and rejected on the driver. Do not restore
        # RNG (or anything else) to make a state-changing observation look neutral.
        width = sampled_ids.shape[1]
        valid = (output.batch['attention_mask'][:, -output.batch['responses'].shape[1]:][:, :width].bool()
                 & sampled_ids.ne(self.tokenizer.pad_token_id))
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
        raw = dict(phase=phase, serial=serial, rank=self.rank,
                   tensors={key: output.batch[key].detach().cpu().clone() for key in
                            ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs')},
                   sampled_ids=sampled_ids, sampled_log_probs=sampled_logp,
                   actor_repeated_log_probs=repeated, valid_comparison_mask=valid,
                   actual_engine_prompt_token_ids=engine_prompts,
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
