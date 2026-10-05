# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F


__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self, loss_scale=1.):
        assert self.config.grad_clip is not None
        from search_r1.baselines.gradient_norm import clip_grad_norm64
        grad_norm = clip_grad_norm64(self.actor_module, self.config.grad_clip, loss_scale)
        if not torch.isfinite(grad_norm):
            raise ValueError('Nonfinite actor gradient norm; optimizer step refused')
        if self.config.get('policy_safety', {}).get('enabled', False):
            self._last_safety_grad_norm_after = clip_grad_norm64(self.actor_module, float('inf'))
        self.actor_optimizer.step()
        return grad_norm

    def _safety_sync_invalid(self, bad, device, identity, phase, writes, tensors=None, current=None, details=None):
        """All ranks meet this boundary before a local validator may raise."""
        import os, json
        from pathlib import Path
        from search_r1.baselines.recovery import atomic_torch_save, cpu_snapshot
        group = getattr(self.actor_module, 'process_group', None)
        flag = torch.tensor(int(bad), device=device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=group)
        if not bool(flag):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        record = dict(identity, phase=phase, actual_optimizer_writes=writes,
            rank=rank, pid=os.getpid(), local_invalid=bool(bad), details=details,
            run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'), commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
            config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
            data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'))
        error = None
        try:
            root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
            if root:
                folder = Path(root) / 'policy-safety';folder.mkdir(parents=True, exist_ok=True)
                stem = f'numeric-step{identity.get("outer_step")}-mini{identity["mini_index"]}-{phase}-rank{rank}'
                atomic_torch_save(dict(record=record, tensors=cpu_snapshot(tensors), current=cpu_snapshot(current)), folder / (stem+'.pt'))
                (folder / (stem+'.json')).write_text(json.dumps(record, allow_nan=False, sort_keys=True))
        except Exception as exc:
            error = repr(exc)
        io_bad = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(io_bad, op=torch.distributed.ReduceOp.MAX, group=group)
        raise RuntimeError('POLICY_SAFETY_REJECTED: synchronized invalid input/state' + ('; diagnostic IO failed' if bool(io_bad) else ''))

    @torch.no_grad()
    def _policy_safety_guard(self, monitor, temperature, control, identity, phase, writes):
        from search_r1.baselines.policy_safety import local_statistics, global_statistics, rejection_reason
        from search_r1.baselines.loss_scaling import TransparentProbe, coordinated_badflag
        from search_r1.baselines.recovery import atomic_torch_save, cpu_snapshot
        import os, json
        from pathlib import Path
        group = getattr(self.actor_module, 'process_group', None)
        # Match one global decision before FSDP forward, including a locally corrupt shard/Adam.
        device = self.actor_module.compute_device if isinstance(self.actor_module, FSDP) else next(self.actor_module.parameters()).device
        state_bad = False
        values = list(self.actor_module.parameters()) + list(self.actor_module.buffers())
        values += [v for state in self.actor_optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
        for value in values:
            for chunk in value.detach().reshape(-1).split(262144):
                state_bad = state_bad or not bool(torch.isfinite(chunk).all())
        self._safety_sync_invalid(state_bad, device, identity, phase+'_state', writes,
                                  monitor, details='NONFINITE_MODEL_BUFFER_OR_ADAM')
        from search_r1.baselines.runtime_audit import tensor_hash
        monitor_hashes = {key: tensor_hash(value) for key, value in monitor.items()}
        monitor_changed = monitor_hashes != self._safety_monitor_identity
        parts, current_logps = [], []
        probe = TransparentProbe(self.actor_module, getattr(self, 'rollout_sharding_manager', None))
        entered = False; entry_error = exit_error = None
        try:
            try:
                probe.__enter__(); entered = True
            except Exception as exc:
                entry_error = repr(exc)
            self._safety_sync_invalid(entry_error is not None or getattr(probe, 'invalid', True),
                device, identity, phase+'_probe_entry', writes, monitor, details=entry_error)
            with torch.no_grad():
                for micro_index, micro in enumerate(monitor.split(self.config.ppo_micro_batch_size)):
                    schema_error = None
                    try:
                        expected = micro['old_log_probs'].shape
                        if any(micro[key].shape != expected for key in ('advantages','policy_weights','step_ids','responses')):
                            raise ValueError('Monitor action tensor shape changed')
                        if micro['step_ids'].dtype not in (torch.int8,torch.int16,torch.int32,torch.int64):
                            raise ValueError('Monitor action identity dtype changed')
                    except Exception as exc:
                        schema_error = repr(exc)
                    self._safety_sync_invalid(schema_error is not None, device, identity,
                        phase+f'_schema{micro_index}', writes, micro, details=schema_error)
                    micro = micro.cuda()
                    _, current = self._forward_micro_batch(micro, temperature)
                    mask = micro['loss_mask'] if self.config.state_masking else micro['attention_mask'][:, -micro['responses'].shape[1]:]
                    stats_error = None
                    try:
                        part = local_statistics(micro['old_log_probs'], current, micro['advantages'], mask,
                            micro['policy_weights'], micro['step_ids'], control['dual_clip'], control['kl_log_bound'])
                    except Exception as exc:
                        stats_error = repr(exc)
                    self._safety_sync_invalid(stats_error is not None, device, identity,
                        phase+f'_stats{micro_index}', writes, micro, current, details=stats_error)
                    parts.append(part)
                    current_logps.append(current.detach().cpu())
        finally:
            if entered:
                try:
                    probe.__exit__(None, None, None)
                except Exception as exc:
                    exit_error = repr(exc)
        self._safety_sync_invalid(exit_error is not None or probe.invalid,
            device, identity, phase+'_probe_exit', writes, monitor, details=exit_error)
        if monitor_changed:
            parts[0][0][3] += 1
        stats = global_statistics(parts, group)
        reason = rejection_reason(stats, control)
        record = dict(identity, phase=phase, actual_optimizer_writes=writes, allowed=reason is None,
                      rejection_reason=reason, statistics=stats, fixed_monitor_hashes=monitor_hashes,
                      temperature=temperature, monitor_identity_changed=monitor_changed)
        mode = control.get('mode', 'production')
        record.update(execution_mode=mode,
            admission_scope={'production':'production_calibrated','native_fixture':'test_thresholds','calibration':'numeric_only'}[mode],
            dual_clip=control['dual_clip'], kl_log_bound=control['kl_log_bound'],
            target_old_policy_kl=control.get('target_old_policy_kl'), negative_tail_limit=control.get('negative_tail_limit'),
            native_binding_sha256=os.environ.get('GIGPO_POLICY_NATIVE_BINDING_SHA256') if mode != 'production' else None,
            calibration_evidence_sha256=control.get('calibration_evidence_sha256') if mode == 'production' else None)
        record.update(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            pid=os.getpid(), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
            commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
            config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
            data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'))
        if phase == 'after':
            record['gradient_norm_after_clip'] = float(self._last_safety_grad_norm_after)
        # Every rank writes its own actual inputs on rejection. No partial outer is completed.
        error = None
        try:
            root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
            if root:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                folder = Path(root) / 'policy-safety';folder.mkdir(parents=True, exist_ok=True)
                with (folder / f'rank-{rank}.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n');stream.flush()
                if reason:
                    atomic_torch_save(dict(record=record, monitor={k:cpu_snapshot(v) for k,v in monitor.items()},
                                           current_log_probs=current_logps),
                                      folder / f'rejected-step{identity.get("outer_step")}-mini{identity["mini_index"]}-{phase}-rank{rank}.pt')
        except Exception as exc:
            error = repr(exc)
        device = parts[0][0].device
        coordinated_badflag(error is not None, device, group)
        if reason:
            raise RuntimeError('POLICY_SAFETY_REJECTED: ' + json.dumps(record, allow_nan=False, sort_keys=True))
        return stats

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        outer_identity = dict(data.meta_info)
        safety = self.config.get('policy_safety', {})
        safety_enabled = safety.get('enabled', False)
        if safety_enabled:
            from search_r1.baselines.policy_safety import validate_config
            safety_device = self.actor_module.compute_device if isinstance(self.actor_module, FSDP) else next(self.actor_module.parameters()).device
            entry_error = None
            try:
                validate_config(safety)
                if self.config.get('baseline_method') != 'gigpo' or self.config.use_dynamic_bsz or self.config.ppo_epochs != 1:
                    raise ValueError('Policy safety requires the reviewed fixed-micro GiGPO epoch1 path')
            except Exception as exc:
                entry_error = repr(exc)
            self._safety_sync_invalid(entry_error is not None, safety_device,
                dict(outer_step=outer_identity.get('baseline_outer_step'), mini_index=-1),
                'entry_config', 0, details=entry_error)

        preparation_error = None
        try:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
            temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
            if self.config.state_masking:
                select_keys.append('loss_mask')
            if self.config.use_kl_loss:
                select_keys.append('ref_log_prob')
            paper_objective = 'policy_weights' in data.batch.keys()
            if paper_objective:
                select_keys.append('policy_weights')
                if self.config.get('baseline_method') == 'gigpo':
                    select_keys.append('step_ids')
            batch = data.select(batch_keys=select_keys).batch

            audit = None
            if self.config.get('baseline_method') in ('grpo', 'gigpo'):
                from search_r1.baselines.runtime_audit import enabled, ActorAudit
                if enabled():
                    audit = ActorAudit(self.actor_module, batch, data.meta_info, self.config.ppo_epochs)

            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            dataloader = batch.split(self.config.ppo_mini_batch_size)
            monitor = None
            if safety_enabled:
                monitor_error = None
                try:
                    monitor = dataloader[0].clone()
                    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                    if len(monitor) * world != safety['monitor_first_global_rows'] or len(dataloader) != 10:
                        raise ValueError('Fixed monitor global256/ten-minibatch coverage mismatch')
                    from search_r1.baselines.runtime_audit import tensor_hash
                    self._safety_monitor_identity = {key: tensor_hash(value) for key, value in monitor.items()}
                except Exception as exc:
                    monitor_error = repr(exc)
                if monitor_error:
                    raise ValueError(monitor_error)

        except Exception as exc:
            if not safety_enabled:
                raise
            preparation_error = repr(exc)
        if safety_enabled:
            self._safety_sync_invalid(preparation_error is not None, safety_device,
                dict(outer_step=outer_identity.get('baseline_outer_step'), mini_index=-1),
                'entry_preparation', 0, details=preparation_error)

        metrics = {}
        updates = skipped = 0
        # Fixed rollout advantages/logprobs are reused across genuine inner epochs.
        # Default E=1 preserves the reviewed reproduction's update schedule.
        for batch_idx, data in enumerate(itertools.chain.from_iterable(
                dataloader for _ in range(self.config.ppo_epochs))):
            if audit is not None and batch_idx % len(dataloader) == 0:
                audit.epoch()
            # split batch into micro_batches
            mini_batch = data
            from search_r1.baselines.distillation import global_token_count
            count_mask = mini_batch['loss_mask'] if self.config.state_masking else mini_batch['attention_mask'][:, -mini_batch['responses'].shape[1]:]
            total, world = global_token_count(count_mask)
            if total.item() == 0:
                if self.config.get('complete_internal_state', False):
                    raise RuntimeError('Fresh Instruct update has an empty global minibatch; cannot skip optimizer budget')
                skipped += 1
                continue
            if self.config.get('baseline_method') in ('gigpo', 'tree_grpo', 'topd', 'cosearch') and not paper_objective:
                raise ValueError('Paper actor objective requires driver-computed policy_weights')
            topd_token_mean = self.config.get('baseline_method') == 'topd'
            if paper_objective:
                from search_r1.baselines.objectives import row_denominator, weighted_policy_loss
                denominator = row_denominator(len(mini_batch), mini_batch.device)
                if audit is not None:
                    audit.row_denominators.append(dict(mini_index=batch_idx, local_rows=len(mini_batch),
                        actual_denominator=float(denominator.detach().item()), world_size=world))
                weight_mass, mass_world = global_token_count(mini_batch['policy_weights'])
                metric_mass = (weight_mass / mass_world).clamp_min(torch.finfo(weight_mass.dtype).tiny)
            elif topd_token_mean:
                from search_r1.baselines.advantages import token_mean_ppo
                denominator = total / world
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            if safety_enabled:
                self._policy_safety_guard(monitor, temperature, safety,
                    dict(outer_step=outer_identity.get('baseline_outer_step'), mini_index=batch_idx), 'before', updates)
            self.actor_optimizer.zero_grad()
            mini_metrics = {}

            loss_scale = 1.
            probe_logps = []
            scaled_joint = self.config.get('joint_loss_scaling', False) and not safety_enabled
            if scaled_joint:
                if not paper_objective or self.config.get('baseline_method') != 'gigpo':
                    raise ValueError('Joint loss scaling is only defined for the GiGPO paper objective')
                from search_r1.baselines.loss_scaling import (TransparentProbe, active_joint_bound,
                                                            coordinated_scale, coordinated_badflag,
                                                            describe_invalid_joint_probe, write_probe_diagnostic)
                group = getattr(self.actor_module, 'process_group', None)
                bound = torch.full((), -float('inf'), dtype=torch.float64, device='cuda')
                bad = torch.zeros((), dtype=torch.int32, device='cuda')
                probe_details = []
                probe_identity = dict(outer_step=outer_identity.get('baseline_outer_step'),
                                      mini_index=batch_idx, optimizer_steps_completed=batch_idx)
                with TransparentProbe(self.actor_module, getattr(self, 'rollout_sharding_manager', None)) as probe:
                    try:
                        coordinated_badflag(probe.invalid, bound.device, group)
                    except ValueError:
                        write_probe_diagnostic(dict(probe_identity, phase='boundary_enter_rejected',
                                                    local_invalid=bool(probe.invalid)))
                        raise
                    with torch.no_grad():
                        for probe_index, probe_data in enumerate(micro_batches):
                            probe_data = probe_data.cuda()
                            _, probe_logp = self._forward_micro_batch(probe_data, temperature)
                            width = probe_data['responses'].shape[1]
                            probe_mask = (probe_data['loss_mask'] if self.config.state_masking else
                                          probe_data['attention_mask'][:, -width:])
                            value, invalid = active_joint_bound(probe_data['old_log_probs'], probe_logp,
                                probe_data['advantages'], probe_mask, probe_data['policy_weights'],
                                probe_data['step_ids'], self.config.clip_ratio)
                            if bool(invalid):
                                try:
                                    detail = describe_invalid_joint_probe(probe_data['old_log_probs'], probe_logp,
                                        probe_data['advantages'], probe_mask, probe_data['policy_weights'],
                                        probe_data['step_ids'], self.config.clip_ratio)
                                except Exception as exc:
                                    detail = dict(diagnostic_error=repr(exc))
                                probe_details.append(dict(micro_index=probe_index, details=detail))
                            bound = torch.maximum(bound, value)
                            bad = torch.maximum(bad, invalid)
                            probe_logps.append(probe_logp.detach().cpu())
                bad = torch.maximum(bad, torch.tensor(int(probe.invalid), dtype=torch.int32, device=bad.device))
                # Existing global badflag/MAX collectives decide validity unchanged.
                # Write separate metadata only after every rank reaches that decision.
                try:
                    loss_scale = coordinated_scale(bound, bad, group)
                except ValueError as exc:
                    write_probe_diagnostic(dict(probe_identity, phase='rejected',
                        local_bad=bool(bad), boundary_exit_invalid=bool(probe.invalid),
                        local_bound=str(float(bound)), details=probe_details, error=str(exc)))
                    raise
                write_probe_diagnostic(dict(probe_identity, phase='accepted',
                    local_bad=bool(bad), boundary_exit_invalid=bool(probe.invalid),
                    local_bound=str(float(bound)), global_loss_scale=float(loss_scale)))
                mini_metrics['actor/joint_loss_scale'] = float(loss_scale)
                mini_metrics['actor/probe_micro_batches'] = len(probe_logps)

            for micro_index, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                if scaled_joint:
                    # Synchronize before the loss helpers can raise locally.
                    value, invalid = active_joint_bound(old_log_prob, log_prob, advantages,
                        response_mask, data['policy_weights'], data['step_ids'], clip_ratio)
                    mismatch = not torch.equal(log_prob.detach().cpu(), probe_logps[micro_index])
                    if self.config.use_kl_loss:
                        invalid = invalid | (~torch.isfinite(data['ref_log_prob'][response_mask.bool()]).all()).to(torch.int32)
                    coordinated_badflag(bool(invalid) or mismatch, log_prob.device, group)

                if safety_enabled:
                    from search_r1.baselines.policy_safety import local_statistics
                    from search_r1.baselines.loss_scaling import coordinated_badflag
                    numeric_error = None
                    try:
                        probe_stats = local_statistics(old_log_prob, log_prob, advantages, response_mask,
                            data['policy_weights'], data['step_ids'], safety['dual_clip'], safety['kl_log_bound'])
                        invalid = bool(probe_stats[0][3])
                        if self.config.use_kl_loss:
                            valid = response_mask.bool()
                            ref_delta = data['ref_log_prob'][valid].double() - log_prob[valid].double()
                            invalid = invalid or not bool(torch.isfinite(ref_delta).all())
                    except Exception as exc:
                        invalid = True;numeric_error = repr(exc)
                    self._safety_sync_invalid(invalid, log_prob.device,
                        dict(outer_step=outer_identity.get('baseline_outer_step'), mini_index=batch_idx),
                        f'gradient_micro{micro_index}', updates, data, log_prob.detach(), details=numeric_error)

                if paper_objective:
                    pg_loss, pg_clipfrac, ppo_kl = weighted_policy_loss(
                        old_log_prob, log_prob, advantages, response_mask, data['policy_weights'],
                        clip_ratio, denominator, data.get('step_ids', None),
                        dual_clip=safety['dual_clip'] if safety_enabled else None)
                elif topd_token_mean:
                    pg_loss, pg_clipfrac, ppo_kl = token_mean_ppo(
                        old_log_prob, log_prob, advantages, response_mask, clip_ratio, denominator)
                else:
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                  log_prob=log_prob,
                                                                                  advantages=advantages,
                                                                                  eos_mask=response_mask,
                                                                                  cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = (verl_F.masked_sum(entropy, data['policy_weights']) / denominator
                                if paper_objective else verl_F.masked_mean(entropy, response_mask))

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=torch.where(response_mask.bool(), log_prob.double() if safety_enabled else log_prob, 0.),
                                                ref_logprob=torch.where(response_mask.bool(), ref_log_prob.double() if safety_enabled else ref_log_prob, 0.),
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = (verl_F.masked_sum(kld, data['policy_weights']) / denominator
                               if paper_objective else masked_mean(kld, response_mask))

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    if paper_objective:
                        mini_metrics['actor/kl_loss'] = mini_metrics.get('actor/kl_loss', 0.) + kl_loss.detach().item()
                    else:
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss if paper_objective or topd_token_mean else policy_loss / self.gradient_accumulation
                if audit is not None:
                    parts = dict(policy=pg_loss, entropy=entropy_loss, total=loss)
                    if self.config.use_kl_loss:
                        parts['kl'] = kl_loss
                    audit.micro(log_prob, old_log_prob, response_mask, loss, data.get('step_ids', None),
                        advantages=advantages, loss_parts=parts,
                        ref_logp=ref_log_prob if self.config.use_kl_loss else None,
                        dual_clip=safety['dual_clip'] if safety_enabled else None)
                if scaled_joint or safety_enabled:
                    from search_r1.baselines.loss_scaling import coordinated_badflag
                    coordinated_badflag(not bool(torch.isfinite(loss)), log_prob.device,
                                        getattr(self.actor_module, 'process_group', None))
                (loss * loss_scale if scaled_joint else loss).backward()

                micro_mass = data['policy_weights'].sum().detach().item() if paper_objective else 0.
                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                if paper_objective:
                    for key, value in data.items():
                        if key in ('actor/pg_clipfrac', 'actor/ppo_kl'):
                            value *= micro_mass / metric_mass.item()
                        mini_metrics[key] = mini_metrics.get(key, 0.) + value
                else:
                    append_to_dict(metrics, data)

            if paper_objective:
                append_to_dict(metrics, mini_metrics)
            try:
                grad_norm = self._optimizer_step(loss_scale) if scaled_joint else self._optimizer_step()
            except BaseException:
                if scaled_joint:
                    import os
                    from pathlib import Path
                    from search_r1.baselines.recovery import atomic_torch_save, cpu_snapshot
                    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
                    if root:
                        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                        atomic_torch_save(dict(outer_identity=outer_identity, mini_index=batch_idx,
                            loss_scale=float(loss_scale), active_logratio_bound=float(bound),
                            mini_batch={key: cpu_snapshot(value) for key, value in mini_batch.items()},
                            real_log_probs=probe_logps, failed_before_adam=True),
                            Path(root) / 'numeric' / f'failed-mini-rank-{rank}.pt')
                raise
            if audit is not None:
                audit.gradient(grad_norm)
            updates += 1
            if safety_enabled:
                self._policy_safety_guard(monitor, temperature, safety,
                    dict(outer_step=outer_identity.get('baseline_outer_step'), mini_index=batch_idx), 'after', updates)
            data = {'actor/grad_norm': grad_norm.detach().item()}
            if safety_enabled:
                data['actor/grad_norm_after_clip'] = float(self._last_safety_grad_norm_after)
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if audit is not None:
            audit.finish(updates, skipped)
        metrics['actor/optimizer_steps'] = updates
        metrics['actor/skipped_empty_minibatches'] = skipped
        if paper_objective:
            from search_r1.baselines.objectives import reduce_paper_metrics
            reduce_paper_metrics(metrics, batch.device)
        return metrics
