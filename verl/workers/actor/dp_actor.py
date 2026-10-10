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
import json
import os
from typing import Iterable, Optional, Tuple

import torch
from search_g1.compute_precision import ComputePrecision
from search_g1.numerics import require_finite
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_of_labels_v2, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

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
        # Fresh workers must expose zero writes before any guard can skip Adam.
        self._optimizer_writes = 0
        self.compute_precision = ComputePrecision.from_config(self.config)
        self.compute_precision.validate_module(self.actor_module,
                                               process_group=getattr(self.actor_module, 'process_group', None))
        self.compute_precision.configure_worker()
        self.last_forward_precision = None
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature,
                             compute_entropy=True) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len), or None when not requested
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        precision_group = getattr(self.actor_module, 'process_group', None)
        with self.compute_precision.autocast(process_group=precision_group):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                response_positions = indices.remainder(seqlen)
                response_positions = ((response_positions >= seqlen - response_length - 1) &
                                      (response_positions < seqlen - 1)).unsqueeze(0)

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
                    response_positions, _, _ = ulysses_pad_and_slice_inputs(response_positions, None,
                                                                               self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                response_positions = response_positions.squeeze(0)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                self.last_forward_precision = self.compute_precision.observe_logits(output.logits,
                                                                                    process_group=precision_group)
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                # Only response prediction rows need a FP32 vocabulary reduction.
                response_logits = logits_rmpad[response_positions].float()
                response_logits.div_(temperature)
                response_log_probs = logprobs_of_labels_v2(response_logits,
                                                          input_ids_rmpad_rolled[response_positions])
                log_probs = response_log_probs.new_zeros(logits_rmpad.size(0)).masked_scatter(
                    response_positions, response_log_probs)
                if compute_entropy:
                    response_entropy = self.compute_entropy_from_logits(response_logits)
                    entropy_rmpad = response_entropy.new_zeros(logits_rmpad.size(0)).masked_scatter(
                        response_positions, response_entropy)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if compute_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                              gather_dim=0,
                                                              unpad_dim=0,
                                                              padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if compute_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = (full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                           if compute_entropy else None)  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                self.last_forward_precision = self.compute_precision.observe_logits(output.logits,
                                                                                    process_group=precision_group)
                logits = output.logits[:, -response_length - 1:-1].float()  # (bsz, response_length, vocab)
                logits.div_(temperature)
                log_probs = logprobs_of_labels_v2(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits) if compute_entropy else None

            if (os.environ.get('SEARCH_G1_PROFILE') == 'common8'
                    and self.compute_precision.compute_dtype == 'bfloat16'
                    and not getattr(self, '_actual_precision_logged', False)):
                observation = dict(self.last_forward_precision,
                    role='actor' if self.actor_optimizer is not None else 'reference',
                    rank=torch.distributed.get_rank(),
                    master_parameter_dtypes=sorted({str(p.dtype) for p in self.actor_module.parameters()}),
                    selected_log_prob_dtype=str(log_probs.dtype),
                    actual_forward=True, optimizer_writes=getattr(self, '_optimizer_writes', 0))
                print('SEARCH_G1_ACTUAL_POLICY_PRECISION=' + json.dumps(observation, sort_keys=True), flush=True)
                self._actual_precision_logged = True
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        group = self.actor_module.process_group if isinstance(self.actor_module, FSDP) else None
        gradients = {f'gradient_{i}':p.grad for i,p in enumerate(self.actor_module.parameters()) if p.grad is not None}
        recoverable = getattr(self, '_recoverable_guard_mode', False)
        require_finite(gradients,stage='before_gradient_clipping',process_group=group,recoverable=recoverable)
        from search_g1.gradient_clipping import clip_global_l2
        grad_norm = clip_global_l2(self.actor_module,self.config.grad_clip,
            context=getattr(self,"_gradient_diagnostic_context",None),recoverable=recoverable)
        require_finite({"grad_norm":grad_norm},stage="before_optimizer_step",process_group=group,recoverable=recoverable)
        self.actor_optimizer.step()
        self._optimizer_writes=getattr(self,"_optimizer_writes",0)+1
        if __import__('os').environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.worker_evidence import append
            append('OPTIMIZER_WRITE', dict(rank=torch.distributed.get_rank(),
                context=self._gradient_diagnostic_context, write_count=self._optimizer_writes,
                gradient_norm_before=float(grad_norm),gradient_norm_after=float(self.actor_module._last_clipped_global_l2),
                learning_rates=[float(part['lr']) for part in self.actor_optimizer.param_groups]),
                collective=recoverable,group=group)
        if (__import__('os').environ.get('SEARCH_G1_PROFILE') == 'common8' or
                not torch.distributed.is_initialized() or torch.distributed.get_rank()==0):
            print("SEARCH_G1_OPTIMIZER_WRITE="+__import__("json").dumps(dict(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,context=getattr(self,"_gradient_diagnostic_context",None),write_count=self._optimizer_writes)),flush=True)
        if __import__('os').environ.get('SEARCH_G1_PROFILE') == 'common8':
            post = {f'parameter_{i}': p.detach() for i,p in enumerate(self.actor_module.parameters())}
            for i,state in enumerate(self.actor_optimizer.state.values()):
                for name,value in state.items():
                    if torch.is_tensor(value):
                        post[f'optimizer_{i}_{name}'] = value
            require_finite(post, stage='after_optimizer_write', process_group=group)
        return grad_norm

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
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature,
                                                        compute_entropy=False)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def _monitor_old_policy(self, frozen, safety, tail_limit, stage):
        from search_g1.policy_safety import monitor
        from reproduction.common8.worker_evidence import append
        from reproduction.common8.topology import monitor_scope
        group = self.actor_module.process_group if isinstance(self.actor_module, FSDP) else None
        import hashlib
        current_hashes={name:hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name,value in frozen.items()}
        device=torch.device('cuda',torch.cuda.current_device()) if torch.distributed.is_initialized() and torch.distributed.get_backend(group)=='nccl' else torch.device('cpu')
        require_finite({'frozen_monitor_identity':torch.tensor(int(current_hashes==self._monitor_identity['tensor_sha256']),device=device)},
                       stage='frozen_monitor_identity',process_group=group,positive=('frozen_monitor_identity',))
        old_mode = self.actor_module.training
        self.actor_module.eval()
        values = []; masks = []; advantages = []; weights = []
        try:
            with torch.no_grad():
                for micro in frozen.split(self.config.ppo_micro_batch_size):
                    micro = micro.cuda(device=torch.cuda.current_device())
                    _, new = self._forward_micro_batch(micro, temperature=self._monitor_temperature,
                                                      compute_entropy=False)
                    mask = micro['loss_mask'] if self.config.state_masking else micro['attention_mask'][:, -new.shape[1]:]
                    valid = mask.bool()
                    delta = torch.where(valid,new,0.).double() - torch.where(valid,micro['old_log_probs'],0.).double()
                    values.append(delta);masks.append(mask);advantages.append(micro['advantages'])
                    weights.append(mask.double()/mask.sum().clamp_min(1).double())
        finally:
            self.actor_module.train(old_mode)
        report = monitor(torch.cat(values),torch.cat(advantages),torch.cat(masks),torch.cat(weights),
                         c=float(safety.dual_clip),kl_log_bound=float(safety.kl_log_bound),
                         target_kl=float(safety.target_old_policy_kl),negative_tail_limit=tail_limit,group=group,
                         execution_role=self._policy_execution_role)
        report.update(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                      stage=stage,context=self._gradient_diagnostic_context,
                      optimizer_writes=getattr(self,'_optimizer_writes',0),
                      calibration_sha256=self._safety_calibration_sha,
                      monitor_identity=self._monitor_identity,
                      objective_weight_definition='equal two-row micro masked_mean; weighted global numerator/denominator',
                      monitoring_scope=monitor_scope(),
                      rejection_count=getattr(self,'_policy_rejections',0)+int(not report['allowed']))
        append('POLICY_MONITOR',report,collective=getattr(self,'_recoverable_guard_mode',False),group=group)
        if not report['allowed']:
            self._policy_rejections=report['rejection_count']
            self.actor_optimizer.zero_grad()
            if getattr(self,'_recoverable_guard_mode',False):
                from search_g1.policy_safety import RecoverableGuard
                raise RecoverableGuard(stage=stage,reasons=report['reasons'],diagnostic=report)
            raise RuntimeError('Old-rollout reuse stopped collectively; already-written steps retained: '+repr(report['reasons']))
        return report

    def update_policy(self, data: DataProto):
        """Finish each formal attempt with a typed all-rank disposition."""
        from search_g1.policy_safety import (RecoverableGuard, recoverable_mode,
            bind_actual_config, agree_attempt_result, _phase_agreement)
        group = self.actor_module.process_group if isinstance(self.actor_module,FSDP) else None
        bind_actual_config(self.config,group=group)
        self._recoverable_guard_mode = recoverable_mode(self.config,group=group)
        self._last_attempt_result = None
        if not self._recoverable_guard_mode:
            return self._update_policy_attempt(data)
        import re
        attempt_id = data.meta_info.get('attempt_id')
        attempt_index = data.meta_info.get('attempt_index')
        logical_update = data.meta_info.get('logical_update')
        writes_before = getattr(self,'_optimizer_writes',0)
        errors = []
        if not (isinstance(attempt_id,str) and re.fullmatch('[0-9a-f]{32}',attempt_id) and
                type(attempt_index) is int and attempt_index==getattr(self,'_last_attempt_index',0)+1 and
                attempt_id not in getattr(self,'_used_attempt_ids',set()) and
                type(logical_update) is int and logical_update==getattr(self,'_complete_outers',0)+1):
            errors.append('Nonsequential or reused formal attempt identity')
        _phase_agreement(dict(attempt_id=attempt_id,attempt_index=attempt_index,
                             logical_update=logical_update,writes_before=writes_before),errors,group)
        self._last_attempt_index = attempt_index
        self._used_attempt_ids = getattr(self,'_used_attempt_ids',set()) | {attempt_id}
        metrics = self._update_policy_attempt(data)
        writes_after = getattr(self,'_optimizer_writes',0)
        processed = len(self._last_opportunities)
        complete = processed==20 and writes_after>writes_before
        result = dict(status='complete' if complete else 'recoverable_guard',
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            logical_update=logical_update,attempt_id=attempt_id,attempt_index=attempt_index,
            optimizer_writes_before=writes_before,optimizer_writes_after=writes_after,
            optimizer_writes_this_attempt=writes_after-writes_before,
            opportunities_processed=processed,
            opportunity_statuses=[row['status'] for row in self._last_opportunities],
            guard_stage=None if complete else 'all_update_opportunities_skipped',
            guard_reasons=[] if complete else ['no_successful_adam_write'],
            complete_outer=complete,retained_partial_adam_writes=not complete,
            no_automatic_rollback=True)
        agree_attempt_result(result,group=group)
        if complete:
            assert 1<=result['optimizer_writes_this_attempt']<=20
            self._complete_outers = logical_update
        else:
            assert 0<=result['optimizer_writes_this_attempt']<=20
            from reproduction.common8.worker_evidence import append
            append('ABORTED_ATTEMPT',dict(rank=result['rank'],logical_update=logical_update,
                attempt_id=attempt_id,attempt_index=attempt_index,
                context=getattr(self,'_gradient_diagnostic_context',None),
                behavior_probability_gate_sha256=data.meta_info['behavior_probability_gate']['sha256'],
                optimizer_writes=result['optimizer_writes_this_attempt'],
                cumulative_writes=writes_after,writes_before=writes_before,
                guard_stage=result['guard_stage'],guard_reasons=result['guard_reasons'],
                opportunities_processed=processed,
                recoverable_guard=True,complete_outer=False,scheduler_advances=0,
                retained_partial_adam_writes=True,no_automatic_rollback=True),collective=True,group=group)
        self._last_attempt_result = result
        return metrics

    def _update_policy_attempt(self, data: DataProto):
        from search_g1.policy_safety import bind_actual_config, RecoverableGuard
        group = self.actor_module.process_group if isinstance(self.actor_module,FSDP) else None
        bind_actual_config(self.config,group=group)
        # Collective finite checks require matching microbatch counts on all ranks.
        assert not self.config.use_dynamic_bsz, 'Search-G1 finite-update protocol requires fixed microbatch scheduling'
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        logical_update=data.meta_info.get('logical_update')
        writes_before = getattr(self, '_optimizer_writes', 0)
        common8 = __import__('os').environ.get('SEARCH_G1_PROFILE') == 'common8'
        if common8:
            from reproduction.common8.behavior_runtime import verify_worker_input
            from reproduction.common8.topology import actor_local_mini, actor_local_micro, local_trajectories, world_size
            verify_worker_input(data, logical_update=logical_update, group=group)
            batch_gate_sha = data.meta_info['behavior_probability_gate']['sha256']
            assert torch.distributed.get_world_size() == world_size()
            assert len(data) == local_trajectories() and self.config.ppo_mini_batch_size == actor_local_mini()
            assert self.config.ppo_micro_batch_size == actor_local_micro() and self.config.ppo_epochs == 1
        temperature = data.meta_info['temperature']


        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        safety = self.config.get('policy_safety', None)
        safety_enabled = safety is not None and safety.enabled
        if safety_enabled:
            from search_g1.policy_safety import bind_policy_phase
            group = self.actor_module.process_group if isinstance(self.actor_module,FSDP) else None
            tail_limit,self._safety_calibration_sha,self._policy_execution_role=bind_policy_phase(
                self.config,self.actor_optimizer,logical_update=logical_update,
                optimizer_writes=writes_before,group=group)
            self._phase_initial_adam_state_entries=len(self.actor_optimizer.state)
            if getattr(self,'_recoverable_guard_mode',False) and not getattr(self,'_inline_initial_optimizer_published',False):
                assert writes_before==0 and logical_update==1
                from reproduction.common8.formal_inline_safety import publish_optimizer_state
                publish_optimizer_state(self.actor_module,self.actor_optimizer,logical_update=logical_update,
                    writes=writes_before,group=group,complete=False,
                    attempt_id=data.meta_info['attempt_id'],attempt_index=data.meta_info['attempt_index'])
                self._inline_initial_optimizer_published=True
            from reproduction.common8.topology import monitor_trajectories_per_rank
            monitor_count=monitor_trajectories_per_rank()
            assert common8 and safety.monitor_trajectories_per_rank == monitor_count
            # Freeze actual old/advantage/mask/input bytes once; select length strata.
            length_mask=batch['loss_mask'] if self.config.state_masking else batch['attention_mask'][:,-batch['responses'].shape[1]:]
            ordered=torch.argsort(length_mask.sum(-1),stable=True)
            indices=ordered[torch.linspace(0,len(batch)-1,monitor_count,device=ordered.device).long()]
            require_finite({'monitor_actor_lengths':length_mask[indices].sum(-1)},
                           stage='frozen_monitor_selection_before_any_minibatch',
                           process_group=group,positive=('monitor_actor_lengths',))
            frozen=batch[indices].clone()
            self._monitor_temperature=temperature
            import hashlib
            self._monitor_identity=dict(selection_indices=indices.detach().cpu().tolist(),
                actor_token_lengths=length_mask[indices].sum(-1).detach().cpu().tolist(),
                tensor_sha256={name:hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                               for name,value in frozen.items()})
        metrics = {}
        attempt_meta = {key:data.meta_info[key] for key in ('attempt_id','attempt_index')} if getattr(self,'_recoverable_guard_mode',False) else {}
        self._last_opportunities = []
        for batch_idx, mini_batch in enumerate(dataloader):
            self._gradient_diagnostic_context=dict(logical_update=logical_update,minibatch_index=batch_idx,prior_optimizer_writes=getattr(self,"_optimizer_writes",0))
            if getattr(self,'_recoverable_guard_mode',False):
                self._gradient_diagnostic_context.update(**attempt_meta)
            if common8:
                self._gradient_diagnostic_context['behavior_probability_gate_sha256'] = batch_gate_sha
            before_opportunity = getattr(self,'_optimizer_writes',0)
            opportunity_metrics = {}; guard = None
            try:
                self._run_minibatch(mini_batch,temperature=temperature,safety_enabled=safety_enabled,
                    safety=safety,frozen=frozen if safety_enabled else None,
                    tail_limit=tail_limit if safety_enabled else None,metrics=opportunity_metrics)
            except RecoverableGuard as rejection:
                if not getattr(self,'_recoverable_guard_mode',False):
                    raise
                guard = rejection
                self.actor_optimizer.zero_grad()
            after_opportunity = getattr(self,'_optimizer_writes',0)
            wrote = after_opportunity==before_opportunity+1
            assert after_opportunity-before_opportunity in (0,1)
            guard_events = [dict(stage=guard.stage,reasons=guard.reasons)] if guard is not None else []
            if guard is not None and not wrote:
                # Every skipped opportunity still gets a real post monitor of
                # the retained parameters, never a substituted or reused report.
                try:
                    self._monitor_old_policy(frozen,safety,tail_limit,'after_minibatch')
                except RecoverableGuard as post_guard:
                    guard_events.append(dict(stage=post_guard.stage,reasons=post_guard.reasons))
            if wrote:
                for key,value in opportunity_metrics.items():
                    if isinstance(value,list):
                        metrics.setdefault(key,[]).extend(value)
                    else:
                        metrics[key]=value
            if getattr(self,'_recoverable_guard_mode',False):
                row = dict(rank=torch.distributed.get_rank(),context=dict(self._gradient_diagnostic_context),
                    opportunity_index=batch_idx,processed=True,
                    status='written_with_post_guard' if wrote and guard is not None else 'written' if wrote else 'skipped_guard',
                    optimizer_write_succeeded=wrote,optimizer_writes_before=before_opportunity,
                    optimizer_writes_after=after_opportunity,
                    guard_stage=guard.stage if guard is not None else None,
                    guard_reasons=guard.reasons if guard is not None else [],guard_events=guard_events)
                from search_g1.policy_safety import _phase_agreement
                _phase_agreement({key:value for key,value in row.items() if key!='rank'},[],group)
                from reproduction.common8.worker_evidence import append
                append('OPPORTUNITY',row,collective=True,group=group)
                self._last_opportunities.append(row)
        self.actor_optimizer.zero_grad()
        if common8:
            writes = self._optimizer_writes - writes_before
            if getattr(self,'_recoverable_guard_mode',False):
                assert len(self._last_opportunities)==20 and 0<=writes<=20
                if writes==0:
                    return {}  # Fully skipped attempt has no completed metrics/budget/proof.
                metrics['actor/update_opportunities_this_attempt']=20
                metrics['actor/skipped_update_opportunities']=20-writes
            else:
                assert writes == 20
            if safety_enabled and self._policy_execution_role != 'production':
                from search_g1.policy_safety import publish_calibration_optimizer_proof
                publish_calibration_optimizer_proof(self.actor_module,self.actor_optimizer,
                    execution_role=self._policy_execution_role,plan_sha256=self._safety_calibration_sha,
                    initial_writes=writes_before,initial_state_entries=self._phase_initial_adam_state_entries,
                    final_writes=self._optimizer_writes,group=group)
            append_to_dict(metrics, {'actor/optimizer_steps_this_update': writes})
            if getattr(self,'_recoverable_guard_mode',False):
                from reproduction.common8.formal_inline_safety import publish_optimizer_state
                publish_optimizer_state(self.actor_module,self.actor_optimizer,logical_update=logical_update,
                    writes=self._optimizer_writes,group=group,complete=True,
                    **attempt_meta)
            from reproduction.common8.worker_evidence import append
            budget = dict(rank=torch.distributed.get_rank(),
                logical_update=logical_update, local_trajectories=len(batch),
                local_mini=self.config.ppo_mini_batch_size, local_micro=self.config.ppo_micro_batch_size, optimizer_writes=writes,
                cumulative_writes=self._optimizer_writes)
            if getattr(self,'_recoverable_guard_mode',False):
                budget.update(**attempt_meta,opportunities_processed=20,
                              optimizer_writes_before=writes_before,optimizer_writes_after=self._optimizer_writes)
            append('OUTER_BUDGET',budget,collective=getattr(self,'_recoverable_guard_mode',False),group=group)
            print('SEARCH_G1_OUTER_BUDGET=' + __import__('json').dumps(budget), flush=True)
        return metrics

    def _run_minibatch(self, mini_batch, *, temperature, safety_enabled, safety, frozen, tail_limit, metrics):
        """One update opportunity; caller records its disposition on every rank."""
        group = self.actor_module.process_group if isinstance(self.actor_module,FSDP) else None
        recoverable = getattr(self,'_recoverable_guard_mode',False)
        if safety_enabled:
            self._monitor_old_policy(frozen,safety,tail_limit,'before_minibatch')
        if self.config.use_dynamic_bsz:
            micro_batches,_ = rearrange_micro_batches(batch=mini_batch,
                max_token_len=self.config.ppo_max_token_len_per_gpu*self.ulysses_sequence_parallel_size)
        else:
            micro_batches=mini_batch.split(self.config.ppo_micro_batch_size)
        self.actor_optimizer.zero_grad()
        for micro in micro_batches:
            micro=micro.cuda(device=torch.cuda.current_device())
            width=micro['responses'].size(1)
            mask=micro['loss_mask'] if self.config.state_masking else micro['attention_mask'][:,-width:]
            old=micro['old_log_probs'];advantages=micro['advantages']
            entropy,new=self._forward_micro_batch(micro_batch=micro,temperature=temperature)
            authored=mask.bool()
            if safety_enabled:
                require_finite(dict(new=torch.where(authored,new,0.),old=torch.where(authored,old,0.),
                    advantage=torch.where(authored,advantages,0.),mass=mask.sum()),
                    stage='before_dual_clip_exp',process_group=group,positive=('mass',),recoverable=recoverable)
            pg_loss,clipfrac,ppo_kl=core_algos.compute_policy_loss(old_log_prob=old,log_prob=new,
                advantages=advantages,eos_mask=mask,cliprange=self.config.clip_ratio,
                dual_clip=float(safety.dual_clip) if safety_enabled else None)
            entropy_loss=verl_F.masked_mean(entropy,mask)
            policy_loss=pg_loss-entropy_loss*self.config.entropy_coeff
            numerical_fields=dict(pg_loss=pg_loss,entropy_loss=entropy_loss,ppo_kl=ppo_kl,
                response_tokens=mask.sum(),authored_log_prob=torch.where(authored,new,0.),
                authored_old_log_prob=torch.where(authored,old,0.),
                authored_advantages=torch.where(authored,advantages,0.))
            if self.config.use_kl_loss:
                masked_new=torch.where(authored,new,torch.zeros_like(new))
                masked_ref=torch.where(authored,micro['ref_log_prob'],torch.zeros_like(micro['ref_log_prob']))
                # Zero reference coefficient does not evaluate an unnecessary exponential.
                if self.config.kl_loss_coef==0:
                    kl_loss=new.new_zeros(())
                else:
                    kl_loss=masked_mean(core_algos.kl_penalty(logprob=masked_new,
                        ref_logprob=masked_ref,kl_penalty=self.config.kl_loss_type),mask)
                    policy_loss=policy_loss+kl_loss*self.config.kl_loss_coef
                numerical_fields.update(kl_loss=kl_loss,authored_ref_log_prob=masked_ref)
                metrics['actor/kl_loss']=kl_loss.detach().item()
                metrics['actor/kl_coef']=self.config.kl_loss_coef
            loss=policy_loss/self.gradient_accumulation
            numerical_fields['loss']=loss
            require_finite(numerical_fields,stage='before_backward',process_group=group,
                           positive=('response_tokens',),recoverable=recoverable)
            loss.backward()
            append_to_dict(metrics,{'actor/entropy_loss':entropy_loss.detach().item(),
                'actor/pg_loss':pg_loss.detach().item(),'actor/pg_clipfrac':clipfrac.detach().item(),
                'actor/ppo_kl':ppo_kl.detach().item()})
        grad_norm=self._optimizer_step()
        append_to_dict(metrics,{'actor/grad_norm':grad_norm.detach().item()})
        if safety_enabled:
            self._monitor_old_policy(frozen,safety,tail_limit,'after_minibatch')
