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
from search_g1.compute_precision import ComputePrecision
from search_g1.numerics import require_finite
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

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
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
                self.last_forward_precision = self.compute_precision.observe_logits(output.logits,
                                                                                    process_group=precision_group)
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
                self.last_forward_precision = self.compute_precision.observe_logits(output.logits,
                                                                                    process_group=precision_group)
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        group = self.actor_module.process_group if isinstance(self.actor_module, FSDP) else None
        gradients = {f'gradient_{i}':p.grad for i,p in enumerate(self.actor_module.parameters()) if p.grad is not None}
        require_finite(gradients,stage='before_gradient_clipping',process_group=group)
        from search_g1.gradient_clipping import clip_global_l2
        grad_norm = clip_global_l2(self.actor_module,self.config.grad_clip,context=getattr(self,"_gradient_diagnostic_context",None))
        require_finite({"grad_norm":grad_norm},stage="before_optimizer_step",process_group=group)
        self.actor_optimizer.step()
        self._optimizer_writes=getattr(self,"_optimizer_writes",0)+1
        if __import__('os').environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.worker_evidence import append
            append('OPTIMIZER_WRITE', dict(rank=torch.distributed.get_rank(),
                context=self._gradient_diagnostic_context, write_count=self._optimizer_writes,
                gradient_norm_before=float(grad_norm),gradient_norm_after=float(self.actor_module._last_clipped_global_l2),
                learning_rates=[float(group['lr']) for group in self.actor_optimizer.param_groups]))
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
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
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
                    _, new = self._forward_micro_batch(micro, temperature=self._monitor_temperature)
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
                      monitoring_scope='fixed four trajectories per rank across length strata; not full rollout',
                      rejection_count=getattr(self,'_policy_rejections',0)+int(not report['allowed']))
        append('POLICY_MONITOR',report)
        if not report['allowed']:
            self._policy_rejections=report['rejection_count']
            self.actor_optimizer.zero_grad()
            raise RuntimeError('Old-rollout reuse stopped collectively; already-written steps retained: '+repr(report['reasons']))
        return report

    def update_policy(self, data: DataProto):
        from search_g1.policy_safety import bind_actual_config
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
            verify_worker_input(data, logical_update=logical_update, group=group)
            batch_gate_sha = data.meta_info['behavior_probability_gate']['sha256']
            assert torch.distributed.get_world_size() == 8
            assert len(data) == 320 and self.config.ppo_mini_batch_size == 16
            assert self.config.ppo_micro_batch_size == 2 and self.config.ppo_epochs == 1
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
            assert common8 and safety.monitor_trajectories_per_rank == 4
            # Freeze actual old/advantage/mask/input bytes once; select length strata.
            length_mask=batch['loss_mask'] if self.config.state_masking else batch['attention_mask'][:,-batch['responses'].shape[1]:]
            ordered=torch.argsort(length_mask.sum(-1),stable=True)
            indices=ordered[torch.linspace(0,len(batch)-1,4,device=ordered.device).long()]
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
        for batch_idx, data in enumerate(dataloader):
            self._gradient_diagnostic_context=dict(logical_update=logical_update,minibatch_index=batch_idx,prior_optimizer_writes=getattr(self,"_optimizer_writes",0))
            if common8:
                self._gradient_diagnostic_context['behavior_probability_gate_sha256'] = batch_gate_sha
            if safety_enabled:
                self._monitor_old_policy(frozen,safety,tail_limit,'before_minibatch')
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda(device=torch.cuda.current_device())  # TensorDict must follow this rank, including offload
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

                if safety_enabled:
                    authored=response_mask.bool()
                    group=self.actor_module.process_group if isinstance(self.actor_module,FSDP) else None
                    require_finite(dict(new=torch.where(authored,log_prob,0.),old=torch.where(authored,old_log_prob,0.),
                                        advantage=torch.where(authored,advantages,0.),mass=response_mask.sum()),
                                   stage='before_dual_clip_exp',process_group=group,positive=('mass',))
                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio,
                                                                              dual_clip=float(safety.dual_clip) if safety_enabled else None)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    masked_log_prob = torch.where(response_mask.bool(),log_prob,torch.zeros_like(log_prob))
                    masked_ref_log_prob = torch.where(response_mask.bool(),ref_log_prob,torch.zeros_like(ref_log_prob))
                    kld = core_algos.kl_penalty(logprob=masked_log_prob,
                                                ref_logprob=masked_ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                numerical_fields = dict(loss=loss,pg_loss=pg_loss,entropy_loss=entropy_loss,ppo_kl=ppo_kl,
                                        response_tokens=response_mask.sum())
                numerical_fields.update(authored_log_prob=torch.where(response_mask.bool(),log_prob,torch.zeros_like(log_prob)),
                    authored_old_log_prob=torch.where(response_mask.bool(),old_log_prob,torch.zeros_like(old_log_prob)),
                    authored_advantages=torch.where(response_mask.bool(),advantages,torch.zeros_like(advantages)))
                if self.config.use_kl_loss:
                    numerical_fields['kl_loss'] = kl_loss
                    numerical_fields['authored_ref_log_prob'] = masked_ref_log_prob
                group = self.actor_module.process_group if isinstance(self.actor_module, FSDP) else None
                require_finite(numerical_fields,stage='before_backward',process_group=group,positive=('response_tokens',))
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            if safety_enabled:
                self._monitor_old_policy(frozen,safety,tail_limit,'after_minibatch')
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if common8:
            writes = self._optimizer_writes - writes_before
            assert writes == 20
            if safety_enabled and self._policy_execution_role != 'production':
                from search_g1.policy_safety import publish_calibration_optimizer_proof
                publish_calibration_optimizer_proof(self.actor_module,self.actor_optimizer,
                    execution_role=self._policy_execution_role,plan_sha256=self._safety_calibration_sha,
                    initial_writes=writes_before,initial_state_entries=self._phase_initial_adam_state_entries,
                    final_writes=self._optimizer_writes,group=group)
            append_to_dict(metrics, {'actor/optimizer_steps_this_update': writes})
            from reproduction.common8.worker_evidence import append
            append('OUTER_BUDGET', dict(rank=torch.distributed.get_rank(),
                logical_update=logical_update, local_trajectories=len(batch),
                local_mini=16, local_micro=2, optimizer_writes=writes,
                cumulative_writes=self._optimizer_writes))
            print('SEARCH_G1_OUTER_BUDGET=' + __import__('json').dumps(dict(
                rank=torch.distributed.get_rank(), logical_update=logical_update,
                local_trajectories=len(batch), local_mini=16, local_micro=2,
                optimizer_writes=writes, cumulative_writes=self._optimizer_writes)), flush=True)
        return metrics
