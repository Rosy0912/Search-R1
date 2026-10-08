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

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        from search_r1.baselines.gradient_norm import clip_grad_norm64
        grad_norm = clip_grad_norm64(self.actor_module, self.config.grad_clip)
        if not torch.isfinite(grad_norm):
            raise ValueError('Nonfinite actor gradient norm; optimizer step refused')
        if grad_norm > self.config.get('recovery_max_grad_norm',float('inf')):
            raise ValueError(f'Recovery preclip gradient gate: {float(grad_norm)}; optimizer step refused')
        safety = getattr(self, "_policy_safety", None)
        if safety is not None:
            self.last_postclip_grad_norm = float(clip_grad_norm64(self.actor_module, float("inf")))
            safety.sync_valid(not torch.isfinite(torch.tensor(self.last_postclip_grad_norm)) or self.last_postclip_grad_norm > float(self.config.grad_clip) + 1e-5, "measured_postclip_gradient")
        self.actor_optimizer.step()
        if safety is not None:
            safety.committed += 1
            safety.last_grad_preclip = float(grad_norm)
            safety.last_grad_postclip = self.last_postclip_grad_norm
            safety.publish("optimizer_committed", dict(allowed=True, reasons=[]))
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

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        method = self.config.get('baseline_method')

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
        elif self.config.get('baseline_method') == 'tree_grpo':
            from search_r1.baselines.tree_rank_audit import enabled, TreeRankAudit
            if enabled():
                audit = TreeRankAudit(self.actor_module, batch, data.meta_info, self.config, __file__)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        safety = None
        safety_config = self.config.get("policy_safety", {})
        if safety_config.get("enabled", False):
            if method != "tree_grpo" or not paper_objective or self.use_ulysses_sp or self.config.use_dynamic_bsz:
                raise ValueError("Reviewed safety path requires fixed-micro Tree DP without sequence parallelism")
            from search_r1.baselines.policy_safety_runtime import PolicySafetyRuntime
            safety = PolicySafetyRuntime(self, batch, data.meta_info, safety_config)
        self._policy_safety = safety
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
                skipped += 1
                continue
            if self.config.get('baseline_method') in ('gigpo', 'tree_grpo', 'topd', 'cosearch') and not paper_objective:
                raise ValueError('Paper actor objective requires driver-computed policy_weights')
            topd_token_mean = self.config.get('baseline_method') == 'topd'
            if paper_objective:
                from search_r1.baselines.objectives import row_denominator, weighted_policy_loss
                denominator = row_denominator(len(mini_batch), mini_batch.device)
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

            if safety is not None:
                safety.measure("before_optimizer")
            self.actor_optimizer.zero_grad()
            mini_metrics = {}

            for data in micro_batches:
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

                if safety is not None:
                    safety.inputs(old_log_prob, log_prob, advantages, response_mask, data["policy_weights"], entropy, data.get("ref_log_prob", None))
                if paper_objective:
                    pg_loss, pg_clipfrac, ppo_kl = weighted_policy_loss(
                        old_log_prob, log_prob, advantages, response_mask, data['policy_weights'],
                        clip_ratio, denominator, data.get('step_ids', None),
                        dual_clip=safety_config['dual_clip'] if safety is not None else None)
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
                    kld = core_algos.kl_penalty(logprob=torch.where(response_mask.bool(), log_prob, 0.),
                                                ref_logprob=torch.where(response_mask.bool(), ref_log_prob, 0.),
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
                    diagnostic = {}
                    if method == 'tree_grpo':
                        diagnostic['diagnostic_data'] = {
                            key: data[key] for key in ('input_ids', 'responses', 'policy_weights', 'attention_mask')
                            if key in data}
                    audit_failed = False
                    try:
                        audit.micro(log_prob, old_log_prob, response_mask, loss, data.get('step_ids', None),
                            advantages=advantages, loss_parts=parts,
                            ref_logp=ref_log_prob if self.config.use_kl_loss else None, **diagnostic)
                    except Exception:
                        if safety is None:
                            raise
                        audit_failed = True
                    if safety is not None:
                        safety.sync_valid(audit_failed, 'rank_audit_micro_before_backward')
                if safety is not None:
                    safety.sync_valid(~torch.isfinite(loss), "micro_loss_before_backward")
                try:
                    loss.backward()
                except Exception as error:
                    if method == 'tree_grpo' and audit is not None:
                        audit.failure(error)
                    raise

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
                grad_norm = self._optimizer_step()
            except Exception as error:
                if method == 'tree_grpo' and audit is not None:
                    audit.failure(error)
                raise
            if audit is not None:
                if safety is None:
                    audit.gradient(grad_norm)
                else:
                    audit_failed = False
                    try:
                        audit.gradient(grad_norm)
                    except Exception:
                        audit_failed = True
                    safety.sync_valid(audit_failed, "rank_audit_after_committed_write")
            updates += 1
            if safety is not None:
                safety.committed = updates
                safety.parameters_and_adam()
                safety.measure('after_optimizer')
                append_to_dict(metrics, {'actor/postclip_grad_norm': self.last_postclip_grad_norm})
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if audit is not None:
            if safety is None:
                audit.finish(updates, skipped)
            else:
                audit_failed = False
                try:
                    audit.finish(updates, skipped)
                except Exception:
                    audit_failed = True
                safety.sync_valid(audit_failed, "rank_audit_finish")
        metrics['actor/optimizer_steps'] = updates
        metrics['actor/skipped_empty_minibatches'] = skipped
        if paper_objective:
            from search_r1.baselines.objectives import reduce_paper_metrics
            reduce_paper_metrics(metrics, batch.device)
        return metrics
