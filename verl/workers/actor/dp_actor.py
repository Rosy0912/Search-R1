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
        if getattr(self, '_active_policy_safety', None):
            from search_r1.baselines.policy_safety import consensus, SkipOptimizerOpportunity
            try:
                grad_norm = clip_grad_norm64(self.actor_module, self.config.grad_clip)
                norm_ok = bool(torch.isfinite(grad_norm))
            except ValueError:
                norm_ok = False
                grad_norm = torch.tensor(0.)
            if not consensus(norm_ok, self._safety_group):
                raise SkipOptimizerOpportunity('gradient_clip_rejected')
        else:
            grad_norm = clip_grad_norm64(self.actor_module, self.config.grad_clip)
        if getattr(self, '_active_policy_safety', None):
            from search_r1.baselines.policy_safety import consensus
            if not consensus(bool(torch.isfinite(grad_norm)), self._safety_group):
                raise SkipOptimizerOpportunity('nonfinite_gradient_norm')
        elif not torch.isfinite(grad_norm):
            raise ValueError('Nonfinite actor gradient norm; optimizer step refused')
        if getattr(self, '_active_policy_safety', None):
            from search_r1.baselines.optimizer_transaction import transactional_adam_step
            from search_r1.baselines.policy_safety import SkipOptimizerOpportunity
            result=transactional_adam_step(self.actor_optimizer,self._safety_group)
            if not result['committed']: raise SkipOptimizerOpportunity(result['reason'])
        else:
            self.actor_optimizer.step()
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
            micro_batch = micro_batch.cuda(device=torch.cuda.current_device())
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

    def _monitor_frozen_policy(self, frozen, safety, stage, monitor_batch=None):
        from search_r1.baselines.policy_safety import monitor, consensus, tensor_identity
        import time
        started = time.monotonic()
        identity_ok = tensor_identity(frozen) == self._safety_monitor_identity
        if not consensus(identity_ok, self._safety_group):
            raise RuntimeError('Frozen rollout identity changed; not a numerical skip')
        training = self.actor_module.training
        self.actor_module.eval()
        current, masks, advantages, weights, old, hf0 = [], [], [], [], [], []
        sampled, sampling_masks = [], []
        try:
            with torch.no_grad():
                for micro in (frozen if monitor_batch is None else monitor_batch).split(self.config.ppo_micro_batch_size):
                    micro = micro.cuda(device=torch.cuda.current_device())
                    _, new = self._forward_micro_batch(micro, self._safety_temperature)
                    mask = micro['loss_mask'] if self.config.state_masking else micro['attention_mask'][:, -new.shape[1]:]
                    current.append(new); masks.append(mask); advantages.append(micro['advantages'])
                    old.append(micro['old_log_probs'])
                    hf0.append(micro['old_log_probs'] if self._baseline_aligned else micro['frozen_hf_log_probs'])
                    weights.append(mask.double() if self._baseline_aligned else
                                   mask.double() / mask.sum(-1, keepdim=True).clamp_min(1).double())
                    if self._baseline_aligned and 'sampling_log_probs' in micro.keys():
                        sampled.append(micro['sampling_log_probs'])
                        sampling_masks.append(mask * micro['sampling_mask'])
        finally:
            self.actor_module.train(training)
        report = monitor(torch.cat(old), torch.cat(current), torch.cat(advantages), torch.cat(masks), torch.cat(weights),
                         c=float(safety['dual_clip']), kl_log_bound=float(safety['kl_log_bound']),
                         target_kl=float(safety['target_old_policy_kl']), negative_tail_limit=self._safety_tail_limit,
                         group=self._safety_group)
        from search_r1.baselines.behavior_policy import hf_drift_summary
        report['hf_diagnostic_drift']=hf_drift_summary(torch.cat(hf0),torch.cat(old),torch.cat(current),torch.cat(masks),torch.cat(weights))
        report.update(denominator_role='actual_native_stochastic_sampling_policy',
                      cross_backend_hf_drift_is_separate=True, stage=stage, monitoring_scope='full_frozen_rollout' if monitor_batch is None else 'full_current_minibatch',
                      monitored_local_rows=len(frozen if monitor_batch is None else monitor_batch),
                      monitored_tensor_sha256=self._safety_monitor_identity if monitor_batch is None else tensor_identity(monitor_batch),
                      objective_weight_definition='sequence_mean reciprocal supervised row length; global weight mass',
                      rollout_identity=self._safety_rollout_identity, tensor_sha256=self._safety_monitor_identity,
                      threshold_origin='user_fixed',fixed_threshold_artifact_sha256=self._safety_calibration_sha, monitor_seconds=time.monotonic()-started)
        if self._baseline_aligned:
            numerical_reasons = {'nonfinite_or_negative_effective_input', 'nonfinite_subtraction'}
            if numerical_reasons.intersection(report['reasons']):
                from search_r1.baselines.policy_safety import SkipOptimizerOpportunity
                raise SkipOptimizerOpportunity('nonfinite_policy_monitor_input')
            report.update(diagnostic_only=True, allowed=True,
                          frozen_hf_old_policy_kl=report.pop('sampled_old_policy_kl'),
                          sampling_policy_kl=None,
                          sampling_policy_kl_status='unavailable_on_retokenized_training_actions',
                          objective_weight_definition='full-scope token mean diagnostic; actor uses microbatch token mean / accumulation',
                          denominator_role='frozen_hf_old_policy', threshold_origin='diagnostic_only')
            report['hf_diagnostic_drift']['role'] = 'frozen_hf_old_policy_denominator_drift'
            if sampled:
                report['raw_sampler_token_transport_diagnostic'] = dict(
                    diagnostic_only=True, probability_alignment='unverified_raw_sampler_only',
                    selected_logprobs_finite=bool(torch.isfinite(torch.cat(sampled)).all()))
            from search_r1.baselines.raw_sampler_diagnostic import evaluate_raw_diagnostic
            report['raw_sampler_diagnostic'] = evaluate_raw_diagnostic(self, self._raw_sampler_diagnostic_cache)
        capture_ok = True
        capture_error = None
        sink = getattr(self, '_safety_evidence_sink', None)
        if sink is not None:
            try:
                observed = frozen if monitor_batch is None else monitor_batch
                sink(dict(report=report, old_log_probs=torch.cat(old).detach().cpu(),
                          current_log_probs=torch.cat(current).detach().cpu(),
                          advantages=torch.cat(advantages).detach().cpu(),
                          mask=torch.cat(masks).detach().cpu(), weights=torch.cat(weights).detach().cpu(),
                          monitor_batch={key: value.detach().cpu() for key, value in observed.items()}))
            except Exception as error:
                capture_ok = False
                capture_error = str(error)
        if not consensus(capture_ok, self._safety_group):
            raise RuntimeError('Diagnostic evidence sink failed collectively: '+str(capture_error))
        self._safety_reports.append(report)
        return report

    def update_policy(self, data: DataProto):
        from search_r1.baselines.policy_safety import PartialPolicyUpdate
        from search_r1.baselines.optimizer_transaction import OptimizerCommitFault
        from search_r1.baselines.gradient_norm import GradientStructureFault
        self._baseline_aligned = self.config.get('baseline_aligned', False)
        self._safety_writes = 0
        self._safety_processed = 0
        self._safety_skips = []
        self._safety_attempt_id = data.meta_info.get('attempt_id')
        outer = data.meta_info.get('actual_outer')
        self._safety_expected_writes = (20 if self._baseline_aligned or outer <= 49 else 10) if type(outer) is int and 1 <= outer <= 200 else None
        self._safety_reports = []
        try:
            return DataParallelPPOActor._update_policy_impl(self, data)
        except OptimizerCommitFault as failure:
            return DataParallelPPOActor._fatal_update(self,'optimizer_commit_fault',failure.commit_progress)
        except GradientStructureFault as failure:
            return DataParallelPPOActor._fatal_update(self,'gradient_structure_fault',dict(error=str(failure),local_commit_complete=False))
        except PartialPolicyUpdate as failure:
            self.actor_optimizer.zero_grad()
            return {'actor/optimizer_steps': self._safety_writes,
                    'fresh200_update_state': failure.event,
                    'fresh200_policy_monitor': self._safety_reports}

    def _fatal_update(self, reason, progress):
        self.actor_optimizer.zero_grad()
        previous=self._safety_writes
        actual=previous+int(progress.get('local_commit_complete') is True and not progress.get('count_already_recorded'))
        event=DataParallelPPOActor._skip_event(self,reason)
        event.update(status='INFRASTRUCTURE_FAULT',update_complete=False,attempt_complete=False,postabort=True,
            optimizer_steps=actual,confirmed_previous_optimizer_steps=previous,failed_commit_progress=progress,
            current_state_requires_native_save=True,failed_opportunities=1,fatal=True)
        if getattr(self, '_baseline_aligned', False):
            from search_r1.baselines.policy_safety import BaselineAlignedUpdateFault
            raise BaselineAlignedUpdateFault(event, self._safety_reports)
        return {'actor/optimizer_steps':actual,'fresh200_update_state':event,'fresh200_policy_monitor':self._safety_reports}

    def _skip_event(self, reason=None, complete=False, postabort=False):
        writes=self._safety_writes;total=self._safety_expected_writes
        finished=bool(complete and writes>0)
        status=('COMPLETE_WITH_SKIPS' if self._safety_skips else 'COMPLETE') if finished else ('ABORTED' if writes else 'SKIPPED')
        return dict(status=status,update_complete=finished,attempt_complete=bool(complete),all_opportunities_processed=bool(complete),
            optimizer_steps=writes,expected_optimizer_steps=total,optimizer_opportunities_total=total,
            processed_opportunities=self._safety_processed,skipped_opportunities=len(self._safety_skips),
            unprocessed_opportunities=None if total is None else total-self._safety_processed,
            skipped_indices=[x['index'] for x in self._safety_skips],skipped_reasons=self._safety_skips,
            reason=reason,postabort=bool(postabort or (writes>0 and not complete)),actual_outer=getattr(self,'_safety_actual_outer',None),
            attempt_id=self._safety_attempt_id,rollout_identity=getattr(self,'_safety_rollout_identity',None),
            baseline_aligned=getattr(self,'_baseline_aligned',False))

    def _reject_safety(self, reason, report=None):
        from search_r1.baselines.policy_safety import PartialPolicyUpdate
        self.actor_optimizer.zero_grad()
        postabort=reason.startswith('post_')
        if not postabort:
            # Full-rollout preflight rejects the attempt before this mini opportunity.
            self._safety_processed -= 1
        event=DataParallelPPOActor._skip_event(self,reason,postabort=postabort)
        event['report']=report
        raise PartialPolicyUpdate(event)

    def _update_policy_impl(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        sequence_mean = self.config.get('loss_reduction', 'token_mean') == 'sequence_mean'
        if sequence_mean and self.config.use_dynamic_bsz and torch.distributed.is_initialized():
            raise ValueError('Distributed sequence_mean requires fixed micro batches')
        from search_r1.baselines.policy_safety import bind_config, consensus, agree_document, tensor_identity
        self._safety_group = self.actor_module.process_group if isinstance(self.actor_module, FSDP) else None
        self._safety_actual_outer = data.meta_info.get('actual_outer')
        self._safety_rollout_identity = data.meta_info.get('rollout_identity')
        safety = bind_config(self.config, dict(actual_outer=self._safety_actual_outer,
                            rollout_identity=self._safety_rollout_identity, local_rows=len(data)), self._safety_group)
        self._active_policy_safety = safety
        mini_size = self.config.ppo_mini_batch_size
        if safety:
            outer = self._safety_actual_outer
            world = torch.distributed.get_world_size(self._safety_group) if torch.distributed.is_initialized() else 1
            layout_ok = type(outer) is int and 1 <= outer <= 200 and bool(self._safety_rollout_identity)
            layout_ok = layout_ok and type(self._safety_attempt_id) is int and self._safety_attempt_id>0
            layout_ok = layout_ok and world == 8 and self.config.ppo_micro_batch_size == 2 and len(data) == 320
            if self._baseline_aligned:
                layout_ok = layout_ok and self.config.ppo_mini_batch_size == 16
            if not consensus(layout_ok, self._safety_group):
                raise RuntimeError('Invalid fresh200 layout; not a recoverable skip')
            mini_size = 16 if self._baseline_aligned or outer <= 49 else 32
            self._safety_expected_writes = 20 if self._baseline_aligned or outer <= 49 else 10
            if self._baseline_aligned:
                self._safety_tail_limit = safety.get('negative_tail_limit')
                self._safety_calibration_sha = None
            else:
                from search_r1.baselines.policy_safety import NonfiniteSamplingPolicy, PartialPolicyUpdate
                from search_r1.baselines.fresh200_contract import verify_fixed_policy_safety_artifact, verify_sampling_policy_binding
                try:
                    tail, sha = verify_fixed_policy_safety_artifact(safety)
                    artifact_ok, admission_error = True, None
                except Exception as error:
                    tail, sha, artifact_ok, admission_error = None, None, False, str(error)
                if not agree_document({'fixed_threshold_artifact_sha256': sha, 'tail_limit': tail}, artifact_ok, self._safety_group):
                    raise RuntimeError('Fixed-threshold admission rejected: '+str(admission_error))
                self._safety_tail_limit, self._safety_calibration_sha = tail, sha
                sampling_sha, numerical_error, structural_error = None, None, None
                try:
                    sampling_sha = verify_sampling_policy_binding(data, safety)
                except NonfiniteSamplingPolicy as error:
                    numerical_error = str(error)
                except Exception as error:
                    structural_error = str(error)
                # Identity/context/transport errors remain fatal even if another rank
                # has numerical rejection. Every rank enters these reductions.
                if not consensus(structural_error is None, self._safety_group):
                    raise RuntimeError('Sampling structural admission rejected: '+str(structural_error))
                # Even a locally numerical-rejected row must bind the same actual
                # sampled snapshot as every other rank before numerical skipping.
                import hashlib
                snapshot_sha = hashlib.sha256(data.meta_info['policy_snapshot_id'].encode()).hexdigest()
                if not agree_document({'sampling_policy_binding': snapshot_sha}, True, self._safety_group):
                    raise RuntimeError('Sampling policy identity differs across ranks')
                if not consensus(numerical_error is None, self._safety_group):
                    event = DataParallelPPOActor._skip_event(self, 'nonfinite_sampling_policy')
                    event['report'] = dict(status='NUMERIC_REJECTED', local_error=numerical_error)
                    raise PartialPolicyUpdate(event)
        if self._baseline_aligned:
            from search_r1.baselines.raw_sampler_diagnostic import prepare_raw_diagnostic
            self._raw_sampler_diagnostic_cache = prepare_raw_diagnostic(data, self.config)
        assert mini_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = mini_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if safety and not self._baseline_aligned:
            select_keys.append('frozen_hf_log_probs')
        if self._baseline_aligned and 'sampling_log_probs' in data.batch.keys():
            select_keys.extend(['sampling_log_probs', 'sampling_mask'])
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

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(mini_size)
        if safety:
            frozen = batch.clone()
            self._safety_monitor_identity = tensor_identity(frozen)
            self._safety_temperature = temperature

        metrics = {}
        updates = skipped = 0
        bapo_logprob_stats = []
        # Fixed rollout advantages/logprobs are reused across genuine inner epochs.
        # Default E=1 preserves the reviewed reproduction's update schedule.
        for batch_idx, data in enumerate(itertools.chain.from_iterable(
                dataloader for _ in range(self.config.ppo_epochs))):
            from search_r1.baselines.policy_safety import SkipOptimizerOpportunity
            self._safety_processed += 1
            try:
                # split batch into micro_batches
                mini_batch = data
                if safety:
                    # Aligned KL/tail diagnostics scan the frozen rollout only
                    # at outer boundaries. Each write still checks actual forward
                    # inputs, loss, gradients and the transactional Adam candidate.
                    if not self._baseline_aligned or batch_idx == 0:
                        report = DataParallelPPOActor._monitor_frozen_policy(self, frozen, safety, 'before_adam')
                        if 'frozen_rollout_identity_changed' in report.get('reasons',[]): raise RuntimeError('Frozen rollout identity changed')
                        if not report['allowed']:
                            DataParallelPPOActor._reject_safety(self, 'policy_monitor', report)
                    if not self._baseline_aligned:
                        report = DataParallelPPOActor._monitor_frozen_policy(self, frozen, safety, 'before_adam_full_minibatch', mini_batch)
                        if not report['allowed']:
                            raise SkipOptimizerOpportunity('policy_monitor_full_minibatch')
                from search_r1.baselines.distillation import global_token_count
                count_mask = mini_batch['loss_mask'] if self.config.state_masking else mini_batch['attention_mask'][:, -mini_batch['responses'].shape[1]:]
                total, world = global_token_count(count_mask)
                if total.item() == 0 and not sequence_mean and not self._baseline_aligned:
                    skipped += 1
                    continue
                if self.config.get('baseline_method') in ('gigpo', 'tree_grpo', 'topd', 'cosearch') and not paper_objective:
                    raise ValueError('Paper actor objective requires driver-computed policy_weights')
                topd_token_mean = self.config.get('baseline_method') == 'topd'
                if sequence_mean or paper_objective:
                    from search_r1.baselines.objectives import row_denominator, weighted_policy_loss
                    denominator = row_denominator(len(mini_batch), mini_batch.device)
                elif topd_token_mean:
                    from search_r1.baselines.advantages import token_mean_ppo
                    denominator = total / world
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()
                mini_metrics = {}

                for data in micro_batches:
                    data = data.cuda(device=torch.cuda.current_device())  # Explicit logical rank device, including offload.
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    if self.config.state_masking:
                        response_mask = data['loss_mask']
                    if sequence_mean:
                        # Empty trajectories contribute zero but remain in the full
                        # global mini row denominator; FSDP averages across ranks.
                        policy_weights = response_mask / response_mask.sum(-1, keepdim=True).clamp_min(1)
                    elif paper_objective:
                        policy_weights = data['policy_weights']
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                    if sequence_mean or paper_objective or topd_token_mean:
                        denominator = denominator.to(log_prob.device)

                    if self.config.get('baseline_method') == 'bapo':
                        valid = response_mask.bool()
                        bapo_logprob_stats.append(torch.stack((
                            (~torch.isfinite(old_log_prob) & valid).sum(),
                            (~torch.isfinite(log_prob) & valid).sum(), valid.sum())).detach())

                    if self._baseline_aligned:
                        # Archive computes a masked token mean separately in each
                        # fixed microbatch, then divides by accumulation count.
                        from search_r1.baselines.objectives import weighted_policy_loss
                        micro_denominator = response_mask.sum().clamp_min(1).to(log_prob.device)
                        pg_loss, pg_clipfrac, ppo_kl = weighted_policy_loss(
                            old_log_prob, log_prob, advantages, response_mask, response_mask,
                            clip_ratio, micro_denominator, safety_group=self._safety_group,
                            baseline_aligned=True)
                    elif sequence_mean or paper_objective:
                        pg_loss, pg_clipfrac, ppo_kl = weighted_policy_loss(
                            old_log_prob, log_prob, advantages, response_mask, policy_weights,
                            clip_ratio, denominator, None if sequence_mean else data.get('step_ids', None),
                            log_ratio_clip=20.0 if self.config.get('baseline_method') == 'bapo' and not self._baseline_aligned else None,
                            dual_clip=float(safety['dual_clip']) if safety and not self._baseline_aligned else None,
                            safety_group=self._safety_group, baseline_aligned=self._baseline_aligned)
                        if sequence_mean:
                            pg_clipfrac = pg_clipfrac * policy_weights.sum() / denominator
                            ppo_kl = ppo_kl * policy_weights.sum() / denominator
                    elif topd_token_mean:
                        pg_loss, pg_clipfrac, ppo_kl = token_mean_ppo(
                            old_log_prob, log_prob, advantages, response_mask, clip_ratio, denominator)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                      log_prob=log_prob,
                                                                                      advantages=advantages,
                                                                                      eos_mask=response_mask,
                                                                                      cliprange=clip_ratio,
                                                                                      log_ratio_clip=20.0 if self.config.get('baseline_method') == 'bapo' else None)
                    # compute entropy loss from entropy
                    entropy_loss = ((torch.where(response_mask.bool(), entropy, 0.) * response_mask).sum() / micro_denominator
                                    if self._baseline_aligned else
                                    verl_F.masked_sum(torch.where(response_mask.bool(), entropy, 0.), policy_weights) / denominator
                                    if sequence_mean or paper_objective else verl_F.masked_mean(torch.where(response_mask.bool(), entropy, 0.), response_mask))

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss and (not safety or self.config.kl_loss_coef != 0):
                        if self._baseline_aligned and not consensus(
                                bool(torch.isfinite(data['ref_log_prob'][response_mask.bool()]).all()), self._safety_group):
                            raise SkipOptimizerOpportunity('nonfinite_reference_log_probability')
                        ref_log_prob = data['ref_log_prob']
                        # Check the reference subtraction collectively before its local kernel guard.
                        if self._baseline_aligned and not consensus(bool(torch.isfinite(
                                (ref_log_prob.double() - log_prob.double())[response_mask.bool()]).all()), self._safety_group):
                            raise SkipOptimizerOpportunity('nonfinite_reference_difference')
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=torch.where(response_mask.bool(), log_prob, 0.),
                                                    ref_logprob=torch.where(response_mask.bool(), ref_log_prob.double() if self._baseline_aligned else ref_log_prob, 0.),
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = ((kld * response_mask).sum() / micro_denominator if self._baseline_aligned else
                                   verl_F.masked_sum(kld, policy_weights) / denominator
                                   if sequence_mean or paper_objective else masked_mean(kld, response_mask))

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        if sequence_mean:
                            mini_metrics['actor/kl_loss'] = mini_metrics.get('actor/kl_loss', 0.) + kl_loss.detach().item()
                        elif self._baseline_aligned:
                            append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach().item()})
                        else:
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    loss = policy_loss if sequence_mean or paper_objective or topd_token_mean else policy_loss / self.gradient_accumulation
                    if safety and not consensus(bool(torch.isfinite(loss)) and bool(torch.isfinite(entropy[response_mask.bool()]).all()), self._safety_group):
                        raise SkipOptimizerOpportunity('nonfinite_loss_or_entropy')
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    if sequence_mean:
                        for key, value in data.items():
                            mini_metrics[key] = mini_metrics.get(key, 0.) + value
                    else:
                        append_to_dict(metrics, data)

                if sequence_mean:
                    append_to_dict(metrics, mini_metrics)
                if safety:
                    finite_gradients = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.actor_module.parameters())
                    if not consensus(finite_gradients, self._safety_group):
                        raise SkipOptimizerOpportunity('nonfinite_gradients')
                grad_norm = self._optimizer_step()
                updates += 1
                self._safety_writes = updates
                if safety:
                    state_finite = all(bool(torch.isfinite(p).all()) for p in self.actor_module.parameters())
                    state_finite = state_finite and all(bool(torch.isfinite(value).all())
                        for state in self.actor_optimizer.state.values() for value in state.values() if torch.is_tensor(value))
                    if not consensus(state_finite, self._safety_group):
                        from search_r1.baselines.optimizer_transaction import OptimizerCommitFault
                        raise OptimizerCommitFault(dict(stage='post_commit_state_nonfinite',
                            local_commit_complete=True,count_already_recorded=True))
                    if not self._baseline_aligned or batch_idx + 1 == self._safety_expected_writes:
                        report = DataParallelPPOActor._monitor_frozen_policy(self, frozen, safety, 'after_adam')
                        if not report['allowed']:
                            DataParallelPPOActor._reject_safety(self, 'post_policy_monitor', report)
                    if not self._baseline_aligned:
                        report = DataParallelPPOActor._monitor_frozen_policy(self, frozen, safety, 'after_adam_full_minibatch', mini_batch)
                        if not report['allowed']:
                            DataParallelPPOActor._reject_safety(self, 'post_policy_monitor_full_minibatch', report)
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
            except SkipOptimizerOpportunity as failure:
                if not safety: raise
                if self._baseline_aligned:
                    return DataParallelPPOActor._fatal_update(self, str(failure), dict(
                        stage='numeric_admission', local_commit_complete=False))
                self.actor_optimizer.zero_grad()
                self._safety_skips.append(dict(index=batch_idx,reason=str(failure)))
                skipped += 1
                continue
        self.actor_optimizer.zero_grad()
        metrics['actor/optimizer_steps'] = updates
        metrics['actor/skipped_empty_minibatches'] = skipped
        if self.config.get('baseline_method') == 'bapo':
            stats = torch.stack(bapo_logprob_stats).sum(0).tolist() if bapo_logprob_stats else [0, 0, 0]
            for key, value in zip(('old_logprob_nonfinite', 'new_logprob_nonfinite', 'loss_tokens'), stats):
                metrics['bapo/' + key] = value
            norms = metrics.get('actor/grad_norm', [])
            metrics['bapo/grad_norm_min'] = min(norms) if norms else 0.
            metrics['bapo/grad_norm_max'] = max(norms) if norms else 0.
        if safety:
            if not consensus(self._safety_processed == self._safety_expected_writes, self._safety_group):
                raise RuntimeError('Optimizer opportunity count mismatch')
            if self._baseline_aligned and not consensus(
                    self._safety_writes == 20 and not self._safety_skips, self._safety_group):
                return DataParallelPPOActor._fatal_update(self, 'incomplete_aligned_adam_writes', dict(
                    stage='completion', local_commit_complete=False))
            metrics['fresh200_update_state'] = DataParallelPPOActor._skip_event(self,complete=True)
            metrics['fresh200_policy_monitor'] = self._safety_reports
        return metrics
