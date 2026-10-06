"""Synchronous Tree actor checks; a refusal exits without partial-outer credit."""
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .policy_safety import update_statistics, admission, global_negative_quantiles


class PolicySafetyRefusal(RuntimeError):
    pass


BEHAVIOR_MU_PROTOCOL = 'tree_behavior_mu_centered_ppo_v1'


def behavior_mu_correction_enabled(config):
    """An explicit profile, never a fallback when a field/protocol is missing."""
    if 'sampling_probability_correction' not in config:
        return False
    correction = config['sampling_probability_correction']
    if correction is None:
        raise ValueError('Present behavior-mu correction configuration cannot be null')
    if (type(correction.get('enabled')) is not bool
            or correction.get('protocol') != BEHAVIOR_MU_PROTOCOL):
        raise ValueError('Invalid explicit behavior-mu correction profile/protocol')
    return correction['enabled']


def validate_actor_probability_profile(actor, data, config):
    """All ranks validate row-local immutable mu bindings before native forwards."""
    device = (torch.device('cuda', torch.cuda.current_device())
              if dist.is_initialized() and dist.get_backend() == 'nccl'
              else data.batch['input_ids'].device)
    error, corrected = None, False
    try:
        corrected = behavior_mu_correction_enabled(config)
        marker = data.meta_info.get('tree_behavior_mu_corrected_profile', False)
        if type(marker) is not bool or marker != corrected:
            raise ValueError('Actor correction profile differs from batch profile')
        if not corrected:
            if ('sampled_old_log_probs' in data.batch.keys()
                    or 'actor_old_log_probs' in data.batch.keys()
                    or 'tree_behavior_mu_binding' in data.non_tensor_batch
                    or 'tree_behavior_mu_protocol' in data.meta_info
                    or 'tree_behavior_mu_runtime_provenance' in data.meta_info):
                raise ValueError('Behavior-mu fields supplied to uncorrected actor profile')
        else:
            if config.get('enabled') is not True or actor.config.get('baseline_method') != 'tree_grpo':
                raise ValueError('Behavior-mu correction requires enabled Tree policy safety')
            from .tree_behavior_mu import validate_behavior_mu_batch, runtime_provenance
            validate_behavior_mu_batch(data, expected_provenance=runtime_provenance(os.environ), required=True)
            sampled, old, beta = (data.batch[key] for key in
                                  ('sampled_old_log_probs', 'old_log_probs', 'actor_old_log_probs'))
            valid = PolicySafetyRuntime.effective_mask(data.batch)
            temperature = data.meta_info['temperature']
            if type(temperature) not in (int, float) or temperature != 1.:
                raise ValueError('Corrected actor forward temperature must equal captured plain mu temperature')
            micro, mini, epochs = (actor.config[key] for key in
                                   ('ppo_micro_batch_size', 'ppo_mini_batch_size', 'ppo_epochs'))
            if (any(type(value) is not int or value <= 0 for value in (micro, mini, epochs))
                    or mini % micro):
                raise ValueError('Corrected actor fixed mini/micro/epoch configuration invalid')
            for key in ('advantages', 'policy_weights') + (('ref_log_prob',) if actor.config.use_kl_loss else ()):
                value = data.batch[key]
                if (value.shape != sampled.shape or not value.is_floating_point()
                        or not torch.isfinite(value[valid]).all()):
                    raise ValueError('Corrected actor required float field invalid: '+key)
            if (sampled.dtype != torch.float32 or old.dtype != torch.float32
                    or not beta.is_floating_point() or old.shape != sampled.shape
                    or beta.shape != sampled.shape or not torch.equal(old, sampled)
                    or not torch.isfinite(beta[valid]).all()):
                raise ValueError('Fixed mu denominator / independent actor beta tensor binding invalid')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:2048]
    flag = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise PolicySafetyRefusal('Global probability-profile startup refusal; local diagnosis: '+str(error))
    signature = torch.tensor([int(corrected), int(bool(config.get('enabled', False)))], device=device)
    lo, hi = signature.clone(), signature.clone()
    if dist.is_initialized():
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.equal(lo, hi):
        raise PolicySafetyRefusal('Global probability-profile startup refusal: rank profile mismatch')
    return corrected


class PolicySafetyRuntime:
    def __init__(self, actor, batch, meta, config, source_data=None):
        self.actor, self.batch, self.config = actor, batch, config
        self.source_data = source_data
        self.device = (torch.device('cuda', torch.cuda.current_device())
                       if dist.is_initialized() and dist.get_backend() == 'nccl'
                       else batch['input_ids'].device)
        self.committed = 0
        self.micro = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.sequence = 0
        self.monitor_native_forward_calls = 0
        self.initial_actor_repeat_native_forward_calls = 0
        self.monitor_forward_wall_seconds = 0.
        self.actor_monitor_beta = None
        self.initial_mu_monitor_beta_max_abs_delta = None
        error = None
        try:
            self._prepare(meta)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        flag = torch.tensor(int(error is not None), device=self.device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise PolicySafetyRefusal(f'Global safety startup refusal; local diagnosis: {error}')
        # Different per-rank row/chunk counts would strand an FSDP monitor forward.
        signature = torch.tensor([len(self.batch), len(self.monitor), actor.config.ppo_micro_batch_size,
                                  self.target_kl, self.tail_limit, self.initial_consistency_limit,
                                  int(self.corrected_profile),
                                  float(config['dual_clip']), float(config['kl_log_bound']),
                                  actor.config.ppo_mini_batch_size, actor.config.ppo_epochs,
                                  actor.config.clip_ratio, actor.config.grad_clip,
                                  actor.config.get('recovery_max_grad_norm', 1e300)],
                                 device=self.device, dtype=torch.float64)
        lo, hi = signature.clone(), signature.clone()
        if dist.is_initialized():
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.equal(lo, hi):
            raise PolicySafetyRefusal('Global safety startup refusal: rank configuration/count mismatch')

    @staticmethod
    def effective_mask(data):
        response_attention = data['attention_mask'][:, -data['responses'].shape[1]:].bool()
        return response_attention & data['loss_mask'].bool()

    def _prepare(self, meta):
        actor, batch, config = self.actor, self.batch, self.config
        if (actor.config.get('baseline_method') != 'tree_grpo'
                or 'policy_weights' not in batch.keys()
                or actor.use_ulysses_sp or actor.config.use_dynamic_bsz
                or not actor.config.state_masking):
            raise ValueError('Safety requires fixed-micro state-masked Tree DP without sequence parallelism')
        self.step = int(meta['baseline_outer_step'])
        self.target_kl = float(config['target_old_policy_kl'])
        self.corrected_profile = behavior_mu_correction_enabled(config)
        if self.corrected_profile:
            if self.source_data is None:
                raise ValueError('Corrected safety startup requires full row-local mu source data')
            from .tree_behavior_mu import validate_behavior_mu_batch, runtime_provenance
            validate_behavior_mu_batch(self.source_data, expected_provenance=runtime_provenance(os.environ), required=True)
            for key in ('input_ids', 'responses', 'attention_mask', 'position_ids', 'loss_mask',
                        'sampled_old_log_probs', 'actor_old_log_probs', 'old_log_probs'):
                if key not in batch.keys():
                    raise ValueError('Corrected safety batch differs from full source data: '+key)
                selected, source = batch[key], self.source_data.batch[key]
                if key == 'actor_old_log_probs':
                    # Beta on nonsupervised observation/padding is never a policy input.
                    # Retain the original effective-token finite contract (NaN padding allowed).
                    valid = self.effective_mask(batch)
                    same = (selected.shape == source.shape and selected.dtype == source.dtype
                            and torch.equal(selected[valid], source[valid]))
                else:
                    same = torch.equal(selected, source)
                if not same:
                    raise ValueError('Corrected safety batch differs from full source data: '+key)
            valid = self.effective_mask(batch)
            if (batch['old_log_probs'].dtype != torch.float32
                    or not torch.equal(batch['old_log_probs'], batch['sampled_old_log_probs'])
                    or not batch['actor_old_log_probs'].is_floating_point()
                    or not torch.isfinite(batch['actor_old_log_probs'][valid]).all()):
                raise ValueError('Corrected safety requires actual fixed mu and finite independent beta')
            if meta.get('tree_behavior_mu_corrected_profile') is not True:
                raise ValueError('Corrected safety profile missing validated actor batch marker')
            repeat_limit = config['sampling_probability_correction']['initial_actor_repeat_max_abs_delta']
            if type(repeat_limit) not in (int, float):
                raise ValueError('Explicit numeric initial actor repeat threshold required')
            self.initial_consistency_limit = float(repeat_limit)
            if any(key not in batch.keys() for key in ('actor_old_log_probs', 'sampled_old_log_probs')):
                raise ValueError('Corrected safety profile missing independent fixed mu/beta fields')
        else:
            self.initial_consistency_limit = float(config['initial_old_logprob_max_abs_delta'])
        self.tail_limit = float(config['negative_tail_limit'])
        dual_clip, bound = float(config['dual_clip']), float(config['kl_log_bound'])
        if (not all(math.isfinite(value) for value in
                    (self.target_kl, self.tail_limit, self.initial_consistency_limit, dual_clip, bound))
                or self.initial_consistency_limit < 0 or self.target_kl <= 0
                or not 0 <= self.tail_limit <= 1 or dual_clip <= 1 + float(actor.config.clip_ratio) or bound <= 0):
            raise ValueError('Invalid explicit safety thresholds')
        count = int(config['monitor_rows_per_rank'])
        if count != config['monitor_rows_per_rank'] or not 0 < count <= len(batch) or count % actor.config.ppo_micro_batch_size:
            raise ValueError('Fixed monitor rows must be positive and micro-divisible')
        lengths = self.effective_mask(batch).sum(-1)
        ordered = torch.argsort(lengths, stable=True)
        self.indices = ordered[torch.linspace(0, len(batch)-1, count, device=ordered.device).long()]
        self.monitor = batch[self.indices].to(self.device)
        self.temperature = meta['temperature']
        self.root = Path(os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR']) / 'policy-safety'
        self.root.mkdir(parents=True, exist_ok=True)
        if any(isinstance(module, torch.nn.Dropout) and module.p > 0
               for module in actor.actor_module.modules()):
            raise ValueError('Safety forward requires zero dropout; no silent model mode change')
        model_config = getattr(actor.actor_module, 'config', None)
        if model_config is not None and float(getattr(model_config, 'attention_dropout', 0.)) != 0:
            raise ValueError('Safety forward requires zero functional attention dropout')

    def publish(self, phase, decision):
        self.sequence += 1
        path = self.root / f'step-{self.step:06d}-rank-{self.rank:02d}-event-{self.sequence:04d}.json'
        record = dict(outer_step=self.step, rank=self.rank, phase=phase,
                      committed_optimizer_calls=self.committed, micro=self.micro,
                      outer_completed=False, scheduler_advanced=False, automatic_rollback=False,
                      monitor_local_row_indices=self.indices.tolist(), monitor_rows=len(self.monitor),
                      local_outer_rows=len(self.batch), config=dict(self.config), decision=decision,
                      scope='Fixed length-stratified local-outer subset; not all rollout tokens')
        record['probability_denominator'] = 'actual_behavior_mu' if self.corrected_profile else 'actor_recomputed_beta'
        record['original_sampler_actor_consistency_accepted'] = False if self.corrected_profile else None
        record['monitor_native_forward_calls'] = self.monitor_native_forward_calls
        record['additional_initial_actor_repeat_native_forward_calls'] = self.initial_actor_repeat_native_forward_calls
        record['monitor_forward_wall_seconds'] = self.monitor_forward_wall_seconds
        record['monitor_forward_wall_seconds_scope'] = 'Host native-call interval; CUDA completion and collective wait may occur outside interval'
        record['measured_gradient_preclip'] = getattr(self, 'last_grad_preclip', None)
        record['measured_gradient_postclip'] = getattr(self, 'last_grad_postclip', None)
        failed = False
        try:
            with path.open('x') as stream:
                json.dump(record, stream, allow_nan=False)
        except Exception:
            failed = True
        flag = torch.tensor(int(failed), device=self.device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise PolicySafetyRefusal('Global safety evidence-write failure; no further optimizer writes')
        if not decision['allowed']:
            raise PolicySafetyRefusal(f'Policy safety refusal after {self.committed} committed calls: '
                                      f'{decision.get("reasons")}; evidence {path}')

    def sync_valid(self, bad, phase):
        flag = torch.as_tensor(bad, dtype=torch.long, device=self.device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            self.publish(phase, dict(allowed=False, reasons=['global_nonfinite_or_invalid_numeric']))

    def inputs(self, old, current, advantages, mask, weights, entropy, reference=None):
        valid = mask.bool()
        bad = torch.zeros((), dtype=torch.bool, device=self.device)
        for value in (old, current, advantages, weights, entropy, reference):
            if value is not None: bad |= ~torch.isfinite(value[valid]).all()
        delta = torch.where(valid, current, 0.).double() - torch.where(valid, old, 0.).double()
        bad |= (~torch.isfinite(delta[valid])).any()
        bad |= (delta[valid] > self.config['kl_log_bound']).any()
        bad |= (weights[valid] < 0).any()
        bad |= (~torch.isfinite(advantages[valid].double().abs()*self.config['dual_clip'])).any()
        if reference is not None:
            bad |= (~torch.isfinite((reference-current)[valid])).any()
        self.sync_valid(bad, 'micro_inputs_before_exp_backward')
        self.micro += 1

    @torch.no_grad()
    def measure(self, phase, publish=True):
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            return self._measure(phase, publish=publish)

    def _native_monitor(self, data, repeat=False):
        started = time.perf_counter()
        _, current = self.actor._forward_micro_batch(data, self.temperature)
        self.monitor_forward_wall_seconds += time.perf_counter() - started
        self.monitor_native_forward_calls += 1
        if repeat: self.initial_actor_repeat_native_forward_calls += 1
        return current

    def _measure(self, phase, publish=True):
        combined, negative, initial_values = None, [], []
        max_abs_delta = torch.zeros((), device=self.device, dtype=torch.float64)
        full_beta_delta = torch.zeros_like(max_abs_delta)
        monitor_beta_delta = torch.zeros_like(max_abs_delta)
        mu_full_beta_delta = torch.zeros_like(max_abs_delta)
        invalid = torch.zeros((), device=self.device, dtype=torch.bool)
        first = self.corrected_profile and self.actor_monitor_beta is None
        offset = 0
        for data in self.monitor.split(self.actor.config.ppo_micro_batch_size):
            current = self._native_monitor(data)
            stats, values = update_statistics(data['old_log_probs'], current, data['advantages'],
                                             self.effective_mask(data), data['policy_weights'],
                                             self.config['dual_clip'], self.config['kl_log_bound'])
            combined = stats if combined is None else {key: combined[key]+stats[key] for key in stats}
            negative.append(values)
            valid = self.effective_mask(data)
            delta = (current.double() - data['old_log_probs'].double())[valid]
            finite_delta = torch.where(torch.isfinite(delta), delta, 0.)
            if finite_delta.numel(): max_abs_delta = torch.maximum(max_abs_delta, finite_delta.abs().max())
            if self.corrected_profile:
                beta = data['actor_old_log_probs']
                invalid |= ~torch.isfinite(beta[valid]).all() | ~torch.isfinite(current[valid]).all()
                beta_mu = (beta.double() - data['old_log_probs'].double())[valid]
                invalid |= ~torch.isfinite(beta_mu).all()
                if beta_mu.numel():
                    mu_full_beta_delta = torch.maximum(mu_full_beta_delta, torch.where(torch.isfinite(beta_mu), beta_mu.abs(), 0.).max())
                full = (current.double() - beta.double())[valid]
                invalid |= ~torch.isfinite(full).all()
                if full.numel():
                    full_beta_delta = torch.maximum(full_beta_delta, torch.where(torch.isfinite(full), full.abs(), 0.).max())
                if first:
                    initial_values.append(current.detach().clone())
                else:
                    frozen = self.actor_monitor_beta[offset:offset+len(data)]
                    change = (current.double() - frozen.double())[valid]
                    invalid |= ~torch.isfinite(change).all()
                    if change.numel():
                        monitor_beta_delta = torch.maximum(monitor_beta_delta, torch.where(torch.isfinite(change), change.abs(), 0.).max())
                offset += len(data)
        # The first beta monitor is frozen at the original same-monitor partition;
        # its second native pass checks repeatability, never mu==beta consistency.
        repeat_delta = torch.zeros_like(max_abs_delta)
        if first:
            frozen = torch.cat(initial_values, dim=0)
            offset = 0
            for data in self.monitor.split(self.actor.config.ppo_micro_batch_size):
                repeated = self._native_monitor(data, repeat=True)
                valid = self.effective_mask(data)
                difference = (repeated.double() - frozen[offset:offset+len(data)].double())[valid]
                invalid |= ~torch.isfinite(difference).all()
                if difference.numel():
                    repeat_delta = torch.maximum(repeat_delta, torch.where(torch.isfinite(difference), difference.abs(), 0.).max())
                offset += len(data)
            self.actor_monitor_beta = frozen
        decision = admission(combined, self.target_kl, self.tail_limit)
        decision.update(global_negative_quantiles(torch.cat(negative)))
        if dist.is_initialized(): dist.all_reduce(max_abs_delta, op=dist.ReduceOp.MAX)
        decision['global_monitor_max_abs_delta'] = max_abs_delta.item()
        if self.corrected_profile:
            diagnostics = torch.stack([full_beta_delta, monitor_beta_delta, repeat_delta, mu_full_beta_delta, invalid.double()])
            if dist.is_initialized(): dist.all_reduce(diagnostics, op=dist.ReduceOp.MAX)
            if first: self.initial_mu_monitor_beta_max_abs_delta = max_abs_delta.item()
            decision.update(probability_denominator='actual_behavior_mu',
                            behavior_kl_scope='Original Tree equal-leaf weighted fixed monitor measure; not unbiased full-policy KL',
                            global_monitor_mu_actor_max_abs_delta=max_abs_delta.item(),
                            global_monitor_mu_full_partition_beta_max_abs_delta=diagnostics[3].item(),
                            global_monitor_initial_mu_monitor_beta_max_abs_delta=self.initial_mu_monitor_beta_max_abs_delta,
                            original_sampler_actor_consistency_accepted=False,
                            initial_mu_kl_baseline_subtracted=False,
                            global_monitor_actor_full_partition_max_abs_delta=diagnostics[0].item(),
                            global_monitor_actor_frozen_monitor_max_abs_delta=diagnostics[1].item(),
                            global_monitor_actor_repeat_max_abs_delta=diagnostics[2].item() if first else None,
                            initial_actor_repeat_checked=first,
                            additional_initial_actor_repeat_native_forward_calls=self.initial_actor_repeat_native_forward_calls)
            if diagnostics[4].item():
                decision['allowed'] = False
                decision['reasons'].append('nonfinite_actor_beta_or_monitor_difference')
            if first and diagnostics[2] > self.initial_consistency_limit:
                decision['allowed'] = False
                decision['reasons'].append('initial_actor_same_partition_repeat_inconsistency')
        elif self.committed == 0 and phase == 'before_optimizer' and max_abs_delta > self.initial_consistency_limit:
            decision['allowed'] = False
            decision['reasons'].append('initial_old_current_probability_inconsistency')
        if publish: self.publish(phase, decision)
        return decision

    @torch.no_grad()
    def parameters_and_adam(self):
        bad = torch.zeros((), dtype=torch.bool, device=self.device)
        tensors = list(self.actor.actor_module.parameters())
        tensors += [value for state in self.actor.actor_optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        for tensor in tensors:
            if tensor.is_floating_point():
                for chunk in tensor.detach().reshape(-1).split(262144):
                    bad |= ~torch.isfinite(chunk).all().to(self.device)
        self.sync_valid(bad, 'post_write_all_local_parameters_and_adam')
