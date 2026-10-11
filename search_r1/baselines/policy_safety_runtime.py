"""Synchronous Tree actor checks; a refusal exits without partial-outer credit."""
import json
import math
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from .policy_safety import update_statistics, admission, global_negative_quantiles


class PolicySafetyRefusal(RuntimeError):
    def __init__(self, message, *, matched_all_ranks=False, safety_guard=False):
        super().__init__(message)
        self.matched_all_ranks = matched_all_ranks
        self.safety_guard = safety_guard


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


def full_actor_beta_available(meta):
    value = meta.get('tree_full_actor_beta_available', True)
    if type(value) is not bool:
        raise ValueError('Full actor beta availability must be explicit boolean')
    if not value and os.environ.get('RECURSIVE_A_OMIT_FULL_BETA') != '1':
        raise ValueError('Missing full beta requires explicit authorized recursive-A profile')
    return value


class _ActorProbabilityValidation:
    """Process-local, single-update proof; never accepted from batch metadata.

    DataProto.select shares these exact tensors. Keep references and mutation
    versions so replacing or writing a tensor cannot reuse a previous check.
    No rollout/driver receipt is trusted as an actor validation result.
    """
    def __init__(self, source_data):
        self.source_data = source_data
        self.meta = source_data.meta_info
        self.non_tensor = source_data.non_tensor_batch
        self.metadata_fingerprint = self._metadata_fingerprint(source_data)
        self.tensors = tuple((key, value, value._version)
                             for key, value in source_data.batch.items())

    @staticmethod
    def _metadata_fingerprint(source_data):
        from .tree_behavior_mu import FIELD, META, PROVENANCE_META, PROVENANCE_KEYS
        from .tree_raw_tokens import FIELD as RAW_FIELD, fingerprint
        meta_keys = (META, PROVENANCE_META, 'tree_behavior_mu_corrected_profile',
                     'tree_raw_token_protocol', 'temperature', 'tree_full_actor_beta_available',
                     'baseline_outer_step', 'baseline_attempt_id')
        meta = [(key, key in source_data.meta_info, source_data.meta_info.get(key))
                for key in meta_keys]
        rows = []
        for key in (FIELD, RAW_FIELD, 'tree_uid', 'trajectory_uid'):
            value = source_data.non_tensor_batch.get(key)
            rows.append((key, key in source_data.non_tensor_batch,
                         value.tolist() if hasattr(value, 'tolist') else value))
        # Hash actual nested contents, not their caller-supplied binding hashes
        # or container identities. This is two batch JSON scans, never another
        # tokenizer/prefix proof scan. Environment provenance is equally bound.
        return fingerprint(dict(meta=meta, rows=rows,
                                runtime={key: os.environ.get(key) for key in PROVENANCE_KEYS}))

    def matches(self, source_data):
        tensors_match = (source_data is self.source_data
                and source_data.meta_info is self.meta
                and source_data.non_tensor_batch is self.non_tensor
                and len(source_data.batch.keys()) == len(self.tensors)
                and all(source_data.batch.get(key) is value and value._version == version
                        for key, value, version in self.tensors))
        if not tensors_match:
            return False
        try:
            return self._metadata_fingerprint(source_data) == self.metadata_fingerprint
        except (TypeError, ValueError):
            # Malformed nested metadata must re-enter the full validator too.
            return False


def validate_actor_probability_profile(actor, data, config, *, return_context=False):
    """All ranks validate row-local immutable mu bindings before native forwards."""
    device = (torch.device('cuda', torch.cuda.current_device())
              if dist.is_initialized() and dist.get_backend() == 'nccl'
              else data.batch['input_ids'].device)
    error, corrected, context = None, False, None
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
            sampled, old = (data.batch[key] for key in ('sampled_old_log_probs', 'old_log_probs'))
            available = full_actor_beta_available(data.meta_info)
            beta = data.batch['actor_old_log_probs'] if 'actor_old_log_probs' in data.batch.keys() else None
            if available != (beta is not None):
                raise ValueError('Actual beta presence differs from declared availability')
            if not available and (not actor.config.use_kl_loss or actor.config.kl_loss_coef != 0.):
                raise ValueError('Omitting full beta requires zero reference KL loss')
            valid = PolicySafetyRuntime.effective_mask(data.batch)
            temperature = data.meta_info['temperature']
            if type(temperature) not in (int, float) or temperature != 1.:
                raise ValueError('Corrected actor forward temperature must equal captured plain mu temperature')
            micro, mini, epochs = (actor.config[key] for key in
                                   ('ppo_micro_batch_size', 'ppo_mini_batch_size', 'ppo_epochs'))
            if (any(type(value) is not int or value <= 0 for value in (micro, mini, epochs))
                    or mini % micro):
                raise ValueError('Corrected actor fixed mini/micro/epoch configuration invalid')
            finite_checks = []
            for key in ('advantages', 'policy_weights') + (('ref_log_prob',) if actor.config.use_kl_loss else ()):
                value = data.batch[key]
                if value.shape != sampled.shape or not value.is_floating_point():
                    raise ValueError('Corrected actor required float field invalid: '+key)
                finite_checks.append(torch.isfinite(value[valid]).all())
            if (sampled.dtype != torch.float32 or old.dtype != torch.float32
                    or old.shape != sampled.shape or not torch.equal(old, sampled)
                    or (available and (not beta.is_floating_point() or beta.shape != sampled.shape))):
                raise ValueError('Fixed mu denominator / independent actor beta tensor binding invalid')
            if available: finite_checks.append(torch.isfinite(beta[valid]).all())
            if not torch.stack(finite_checks).all():
                raise ValueError('Corrected actor required float field / independent beta nonfinite')
            context = _ActorProbabilityValidation(data)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:2048]
    flag = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise PolicySafetyRefusal('Global probability-profile startup refusal; local diagnosis: '+str(error), matched_all_ranks=True)
    signature = torch.tensor([int(corrected), int(bool(config.get('enabled', False))),
                              int(full_actor_beta_available(data.meta_info)) if corrected else 0], device=device)
    lo, hi = signature.clone(), signature.clone()
    if dist.is_initialized():
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.equal(lo, hi):
        raise PolicySafetyRefusal('Global probability-profile startup refusal: rank profile mismatch', matched_all_ranks=True)
    return (corrected, context) if return_context else corrected


class PolicySafetyRuntime:
    def __init__(self, actor, batch, meta, config, source_data=None, validation_context=None):
        self.actor, self.batch, self.config = actor, batch, config
        self.source_data = source_data
        self.validation_context = validation_context
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
        self.reused_monitor_measurements = 0
        self._after_monitor = None
        self.initial_state_checked = False
        error = None
        try:
            self._prepare(meta)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        flag = torch.tensor(int(error is not None), device=self.device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise PolicySafetyRefusal(f'Global safety startup refusal; local diagnosis: {error}', matched_all_ranks=True)
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
            raise PolicySafetyRefusal('Global safety startup refusal: rank configuration/count mismatch', matched_all_ranks=True)

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
        from .tree_guard_continue import evidence_suffix
        self.attempt_suffix = evidence_suffix(meta)
        self.attempt_id = meta.get('baseline_attempt_id')
        self.target_kl = float(config['target_old_policy_kl'])
        self.corrected_profile = behavior_mu_correction_enabled(config)
        self.full_actor_beta_available = full_actor_beta_available(meta) if self.corrected_profile else False
        if self.corrected_profile:
            if self.source_data is None:
                raise ValueError('Corrected safety startup requires full row-local mu source data')
            shared_validation = (isinstance(self.validation_context, _ActorProbabilityValidation)
                                 and self.validation_context.matches(self.source_data))
            if not shared_validation:
                from .tree_behavior_mu import validate_behavior_mu_batch, runtime_provenance
                validate_behavior_mu_batch(self.source_data, expected_provenance=runtime_provenance(os.environ), required=True)
            for key in ('input_ids', 'responses', 'attention_mask', 'position_ids', 'loss_mask',
                        'sampled_old_log_probs', 'old_log_probs') + (('actor_old_log_probs',) if self.full_actor_beta_available else ()):
                if key not in batch.keys():
                    raise ValueError('Corrected safety batch differs from full source data: '+key)
                selected, source = batch[key], self.source_data.batch[key]
                if selected is source:
                    same = True
                elif key == 'actor_old_log_probs':
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
            if not shared_validation and (batch['old_log_probs'].dtype != torch.float32
                    or not torch.equal(batch['old_log_probs'], batch['sampled_old_log_probs'])
                    or (self.full_actor_beta_available and (not batch['actor_old_log_probs'].is_floating_point()
                    or not torch.isfinite(batch['actor_old_log_probs'][valid]).all()))):
                raise ValueError('Corrected safety requires actual fixed mu and finite independent beta')
            if meta.get('tree_behavior_mu_corrected_profile') is not True:
                raise ValueError('Corrected safety profile missing validated actor batch marker')
            repeat_limit = config['sampling_probability_correction']['initial_actor_repeat_max_abs_delta']
            if type(repeat_limit) not in (int, float):
                raise ValueError('Explicit numeric initial actor repeat threshold required')
            self.initial_consistency_limit = float(repeat_limit)
            if any(key not in batch.keys() for key in (('actor_old_log_probs', 'sampled_old_log_probs') if self.full_actor_beta_available else ('sampled_old_log_probs',))):
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
        path = self.root / f'step-{self.step:06d}{self.attempt_suffix}-rank-{self.rank:02d}-event-{self.sequence:04d}.json'
        record = dict(outer_step=self.step, attempt_id=self.attempt_id, rank=self.rank, phase=phase,
                      committed_optimizer_calls=self.committed, micro=self.micro,
                      outer_completed=False, scheduler_advanced=False, automatic_rollback=False,
                      monitor_local_row_indices=self.indices.tolist(), monitor_rows=len(self.monitor),
                      local_outer_rows=len(self.batch), config=self.config, decision=decision,
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
        error = None
        try:
            record['config'] = (OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
                                if OmegaConf.is_config(self.config) else dict(self.config))
            payload = json.dumps(record, allow_nan=False)
            with path.open('x') as stream:
                stream.write(payload)
        except Exception as exc:
            failed = True
            error = f'{type(exc).__name__}: {exc}'[:2048]
        flag = torch.tensor(int(failed), device=self.device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise PolicySafetyRefusal('Global safety evidence-write failure; no further optimizer writes; '
                                      f'local diagnosis: {error}', matched_all_ranks=True)
        if not decision['allowed']:
            reasons = decision.get('reasons', [])
            faults = {'initial_actor_same_partition_repeat_inconsistency',
                      'initial_old_current_probability_inconsistency'}
            safety_guard = bool(reasons) and not any(reason in faults for reason in reasons)
            raise PolicySafetyRefusal(f'Policy safety refusal after {self.committed} committed calls: '
                                      f'{reasons}; evidence {path}', matched_all_ranks=True,
                                      safety_guard=safety_guard and not phase.startswith('rank_audit'))

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
        # After Adam and before the next opportunity are the same policy state.
        # Require agreement before bypassing FSDP forwards; a local-only cache
        # decision could otherwise strand ranks inside the monitor collective.
        if phase == 'before_optimizer':
            cached = self._after_monitor
            reusable = (cached is not None and cached[0] == self._monitor_version())
            reuse = torch.tensor(int(reusable), device=self.device)
            if dist.is_initialized(): dist.all_reduce(reuse, op=dist.ReduceOp.MIN)
            if reuse.item():
                decision = dict(cached[1], reasons=list(cached[1]['reasons']),
                                monitor_measurement_reused=True,
                                monitor_measurement_source_phase='after_optimizer',
                                monitor_measurement_optimizer_calls=self.committed)
                self.reused_monitor_measurements += 1
                if publish: self.publish(phase, decision)
                return decision
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            decision = self._measure(phase, publish=publish)
        if phase == 'after_optimizer' and decision['allowed']:
            self._after_monitor = (self._monitor_version(),
                                   dict(decision, reasons=list(decision['reasons'])))
        return decision

    def _monitor_version(self):
        return (self.committed, self.actor.actor_module.training,
                tuple((id(value), value._version) for value in self.actor.actor_module.parameters()),
                tuple((key, id(value), value._version) for key, value in self.monitor.items()))

    def _native_monitor(self, data, repeat=False):
        started = time.perf_counter()
        forward = getattr(self.actor, '_forward_monitor_micro_batch', self.actor._forward_micro_batch)
        _, current = forward(data, self.temperature)
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
                invalid |= ~torch.isfinite(current[valid]).all()
                if self.full_actor_beta_available:
                    beta = data['actor_old_log_probs']
                    invalid |= ~torch.isfinite(beta[valid]).all()
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
        decision['monitor_measurement_reused'] = False
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
                            full_actor_beta_available=self.full_actor_beta_available,
                            global_monitor_mu_full_partition_beta_max_abs_delta=diagnostics[3].item() if self.full_actor_beta_available else None,
                            global_monitor_initial_mu_monitor_beta_max_abs_delta=self.initial_mu_monitor_beta_max_abs_delta,
                            original_sampler_actor_consistency_accepted=False,
                            initial_mu_kl_baseline_subtracted=False,
                            global_monitor_actor_full_partition_max_abs_delta=diagnostics[0].item() if self.full_actor_beta_available else None,
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
    def parameters_and_adam(self, phase='post_write_all_local_parameters_and_adam'):
        # Aggregate on the tensor's own device before transferring one flag.
        # Adam step scalars can live on CPU even when its moment buffers are on
        # CUDA; copying a flag per tensor adds hundreds of tiny synchronizations.
        by_device = {}
        tensors = list(self.actor.actor_module.parameters())
        tensors += [value for state in self.actor.actor_optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        for tensor in tensors:
            if tensor.is_floating_point():
                by_device.setdefault(tensor.device, []).append(tensor.detach())
        bad = torch.zeros((), dtype=torch.bool, device=self.device)
        for device, values in by_device.items():
            local_bad = torch.zeros((), dtype=torch.bool, device=device)
            scalars = [value.reshape(()) for value in values if value.numel() == 1]
            if scalars:
                local_bad |= ~torch.isfinite(torch.stack(scalars)).all()
            for value in values:
                if value.numel() != 1:
                    # Same exhaustive finite test, with bounded 4 Mi-element
                    # boolean temporaries and 16x fewer chunks than before.
                    for chunk in value.reshape(-1).split(4194304):
                        local_bad |= ~torch.isfinite(chunk).all()
            bad |= local_bad.to(self.device)
        self.sync_valid(bad, phase)
