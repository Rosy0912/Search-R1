"""Synchronous Tree actor checks; a refusal exits without partial-outer credit."""
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from .policy_safety import update_statistics, admission, global_negative_quantiles


class PolicySafetyRefusal(RuntimeError):
    pass


class PolicySafetyRuntime:
    def __init__(self, actor, batch, meta, config):
        self.actor, self.batch, self.config = actor, batch, config
        self.device = (torch.device('cuda', torch.cuda.current_device())
                       if dist.is_initialized() and dist.get_backend() == 'nccl'
                       else batch['input_ids'].device)
        self.committed = 0
        self.micro = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.sequence = 0
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
                                  float(config['dual_clip']), float(config['kl_log_bound'])],
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
    def measure(self, phase):
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            return self._measure(phase)

    def _measure(self, phase):
        combined = None
        negative = []
        max_abs_delta = torch.zeros((), device=self.device, dtype=torch.float64)
        for data in self.monitor.split(self.actor.config.ppo_micro_batch_size):
            _, current = self.actor._forward_micro_batch(data, self.temperature)
            stats, values = update_statistics(data['old_log_probs'], current, data['advantages'],
                                             self.effective_mask(data), data['policy_weights'],
                                             self.config['dual_clip'], self.config['kl_log_bound'])
            combined = stats if combined is None else {key: combined[key]+stats[key] for key in stats}
            negative.append(values)
            valid = self.effective_mask(data)
            delta = (current.double() - data['old_log_probs'].double())[valid]
            finite_delta = torch.where(torch.isfinite(delta), delta, 0.)
            if finite_delta.numel(): max_abs_delta = torch.maximum(max_abs_delta, finite_delta.abs().max())
        decision = admission(combined, self.target_kl, self.tail_limit)
        decision.update(global_negative_quantiles(torch.cat(negative)))
        if dist.is_initialized(): dist.all_reduce(max_abs_delta, op=dist.ReduceOp.MAX)
        decision['global_monitor_max_abs_delta'] = max_abs_delta.item()
        if self.committed == 0 and phase == 'before_optimizer' and max_abs_delta > self.initial_consistency_limit:
            decision['allowed'] = False
            decision['reasons'].append('initial_old_current_probability_inconsistency')
        self.publish(phase, decision)
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
