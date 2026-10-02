"""OPSD actor uses full-vocabulary loss with a separate frozen, privileged teacher."""
import math
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.py_functional import append_to_dict
from .distillation import FrozenTeacher, opsd_loss, global_token_count
from .rollout import select_rows


class DistillationActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer, tokenizer):
        super().__init__(config, actor_module, actor_optimizer)
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError('OPSD padded alignment currently requires sequence parallel size 1')
        if config.use_kl_loss or config.entropy_coeff:
            raise ValueError('Pure OPSD must disable reference KL and entropy bonus')
        self.teacher = FrozenTeacher(config.distillation.teacher_path, tokenizer, 'cpu')
        self.tokenizer = tokenizer

    def update_policy(self, data):
        self.actor_module.train()
        # Disable dropout for deterministic distillation distributions. Gradients remain on.
        for module in self.actor_module.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        metrics = {}
        settings = self.config.distillation
        updates = 0
        skipped = 0
        mini_size = self.config.ppo_mini_batch_size
        micro_size = self.config.ppo_micro_batch_size
        if micro_size < 1 or mini_size % micro_size:
            raise ValueError('Invalid OPSD minibatch/microbatch sizes')
        # Split DataProto itself to retain teacher-only ground truths in row order.
        for _ in range(self.config.ppo_epochs):
            for offset in range(0, len(data), mini_size):
                mini = select_rows(data, slice(offset, offset + mini_size))
                width = mini.batch['responses'].shape[1]
                total, world = global_token_count(mini.batch['loss_mask'])
                if total.item() == 0:
                    skipped += 1
                    continue
                denominator = total / world
                self.actor_optimizer.zero_grad()
                for micro_offset in range(0, len(mini), micro_size):
                    micro = select_rows(mini, slice(micro_offset, micro_offset + micro_size))
                    tensors = micro.batch
                    targets = [r['ground_truth']['target'] for r in micro.non_tensor_batch['reward_model']]
                    loss = opsd_loss(self.actor_module, self.teacher, tensors, self.tokenizer,
                                     targets, settings, denominator)
                    loss.backward()
                    append_to_dict(metrics, {'actor/opsd_loss': float(loss.detach())})
                grad_norm = self._optimizer_step()
                updates += 1
                append_to_dict(metrics, {'actor/grad_norm': float(grad_norm.detach())})
        self.actor_optimizer.zero_grad()
        metrics['actor/optimizer_steps'] = updates
        metrics['actor/skipped_empty_minibatches'] = skipped
        metrics['teacher/parameters_per_rank'] = self.teacher.parameter_count
        metrics['teacher/scored_tokens_cumulative_per_rank'] = self.teacher.scored_tokens
        metrics.update(self.teacher.cost_metrics())
        metrics.update(self.teacher.audit_metrics(self.actor_optimizer))
        observations = {key: value for key, value in metrics.items()
                        if key.startswith(("teacher/", "teacher_worker/"))}
        observations['actor/samples'] = len(data)
        old_log_probs = data.batch.get('old_log_probs', None)
        observations['actor/old_log_probs_present'] = int(old_log_probs is not None)
        if old_log_probs is not None:
            observations['actor/old_log_probs_finite'] = int(torch.isfinite(old_log_probs).all().item())
        mask = data.batch['loss_mask']
        observations['actor/supervised_tokens'] = int(mask.sum().item())
        attention = data.batch.get('attention_mask', None)
        info = data.batch.get('info_mask', None)
        if attention is not None and info is not None:
            width = data.batch['responses'].shape[-1]
            valid = torch.all((mask == 0) | (mask == 1))
            valid &= torch.all(mask <= attention[:, -width:])
            valid &= torch.equal(mask, info[:, -width:])
            observations['actor/loss_mask_valid'] = int(valid.item())
        observations['actor/optimizer_steps'] = updates
        observations['actor/skipped_empty_minibatches'] = skipped
        for key in ('actor/opsd_loss', 'actor/grad_norm'):
            values = metrics.get(key, [])
            if values:
                observations[key + '_mean'] = sum(values) / len(values)
                observations[key + '_finite'] = int(all(math.isfinite(v) for v in values))
        if torch.cuda.is_available():
            observations['actor/cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            observations['actor/cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        # DataProto.concat retains only rank 0 meta_info. Replicate complete rank
        # observations on every worker, without changing loss/gradient reduction.
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, observations)
        else:
            gathered = [observations]
        for worker_rank, values in enumerate(gathered):
            metrics.update({f'{key}/rank_{worker_rank}': value
                            for key, value in values.items()})
        return metrics
