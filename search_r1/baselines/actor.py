"""OPSD actor uses full-vocabulary loss with a separate frozen, privileged teacher."""
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
        return metrics
