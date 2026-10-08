"""Frozen e43 OPSD loss loop for empty-chunk regression only.

Original e43 opsd_loss body retained verbatim. Shared helper math/teacher class
are unchanged by A; this fixture is never imported by production training.
"""
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint
from search_r1.baselines.distillation import teacher_inputs, generalized_jsd

def opsd_loss(student_model, teacher, tensors, tokenizer, targets, settings, denominator):
    """Identical CPU/GPU loss path; full logits retain gradients only for student."""
    width = tensors['responses'].shape[1]
    tid, tmask, positions = teacher_inputs(tokenizer, tensors['prompts'], tensors['responses'],
                                           tensors['attention_mask'][:, tensors['attention_mask'].shape[1] - width:], targets,
                                           teacher.model.config.max_position_embeddings)
    with teacher.measure(), torch.no_grad():
        hidden = teacher.model.base_model(input_ids=tid.to(teacher.device),
                                          attention_mask=tmask.to(teacher.device),
                                          use_cache=False).last_hidden_state
    teacher.scored_tokens += int(tmask.sum())
    device = tensors['input_ids'].device
    # Chunking loss does not bound the model's full logits allocation.
    if hasattr(settings, 'max_student_logits_gib'):
        bytes_per_element = 2 if device.type == 'cuda' else 4
        allocation = tensors['input_ids'].numel() * student_model.config.vocab_size * bytes_per_element
        if allocation > settings.max_student_logits_gib * 1024 ** 3:
            raise ValueError('OPSD full-logits estimate exceeds explicit budget; reduce microbatch/length')
    context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with context:
        logits = student_model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'],
                               position_ids=tensors['position_ids'], use_cache=False).logits
        student = logits[:, -width - 1:-1]
    # A connected scalar zero needs one vocabulary column, not a second full
    # response-logits allocation. All supervised columns are validated below.
    seed = student[..., :1]
    total = torch.where(torch.isfinite(seed), seed, torch.zeros_like(seed)).sum() * 0
    for start in range(0, width, settings.chunk_size):
        stop = min(start + settings.chunk_size, width)
        pos = positions[:, start:stop].clamp(0, hidden.shape[1] - 1).to(teacher.device)
        features = hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))
        with teacher.measure(), torch.no_grad():
            target_logits = teacher.model.get_output_embeddings()(features)
        # CUDA copies may wait for the student stream; exclude that from CPU teacher timing.
        target_logits = target_logits.to(device)
        valid = tensors['loss_mask'][:, start:stop].bool()
        source_logits = torch.where(valid[..., None], student[:, start:stop], 0.)
        target_logits = torch.where(valid[..., None], target_logits, 0.)
        if not torch.isfinite(source_logits).all() or not torch.isfinite(target_logits).all():
            raise ValueError('Nonfinite OPSD logits on supervised tokens')
        divergence_fn = checkpoint if getattr(settings, 'loss_checkpointing', False) else None
        args = (source_logits, target_logits, settings.beta, settings.temperature, settings.component_clip)
        if divergence_fn is not None:
            # Explicit immutable arguments avoid capturing another loop chunk.
            # Only softmax/KL/clipping recompute; teacher/student projection and
            # context construction remain outside this non-reentrant checkpoint.
            divergence = divergence_fn(generalized_jsd, *args, use_reentrant=False)
        else:
            divergence = generalized_jsd(*args)
        total = total + (divergence * tensors['loss_mask'][:, start:stop]).sum()
    return total / denominator

