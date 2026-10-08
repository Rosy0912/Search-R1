"""Full-vocabulary OPSD loss and explicit teacher/student target alignment."""
import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F


def generalized_jsd(student_logits, teacher_logits, beta=0., temperature=1., component_clip=.05):
    if not 0 <= beta <= 1 or temperature <= 0:
        raise ValueError('Invalid JSD beta/temperature')
    s = F.log_softmax(student_logits.float() / temperature, -1)
    t = F.log_softmax(teacher_logits.detach().float() / temperature, -1)
    if beta == 0:
        parts = F.kl_div(s, t, log_target=True, reduction='none')
    elif beta == 1:
        parts = F.kl_div(t, s, log_target=True, reduction='none')
    else:
        m = torch.logaddexp(t + math.log(beta), s + math.log1p(-beta))
        parts = beta * F.kl_div(m, t, log_target=True, reduction='none') + (1 - beta) * F.kl_div(m, s, log_target=True, reduction='none')
    if component_clip is not None:
        if component_clip <= 0:
            raise ValueError('component_clip must be positive or null')
        parts = parts.clamp(max=component_clip)
    return parts.sum(-1)


def teacher_inputs(tokenizer, prompts, responses, response_attention, targets=None, max_length=None):
    """Append private training-answer hint to prompt, then identical student response.

    Return absolute prediction positions for every response target. Observations
    remain in response and attention, but the caller excludes them from loss.
    """
    from .rollout import padded
    pad = tokenizer.pad_token_id
    prefixes, rows = [], []
    for i, prompt in enumerate(prompts):
        prefix = prompt[prompt != pad].tolist()
        if targets is not None:
            answers = targets[i]
            if isinstance(answers, str):
                answers = [answers]
            hint = '\n[Teacher-only training context: accepted answers: ' + repr(list(answers)) + ']\n'
            prefix += tokenizer.encode(hint, add_special_tokens=False)
        prefixes.append(len(prefix))
        response = responses[i][response_attention[i].bool()].tolist()
        row = prefix + response
        if max_length is not None and len(row) > max_length:
            raise ValueError('Teacher context exceeds model limit; no silent truncation')
        rows.append(row)
    ids = padded(rows, pad)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        attention[i, :len(row)] = 1
    positions = torch.tensor(prefixes)[:, None] + response_attention.long().cpu().cumsum(-1) - 2
    return ids, attention, positions


def opsd_loss(student_model, teacher, tensors, tokenizer, targets, settings, denominator):
    """Identical CPU/GPU loss path; full logits retain gradients only for student."""
    width = tensors['responses'].shape[1]
    tid, tmask, positions = teacher_inputs(tokenizer, tensors['prompts'], tensors['responses'],
                                           tensors['attention_mask'][:, -width:], targets,
                                           teacher.model.config.max_position_embeddings)
    with torch.no_grad():
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
    total = torch.where(torch.isfinite(student), student, torch.zeros_like(student)).sum() * 0
    for start in range(0, width, settings.chunk_size):
        stop = min(start + settings.chunk_size, width)
        pos = positions[:, start:stop].clamp(0, hidden.shape[1] - 1).to(teacher.device)
        features = hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))
        with torch.no_grad():
            target_logits = teacher.model.get_output_embeddings()(features).to(device)
        valid = tensors['loss_mask'][:, start:stop].bool()
        source_logits = torch.where(valid[..., None], student[:, start:stop], 0.)
        target_logits = torch.where(valid[..., None], target_logits, 0.)
        if not torch.isfinite(source_logits).all() or not torch.isfinite(target_logits).all():
            raise ValueError('Nonfinite OPSD logits on supervised tokens')
        divergence = generalized_jsd(source_logits, target_logits,
                                     settings.beta, settings.temperature, settings.component_clip)
        total = total + (divergence * tensors['loss_mask'][:, start:stop]).sum()
    return total / denominator


def global_token_count(mask):
    """Raw collective count; every rank makes the same empty-minibatch decision."""
    total = mask.sum().float()
    world = 1
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        world = torch.distributed.get_world_size()
    return total, world


def global_token_denominator(mask):
    """Account for averaged gradients; an empty count is handled by the actor."""
    total, world = global_token_count(mask)
    return total.clamp_min(1) / world


class FrozenTeacher:
    """Unsharded auxiliary teacher, CPU by default; never part of actor optimizer.

    The CPU choice permits offline clusters and protects the original GPU layout.
    It costs one frozen model per actor rank for OPSD; production users should
    explicitly size RAM/microbatches before launching.
    """
    def __init__(self, path, student_tokenizer, device='cpu'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.get_vocab() != student_tokenizer.get_vocab():
            raise ValueError('Teacher/student token-to-id vocabularies differ')
        # Scoring consumes existing shared-vocabulary token IDs; it does not
        # generate with teacher EOS. Preserve both official tokenizer identities.
        self.tokenizer_special_ids = {
            'teacher': {name: getattr(self.tokenizer, name) for name in ('bos_token_id','eos_token_id')},
            'student': {name: getattr(student_tokenizer, name) for name in ('bos_token_id','eos_token_id')},
            'scope': 'FULL_TOKEN_TO_ID_EQUALITY_REQUIRED_SPECIAL_GENERATION_IDS_RECORDED'}
        self.tokenizer.pad_token_id = student_tokenizer.pad_token_id
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                        attn_implementation='eager').to(self.device)
        self.model.eval().requires_grad_(False)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.scored_tokens = 0

    @torch.no_grad()
    def log_probs(self, prompts, responses, attention, temperature=1., targets=None, chunk=16):
        ids, mask, positions = teacher_inputs(self.tokenizer, prompts, responses, attention,
                                              targets, self.model.config.max_position_embeddings)
        ids, mask = ids.to(self.device), mask.to(self.device)
        self.scored_tokens += int(mask.sum())
        hidden = self.model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        result = torch.zeros_like(responses, dtype=torch.float32)
        for start in range(0, responses.shape[1], chunk):
            stop = min(start + chunk, responses.shape[1])
            pos = positions[:, start:stop].clamp(0, hidden.shape[1] - 1).to(self.device)
            features = hidden.gather(1, pos[..., None].expand(-1, -1, hidden.shape[-1]))
            logits = self.model.get_output_embeddings()(features).float() / temperature
            labels = responses[:, start:stop].to(self.device)
            logp = F.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
            result[:, start:stop] = logp.cpu() * attention[:, start:stop].cpu()
        return result
