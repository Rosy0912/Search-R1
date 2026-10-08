"""Online E1 objective; immutable E1 prompt/mask and fixed-base teacher.

Import is CPU-only. GPU inference is performed only by the explicit trainer.
The token records below come from the actual pre-update training forward,
not a later replay or an autoregressive generation probability claim.
"""
import hashlib
import json
import math


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate_component_clip(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('component_clip must explicitly be null or a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError('component_clip must explicitly be null or a finite positive number')
    return float(value)


def encoded_pair(tokenizer, pair, core, max_context=8192):
    """Use the frozen prompt and response_encoding; never truncate either side."""
    spans = pair.get('student_policy_spans')
    if spans is None:
        raise ValueError('Original student policy span provenance is required')
    previous = 0
    for start, end in spans:
        if not previous <= start <= end <= len(pair['student']):
            raise ValueError('Invalid policy span provenance')
        previous = end
    response, mask = core.response_encoding(tokenizer, pair['student'], spans)
    prefixes = [tokenizer.apply_chat_template(core.prompt(pair['question'], ref),
                   tokenize=True, add_generation_prompt=True)
                for ref in (None, pair['reference'])]
    offsets = tokenizer(pair['student'], add_special_tokens=False,
                        return_offsets_mapping=True)
    if offsets['input_ids'] != response or len(mask) != len(response):
        raise ValueError('Response token/mask alignment changed')
    if not response:
        return None, 'empty_response'
    if not sum(mask):
        return None, 'empty_policy_mask'
    if any(len(p) + len(response) > max_context for p in prefixes):
        return None, 'context_limit'
    identity = dict(question_id=pair['question_id'], pair_row_sha256=canonical_sha(pair),
                    reference_response_sha256=hashlib.sha256(pair['reference'].encode()).hexdigest(),
                    student_response_sha256=hashlib.sha256(pair['student'].encode()).hexdigest(),
                    reference_reward=pair['reference_reward'], student_reward=pair['student_reward'])
    if identity['reference_reward'] != 1:
        raise ValueError('Teacher reference must be a successful original trajectory')
    encoded = dict(prefixes=prefixes, response=response, mask=mask,
                   offsets=offsets['offset_mapping'], identity=identity)
    encoded['sha256'] = canonical_sha(encoded)
    return encoded, None


def vocabulary_metrics(student_logits, teacher_logits, component_clip):
    """FP32 forward KL and optional *per vocabulary component* upper clip."""
    import torch
    clip = validate_component_clip(component_clip)
    s = torch.log_softmax(student_logits.float(), dim=-1)
    t = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    q = t.exp()
    contributions = q * (t - s)
    clipped = contributions if clip is None else contributions.clamp(max=clip)
    hit = torch.zeros_like(contributions, dtype=torch.bool) if clip is None else contributions > clip
    return dict(raw=contributions.sum(dim=-1), post=clipped.sum(dim=-1),
                clipped_count=hit.sum(dim=-1), clipped_teacher_mass=(q * hit).sum(dim=-1),
                student_logp=s, teacher_logp=t)


def trajectory_batch_objective(loss_sums, valid_counts):
    """CPU fixture utility, preserving the actual trajectory-mean target."""
    if len(loss_sums) != len(valid_counts) or not loss_sums:
        raise ValueError('Nonempty real trajectory batch required')
    if any(n <= 0 for n in valid_counts):
        raise ValueError('Every trainable trajectory must have effective tokens')
    return sum(x / n for x, n in zip(loss_sums, valid_counts)) / len(loss_sums)


def pair_objective(model, encoded, component_clip, chunk, global_actual_pairs,
                   record=None, record_topk_positions=(), tokenizer=None):
    """One micro=1 trajectory. Returns its mean; trainer divides by real B.

    Both sides teacher-force the same stored student response. Logits are
    chunked on valid positions exactly as in frozen pair_loss. Detached
    metrics leave checkpoint closures as outputs, so backward recomputation
    never duplicates token records or consumes a logging cap twice.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    validate_component_clip(component_clip)
    if chunk != 32 or global_actual_pairs <= 0:
        raise ValueError('chunk32 and actual nonempty global batch are required')
    prefixes, response, mask = (encoded[k] for k in ('prefixes', 'response', 'mask'))
    device = next(model.parameters()).device
    base = model.get_base_model()
    transformer, head = base.base_model, base.get_output_embeddings()
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError('Output head must remain frozen')
    sid = torch.tensor([prefixes[0] + response], device=device)
    tid = torch.tensor([prefixes[1] + response], device=device)
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        teacher = transformer(input_ids=tid, use_cache=False).last_hidden_state[0]
        teacher = teacher[len(prefixes[1])-1:len(prefixes[1])+len(response)-1].detach()
    model.train()  # Enables checkpointing. Trainer verifies all dropout is zero.
    student = transformer(input_ids=sid, use_cache=False).last_hidden_state[0]
    student = student[len(prefixes[0])-1:len(prefixes[0])+len(response)-1]
    positions = torch.tensor(mask, device=device, dtype=torch.bool).nonzero().flatten()
    count = len(positions)
    if count == 0:
        raise ValueError('Zero-mask trajectory is not a trainable pair')
    loss = student.sum() * 0.
    raw_sum, post_sum, clipped_count = 0., 0., 0
    wanted = set(record_topk_positions)
    vocabulary_size=head.weight.shape[0]
    for indices in positions.split(chunk):
        sh, th = student[indices], teacher[indices]
        include_top = any(int(i) in wanted for i in indices.detach().cpu())

        def calculate(s, t, include_top=include_top):
            m = vocabulary_metrics(head(s), head(t), component_clip)
            outputs = (m['post'].sum(), m['raw'].detach(), m['post'].detach(),
                       m['clipped_count'].detach(), m['clipped_teacher_mass'].detach())
            if include_top:
                k = min(32, m['teacher_logp'].shape[-1])
                qv, qi = torch.topk(m['teacher_logp'].detach(), k, dim=-1)
                pv, pi = torch.topk(m['student_logp'].detach(), k, dim=-1)
                outputs += (qv, qi, pv, pi)
            return outputs

        values = checkpoint(calculate, sh, th, use_reentrant=False)
        if not all(torch.isfinite(v).all().item() for v in values if v.is_floating_point()):
            raise FloatingPointError('Nonfinite pre/post KL or token distributions; refuse optimizer')
        loss = loss + values[0]
        raw, post, hits, mass = [v.detach().cpu().tolist() for v in values[1:5]]
        raw_sum += sum(raw); post_sum += sum(post); clipped_count += sum(hits)
        if record:
            top = [v.detach().cpu().tolist() for v in values[5:]]
            for row, pos in enumerate(indices.detach().cpu().tolist()):
                item = dict(token_index=pos, token_id=response[pos],
                            token_offsets=encoded['offsets'][pos], policy_mask=bool(mask[pos]),
                            pre_clip_full_vocabulary_kl=raw[row], post_clip_full_vocabulary_loss=post[row],
                            clipped_vocabulary_count=hits[row], clipped_vocabulary_fraction=hits[row]/vocabulary_size,
                            vocabulary_size=vocabulary_size, clipped_teacher_probability_mass=mass[row],
                            trajectory_contribution=post[row]/count,
                            global_batch_contribution=post[row]/count/global_actual_pairs,
                            loss_scope='actual_training_forward_before_optimizer',
                            teacher_forced_fixed_student_prefix=True, actual_generation_scores=False)
                if tokenizer is not None:
                    item['token_text'] = tokenizer.decode([response[pos]])
                if include_top and pos in wanted:
                    item['teacher_top32_logp'], item['teacher_top32_ids'] = top[0][row], top[1][row]
                    item['student_top32_logp'], item['student_top32_ids'] = top[2][row], top[3][row]
                record(item)
    return loss / count, dict(valid_tokens=count, raw_loss=raw_sum/count,
                              post_loss=post_sum/count, clipped_vocabulary_count=clipped_count)
