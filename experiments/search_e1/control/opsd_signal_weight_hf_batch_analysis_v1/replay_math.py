"""Same-response posthoc probability metrics; no generation or training."""
import math
try:
    from .common import require, semantic_sha
except ImportError:
    from common import require, semantic_sha

FEATURES = ('student_entropy', 'answer_student_logp', 'answer_teacher_logp', 'answer_gap',
    'raw_teacher_reverse_kl', 'post_clip_teacher_loss', 'source100_ordinary_reverse_kl',
    'policy_student_logp', 'policy_teacher_logp')


def encode_response(tokenizer, question, result, reference, core, regions, max_context=8192):
    """Ordinary context legality is independent of missing/overlong teacher B."""
    response = result['response']; spans = result['policy_spans']
    previous = 0
    for a, b in spans:
        require(type(a) is int and type(b) is int and previous <= a <= b <= len(response),
            'Native authored policy span provenance invalid')
        previous = b
    ids, mask = core.response_encoding(tokenizer, response, spans)
    offsets = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    require(ids == offsets['input_ids'] and len(ids) == len(mask) == len(offsets['offset_mapping'])
        and all(type(x) in (bool, int) and x in (0, 1) for x in mask), 'Native response ID/mask/offset alignment changed')
    ordinary = tokenizer.apply_chat_template(core.prompt(question, None), tokenize=True, add_generation_prompt=True)
    teacher = None if reference is None else tokenizer.apply_chat_template(core.prompt(question, reference),
        tokenize=True, add_generation_prompt=True)
    ordinary_reason = 'empty_response' if not ids else 'empty_policy_mask' if not sum(mask) else (
        'ordinary_context_limit' if len(ordinary) + len(ids) > max_context else None)
    teacher_reason = 'no_common_zero_successful_reference' if teacher is None else (
        'teacher_joint_context_limit' if len(teacher) + len(ids) > max_context else ordinary_reason)
    segments = regions.character_regions(response, spans)
    labels = [regions.category(offset, segments) for offset in offsets['offset_mapping']]
    item = dict(response=ids, mask=mask, offsets=offsets['offset_mapping'], ordinary_prefix=ordinary,
        teacher_prefix=None if teacher_reason else teacher, regions=labels,
        answer_positions=[i for i, label in enumerate(labels) if label == 'final_answer' and mask[i]],
        ordinary_missing_reason=ordinary_reason, teacher_missing_reason=teacher_reason,
        ordinary_context_tokens=len(ordinary) + len(ids),
        teacher_context_tokens=None if teacher is None else len(teacher) + len(ids), no_truncation=True,
        causal_alignment='prefix_length-1 : prefix_length+response_length-1')
    item['sha256'] = semantic_sha(item)
    return item


def scalar_metrics(student_logits, ordinary_source_logits, teacher_logits, objective):
    """FP32 all-vocabulary math, borrowing frozen component upper-clip10 code."""
    import torch
    substitute = ordinary_source_logits if teacher_logits is None else teacher_logits
    values = objective.vocabulary_metrics(student_logits, substitute, 10., ordinary_source_logits,
        alpha=1., use_mixture=False)
    s = values['student_logp']; b = values['old_logp']
    t = None if teacher_logits is None else values['teacher_logp']
    require(all(torch.isfinite(x).all().item() for x in (s, b)), 'Nonfinite student/source vocabulary')
    return dict(student_logp=s, source_logp=b, teacher_logp=t,
        student_entropy=-(s.exp() * s).sum(-1),
        source100_ordinary_reverse_kl=values['student_to_outer_old_raw_kl'],
        raw_teacher_reverse_kl=None if t is None else values['raw'],
        post_clip_teacher_loss=None if t is None else values['post'],
        clipped_vocabulary_count=None if t is None else values['clipped_count'],
        clipped_student_probability_mass=None if t is None else values['clipped_student_mass'],
        clipped_teacher_probability_mass=None if t is None else values['clipped_teacher_mass'])


def summarize_tokens(encoded, rows):
    wanted = [] if encoded['ordinary_missing_reason'] else [i for i, mask in enumerate(encoded['mask']) if mask]
    require([r['token_index'] for r in rows] == wanted, 'Replay must cover every effective policy token exactly once')
    answer = set(encoded['answer_positions'])
    for row in rows:
        i = row['token_index']
        require(row['token_id'] == encoded['response'][i] and row['token_offsets'] == encoded['offsets'][i]
            and row['region'] == encoded['regions'][i], 'Posthoc token ID/offset/region alignment differs')
        require({'student_entropy', 'source100_ordinary_reverse_kl', 'raw_teacher_reverse_kl',
            'post_clip_teacher_loss', 'student_observed_logp', 'teacher_observed_logp', 'source_observed_logp',
            'clipped_vocabulary_count', 'clipped_student_probability_mass', 'clipped_teacher_probability_mass'} <= set(row),
            'Every effective token requires all native scalar keys; unavailable teacher fields must explicitly be None')
        require(all(value is None or type(value) in (int, float) and math.isfinite(value)
            for key, value in row.items() if key in FEATURES or key.endswith('_logp')
            or key.startswith('clipped_')), 'Nonfinite token metric')
    def mean(key, selected=rows):
        values = [r[key] for r in selected if r.get(key) is not None]
        require(not values or len(values) == len(selected), 'Feature cannot silently drop some effective tokens')
        return math.fsum(values) / len(values) if values else None
    ar = [r for r in rows if r['token_index'] in answer]
    features = {key: mean(key) for key in ('student_entropy', 'source100_ordinary_reverse_kl',
        'raw_teacher_reverse_kl', 'post_clip_teacher_loss')}
    features['answer_student_logp'] = mean('student_observed_logp', ar)
    features['answer_teacher_logp'] = mean('teacher_observed_logp', ar)
    features['answer_gap'] = None if (not ar or features['answer_teacher_logp'] is None) else (
        math.fsum(r['student_observed_logp'] - r['teacher_observed_logp'] for r in ar) / len(ar))
    features['policy_student_logp'] = mean('student_observed_logp')
    features['policy_teacher_logp'] = mean('teacher_observed_logp')
    reasons = {}
    for key in FEATURES:
        if features[key] is not None:
            reasons[key] = None
        elif encoded['ordinary_missing_reason']:
            reasons[key] = encoded['ordinary_missing_reason']
        elif key in ('answer_gap', 'answer_student_logp', 'answer_teacher_logp') and not ar:
            reasons[key] = 'no_effective_final_answer_payload_tokens'
        else:
            reasons[key] = encoded['teacher_missing_reason']
    count = len(rows); regions = {}
    for label in sorted(set(r['region'] for r in rows)):
        selected = [r for r in rows if r['region'] == label]
        contributions = {}
        for key in ('raw_teacher_reverse_kl', 'post_clip_teacher_loss', 'source100_ordinary_reverse_kl', 'student_entropy'):
            values = [r[key] for r in selected if r[key] is not None]
            contributions[key] = dict(sum=math.fsum(values) if values else None,
                region_mean=math.fsum(values) / len(values) if values else None,
                trajectory_mean_contribution=math.fsum(values) / count if values else None)
        regions[label] = dict(effective_tokens=len(selected), contributions=contributions)
    return dict(valid_policy_tokens=count, answer_valid_tokens=len(ar), metrics=features, missing_reasons=reasons,
        regions=regions, teacher_available=encoded['teacher_missing_reason'] is None,
        encoded_sha256=encoded['sha256'], ordinary_context_tokens=encoded['ordinary_context_tokens'],
        teacher_context_tokens=encoded['teacher_context_tokens'],
        clipped_vocabulary_count=None if features['raw_teacher_reverse_kl'] is None else sum(r['clipped_vocabulary_count'] for r in rows),
        clipped_student_probability_mass_mean=mean('clipped_student_probability_mass'),
        clipped_teacher_probability_mass_mean=mean('clipped_teacher_probability_mass'),
        region_contributions_are_loss_not_parameter_gradients=True,
        original_training_objective=False, metrics_scope='posthoc_same_saved_response_teacher_forcing',
        actual_generation_scores=False, full_vocabulary_arrays_saved=False)


def replay_response(model, tokenizer, encoded, modules, token_record, adapter_name):
    """Three fixed-prefix forwards; frozen student/reference/source, no generate."""
    import torch
    if encoded['ordinary_missing_reason']:
        return summarize_tokens(encoded, [])
    response = encoded['response']; device = next(model.parameters()).device
    precision = modules['precision'].PrecisionContract('bf16')
    transformer = model.get_base_model().base_model
    head = model.get_base_model().get_output_embeddings()
    require(not any(p.requires_grad for p in model.parameters()), 'Replay parameters must all be frozen')
    require(not any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()), 'Unexpected dropout in replay')
    def forward(prefix, role):
        ids = torch.tensor([prefix + response], device=device)
        with precision.context(role):
            hidden = transformer(input_ids=ids, use_cache=False).last_hidden_state[0]
        require(torch.isfinite(hidden).all().item(), 'Nonfinite causal hidden state')
        return modules['targets'].causal_hidden(hidden, len(prefix), len(response)).detach()
    with torch.no_grad():
        model.eval()
        with model.disable_adapter():
            source = forward(encoded['ordinary_prefix'], 'source100_ordinary')
            teacher = None if encoded['teacher_prefix'] is None else forward(encoded['teacher_prefix'], 'teacher')
        model.set_adapter(adapter_name)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.train()  # Same frozen dropout0 current role as original sibling scorer.
        student = forward(encoded['ordinary_prefix'], 'current')
        positions = torch.tensor(encoded['mask'], device=device, dtype=torch.bool).nonzero().flatten()
        rows = []
        for indices in positions.split(32):
            with precision.context('current_head'): sl = head(student[indices])
            with precision.context('source100_ordinary_head'): bl = head(source[indices])
            with precision.context('teacher_head'): tl = None if teacher is None else head(teacher[indices])
            values = scalar_metrics(sl, bl, tl, modules['objective'])
            tokenids = torch.tensor(response, device=device)[indices]
            observed = {name: None if values[name] is None else values[name].gather(-1, tokenids[:, None]).squeeze(-1)
                for name in ('student_logp', 'source_logp', 'teacher_logp')}
            small = {k: None if v is None else v.detach().cpu().tolist() for k, v in values.items() if not k.endswith('_logp')}
            small.update({k.replace('_logp', '_observed_logp'): None if v is None else v.detach().cpu().tolist()
                for k, v in observed.items()})
            for local, pos in enumerate(indices.detach().cpu().tolist()):
                row = dict(token_index=pos, token_id=response[pos], token_text=tokenizer.decode([response[pos]]),
                    token_offsets=encoded['offsets'][pos], region=encoded['regions'][pos], policy_mask=True,
                    original_training_objective=False, actual_generation_scores=False,
                    metrics_scope='posthoc_same_saved_response_teacher_forcing',
                    **{k: None if v is None else v[local] for k, v in small.items()})
                token_record(row); rows.append(row)
        model.eval()
    return summarize_tokens(encoded, rows)
