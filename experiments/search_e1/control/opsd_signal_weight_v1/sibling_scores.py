"""Current-preupdate five-sibling scoring; no optimizer, sampling, or GPU at import.

The region functions below are byte-identical to the audited extract.py region
functions. Scores replay each sibling's OWN history under one shared successful B.
Only the miner's original selected D receives its detached scalar weight.
"""
import copy
import hashlib
import json
import math
import pickle
import random
import re
from contextlib import contextmanager

try:
    from .objective import canonical_sha, encoded_pair, vocabulary_metrics
    from .targets import causal_hidden
except ImportError:
    from objective import canonical_sha, encoded_pair, vocabulary_metrics
    from targets import causal_hidden

VERSION = 'opsd-current-preupdate-five-sibling-scores-v1'
WEIGHTS_VERSION = 'opsd-five-sibling-centered-signal-weights-v1'
AMPLITUDE = .9
NUMERICAL_FLOOR = 1e-6
VARIANT_SIGNALS = {'G_POS': ('answer_gap', 1), 'G_NEG': ('answer_gap', -1),
                   'H_POS': ('student_entropy', 1), 'H_NEG': ('student_entropy', -1)}
TAG = re.compile(r'</?(think|search|answer|information)>')

def character_regions(response, policy_spans):
    """Exact pure category logic reused from opsd_token_region_analysis v1.

    Text labels never redefine the frozen authored-final protocol scorer.
    """
    previous = 0
    for a, b in policy_spans:
        if type(a) != int or type(b) != int or not previous <= a <= b <= len(response):
            raise ValueError('Invalid authored span bounds')
        previous = b
    final = None
    if policy_spans:
        a, b = policy_spans[-1]
        candidates = list(re.finditer(r'<answer>(.*?)</answer>', response[a:b], re.S))
        if candidates:
            m = candidates[-1]; final = (a + m.start(1), a + m.end(1))
    segments = []; stack = []; cursor = 0
    def content(a, b):
        if a >= b:
            return
        category = 'search_query' if stack and stack[-1] == 'search' else 'think_text' if stack and stack[-1] == 'think' else 'other'
        cuts = sorted({a, b} | ({x for x in final if a < x < b} if final else set()))
        for x, y in zip(cuts, cuts[1:]):
            segments.append((x, y, 'final_answer' if final and final[0] <= x and y <= final[1] else category))
    for m in TAG.finditer(response):
        content(cursor, m.start()); segments.append((m.start(), m.end(), 'structural_tags'))
        tag = m.group(1)
        if m.group().startswith('</'):
            if tag in stack:
                stack = stack[:len(stack) - 1 - stack[::-1].index(tag)]
        else:
            stack.append(tag)
        cursor = m.end()
    content(cursor, len(response))
    return segments


def category(offset, segments):
    a, b = offset
    if a == b:
        return 'boundary'
    labels = {label for x, y, label in segments if a < y and b > x}
    return next(iter(labels)) if len(labels) == 1 else 'boundary'



def weights(scores, direction, *, neutral_reason=None):
    """Return five detached scalar weights; each direction uses the same scores.

    Inputs are current-preupdate HF replay scalars, never generation scores.
    A group-level encoding failure must supply neutral_reason for ALL siblings.
    Nonfinite inputs always raise, even when neutral fallback was requested.
    """
    scores = list(scores)
    if len(scores) != 5 or type(direction) is not int or direction not in (-1, 1):
        raise ValueError('Exactly five sibling scores and integer direction +/-1 required')
    if neutral_reason is not None and (not isinstance(neutral_reason, str) or not neutral_reason):
        raise ValueError('Neutral fallback requires an explicit nonempty reason')
    for x in scores:
        if x is not None and (type(x) not in (float, int) or not math.isfinite(x)):
            raise ValueError('A score must be finite numeric or explicit None; never hide NaN')
    valid = [j for j, x in enumerate(scores) if x is not None]
    centered = [0.0] * 5
    scale = None
    mean = None
    raw_maxabs = None
    if neutral_reason is None and len(valid) >= 2:
        # Center differences from an anchor to avoid a large common offset
        # amplifying floating-point error in the group mean.
        anchor = scores[valid[0]]
        delta = [scores[j] - anchor for j in valid]
        if not all(math.isfinite(x) for x in delta):
            raise ValueError('Finite scores overflowed during subtraction')
        mean_delta = math.fsum(delta) / len(valid)
        mean = anchor + mean_delta
        for j, x in zip(valid, delta):
            centered[j] = x - mean_delta
        raw_maxabs = max(abs(x) for x in centered)
        scale = max(NUMERICAL_FLOOR, raw_maxabs)
    elif neutral_reason is None:
        neutral_reason = 'fewer_than_two_available_scores'
    result = [1.0 + direction * AMPLITUDE * x / scale if scale is not None else 1.0
              for x in centered]
    if (not all(math.isfinite(x) and 0.1 - 1e-12 <= x <= 1.9 + 1e-12 for x in result)
            or abs(math.fsum(result) - 5.0) > 1e-10):
        raise ArithmeticError('Weight range/group-mean invariant failed')
    return dict(version=WEIGHTS_VERSION, direction=direction, amplitude=AMPLITUDE,
                numerical_floor=NUMERICAL_FLOOR, scores=scores,
                available_siblings=valid, centered_scores=centered,
                available_score_mean=mean, raw_maxabs=raw_maxabs,
                normalization_scale=scale, weights=result,
                neutral_reason=neutral_reason, group_mean=math.fsum(result) / 5.0)


def selected_batch_summary(selected_weights):
    """Describe actual selected-D scale; do not normalize the optimizer loss."""
    values = list(selected_weights)
    if not values or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in values):
        raise ValueError('A nonempty batch of finite positive weights is required')
    total = math.fsum(values)
    squares = math.fsum(w * w for w in values)
    return dict(actual_pairs=len(values), selected_weight_sum=total,
                selected_weight_sumsq=squares,
                selected_weight_mean=total / len(values),
                selected_weight_min=min(values), selected_weight_max=max(values),
                effective_sample_size=total * total / squares)


def encode_sibling_group(tokenizer, selected_pair, siblings, core, *,
                         training_encoded=None, max_context=8192):
    """Bind all five immutable rows to selected pair's SAME successful reference.

    Pass the trainer's existing selected encoding to retain its exact identity.
    Ancillary sibling failure neutralizes weights, never changes selected-D K.
    """
    rows = sorted(list(siblings), key=lambda row: row['sibling'])
    if len(rows) != 5 or [row['sibling'] for row in rows] != list(range(5)):
        raise ValueError('Five unique original sibling slots required')
    mining = selected_pair.get('mining', {})
    selected = mining.get('student_sibling')
    if type(selected) is not int or not 0 <= selected < 5:
        raise ValueError('Original selected-D sibling identity required')
    row_shas = [canonical_sha(row) for row in rows]
    if mining.get('sibling_row_sha256') != row_shas:
        raise ValueError('Sibling original-row SHA differs from bound miner')
    if selected_pair.get('reference_reward') != 1:
        raise ValueError('Same successful B reference required')
    reference_slot = mining.get('reference_sibling')
    if type(reference_slot) is not int or not 0 <= reference_slot < 5 or (
            rows[reference_slot]['response'] != selected_pair['reference'] or
            rows[reference_slot]['reward'] != 1 or
            mining.get('reference_row_sha256') != row_shas[reference_slot]):
        raise ValueError('Reference B changed from original sibling')
    if rows[selected]['response'] != selected_pair['student'] or (
            rows[selected]['policy_spans'] != selected_pair['student_policy_spans'] or
            rows[selected]['reward'] != selected_pair['student_reward'] or
            mining.get('student_row_sha256') != row_shas[selected]):
        raise ValueError('Selected D changed from miner')
    encoded = []; neutral = []; region_labels = []; answer_positions = []
    for row in rows:
        if any(row[key] != selected_pair[key] for key in ('question_id', 'question', 'answers')):
            raise ValueError('Sibling question/gold identity differs')
        pair = {key: selected_pair[key] for key in
                ('question_id', 'question', 'answers', 'reference', 'reference_reward')}
        pair.update(student=row['response'], student_reward=row['reward'],
                    student_policy_spans=row['policy_spans'])
        item, reason = encoded_pair(tokenizer, pair, core, max_context)
        if reason is not None:
            neutral.append(dict(sibling=row['sibling'], reason=reason))
            encoded.append(None); region_labels.append(None); answer_positions.append([])
            continue
        segments = character_regions(row['response'], row['policy_spans'])
        labels = [category(offset, segments) for offset in item['offsets']]
        positions = [i for i, label in enumerate(labels)
                     if label == 'final_answer' and item['mask'][i]]
        if row['sibling'] == selected and training_encoded is not None:
            for key in ('prefixes', 'response', 'mask', 'offsets'):
                if training_encoded[key] != item[key]:
                    raise ValueError('Selected-D training/scorer encoding differs: '+key)
            item = copy.deepcopy(training_encoded)
        encoded.append(item); region_labels.append(labels); answer_positions.append(positions)
    group = dict(version=VERSION, question_id=selected_pair['question_id'],
        selected_sibling=selected, reference_sibling=reference_slot,
        selected_pair_sha256=canonical_sha(selected_pair),
        reference_response_sha256=hashlib.sha256(selected_pair['reference'].encode()).hexdigest(),
        sibling_row_sha256=row_shas, encoded=encoded,
        encoded_sha256=[None if item is None else item['sha256'] for item in encoded],
        regions=region_labels, answer_positions=answer_positions,
        neutral_reason='sibling_encoding_failure' if neutral else None,
        neutral_details=neutral)
    group['sha256'] = canonical_sha(group)
    return group


def _rng_state(torch):
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                cuda_all=[value.clone() for value in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_initialized() else None)


def _rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state['python'], state['numpy']), protocol=4))
    for name, values in (('torch_cpu', [state['torch_cpu']]),
                         ('cuda_all', state['cuda_all'])):
        digest.update(json.dumps([name, None if values is None else len(values)]).encode())
        for value in values or []:
            digest.update(value.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _rng_equal(left, right, torch):
    import numpy as np
    return (left['python'] == right['python'] and
        left['numpy'][0] == right['numpy'][0] and
        np.array_equal(left['numpy'][1], right['numpy'][1]) and
        left['numpy'][2:] == right['numpy'][2:] and
        torch.equal(left['torch_cpu'], right['torch_cpu']) and
        ((left['cuda_all'] is None and right['cuda_all'] is None) or
         (left['cuda_all'] is not None and right['cuda_all'] is not None and
          len(left['cuda_all']) == len(right['cuda_all']) and
          all(torch.equal(a, b) for a, b in zip(left['cuda_all'], right['cuda_all'])))))


def _restore_rng(state, torch):
    import numpy as np
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['cuda_all'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_all'])


@contextmanager
def preserved_scoring_state(model, precision, targets):
    """No-grad scoring lease; exact mode/adapter/flags/RNG restore on all exits.

    Diagnostic precision events accumulate intentionally; compute role/stage do
    not. CUDA RNG inventory matches trainer's existing all-visible RNG capsule;
    CPU-only calls never initialize CUDA. Parameter/gradient mutation is fatal.
    """
    import torch
    if targets.busy:
        raise ValueError('Scoring cannot overlap a current backward graph')
    active = model.active_adapter
    if active != targets.current_adapter:
        raise ValueError('Current adapter must be active before scoring')
    targets.check_old()
    modules = [(module, module.training) for module in model.modules()]
    disabled_flags = [(module, module._disable_adapters) for module in model.modules()
                      if hasattr(module, '_disable_adapters')]
    params = [(name, p, p.requires_grad, p._version,
               None if p.grad is None else (id(p.grad), p.grad._version))
              for name, p in model.named_parameters()]
    role, stage = precision.role, precision.stage
    before = _rng_state(torch)
    proof = dict(rng_before_sha256=_rng_hash(before), state_restored=False,
                 rng_consumed=False, cuda_rng_scope='all_visible_already_initialized'
                    if before['cuda_all'] is not None else 'not_initialized')
    try:
        with torch.no_grad():
            yield proof
    finally:
        after = _rng_state(torch)
        proof['rng_after_sha256'] = _rng_hash(after)
        proof['rng_consumed'] = not _rng_equal(before, after, torch)
        model.set_adapter(active)
        for _, p, flag, _, _ in params:
            p.requires_grad_(flag)
        for module, training in modules:
            module.training = training
        for module, disabled in disabled_flags:
            module._disable_adapters = disabled
        precision.role, precision.stage = role, stage
        _restore_rng(before, torch)
        targets.check_old()
        if any(p._version != version or
               (None if p.grad is None else (id(p.grad), p.grad._version)) != grad
               for _, p, _, version, grad in params):
            raise ValueError('Scoring mutated model parameters or preexisting gradients')
        if not _rng_equal(before, _rng_state(torch), torch):
            raise ValueError('Scoring RNG restore failed')
        proof['state_restored'] = True
        proof['restored_rng_sha256'] = _rng_hash(_rng_state(torch))
        if proof['rng_consumed']:
            raise ValueError('No-sampling scorer consumed production RNG; state restored')


def _score_encoded(model, encoded, answer_positions, precision, targets, chunk):
    import torch
    head = model.get_base_model().get_output_embeddings()
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError('Original frozen output head required')
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0
           for module in model.modules()):
        raise ValueError('Scoring requires the original dropout0 contract')
    device = next(model.parameters()).device
    prefixes, response = encoded['prefixes'], encoded['response']
    transformer = model.get_base_model().base_model
    def forward(prefix, role):
        ids = torch.tensor([prefix+response], device=device)
        with precision.context(role):
            result = transformer(input_ids=ids, use_cache=False).last_hidden_state[0]
        if not torch.isfinite(result).all().item():
            raise FloatingPointError('Nonfinite '+role+' scoring activation')
        return causal_hidden(result, len(prefix), len(response))
    # Match SharedTargets.hidden's native role-specific mode and causal slice.
    model.eval()
    with model.disable_adapter():
        teacher = forward(prefixes[1], 'teacher').detach()
    targets.activate_current(); model.train()
    student = forward(prefixes[0], 'current')
    positions = torch.tensor(encoded['mask'], device=device, dtype=torch.bool).nonzero().flatten()
    if not len(positions):
        raise ValueError('Empty effective scoring mask')
    wanted = set(answer_positions); entropy = []; raw = []; post = []
    slogp = []; tlogp = []; seen = []; answer_s = []; answer_t = []
    for indices in positions.split(chunk):
        with precision.context('current_head'):
            sl = head(student[indices])
        with precision.context('teacher_head'):
            tl = head(teacher[indices])
        metrics = vocabulary_metrics(sl, tl, 10., use_mixture=False)
        s, t = metrics['student_logp'], metrics['teacher_logp']
        ids = torch.tensor(response, device=device)[indices]
        sv = s.gather(-1, ids[:, None]).squeeze(-1)
        tv = t.gather(-1, ids[:, None]).squeeze(-1)
        ent = -(s.exp()*s).sum(-1)
        if any(not torch.isfinite(value).all().item()
               for value in (sv, tv, ent, metrics['raw'], metrics['post'])):
            raise FloatingPointError('Nonfinite full-vocabulary sibling scores')
        values = indices.detach().cpu().tolist()
        ss, tt = sv.cpu().tolist(), tv.cpu().tolist()
        entropy.extend(ent.cpu().tolist()); raw.extend(metrics['raw'].cpu().tolist())
        post.extend(metrics['post'].cpu().tolist()); slogp.extend(ss); tlogp.extend(tt)
        seen.extend(values)
        for row, position in enumerate(values):
            if position in wanted:
                answer_s.append(ss[row]); answer_t.append(tt[row])
    if set(wanted) - set(seen):
        raise ValueError('Answer positions escaped valid policy mask')
    count = len(seen); answer_count = len(answer_s)
    gap = (math.fsum(a-b for a, b in zip(answer_s, answer_t))/answer_count
           if answer_count else None)
    return dict(valid_policy_tokens=count, answer_valid_tokens=answer_count,
        answer_gap=gap,
        answer_student_logp=None if not answer_count else math.fsum(answer_s)/answer_count,
        answer_teacher_logp=None if not answer_count else math.fsum(answer_t)/answer_count,
        student_entropy=math.fsum(entropy)/count,
        policy_student_logp=math.fsum(slogp)/count,
        policy_teacher_logp=math.fsum(tlogp)/count,
        raw_teacher_reverse_kl=math.fsum(raw)/count,
        post_clip_teacher_loss=math.fsum(post)/count,
        answer_token_indices=list(answer_positions),
        valid_policy_token_indices=seen,
        answer_token_ids=[response[i] for i in answer_positions],
        answer_token_offsets=[encoded['offsets'][i] for i in answer_positions],
        gap_missing_reason='no_valid_final_answer_payload_tokens' if not answer_count else None)


def score_sibling_group(model, group, variant, *, precision, targets,
                        preupdate_identity, chunk=32):
    """Return JSON-safe detached weights; caller persists before training graph.

    All five student/teacher replay forwards use one preupdate model and B.
    No selected-D reselection, optimizer step, hidden sampling, or full-vocab dump.
    """
    if variant not in VARIANT_SIGNALS or chunk != 32:
        raise ValueError('Explicit four-group variant and chunk32 required')
    if not isinstance(preupdate_identity, dict) or not preupdate_identity:
        raise ValueError('Explicit current-preupdate model/Adam identity required')
    if canonical_sha({k:v for k,v in group.items() if k != 'sha256'}) != group['sha256']:
        raise ValueError('Sibling scoring input identity changed')
    scores = []
    with preserved_scoring_state(model, precision, targets) as state:
        if group['neutral_reason'] is None:
            for slot, encoded in enumerate(group['encoded']):
                if encoded is None:
                    raise ValueError('Missing sibling without explicit whole-group fallback')
                item = _score_encoded(model, encoded, group['answer_positions'][slot],
                                      precision, targets, chunk)
                item.update(sibling=slot, original_row_sha256=group['sibling_row_sha256'][slot],
                            encoded_sha256=encoded['sha256'])
                scores.append(item)
        else:
            # Illegal original ancillary encoding never produces fabricated logits.
            scores = [dict(sibling=j, answer_gap=None, student_entropy=None,
                original_row_sha256=group['sibling_row_sha256'][j],
                encoded_sha256=group['encoded_sha256'][j],
                skipped_reason=group['neutral_reason']) for j in range(5)]
        signal, direction = VARIANT_SIGNALS[variant]
        normalized = weights([item[signal] for item in scores], direction,
                             neutral_reason=group['neutral_reason'])
    receipt = dict(schema=VERSION, accepted=True, variant=variant, signal=signal,
        scoring_scope='no_grad_current_preupdate_hf_replay_not_generation_scores',
        original_training_objective=False, selected_D_only_training=True,
        question_id=group['question_id'], group_input_sha256=group['sha256'],
        selected_pair_sha256=group['selected_pair_sha256'],
        reference_response_sha256=group['reference_response_sha256'],
        selected_sibling=group['selected_sibling'],
        preupdate_identity=dict(preupdate_identity), siblings=scores,
        normalization=normalized, selected_score=scores[group['selected_sibling']],
        selected_pair_weight=normalized['weights'][group['selected_sibling']],
        regions=group['regions'], neutral_details=group['neutral_details'],
        state_restore=state)
    receipt['sha256'] = canonical_sha(receipt)
    return receipt


def compare_selected_training_metrics(receipt, token_rows, *, absolute_tolerance=2e-5,
                                      relative_tolerance=2e-5):
    """Compare scorer with actual forward, not sampled processed-token scores.

    Excludes fallback groups (no scorer logits); caller retains this explicit skip.
    Caller must bind receipt preupdate identity to its actual forward header.
    """
    expected = receipt['siblings'][receipt['selected_sibling']]
    if expected.get('skipped_reason') is not None:
        return dict(accepted=True, checked=False, reason=expected['skipped_reason'])
    rows = [row for row in token_rows if 'student_observed_token_logp' in row and
            'teacher_observed_token_logp' in row and 'token_index' in row]
    by_position = {row['token_index']: row for row in rows}
    if (len(by_position) != len(rows) or
            sorted(by_position) != expected['valid_policy_token_indices']):
        raise ValueError('Actual training token inventory differs from sibling scorer')
    answer = [by_position[i] for i in expected['answer_token_indices']]
    actual = dict(student_entropy=math.fsum(row['student_entropy'] for row in rows)/len(rows),
        policy_student_logp=math.fsum(row['student_observed_token_logp'] for row in rows)/len(rows),
        policy_teacher_logp=math.fsum(row['teacher_observed_token_logp'] for row in rows)/len(rows),
        raw_teacher_reverse_kl=math.fsum(row['student_to_teacher_raw_kl'] for row in rows)/len(rows),
        post_clip_teacher_loss=math.fsum(row['post_clip_full_vocabulary_loss'] for row in rows)/len(rows),
        answer_gap=math.fsum(row['student_observed_token_logp']-row['teacher_observed_token_logp']
                            for row in answer)/len(answer) if answer else None)
    differences = {}
    for key, value in actual.items():
        target = expected[key]
        if value is None or target is None:
            if value is not None or target is not None:
                raise ValueError('Scorer/training answer NA differs')
            differences[key] = None
        else:
            if not math.isfinite(value) or not math.isclose(value, target,
                    abs_tol=absolute_tolerance, rel_tol=relative_tolerance):
                raise ValueError('Scorer/actual training distribution differs: '+key)
            differences[key] = value-target
    return dict(accepted=True, checked=True, differences=differences,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)


def compare_score_receipts(first, second, *, absolute_tolerance=2e-5,
                           relative_tolerance=2e-5, weight_tolerance=1e-3):
    """Diagnostic repeat budget, not a theoretical guarantee of score accuracy.

    Same state/encoding/precision/replay shape required. Failed budgets HOLD;
    never silently change amplitude, floor, model, or original selection.
    """
    for key in ('variant', 'signal', 'preupdate_identity', 'group_input_sha256'):
        if first[key] != second[key]:
            raise ValueError('Repeated scorer input/model identity differs: '+key)
    for value in (first, second):
        if value['accepted'] is not True or value['state_restore']['state_restored'] is not True:
            raise ValueError('Repeated scorer lacks restored accepted state')
    deltas = []
    for left, right in zip(first['siblings'], second['siblings']):
        delta = {}
        for key in ('answer_gap', 'student_entropy'):
            a, b = left[key], right[key]
            if a is None or b is None:
                if a is not None or b is not None:
                    raise ValueError('Repeated scorer NA coverage differs')
                delta[key] = None
            else:
                if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(
                        a,b,abs_tol=absolute_tolerance,rel_tol=relative_tolerance):
                    raise ValueError('Repeated scorer score budget exceeded: '+key)
                delta[key] = b-a
        deltas.append(delta)
    weight_deltas = [b-a for a,b in zip(first['normalization']['weights'],
                                       second['normalization']['weights'])]
    if max(abs(value) for value in weight_deltas) > weight_tolerance:
        raise ValueError('Repeated scorer normalized weight budget exceeded')
    return dict(accepted=True, score_deltas=deltas, weight_deltas=weight_deltas,
        absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
        weight_tolerance=weight_tolerance,
        d_over_floor=[None if receipt['normalization']['raw_maxabs'] is None else
            receipt['normalization']['raw_maxabs']/NUMERICAL_FLOOR for receipt in (first,second)])
