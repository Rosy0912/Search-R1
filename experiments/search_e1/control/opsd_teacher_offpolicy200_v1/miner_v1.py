"""Versioned B/D selection shared by sampling, resume, and trainer verification.

The supplied `core` must be the caller's SHA-bound original frozen scientific
module. Reward is called directly; no simplified or duplicated scorer exists.
"""
from collections import Counter
import hashlib
import json

try:
    from .weights_v1 import correctness_weights, WEIGHTS_VERSION
except ImportError:
    from weights_v1 import correctness_weights, WEIGHTS_VERSION

MINER_VERSION = 'search-e1-opsd-ablation-miner-v1'
RANDOM_VERSION = 'search-e1-opsd-uniform-five-sha256-u64-rejection-v1'
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
RANDOM_VARIANTS = ('R', 'W', 'J')
WEIGHTED_VARIANTS = ('W', 'J')
NATIVE_FIELDS = ('question_id', 'question', 'answers', 'reference', 'student',
                 'reference_reward', 'student_reward', 'student_policy_spans')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(raw):
    return hashlib.sha256(raw).digest()


def uniform_sibling(question_id, outer_index):
    """Exact unbiased rejection of 2**64-1, not direct digest % 5."""
    if not isinstance(question_id, str) or not question_id or type(outer_index) is not int or not 0 <= outer_index < 200:
        raise ValueError('Explicit question ID and outer0..9 required')
    limit = (1 << 64) - ((1 << 64) % 5)
    counter = 0; rejected = []
    while True:
        # Each fresh counter is its own canonical SHA256 domain-separated word.
        payload = [RANDOM_VERSION, 42, outer_index, question_id, counter]
        digest = _digest(canonical(payload))
        if len(digest) != 32:
            raise ValueError('SHA256 stream word has wrong shape')
        word = int.from_bytes(digest[:8], 'big')
        if word < limit:
            selected = word % 5
            return selected, dict(version=RANDOM_VERSION, algorithm='SHA256', seed=42,
                encoding='UTF8 canonical JSON list [version,seed,outer,question_id,counter]; ensure_ascii=false; separators=comma/colon',
                outer_index=outer_index, question_id=question_id, word_bits=64,
                word_endianness='big', digest_slice=[0, 8], rejection_limit=limit,
                rejected_attempts=counter, rejected_sha256=rejected, accepted_counter=counter,
                accepted_sha256=digest.hex(), accepted_word_hex=digest[:8].hex(), selected_sibling=selected)
        rejected.append(digest.hex()); counter += 1


def _groups(rows, core):
    groups = {}; slots = set()
    for row in rows:
        qid = row['question_id']; sibling = row['sibling']
        if not isinstance(qid, str) or not qid or type(sibling) is not int or not 0 <= sibling < 5 or (qid, sibling) in slots:
            raise ValueError('Duplicate/invalid question-sibling identity')
        if type(row['reward']) is not int or row['reward'] not in (0, 1):
            raise ValueError('Stored reward must be native strict integer0/1')
        # Original authored spans and strict final-action scorer are authoritative.
        if row['reward'] != core.reward(row['response'], row['answers'], row['policy_spans']):
            raise ValueError('Stored reward differs from frozen strict authored-final scorer')
        if type(row['search_calls']) is not int or not 0 <= row['search_calls'] <= 4:
            raise ValueError('Invalid original executed search count')
        if 'token_count' in row and (type(row['token_count']) is not int or row['token_count'] < 0):
            raise ValueError('Present token_count must be a nonnegative integer; no silent fallback')
        if not isinstance(row['response'], str):
            raise ValueError('Response must preserve the original complete string')
        group = groups.setdefault(qid, [])
        if group and any(row[k] != group[0][k] for k in ('question', 'answers')):
            raise ValueError('Sibling question/gold identity differs')
        group.append(row); slots.add((qid, sibling))
    if any(len(g) != 5 or {r['sibling'] for r in g} != set(range(5)) for g in groups.values()):
        raise ValueError('Every question must have all five unique native siblings')
    return groups


def mine_pairs(rows, variant, outer_index, core):
    if variant not in VARIANTS or type(outer_index) is not int or not 0 <= outer_index < 200:
        raise ValueError('Explicit seven-group variant and outer0..9 required')
    from rapidfuzz.distance import Levenshtein
    rows = list(rows); groups = _groups(rows, core)
    pairs = []; classes = Counter(); counts = Counter(); native_projection = []
    for qid, siblings in groups.items():
        correct = [r for r in siblings if r['reward'] == 1]
        classes['all_wrong' if not correct else 'all_correct' if len(correct) == 5 else 'mixed'] += 1
        if not correct:
            continue  # Preserve all raw siblings; no fabricated successful B.
        # Deliberately preserve source-line ordering and Python first-tie behavior.
        reference = min(correct, key=lambda r: (r['search_calls'], r.get('token_count', len(r['response']))))
        ordered = sorted(siblings, key=lambda r: r['sibling'])
        correctness = correctness_weights([r['reward'] for r in ordered])
        random_draw = None
        if variant in RANDOM_VARIANTS:
            selected, random_draw = uniform_sibling(qid, outer_index)
            student = ordered[selected]  # All slots0..4, including reference itself.
            selection_kind = 'uniform_all_five_including_reference'
        else:
            candidates = [r for r in siblings if r is not reference and r['reward'] == 0]
            selection_kind = 'fail_preferred_farthest_full_response'
            if not candidates:
                candidates = [r for r in siblings if r is not reference]
                selection_kind = 'all_correct_native_success_fallback_farthest'
            student = max(candidates, key=lambda r: Levenshtein.normalized_distance(reference['response'], r['response']))
        native = dict(question_id=qid, question=reference['question'], answers=reference['answers'],
                      reference=reference['response'], student=student['response'],
                      reference_reward=reference['reward'], student_reward=student['reward'],
                      student_policy_spans=student.get('policy_spans'))
        native_projection.append(native)
        self_pair = reference['sibling'] == student['sibling']
        text_equal = reference['response'] == student['response']
        pair_weight = correctness['weights'][student['sibling']] if variant in WEIGHTED_VARIANTS else 1.0
        cc = student['reward'] == 1
        counts['CC' if cc else 'CW'] += 1
        counts['self_pair'] += int(self_pair); counts['response_text_equal'] += int(text_equal)
        mining = dict(version=MINER_VERSION, variant=variant, outer_index=outer_index,
            selection_kind=selection_kind, source_order_siblings=[r['sibling'] for r in siblings],
            reference_sibling=reference['sibling'], student_sibling=student['sibling'],
            reference_row_sha256=object_sha(reference), student_row_sha256=object_sha(student),
            sibling_row_sha256=[object_sha(r) for r in ordered], rewards_by_sibling=correctness['rewards'],
            self_pair_by_sibling=self_pair, response_text_equal=text_equal, pair_class='CC' if cc else 'CW',
            full_response_normalized_levenshtein=Levenshtein.normalized_distance(reference['response'], student['response']),
            random_draw=random_draw, correctness=correctness,
            weights_applied=variant in WEIGHTED_VARIANTS, selected_pair_weight=pair_weight,
            selected_D_reward=student['reward'], weight_semantics='Positive D weight outside clipped trajectory mean; divide by actual pair count, never weight sum')
        pairs.append(dict(native, pair_weight=pair_weight, mining_version=MINER_VERSION, mining=mining))
    if variant not in RANDOM_VARIANTS and native_projection != core.mine_pairs(rows):
        raise ValueError('B/native D selection differs from original exact core.mine_pairs shadow')
    if len(pairs) != len(groups) - classes['all_wrong']:
        raise ValueError('Eligible per-question pair accounting differs')
    return pairs, dict(version=MINER_VERSION, random_version=RANDOM_VERSION, weights_version=WEIGHTS_VERSION,
        variant=variant, outer_index=outer_index, questions=len(groups), generated_trajectories=len(rows),
        P=len(pairs), classes={k: classes[k] for k in ('all_wrong', 'all_correct', 'mixed')},
        pair_classes={k: counts[k] for k in ('CC', 'CW')},
        self_pair=counts['self_pair'], response_text_equal=counts['response_text_equal'], all_wrong_skipped=classes['all_wrong'],
        all_correct_kept=classes['all_correct'], pair_weight_sum=sum(p['pair_weight'] for p in pairs),
        weight_normalization='Each five-sibling mean1 only; global optimization denominator is actual pairs',
        raw_rows_sha256=object_sha(rows), pairs_sha256=object_sha(pairs),
        native_selection_shadow_checked=variant not in RANDOM_VARIANTS,
        K_scope='P is eligible pairs; usable K/context/mask skips belong to frozen encoding downstream')


def verify_pairs(rows, pairs, variant, outer_index, core, expected_version=MINER_VERSION):
    if expected_version != MINER_VERSION:
        raise ValueError('Miner version differs across sampling/resume/trainer')
    regenerated, stats = mine_pairs(rows, variant, outer_index, core)
    if pairs != regenerated:
        raise ValueError('Saved pairs/weights/identities differ from exact versioned miner')
    return stats


def verify_rollouts(rows, questions, variant, outer_index, core):
    """Production wrapper: bind the full512x5 generation to frozen cohort rows."""
    questions = list(questions); rows = list(rows)
    expected = {q['question_id']: q for q in questions}
    if len(questions) != 512 or len(expected) != 512 or len(rows) != 2560:
        raise ValueError('Production sampling requires all512 unique questions x five siblings')
    if any(r['question_id'] not in expected or
           any(r[k] != expected[r['question_id']][k] for k in ('question', 'answers')) for r in rows):
        raise ValueError('Generated question/gold does not match the frozen TRAIN cohort')
    pairs, stats = mine_pairs(rows, variant, outer_index, core)
    if stats['questions'] != 512 or {r['question_id'] for r in rows} != set(expected):
        raise ValueError('Generated slots do not completely cover the frozen cohort')
    return pairs, stats
