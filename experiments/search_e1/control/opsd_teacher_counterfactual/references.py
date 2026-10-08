"""Deterministic CPU selection/reference interventions, using frozen QA/core."""
from collections import Counter
import hashlib
import re

try:
    from .state import sha
except ImportError:
    from state import sha

CELLS = [(d, g) for d in ('nq', 'hotpotqa') for g in ('correct_correct', 'correct_wrong')]


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else None


def select(cases, seed=42):
    if seed != 42:
        raise ValueError('Fixed M3 seed42')
    if len({c['case_id'] for c in cases}) != len(cases) or len({c['question_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate cases/questions')
    selected = []
    for dataset, group in CELLS:
        values = [c for c in cases if (c['dataset'], c['group']) == (dataset, group)]
        if len(values) != 32:
            raise ValueError('M1 successful-reference cell must contain32')
        # M1 was already chosen by this seed/hash, so use the approved same key.
        for case in values:
            expected = hashlib.sha256(f"{seed}|{dataset}|{group}|{case['question_id']}".encode()).hexdigest()
            if case['selection_sha256'] != expected:
                raise ValueError('Original fixed seed/hash identity differs')
        chosen = sorted(values, key=lambda c: (c['selection_sha256'], c['question_id']))[:16]
        selected.extend(dict(case=case, pilot=i < 4, selected_cell_index=i) for i, case in enumerate(chosen))
    return selected


def reference_row(case, core):
    pair = case['original_pair']
    if pair is None or case['group'] == 'all5_wrong':
        raise ValueError('No successful teacher reference')
    items = sorted(case['original_siblings'], key=lambda x: x['provenance']['line'])
    rows = [x['row'] for x in items]
    if len(rows) != 5 or sorted(x['sibling'] for x in rows) != list(range(5)):
        raise ValueError('Reference needs five original source siblings')
    correct = [r for r in rows if r['reward'] == 1]
    if not correct:
        raise ValueError('Reference fabricated')
    chosen = min(correct, key=lambda r: (r['search_calls'], r.get('token_count', len(r['response']))))
    if chosen['response'] != pair['row']['reference'] or core.reward(chosen['response'], chosen['answers'], chosen['policy_spans']) != 1:
        raise ValueError('Native first-in-source-order reference differs from actual pair')
    return chosen


def delete_final(row, qa):
    text = row['response']; spans = row['policy_spans']; authored = [text[a:b] for a, b in spans]
    answer = qa.extract_final_answer(authored)
    if answer is None or not spans:
        raise ValueError('Reference final authored answer is not accepted')
    start, end = spans[-1]
    candidates = list(re.finditer(r'<answer>(.*?)</answer>', text[start:end], re.S))
    if len(candidates) != 1 or candidates[0].group(1).strip() != answer:
        raise ValueError('Cannot locate unique accepted final answer block')
    a, b = start + candidates[0].start(), start + candidates[0].end()
    return text[:a] + text[b:], dict(deleted_character_range=[a, b], original_reference_sha256=text_sha(text),
                                    changed_reference_sha256=text_sha(text[:a] + text[b:]),
                                    interpretation='Final authored answer block only; earlier/info answer information retained')


def conditions(selected, core, qa):
    refs = {x['case']['case_id']: reference_row(x['case'], core) for x in selected}
    by_cell = {}
    for x in selected:
        c = x['case']; by_cell.setdefault((c['dataset'], c['group']), []).append(c)
    result = {}
    for x in selected:
        case = x['case']; key = case['case_id']; row = refs[key]
        deleted, deletion = delete_final(row, qa)
        values = by_cell[(case['dataset'], case['group'])]; index = next(i for i, c in enumerate(values) if c['case_id'] == key)
        donor = None
        answers = case['source_train_row']['row']['answers']; answers = [answers] if isinstance(answers, str) else answers
        for step in range(1, len(values)):
            candidate = values[(index + step) % len(values)]; rr = refs[candidate['case_id']]
            answer = qa.extract_final_answer([rr['response'][a:b] for a, b in rr['policy_spans']])
            if candidate['question_id'] != case['question_id'] and answer is not None and not any(core.normalize(answer) == core.normalize(a) for a in answers):
                donor = candidate; break
        result[key] = dict(reference_sibling=row['sibling'], reference_policy_spans=row['policy_spans'],
            conditions=[dict(name='original', reference=row['response'], reference_sha256=text_sha(row['response']), status='ready'),
                        dict(name='none', reference=None, reference_sha256=None, status='ready'),
                        dict(name='remove_final_answer', reference=deleted, reference_sha256=text_sha(deleted), status='ready', deletion=deletion),
                        dict(name='mismatched', reference=refs[donor['case_id']]['response'] if donor else None,
                             reference_sha256=text_sha(refs[donor['case_id']]['response']) if donor else None,
                             donor_case_id=donor['case_id'] if donor else None, donor_question_id=donor['question_id'] if donor else None,
                             donor_source_case_sha256=donor['case_semantic_sha256'] if donor else None,
                             status='ready' if donor else 'skipped', reason=None if donor else 'no_other_question_different_gold_answer_donor',
                             interpretation='Cross-question reference; not validated for recipient')])
    return result
