"""Bounded-memory, CPU-only analysis of immutable completed OPSD debug probes.

Private output contains identities/statistics, never question/answer/response text.
No torch, tokenizer, model, scoring, GPU or network imports/actions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

VERSIONS = ('source100', 'update1', 'update5', 'epoch1')
REGIONS = ('think_text', 'search_query', 'structural_tags', 'final_answer', 'other', 'boundary')
METRICS = ('teacher_student_clipped_kl', 'teacher_student_unclipped_kl',
           'base_ordinary_to_student_kl', 'clipped_vocabulary_contributions',
           'clipped_teacher_probability_mass')
TAG = re.compile(r'</?(think|search|answer|information)>')
ATOL, RTOL = 2e-5, 2e-4
MAX_LINE = 16 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stat(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def close(a, b):
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite/non-numeric metric')
    return value


def stable_json(path):
    path = Path(path); before = stat(path)
    if before['bytes'] > MAX_LINE:
        raise ValueError('Bounded JSON file limit exceeded')
    data = path.read_bytes()
    if stat(path) != before:
        raise ValueError('Immutable JSON changed during read')
    return json.loads(data), dict(path=str(path.resolve()), stat=before, sha256=digest(data))


def lines(path, prefix=False, proof=None):
    """Streaming lines; prefix mode admits append only and never claims final SHA."""
    path = Path(path); before = stat(path); remaining = before['bytes']; h = hashlib.sha256(); used = 0
    partial = False
    with path.open('rb') as f:
        while remaining:
            raw = f.readline(min(remaining, MAX_LINE + 1))
            if not raw or len(raw) > MAX_LINE:
                raise ValueError('Truncated/oversized JSONL row')
            remaining -= len(raw)
            if not raw.endswith(b'\n'):
                if not prefix:
                    raise ValueError('Incomplete immutable JSONL')
                partial = True; break
            h.update(raw); used += len(raw)
            yield json.loads(raw)
    after = stat(path)
    if prefix:
        if any(after[k] != before[k] for k in ('device', 'inode')) or after['bytes'] < before['bytes']:
            raise ValueError('Results prefix replaced/truncated')
    elif after != before:
        raise ValueError('Immutable JSONL changed during read')
    if proof is not None:
        proof.update(path=str(path.resolve()), stat=before, sha256=h.hexdigest(), bytes_hashed=used,
                     prefix_only=prefix, trailing_partial_line_skipped=partial)


def character_regions(response, policy_spans):
    """Textual categories, independent of the frozen policy mask.

    Exact recognized tag bytes take precedence; content uses innermost open tag.
    Only the last complete answer payload wholly in last authored span is final.
    Malformed/unknown tags remain ordinary text. Any token crossing categories is
    boundary; no category changes validity or invents strict protocol validity.
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


def probe(summary, token_path, response, spans):
    if summary['status'] != 'complete':
        return dict(status='skipped', reason=summary.get('reason', summary['status']))
    m = summary['valid_policy_tokens']; count = summary['response_tokens']
    if type(m) != int or m <= 0 or type(count) != int or count < m:
        raise ValueError('Invalid token denominator')
    has_teacher = summary['reference_status'] == 'successful_original_reference'
    segments = character_regions(response, spans)
    info_spans = [x.span() for x in re.finditer(r'<information>.*?(?:</information>|$)', response, re.S)]
    regions = {r: dict(valid_tokens=0, sums={k: 0.0 if has_teacher or k == 'base_ordinary_to_student_kl' else None for k in METRICS}) for r in REGIONS}
    proof = {}; mask_hash = hashlib.sha256(); total_valid = 0; contribution = 0.; rows = 0
    for row in lines(token_path, proof=proof):
        if row['token_index'] != rows or type(row['token_id']) != int:
            raise ValueError('Token row order/identity mismatch')
        a, b = row['character_offset']
        if type(a) != int or type(b) != int or not 0 <= a <= b <= len(response) or row['text'] != response[a:b]:
            raise ValueError('Token offset/text mismatch')
        if type(row['policy_mask']) != bool or type(row['information_mask']) != bool:
            raise ValueError('Invalid mask type')
        information = any(a < y and b > x for x, y in info_spans)
        expected_policy = b > a and not information and any(x <= a and b <= y for x, y in spans)
        if row['information_mask'] != information or row['policy_mask'] != expected_policy:
            raise ValueError('Recorded mask differs from frozen offset/span rule')
        if row['policy_mask'] and (row['information_mask'] or a == b):
            raise ValueError('Frozen mask includes information/zero-width token')
        if row['policy_mask'] and not any(x <= a and b <= y for x, y in spans):
            raise ValueError('Frozen mask includes token outside authored spans')
        mask_hash.update(canonical([rows, row['token_id'], a, b, row['policy_mask'], row['information_mask']]))
        values = {}
        for key in METRICS:
            value = row[key]
            values[key] = number(value) if has_teacher or key == 'base_ordinary_to_student_kl' else None
            if not has_teacher and key != 'base_ordinary_to_student_kl' and value is not None:
                raise ValueError('No-reference trajectory has fabricated teacher metric')
        if has_teacher:
            clips = values['clipped_vocabulary_contributions']; mass = values['clipped_teacher_probability_mass']
            if type(clips) != int or clips < 0 or not 0 <= mass <= 1 + 1e-6:
                raise ValueError('Invalid clip count/mass')
            actual = number(row['actual_masked_trajectory_loss_contribution'])
            if not close(actual, values['teacher_student_clipped_kl'] / m if row['policy_mask'] else 0.):
                raise ValueError('Frozen masked contribution mismatch')
            contribution += actual
        elif row['actual_masked_trajectory_loss_contribution'] is not None:
            raise ValueError('No-reference loss contribution is not null')
        if row['policy_mask']:
            total_valid += 1; target = regions[category((a, b), segments)]; target['valid_tokens'] += 1
            for key, value in values.items():
                if value is not None:
                    target['sums'][key] += value
        rows += 1
    expected = summary['tokens_file']
    if rows != count or rows != expected['rows'] or total_valid != m or proof['sha256'] != expected['sha256'] or proof['bytes_hashed'] != expected['bytes']:
        raise ValueError('Probe rows/count/hash mismatch')
    means = {}
    for key in METRICS:
        means[key] = sum(r['sums'][key] for r in regions.values()) / m if has_teacher or key == 'base_ordinary_to_student_kl' else None
        if means[key] is not None:
            number(means[key])
    if has_teacher:
        checks = summary['frozen_pair_loss_check']
        if checks.get('passed') is not True or checks.get('per_token_contribution_sum_matches') is not True:
            raise ValueError('Native probe did not pass')
        for x in (summary['trajectory_mean'], summary['frozen_pair_loss'], summary['per_token_contribution_sum_float64'], contribution):
            if not close(means['teacher_student_clipped_kl'], number(x)):
                raise ValueError('Region sum does not reconstruct native trajectory mean')
        if sum(r['sums']['clipped_vocabulary_contributions'] for r in regions.values()) != summary['clipped_policy_vocabulary_contributions']:
            raise ValueError('Clip total disagrees with native summary')
    for r in regions.values():
        n = r['valid_tokens']
        r['token_means'] = {k: v / n if v is not None and n else None for k, v in r['sums'].items()}
        r['trajectory_contributions'] = {k: v / m if v is not None else None for k, v in r['sums'].items()}
    return dict(status='complete', valid_tokens=m, response_tokens=count, means=means, regions=regions,
                response_sha256=digest(response.encode()), policy_spans_sha256=digest(canonical(spans)),
                token_mask_alignment_sha256=mask_hash.hexdigest(), token_file=proof,
                full_vocabulary_dump_status=summary.get('full_vocabulary_dump', {}).get('status'),
                reconstruction_passed=True, teacher_present=has_teacher)


def mean_records(records):
    """Equal trajectories (five hard siblings), then equal questions downstream."""
    if not records or any(x['status'] != 'complete' for x in records):
        return None
    def avg(values):
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None
    return dict(trajectories=len(records), mean_valid_tokens=avg([x['valid_tokens'] for x in records]),
                means={k: avg([x['means'][k] for x in records]) for k in METRICS},
                regions={r: dict(mean_valid_tokens=avg([x['regions'][r]['valid_tokens'] for x in records]),
                  mean_trajectory_contributions={k: avg([x['regions'][r]['trajectory_contributions'][k] for x in records]) for k in METRICS},
                  mean_trajectory_token_means={k: avg([x['regions'][r]['token_means'][k] for x in records]) for k in METRICS},
                  token_mean_nonempty_trajectories=sum(x['regions'][r]['valid_tokens'] > 0 for x in records)) for r in REGIONS})


def teacher_invariance(left_path, right_path, left_proof, right_proof):
    """Two streaming passes, no retained full-vocabulary arrays or token lists."""
    missing = Counter(); comparisons = Counter(); maxima = defaultdict(float); exact = True
    lp = {}; rp = {}; rows = 0
    left = lines(left_path, proof=lp); right = lines(right_path, proof=rp)
    sentinel = object()
    while True:
        a = next(left, sentinel); b = next(right, sentinel)
        if a is sentinel or b is sentinel:
            if a is not b:
                raise ValueError('Teacher invariance row count differs')
            break
        for field in ('token_index', 'token_id', 'character_offset', 'policy_mask', 'information_mask'):
            if a[field] != b[field]:
                raise ValueError('Teacher invariance history/alignment differs')
        qa = a.get('distributions', {}).get('teacher_privileged', {})
        qb = b.get('distributions', {}).get('teacher_privileged', {})
        def compare(field, x, y):
            nonlocal exact
            if x is None or y is None:
                missing[field + (':both' if x is None and y is None else ':one')] += 1
                return
            x = number(x); y = number(y); comparisons[field] += 1
            exact = exact and x == y; maxima[field] = max(maxima[field], abs(x - y))
            if not math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-6):
                raise ValueError('Frozen teacher q differs across common checkpoints: ' + field)
        for field in ('target_logprob', 'target_probability', 'entropy'):
            compare(field, qa.get(field), qb.get(field))
        ta, tb = qa.get('topk'), qb.get('topk')
        if ta is None or tb is None:
            missing['topk' + (':both' if ta is None and tb is None else ':one')] += 1
        else:
            if len(ta) != len(tb):
                raise ValueError('Frozen teacher topk count differs')
            comparisons['topk_rows'] += 1
            for x, y in zip(ta, tb):
                if x.get('token_id') is None or y.get('token_id') is None:
                    missing['topk_token_id'] += 1
                elif x['token_id'] != y['token_id']:
                    raise ValueError('Frozen teacher topk IDs differ')
                for field in ('logprob', 'probability'):
                    compare('topk_' + field, x.get(field), y.get(field))
        rows += 1
    for actual, expected in ((lp, left_proof), (rp, right_proof)):
        if actual != expected:
            raise ValueError('Teacher invariant comparison input changed after probe audit')
    return dict(status='available_fields_invariant_with_missing_fields' if missing else 'verified',
                missing_fields=dict(missing), comparisons=dict(comparisons), maximum_absolute_difference=dict(maxima),
                all_available_numeric_fields_exact=exact, numeric_atol=1e-6, numeric_rtol=1e-6,
                rows=rows, scope='Same preserved response/ref/frozen base only; not teacher own-generation history')


def index_cases(path):
    """Keep offsets and hashes, not 192 full question/five-trajectory payloads."""
    path = Path(path); before = stat(path); h = hashlib.sha256(); cases = {}; offset = 0; questions = set()
    with path.open('rb') as f:
        for raw in iter(lambda: f.readline(MAX_LINE + 1), b''):
            if len(raw) > MAX_LINE or not raw.endswith(b'\n') or len(cases) >= 4096:
                raise ValueError('Case row/count bound exceeded or incomplete')
            case = json.loads(raw)
            if digest(canonical({k: v for k, v in case.items() if k != 'case_semantic_sha256'})) != case['case_semantic_sha256'] or case['case_id'] in cases:
                raise ValueError('Case semantic identity mismatch/duplicate')
            if case['question_id'] in questions or re.fullmatch(r'[A-Za-z0-9_-]+', case['case_id']) is None:
                raise ValueError('Duplicate question/unsafe case ID')
            questions.add(case['question_id'])
            cases[case['case_id']] = (offset, len(raw), digest(raw)); offset += len(raw); h.update(raw)
    if stat(path) != before:
        raise ValueError('Cases changed during index')
    return cases, dict(path=str(path.resolve()), stat=before, sha256=h.hexdigest())


def indexed_case(path, index, proof):
    if stat(path) != proof['stat']:
        raise ValueError('Indexed case source changed')
    offset, size, wanted = index
    with Path(path).open('rb') as f:
        f.seek(offset); raw = f.read(size)
    if digest(raw) != wanted or stat(path) != proof['stat']:
        raise ValueError('Indexed case row changed')
    return json.loads(raw)


def run(cases_dir, run_dir, output, protocol_cases=None):
    cases_dir = Path(cases_dir); run_dir = Path(run_dir); output = Path(output)
    if output.exists():
        raise FileExistsError('Output must be a new exclusive directory')
    helper_path = Path(__file__).resolve(); helper_stat = stat(helper_path)
    helper_hash = digest(helper_path.read_bytes())
    if stat(helper_path) != helper_stat:
        raise ValueError('Helper source changed during read')
    manifest, manifest_proof = stable_json(cases_dir / 'manifest.json')
    cases, case_proof = index_cases(cases_dir / 'cases.jsonl')
    if case_proof['sha256'] != manifest['cases']['sha256']:
        raise ValueError('Case manifest mismatch')
    config, config_proof = stable_json(run_dir / 'config.json')
    protocol = {}; protocol_proof = {}
    if protocol_cases:
        for item in lines(protocol_cases, proof=protocol_proof):
            if item['case_id'] in protocol or item['case_id'] not in cases:
                raise ValueError('Duplicate protocol case')
            protocol[item['case_id']] = dict(source_case_sha256=item['source_case_sha256'],
                completed_case_result_sha256=item['completed_case_result_sha256'],
                versions={v: {k: item['versions'][v][k] for k in ('strict_em', 'trace', 'final_answer_protocol_valid', 'invalid_tag_candidate_matches_gold', 'termination')} for v in VERSIONS})
    skips = Counter(); seen = set(); prefixes = []; cells = defaultdict(list); generated_cells = defaultdict(list); flips = defaultdict(list)
    teacher_checks = Counter(); teacher_missing = Counter()
    output.mkdir(parents=True, exist_ok=False)
    output_file = output / 'cases.jsonl'; out_hash = hashlib.sha256(); result_count = 0
    try:
        with output_file.open('xb') as out:
            for result_path in sorted(run_dir.glob('worker-*/case-results.jsonl')):
                prefix = {}
                for result in lines(result_path, prefix=True, proof=prefix):
                    if result.get('status') != 'complete':
                        skips['incomplete_result'] += 1; continue
                    key = result['case_id']
                    if key in seen or key not in cases:
                        raise ValueError('Duplicate/foreign completed case')
                    case = indexed_case(cases_dir / 'cases.jsonl', cases[key], case_proof); seen.add(key); directory = result_path.parent / key
                    if result['source_case_sha256'] != case['case_semantic_sha256'] or (result['dataset'], result['group']) != (case['dataset'], case['group']):
                        raise ValueError('Completed case identity mismatch')
                    pair = case['original_pair']; hard = pair is None
                    trajectories = [("original-mined-student", pair['row']['student'], pair['row']['student_policy_spans'])] if pair else [("original-failed-sibling-" + str(x['row']['sibling']), x['row']['response'], x['row']['policy_spans']) for x in case['original_siblings']]
                    if hard and (len(trajectories) != 5 or len({x[0] for x in trajectories}) != 5):
                        raise ValueError('Hard case must contain exactly five unique siblings')
                    item = dict(case_id=key, question_id=case['question_id'], dataset=case['dataset'], group=case['group'],
                                source_case_sha256=case['case_semantic_sha256'], result_sha256=digest(canonical(result)),
                                reference_sha256=digest(pair['row']['reference'].encode()) if pair and 'reference' in pair['row'] else None,
                                original={}, generated={}, strict_em={}, common_original_history=False,
                                teacher_invariance={})
                    alignment = {}; all_common = True
                    for version in VERSIONS:
                        original = []
                        for name, response, spans in trajectories:
                            summary, proof = stable_json(directory / ('token-summary-' + version + '-' + name + '.json'))
                            if summary['case_id'] != key or summary['version'] != version or summary['response_origin'] != ('preserved_failed_sibling' if hard else 'preserved_student') or summary['preserved_trajectory_name'] != name:
                                raise ValueError('Summary origin/identity mismatch')
                            if summary['original_training_objective'] != (not hard):
                                raise ValueError('Original objective label mismatch')
                            embedded = {x['preserved_trajectory_name']: x for x in result['versions'][version]['token_probes']}
                            if embedded.get(name) != summary:
                                raise ValueError('Saved summary differs from completed result')
                            analyzed = probe(summary, directory / ('tokens-' + version + '-' + name) / 'tokens.jsonl', response, spans)
                            analyzed.update(trajectory=name, response_origin=summary['response_origin'], summary_file=proof)
                            original.append(analyzed)
                            if analyzed['status'] != 'complete':
                                skips['probe_skip:' + summary['response_origin'] + ':' + version + ':' + analyzed['reason']] += 1; all_common = False
                            else:
                                identity = tuple(analyzed[k] for k in ('response_sha256', 'policy_spans_sha256', 'token_mask_alignment_sha256'))
                                if name in alignment and identity != alignment[name]:
                                    raise ValueError('Four-checkpoint original history/token/mask mismatch')
                                alignment[name] = identity
                                if not hard and version != 'source100':
                                    baseline = next(x for x in item['original']['source100']['trajectories'] if x['trajectory'] == name)
                                    if baseline['status'] == 'complete':
                                        item['teacher_invariance'][version + ':' + name] = teacher_invariance(
                                            directory / ('tokens-source100-' + name) / 'tokens.jsonl',
                                            directory / ('tokens-' + version + '-' + name) / 'tokens.jsonl',
                                            baseline['token_file'], analyzed['token_file'])
                                        invariant = item['teacher_invariance'][version + ':' + name]
                                        teacher_checks[invariant['status']] += 1; teacher_missing.update(invariant['missing_fields'])
                            skips['full_vocab_status:' + summary['response_origin'] + ':' + version + ':' + str(analyzed.get('full_vocabulary_dump_status'))] += 1
                        item['original'][version] = dict(trajectories=original, question_average=mean_records(original))
                        generation, gen_proof = stable_json(directory / ('student-generation-' + version + '.json'))
                        if generation['active_adapter'] != version or generation['privileged_reference'] is not False or generation['kind'] != 'actual_autoregressive_generation' or generation['teacher_forced'] is not False:
                            raise ValueError('Generation role/origin mismatch')
                        em = generation['protocol_em']
                        if em not in (0, 1) or result['versions'][version]['generation_em'] != em:
                            raise ValueError('Recorded strict EM mismatch')
                        item['strict_em'][version] = em
                        summary, proof = stable_json(directory / ('token-summary-' + version + '-actual-student-generation.json'))
                        if (summary['case_id'], summary['version'], summary['response_origin']) != (key, version, 'new_student_generation') or summary['original_training_objective'] is not False:
                            raise ValueError('Generated probe identity mismatch')
                        if summary != result['versions'][version]['actual_student_response_teacher_forced_probe']:
                            raise ValueError('Generated summary differs from completed result')
                        analyzed = probe(summary, directory / ('tokens-' + version + '-actual-student-generation') / 'tokens.jsonl', generation['response'], generation['policy_spans'])
                        analyzed.update(response_origin='new_student_generation', summary_file=proof, generation_file=gen_proof,
                                        paired_checkpoint_loss_comparison_allowed=False)
                        item['generated'][version] = analyzed
                        if analyzed['status'] == 'complete':
                            generated_cells[(version, case['dataset'], case['group'])].append(mean_records([analyzed]))
                        else:
                            skips['probe_skip:new_student_generation:' + version + ':' + analyzed['reason']] += 1
                        skips['full_vocab_status:new_student_generation:' + version + ':' + str(analyzed.get('full_vocabulary_dump_status'))] += 1
                    # Teacher's own new history is separate, never the training q or a checkpoint pair.
                    teacher, teacher_proof = stable_json(directory / 'teacher-generation.json')
                    if hard:
                        if teacher.get('status') != 'not_applicable_no_successful_reference':
                            raise ValueError('Hard case has fabricated teacher')
                        item['generated']['teacher'] = dict(status='skipped', reason='no_successful_reference', generation_file=teacher_proof)
                        skips['probe_skip:new_teacher_generation:no_successful_reference'] += 1
                    else:
                        if teacher.get('active_adapter') != 'disabled' or teacher.get('privileged_reference') is not True:
                            raise ValueError('Teacher identity mismatch')
                        summary, proof = stable_json(directory / 'token-summary-source100-actual-teacher-generation.json')
                        if (summary['case_id'], summary['version'], summary['response_origin']) != (key, 'source100', 'new_teacher_generation') or summary['original_training_objective'] is not False:
                            raise ValueError('Teacher probe identity mismatch')
                        if summary != result['actual_teacher_response_teacher_forced_probe']:
                            raise ValueError('Teacher summary differs from completed result')
                        analyzed = probe(summary, directory / 'tokens-source100-actual-teacher-generation' / 'tokens.jsonl', teacher['response'], teacher['policy_spans'])
                        analyzed.update(response_origin='new_teacher_generation', generation_file=teacher_proof, summary_file=proof,
                                        represents_training_teacher_distribution=False)
                        item['generated']['teacher'] = analyzed
                        if analyzed['status'] == 'complete':
                            generated_cells[('teacher', case['dataset'], case['group'])].append(mean_records([analyzed]))
                        else:
                            skips['probe_skip:new_teacher_generation:' + analyzed['reason']] += 1
                        skips['full_vocab_status:new_teacher_generation:' + str(analyzed.get('full_vocabulary_dump_status'))] += 1
                    if key in protocol:
                        pc = protocol[key]
                        if pc['source_case_sha256'] != case['case_semantic_sha256'] or pc['completed_case_result_sha256'] != item['result_sha256']:
                            raise ValueError('Optional protocol output provenance mismatch')
                        item['protocol'] = {}
                        for v in VERSIONS:
                            pv = pc['versions'][v]
                            if pv['strict_em'] != item['strict_em'][v] or pv['trace']['sha256'] != item['generated'][v]['generation_file']['sha256']:
                                raise ValueError('Optional protocol EM/trace identity mismatch')
                            item['protocol'][v] = {k: pv[k] for k in ('final_answer_protocol_valid', 'invalid_tag_candidate_matches_gold', 'termination')}
                    elif protocol_cases:
                        skips['optional_protocol_case_missing'] += 1
                    item['common_original_history'] = all_common
                    item['paired_changes'] = {}
                    if all_common:
                        for v in VERSIONS:
                            cells[(v, case['dataset'], case['group'])].append(item['original'][v]['question_average'])
                        base = item['original']['source100']['question_average']['means']
                        for v in VERSIONS[1:]:
                            target = item['original'][v]['question_average']['means']
                            transition = str(item['strict_em']['source100']) + '->' + str(item['strict_em'][v])
                            change = {k: target[k] - base[k] if target[k] is not None else None for k in METRICS}
                            item['paired_changes'][v] = dict(original_history_mean_metric_change=change, strict_flip=transition)
                            flips[(v, case['dataset'], case['group'], transition)].append(change)
                    else:
                        skips['case_excluded_from_four_checkpoint_common_original'] += 1
                    data = canonical(item) + b'\n'; out.write(data); out_hash.update(data); result_count += 1
                prefixes.append(prefix)
                skips['trailing_partial_result_line'] += int(prefix['trailing_partial_line_skipped'])
        def aggregate(items):
            return dict(questions=len(items), means={k: sum(x['means'][k] for x in items) / len(items) if items and all(x['means'][k] is not None for x in items) else None for k in METRICS},
                        regions={r: dict(mean_question_token_count=sum(x['regions'][r]['mean_valid_tokens'] for x in items) / len(items) if items else None,
                          mean_question_trajectory_contributions={k: sum(x['regions'][r]['mean_trajectory_contributions'][k] for x in items) / len(items) if items and all(x['regions'][r]['mean_trajectory_contributions'][k] is not None for x in items) else None for k in METRICS},
                          mean_question_trajectory_token_means={k: sum(x['regions'][r]['mean_trajectory_token_means'][k] for x in items if x['regions'][r]['mean_trajectory_token_means'][k] is not None) / sum(x['regions'][r]['mean_trajectory_token_means'][k] is not None for x in items) if any(x['regions'][r]['mean_trajectory_token_means'][k] is not None for x in items) else None for k in METRICS},
                          token_mean_nonempty_questions=sum(x['regions'][r]['mean_trajectory_token_means']['base_ordinary_to_student_kl'] is not None for x in items)) for r in REGIONS})
        def tables(storage):
            totals = defaultdict(list)
            for (v, d, g), values in storage.items():
                totals[v].extend(values)
            return dict(cells=[dict(version=v, dataset=d, group=g, **aggregate(x)) for (v, d, g), x in sorted(storage.items())],
                        overall={v: aggregate(x) for v, x in sorted(totals.items())})
        associations = []
        overall_flips = defaultdict(list)
        for (v, d, g, flip), values in sorted(flips.items()):
            overall_flips[(v, flip)].extend(values)
            associations.append(dict(version=v, dataset=d, group=g, strict_flip=flip, questions=len(values),
                                     mean_original_history_metric_change={k: sum(x[k] for x in values) / len(values) if all(x[k] is not None for x in values) else None for k in METRICS}))
        skips['selected_not_yet_complete'] = len(cases) - len(seen)
        if stat(helper_path) != helper_stat:
            raise ValueError('Helper source changed during analysis')
        summary = dict(schema='opsd-token-region-analysis-v1', run_id=config['run_id'], created_unix=time.time(),
                       interpretation='Loss contributions are not gradients; posthoc fixed-response loss is not historical online loss. Contributions average all five hard trajectories, including zero region contribution; region token means average only nonempty trajectories, with counts. Then equal questions. Stratified training cases, not native512/test estimates.',
                       regions_definition='Exact tags precede innermost open think/search content; last complete answer payload inside last authored span is final_answer, not a protocol validity claim; mixed-category or zero-width tokens are boundary. Frozen mask alone selects supervision.',
                       completed_cases=result_count, selected_cases=len(cases), skip_counts=dict(skips),
                       original_common_history=tables(cells), new_different_responses_unpaired=tables(generated_cells),
                       paired_change_strict_flip_associations=associations,
                       original_common_history_teacher_invariance=dict(checks_by_status=dict(teacher_checks), missing_field_token_counts=dict(teacher_missing),
                         interpretation='Only observed target/topk/logprob/entropy equality, not full-vocabulary proof; missing fields are not evidence of equality. No invariant requirement for new own-history responses.'),
                       overall_paired_change_strict_flip_associations=[dict(version=v, strict_flip=flip, questions=len(values),
                         mean_original_history_metric_change={k: sum(x[k] for x in values) / len(values) if all(x[k] is not None for x in values) else None for k in METRICS}) for (v, flip), values in sorted(overall_flips.items())],
                       source=dict(cases=case_proof, case_manifest=manifest_proof, run_config=config_proof,
                                   producer_source_manifest=config.get('source_manifest'),
                                   completed_result_prefixes=prefixes, optional_protocol=protocol_proof,
                                   helper=dict(path=str(helper_path), sha256=helper_hash, stat=helper_stat)),
                       cases_output=dict(path=str(output_file.resolve()), sha256=out_hash.hexdigest(), rows=result_count))
        with (output / 'summary.json').open('x') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        return summary
    except Exception as exc:
        with (output / 'failed.json').open('x') as f:
            json.dump(dict(status='failed_partial_not_accepted', error_type=type(exc).__name__, error=str(exc)), f)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases-dir', 'run-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-cases', type=Path)
    args = parser.parse_args()
    result = run(args.cases_dir, args.run_dir, args.output, args.protocol_cases)
    print(json.dumps(dict(completed_cases=result['completed_cases'], output=str(args.output))))


if __name__ == '__main__':
    main()
