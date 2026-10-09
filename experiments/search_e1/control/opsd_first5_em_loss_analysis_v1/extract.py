#!/usr/bin/env python3
"""Read only 15 immutable native outers; emit compact private NDJSON features.

No Torch, model, optimizer, HTTP, sampling, Node write, or token-copy operation.
Production execution requires prior publication and an explicit SHA-bound request.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

GROUPS = ('C0', 'C1', 'M')
REGIONS = ('all_policy', 'think_text', 'search_query', 'structural_tags',
           'final_answer', 'other', 'boundary')
TAG = re.compile(r'</?(think|search|answer|information)>')
CORE_SHA = '0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc'
QA_SHA = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
MAX_LINE = 16 << 20
METRICS = {
    'teacher_kl': 'student_to_teacher_raw_kl',
    'outer_old_kl': 'student_to_outer_old_raw_kl',
    'mixed_kl': 'student_to_mixture_raw_kl',
    'target_raw_kl': 'pre_clip_full_vocabulary_kl',
    'target_post_loss': 'post_clip_full_vocabulary_loss',
    'clipped_vocabulary_count': 'clipped_vocabulary_count',
    'clipped_vocabulary_fraction': 'clipped_vocabulary_fraction',
    'clipped_student_probability_mass': 'clipped_student_probability_mass',
    'clipped_teacher_probability_mass': 'clipped_teacher_probability_mass',
    'clipped_target_probability_mass': 'clipped_mixture_probability_mass',
    'student_entropy': 'student_entropy', 'teacher_entropy': 'teacher_entropy',
    'outer_old_entropy': 'outer_old_entropy', 'target_entropy': 'mixture_entropy',
    'student_observed_logp': 'student_observed_token_logp',
    'teacher_observed_logp': 'teacher_observed_token_logp',
    'outer_old_observed_logp': 'outer_old_observed_token_logp',
    'target_observed_logp': 'mixture_observed_token_logp',
}
DERIVED = ('student_teacher_observed_logp_gap', 'clip_loss_removed')
ALL_METRICS = tuple(METRICS) + DERIVED


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/non-numeric ' + label)
    return float(value)


def close(left, right, label):
    require(math.isclose(finite(left, label), finite(right, label),
                         rel_tol=2e-6, abs_tol=2e-6), 'Recorded reconstruction differs: ' + label)


def metadata(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


class Reader:
    def __init__(self):
        self.proofs = {}; self.json_cache = {}

    def opened(self, ref):
        path = Path(ref['path'])
        require(path.is_absolute() and '..' not in path.parts and
                all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink input required')
        before = metadata(path)
        require(stat.S_ISREG(before['mode']), 'Regular input required')
        if 'bytes' in ref:
            require(before['bytes'] == ref['bytes'], 'Input byte count differs')
        for key in ('device', 'inode', 'mtime_ns', 'ctime_ns', 'mode'):
            if key in ref:
                require(before[key] == ref[key], 'Pinned input metadata differs: ' + key)
        handle = path.open('rb')
        require(os.fstat(handle.fileno()).st_ino == before['inode'], 'Opened input inode differs')
        return path, handle, before

    def finish(self, ref, path, handle, before, digest, size):
        require(metadata(path) == before and os.fstat(handle.fileno()).st_size == before['bytes'],
                'Input changed during read')
        require(size == before['bytes'] and digest == ref['sha256'], 'Fresh input SHA/bytes differ')
        proof = dict(path=str(path), sha256=digest, **before)
        prior = self.proofs.get(str(path))
        require(prior is None or prior == proof, 'Repeated input changed')
        self.proofs[str(path)] = proof
        return proof

    def bytes(self, ref):
        path, handle, before = self.opened(ref)
        with handle:
            require(before['bytes'] <= 64 << 20, 'Bounded JSON/source limit exceeded')
            raw = handle.read()
            self.finish(ref, path, handle, before, hashlib.sha256(raw).hexdigest(), len(raw))
        return raw

    def json(self, ref):
        path = ref['path']
        if path in self.json_cache:
            require(ref['sha256'] == self.proofs[path]['sha256'] and metadata(Path(path)) ==
                    {k: self.proofs[path][k] for k in metadata(Path(path))}, 'Cached input identity changed')
            return self.json_cache[path]
        value = json.loads(self.bytes(ref)); self.json_cache[path] = value
        return value

    def lines(self, ref):
        path, handle, before = self.opened(ref); digest = hashlib.sha256(); size = 0
        with handle:
            for raw in handle:
                require(len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'Partial/oversized immutable JSONL row')
                digest.update(raw); size += len(raw)
                yield json.loads(raw)
            self.finish(ref, path, handle, before, digest.hexdigest(), size)

    def stable_end(self):
        for path, proof in self.proofs.items():
            require(metadata(Path(path)) == {k: proof[k] for k in metadata(Path(path))},
                    'Previously read input changed at extraction end')


def load_science(reader, config):
    manifest = reader.json(config['runtime_manifest'])
    files = manifest.get('files', manifest)
    require(len(files) == 305, 'Exact frozen305 manifest required')
    root = Path(config['paths']['runtime'])
    modules = []
    for name, relative, expected in (
            ('search_r1.qa_protocol', 'upstream/Search-R1/search_r1/qa_protocol.py', QA_SHA),
            ('_first5_frozen_core', 'search_e1/core.py', CORE_SHA)):
        entry = files[relative]; digest = entry['sha256'] if isinstance(entry, dict) else entry
        require(digest == expected, 'Frozen scoring identity differs')
        raw = reader.bytes(dict(path=str(root / relative), sha256=expected))
        module = types.ModuleType(name); module.__file__ = str(root / relative)
        if name == 'search_r1.qa_protocol':
            package = types.ModuleType('search_r1'); package.__path__ = []
            sys.modules['search_r1'] = package; sys.modules[name] = module
        exec(compile(raw, module.__file__, 'exec'), module.__dict__)
        modules.append(module)
    require('torch' not in sys.modules, 'Frozen scoring must remain dependency-free')
    return modules[1], modules[0]


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


def empty_region():
    return dict(valid_tokens=0, sums={key: 0. for key in ALL_METRICS}, hit_token_count=0)


def finish_region(region, trajectory_tokens, actual_pairs, weight):
    n = region['valid_tokens']; sums = region['sums']
    means = {key: value / n if n else None for key, value in sums.items()}
    return dict(region, means=means,
        trajectory_contributions={key: value / trajectory_tokens for key, value in sums.items()},
        global_batch_contributions={key: weight * value / trajectory_tokens / actual_pairs for key, value in sums.items()},
        probability_geometric_means={name: math.exp(means[name + '_observed_logp']) if n else None
                                     for name in ('student', 'teacher', 'outer_old', 'target')},
        hit_token_fraction=region['hit_token_count'] / n if n else None)


def token_features(reader, ref, pair, metric, receipt, rank, encoded_sha, config):
    rows = reader.lines(ref)
    try:
        header = next(rows)
    except StopIteration:
        raise ValueError('Empty committed token file')
    require(header['schema'] == 'opsd-online-training-token-header-v1' and
            header['run_id'] == receipt['run_id'] and header['outer_index'] == receipt['outer_index'] and
            header['optimizer_update'] == receipt['optimizer_updates'] and header['rank'] == rank and
            header['pair_index'] == metric['pair_index'] and header['encoded_sha256'] == encoded_sha,
            'Token header committed identity differs')
    require(header['identity']['pair_row_sha256'] == object_sha(pair) and
            header['identity']['question_id'] == pair['question_id'] and
            header['identity']['student_reward'] == pair['student_reward'] and
            header['identity']['reference_reward'] == pair['reference_reward'] and
            header['identity']['student_response_sha256'] == hashlib.sha256(pair['student'].encode()).hexdigest() and
            header['identity']['reference_response_sha256'] == hashlib.sha256(pair['reference'].encode()).hexdigest(),
            'Original response/pair identity differs')
    require(header['contract'] == receipt['contract'] and header['component_clip'] == 10 and
            header['retry'] == receipt['retries'] and header['pair_weight'] == pair['pair_weight'] == metric['pair_weight'] and
            header['student_pre_update_adapter_sha256'] == receipt['pre_update_adapter_sha256'] and
            header['outer_old_adapter'] == receipt['outer_old_adapter'] and
            header['frozen_teacher_identity'] == config['source_model_manifest']['sha256'] and
            header['loss_scope'] == 'actual_training_forward_before_optimizer', 'Pre-update/target/scope differs')
    ids, offsets, mask = header['token_ids'], header['token_offsets'], header['policy_mask']
    require(len(ids) == len(offsets) == len(mask) and all(type(v) is int and v >= 0 for v in ids), 'Response arrays differ')
    require(all(type(v) in (int, bool) and v in (0, 1) for v in mask), 'Invalid policy mask')
    text = pair['student']; spans = pair['student_policy_spans']; segments = character_regions(text, spans)
    info = [m.span() for m in re.finditer(r'<information>.*?(?:</information>|$)', text, re.S)]
    for offset, enabled in zip(offsets, mask):
        require(isinstance(offset, (list, tuple)) and len(offset) == 2 and
                all(type(v) is int for v in offset), 'Invalid offset type')
        a, b = offset
        require(0 <= a <= b <= len(text), 'Offset outside original response')
        expected = b > a and not any(a < y and b > x for x, y in info) and any(x <= a and b <= y for x, y in spans)
        require(bool(enabled) == expected, 'Frozen authored/information mask differs')
    positions = [i for i, value in enumerate(mask) if value]; count = len(positions)
    require(count == metric['valid_tokens'] > 0, 'Trajectory denominator differs')
    regions = {name: empty_region() for name in REGIONS}; auxiliary = 0; actual = receipt['global_actual_pairs']
    for row in rows:
        if 'full_vocab_dump' in row:
            require(row['full_vocab_dump'] in ('complete', 'cap_skip') and row['token_index'] in positions,
                    'Unexpected auxiliary vocabulary row')
            auxiliary += 1; continue
        used = regions['all_policy']['valid_tokens']
        require(used < count, 'Extra token metric row'); pos = positions[used]
        require(row['token_index'] == pos and row['token_id'] == ids[pos] and row['token_offsets'] == offsets[pos] and
                row['policy_mask'] is True and row['valid_trajectory_tokens'] == count and
                row['global_actual_pairs'] == actual and row['pair_weight'] == pair['pair_weight'],
                'Token alignment/normalization differs')
        require(row['direction'] == 'student_to_fixed_target_reverse' and
                row['target_kind'] == ('mixture' if config['variant'] == 'M' else 'teacher') and
                row['loss_scope'] == 'actual_training_forward_before_optimizer' and
                row['teacher_forced_fixed_student_prefix'] is True and row['actual_generation_scores'] is False and
                row['outer_old_identity']['manifest'] == receipt['outer_old_adapter'], 'Token target provenance differs')
        values = {name: finite(row[field], field) for name, field in METRICS.items()}
        hit = row['clipped_vocabulary_count']; vocab = row['vocabulary_size']
        require(type(hit) is int and type(vocab) is int and 0 <= hit <= vocab == 151936, 'Vocabulary/count differs')
        close(values['clipped_vocabulary_fraction'], hit / vocab, 'clip vocabulary fraction')
        require(values['target_raw_kl'] + 2e-6 >= values['target_post_loss'], 'Upper clipping increased loss')
        for key in ('clipped_student_probability_mass', 'clipped_teacher_probability_mass', 'clipped_target_probability_mass'):
            require(0 <= values[key] <= 1 + 2e-6 and (hit or values[key] == 0), 'Invalid clip probability mass')
        for name in ('student', 'teacher', 'outer_old', 'target'):
            require(values[name + '_observed_logp'] <= 2e-6 and values[name + '_entropy'] >= -2e-6,
                    'Invalid log probability/entropy')
        if config['variant'] != 'M':
            close(values['target_raw_kl'], values['teacher_kl'], 'Teacher target raw KL')
            close(values['target_observed_logp'], values['teacher_observed_logp'], 'Teacher target observed logp')
            close(values['clipped_target_probability_mass'], values['clipped_teacher_probability_mass'], 'Teacher target clip mass')
        else:
            left = values['teacher_observed_logp'] + math.log(.1)
            right = values['outer_old_observed_logp'] + math.log(.9)
            maximum = max(left, right); expected = maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))
            require(math.isclose(values['target_observed_logp'], expected, rel_tol=2e-6, abs_tol=2e-5),
                    'Recorded arithmetic mixture observed logp differs')
        close(values['target_raw_kl'], values['mixed_kl'], 'Raw current target KL')
        close(row['trajectory_contribution'], values['target_post_loss'] / count, 'Token trajectory contribution')
        close(row['global_batch_contribution'], pair['pair_weight'] * values['target_post_loss'] / count / actual,
              'Token actual-batch contribution')
        values['student_teacher_observed_logp_gap'] = values['student_observed_logp'] - values['teacher_observed_logp']
        values['clip_loss_removed'] = values['target_raw_kl'] - values['target_post_loss']
        label = category(offsets[pos], segments)
        for name in ('all_policy', label):
            regions[name]['valid_tokens'] += 1; regions[name]['hit_token_count'] += bool(hit)
            for key, value in values.items(): regions[name]['sums'][key] += value
    require(regions['all_policy']['valid_tokens'] == count and
            sum(regions[name]['valid_tokens'] for name in REGIONS[1:]) == count, 'Incomplete region/token cover')
    result = {name: finish_region(value, count, actual, pair['pair_weight']) for name, value in regions.items()}
    for name, native in (('target_raw_kl', 'raw_loss'), ('target_post_loss', 'post_loss'),
                         ('teacher_kl', 'student_to_teacher_raw_kl'), ('outer_old_kl', 'student_to_outer_old_raw_kl'),
                         ('mixed_kl', 'student_to_mixture_raw_kl')):
        close(result['all_policy']['means'][name], metric[native], 'Native pair ' + native)
        close(sum(result[r]['trajectory_contributions'][name] for r in REGIONS[1:]), metric[native], 'Region native reconstruction')
    close(pair['pair_weight'] * result['all_policy']['means']['target_post_loss'], metric['weighted_post_loss'], 'Weighted pair mean')
    require(int(result['all_policy']['sums']['clipped_vocabulary_count']) == metric['clipped_vocabulary_count'], 'Clip count reconstruction differs')
    return result, dict(token_log=reader.proofs[ref['path']], encoded_sha256=encoded_sha,
                        header_identity=header['identity'], sampling_model_identity=header['sampling_model_identity'],
                        pre_update_adapter_sha256=header['student_pre_update_adapter_sha256'],
                        outer_old_adapter=header['outer_old_adapter'], auxiliary_vocabulary_rows=auxiliary,
                        header_mask_sha256=object_sha([ids, mask, offsets]), response_sha256=header['identity']['student_response_sha256'])


def native_diagnostics(row, core, qa):
    text = row['response']; spans = row['policy_spans']; character_regions(text, spans)
    require(type(row['reward']) is int and row['reward'] in (0, 1), 'Native binary reward required')
    require(core.reward(text, row['answers'], spans) == row['reward'], 'Frozen strict reward mismatch')
    action = qa.parse_model_action(text[spans[-1][0]:spans[-1][1]]) if spans else None
    d = row['diagnostics']
    require(d['strict_reward'] == row['reward'] and d['strict_last_action_valid'] == (action is not None) and
            d['strict_final_answer_valid'] == (action is not None and action[0] == 'answer') and
            d['strict_final_action_kind'] == (action[0] if action else None) and d['termination'] == row['termination'] and
            d['executed_search_calls'] == row['search_calls'] and d['no_executed_search'] == (row['search_calls'] == 0),
            'Native strict/behavior diagnostic mismatch')
    require(type(row['search_calls']) is int and 0 <= row['search_calls'] <= 4, 'Native search budget differs')
    return dict(strict_em=row['reward'], strict_last_action_valid=d['strict_last_action_valid'],
                strict_final_answer_valid=d['strict_final_answer_valid'], strict_final_action_kind=d['strict_final_action_kind'],
                executed_search_calls=row['search_calls'], no_executed_search=row['search_calls'] == 0,
                termination=row['termination'], all_policy_spans_tag_balanced=d['all_policy_spans_tag_balanced'])


class Emitter:
    def __init__(self):
        self.counts = Counter(); self.hashes = {}; self.stream_hash = hashlib.sha256()

    def emit(self, kind, value):
        row = dict(record_kind=kind, **value); raw = canonical(row) + b'\n'
        sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
        self.counts[kind] += 1; self.hashes.setdefault(kind, hashlib.sha256()).update(raw); self.stream_hash.update(raw)


def extract_outer(reader, emit, config, config_ref, fact, core, qa):
    group = config['variant']; index = fact['outer']; run = config['run_id']; sources = fact['sources']
    require(fact['variant'] == group and fact['run_id'] == run and fact['status'] == 'COMPLETE_FACTS' and 0 <= index < 5,
            'Only explicit first five closed native facts permitted')
    completion = reader.json(sources['completion']); checkpoint = reader.json(sources['checkpoint'])
    outer = reader.json(sources['outer_manifest']); encoding = reader.json(sources['encoding']); order = reader.json(sources['order'])
    require(completion['accepted'] is True and checkpoint['accepted'] is True and completion['run_id'] == checkpoint['run_id'] == outer['run_id'] == run and
            completion['outer_index'] == outer['outer_index'] == checkpoint['cursor']['outer_index'] == index,
            'Native closed outer/run identity differs')
    require(outer['config_sha256'] == checkpoint['config_sha256'] == config_ref['sha256'] and
            completion['checkpoint']['sha256'] == sources['checkpoint']['sha256'] and
            completion['contract'] == checkpoint['contract'] == outer['contract'] == fact['contract'], 'Closed contract/commit differs')
    require(checkpoint['source_model_identity'] == outer['source_model_identity'] == config['source_model_manifest']['sha256'] and
            checkpoint['runtime_identity'] == outer['runtime_identity'] == config['runtime_manifest']['sha256'], 'Source model/runtime identity differs')
    K = completion['K']; P = completion['P']; delta = completion['optimizer_updates_delta']; before = outer['cumulative_optimizer_updates_before']
    require(type(K) is int and 0 <= K <= P <= 512 and delta == math.ceil(K / 32) and
            completion['optimizer_updates'] == checkpoint['optimizer_updates'] == before + delta and
            checkpoint['cursor']['next_pair_cursor'] == K == checkpoint['cursor']['usable_pair_count'], 'True update/cursor count differs')
    require(all(completion[k] == fact['counts'][k] for k in ('P', 'K', 'skipped', 'optimizer_updates', 'optimizer_updates_delta')),
            'Accepted facts/native counts differ')
    require(before == fact['counts']['optimizer_updates_before'], 'Accepted/native updates-before differ')
    require(checkpoint['training_input']['outer_manifest']['sha256'] == sources['outer_manifest']['sha256'] and
            checkpoint['training_input']['encoding']['sha256'] == sources['encoding']['sha256'] and
            checkpoint['training_input']['order']['sha256'] == sources['order']['sha256'] and
            checkpoint['training_input']['pairs'] == outer['pairs'], 'Native training input references differ')
    require(encoding['K'] == K and encoding['P'] == P and len(encoding['rows']) == K and
            sorted(order['pair_indices']) == list(range(K)) and order['encoded_sha256'] == encoding['sha256'], 'Encoded/order coverage differs')
    questions = list(reader.lines(outer['questions'])); require(len(questions) == 512, 'Native cohort must have512 questions')
    qmap = {q['question_id']: q for q in questions}; require(len(qmap) == 512, 'Duplicate native question')
    require(object_sha(list(qmap)) == outer['train_cursor']['question_ids_sha256'], 'Native question order SHA differs')
    sampled = {}; strict = {}; reward_hits = 0
    for row in reader.lines(outer['rollouts']):
        qid, sibling = row['question_id'], row['sibling']; require(qid in qmap and type(sibling) is int and 0 <= sibling < 5, 'Unexpected rollout slot')
        require(all(row[k] == qmap[qid][k] for k in ('question', 'answers', 'data_source')), 'Original cohort row differs')
        require((qid, sibling) not in sampled, 'Duplicate rollout sibling')
        strict[qid, sibling] = native_diagnostics(row, core, qa); sampled[qid, sibling] = row; reward_hits += row['reward']
    require(len(sampled) == 2560 and all((qid, s) in sampled for qid in qmap for s in range(5)), 'Five-sibling native coverage differs')
    require(reward_hits == fact['sampling']['strict_reward_hits'], 'Native sampling reward hits differ')
    pairs = list(reader.lines(outer['pairs'])); require(len(pairs) == P, 'Eligible pair count differs')
    pair_by_sha = {}; pair_by_qid = {}; classes = Counter()
    for pair in pairs:
        qid = pair['question_id']; require(qid in qmap and qid not in pair_by_qid, 'Unexpected/duplicate eligible qid')
        mining = pair['mining']; siblings = [sampled[qid, s] for s in range(5)]; successes = sum(r['reward'] for r in siblings)
        ref, student = siblings[mining['reference_sibling']], siblings[mining['student_sibling']]
        require(successes > 0 and pair['reference_reward'] == ref['reward'] == 1 and pair['student_reward'] == student['reward'] and
                pair['reference'] == ref['response'] and pair['student'] == student['response'] and pair['student_policy_spans'] == student['policy_spans'] and
                pair['question'] == qmap[qid]['question'] and pair['answers'] == qmap[qid]['answers'], 'Original B/D pair differs')
        require(mining['reference_row_sha256'] == object_sha(ref) and mining['student_row_sha256'] == object_sha(student) and
                mining['sibling_row_sha256'] == [object_sha(r) for r in siblings] and mining['rewards_by_sibling'] == [r['reward'] for r in siblings],
                'Original sibling row hashes/rewards differ')
        require(pair['pair_weight'] == 1 and mining['weights_applied'] is False and mining['variant'] == group and
                mining['outer_index'] == index and mining['selected_D_reward'] == student['reward'] and
                bool(student['reward']) == (successes == 5), 'C0/C1/M selected-D correctness rule violated')
        require(mining['selection_kind'] == ('all_correct_native_success_fallback_farthest' if successes == 5 else 'fail_preferred_farthest_full_response'),
                'Native fail-preferred/fallback selection differs')
        original_order = mining['source_order_siblings']
        require(sorted(original_order) == list(range(5)), 'Native sibling source order differs')
        require(ref is min((siblings[s] for s in original_order if siblings[s]['reward']), key=lambda r:(r['search_calls'], r.get('token_count',len(r['response'])))),
                'Native successful reference choice differs')
        digest = object_sha(pair); require(digest not in pair_by_sha, 'Duplicate pair digest')
        pair_by_sha[digest] = pair; pair_by_qid[qid] = pair; classes[mining['pair_class']] += 1
    require(len(pairs) == sum(any(sampled[qid, s]['reward'] for s in range(5)) for qid in qmap), 'All-wrong exclusion differs')
    require(dict(classes) == {k:v for k,v in outer['sampling_stats']['pair_classes'].items() if v}, 'Native pair classes differ')
    committed = checkpoint['outer_update_receipts']; require(len(committed) == delta, 'Committed update chain length differs')
    seen_pairs = set(); total_tokens = 0; pair_features = {}; native_updates = []
    for local_update, ref in enumerate(committed, 1):
        receipt = reader.json(ref); update = before + local_update
        require(receipt['accepted'] is True and receipt['run_id'] == run and receipt['outer_index'] == index and
                receipt['optimizer_updates'] == update and receipt['contract'] == outer['contract'] and
                receipt['loss_scope'] == 'actual_training_forward_before_optimizer' and receipt['loss_reduction'] == 'trajectory_mean' and
                receipt['frozen_teacher'] is True and receipt['reference_student_prefix_shared'] is True,
                'Committed actual update identity/scope differs')
        batch = order['pair_indices'][(local_update - 1) * 32:local_update * 32]
        require(receipt['global_actual_pairs'] == len(batch) > 0 and sorted(r['rank'] for r in receipt['all_rank_actual_token_metrics']) == list(range(8)),
                'Actual global tail/rank denominator differs')
        update_pairs = []; batch_loss_sum = 0.; batch_contribution_sum = 0.; batch_tokens = 0
        for rank_metrics in receipt['all_rank_actual_token_metrics']:
            rank = rank_metrics['rank']
            require([m['pair_index'] for m in rank_metrics['pairs']] == batch[rank::8], 'Native rank pair assignment differs')
            for metric in rank_metrics['pairs']:
                pair_index = metric['pair_index']; require(pair_index not in seen_pairs, 'Duplicate committed pair exposure')
                # Header first read uses the same immutable reference; no token copy or second token pass.
                token_ref = metric['token_log']; path, handle, meta = reader.opened(token_ref)
                with handle:
                    first = handle.readline(MAX_LINE + 1); require(first.endswith(b'\n') and len(first) <= MAX_LINE, 'Partial token header')
                    require(metadata(path) == meta, 'Header input changed'); header = json.loads(first)
                pair = pair_by_sha.get(header['identity']['pair_row_sha256']); require(pair is not None, 'Token does not bind an original pair')
                regions, provenance = token_features(reader, token_ref, pair, metric, receipt, rank, encoding['rows'][pair_index], config)
                qid = pair['question_id']; mining = pair['mining']; refrow = sampled[qid, mining['reference_sibling']]; student = sampled[qid, mining['student_sibling']]
                count = metric['valid_tokens']; total_tokens += count; batch_tokens += count; seen_pairs.add(pair_index); update_pairs.append(pair_index)
                batch_loss_sum += metric['weighted_post_loss']; batch_contribution_sum += regions['all_policy']['global_batch_contributions']['target_post_loss']
                d = strict[qid, mining['student_sibling']]; successes = sum(sampled[qid, s]['reward'] for s in range(5))
                feature = dict(schema='opsd-first5-pair-feature-v1', question_id=qid, group=group, outer_index=index,
                    optimizer_update=update, outer_optimizer_update_number=local_update, outer_total_optimizer_updates=delta,
                    normalized_update_progress=(local_update-1)/max(delta-1,1), rank=rank, pair_index=pair_index, run_id=run,
                    data_source=qmap[qid]['data_source'], valid_tokens=count, global_actual_pairs=receipt['global_actual_pairs'],
                    siblings_success_count=successes, sampling_success_fraction=successes/5, sampling_pass_at5=1,
                    selected_student_strict_em=pair['student_reward'], reference_strict_em=1, pair_class=mining['pair_class'],
                    student_search_calls=student['search_calls'], student_termination=student['termination'],
                    student_strict_final_action_valid=d['strict_last_action_valid'], student_strict_final_answer_valid=d['strict_final_answer_valid'],
                    student_strict_final_action_kind=d['strict_final_action_kind'], student_no_executed_search=d['no_executed_search'],
                    student_all_policy_spans_tag_balanced=d['all_policy_spans_tag_balanced'],
                    student_response_characters=len(student['response']), student_recorded_token_count=student.get('token_count'),
                    reference_search_calls=refrow['search_calls'], reference_response_characters=len(refrow['response']),
                    reference_token_count=refrow.get('token_count'), reference_token_count_scope='Original sampling recorded token_count; not newly tokenized',
                    student_sibling=mining['student_sibling'], reference_sibling=mining['reference_sibling'],
                    source_model_identity=config['source_model_manifest']['sha256'], runtime_identity=config['runtime_manifest']['sha256'],
                    config_sha256=config_ref['sha256'], deployed_science_source_commit=config['source_commit'],
                    pair_row_sha256=object_sha(pair), update_receipt=dict(path=ref['path'],sha256=ref['sha256'],bytes=ref['bytes']),
                    student_response_sha256=provenance['response_sha256'],
                    reference_response_sha256=provenance['header_identity']['reference_response_sha256'],
                    token_mask_alignment_sha256=provenance['header_mask_sha256'],
                    regions=regions, provenance=provenance,
                    timing_scope='EM and five rewards are original outer-start autonomous sampling; current logp/loss is this committed batch pre-update teacher-forced forward',
                    selection_scope='For C0/C1/M selected D correct iff all five siblings correct; not independent correctness sampling',
                    generation_processed_token_scores=False, region_loss_is_parameter_gradient=False)
                require(qid not in pair_features, 'Duplicate usable question'); pair_features[qid] = feature
                emit.emit('pair_feature', feature)
        require(sorted(update_pairs) == sorted(batch) and batch_tokens == receipt['global_valid_tokens'], 'Native batch/token coverage differs')
        close(batch_contribution_sum, batch_loss_sum / len(batch), 'Global batch objective reconstruction')
        native_updates.append(dict(update=update,actual_pairs=len(batch),valid_tokens=batch_tokens,
                                   reconstructed_batch_objective=batch_contribution_sum, receipt=reader.proofs[ref['path']]))
    require(seen_pairs == set(range(K)) and total_tokens == completion['global_valid_tokens'] == fact['counts']['valid_tokens'],
            'Complete usable/token cover differs')
    for qid, question in qmap.items():
        rewards = [sampled[qid, s]['reward'] for s in range(5)]; successes = sum(rewards); pair = pair_by_qid.get(qid); feature = pair_features.get(qid)
        emit.emit('cohort_feature',dict(schema='opsd-first5-cohort-feature-v1', question_id=qid, group=group, outer_index=index,
            run_id=run, data_source=question['data_source'], raw_rewards=rewards, siblings_success_count=successes,
            sampling_success_fraction=successes/5, sampling_pass_at5=int(successes>0),
            sampling_pass1_estimator='Observed five independent configured samples success fraction; not population probability',
            sibling_diagnostics=[strict[qid,s] for s in range(5)],
            selected_student_strict_em=pair['student_reward'] if pair else None,
            pair_row_sha256=object_sha(pair) if pair else None, selected_pair_available=pair is not None,
            loss_available=feature is not None, loss_missing_reason=None if feature else 'all_five_wrong' if successes==0 else 'native_encoding_skip',
            question_row_sha256=object_sha(question),
            question_semantic_row_sha256=object_sha({k:question[k] for k in ('question_id','question','answers','data_source')}),
            sampling_model_identity=outer['sampling_model_identity'],
            config_sha256=config_ref['sha256'], train_source_sha256=config['train_identity']['sha256'],
            source_model_identity=config['source_model_manifest']['sha256'],
            source_refs=dict(questions=outer['questions'],rollouts=outer['rollouts'],pairs=outer['pairs']),
            train_sampling_not_fixed_heldout=True))
    emit.emit('outer_proof',dict(schema='opsd-first5-outer-extraction-proof-v1', accepted=True, group=group,outer_index=index,run_id=run,
        K=K,P=P,questions=512,sampled_trajectories=2560,strict_reward_hits=reward_hits,committed_updates=delta,
        optimizer_updates_before=before,optimizer_updates_after=before+delta,valid_tokens=total_tokens,
        source_refs=sources,native_update_reconstructions=native_updates,strict_scorer_sha256=QA_SHA,core_sha256=CORE_SHA,
        selected_D_correct_iff_all_five_correct_verified=True, original_farthest_distance_recomputed=False,
        all_policy_region_cover_and_native_scalar_reconstruction=True))
    return set(qmap), dict(group=group,outer=index,K=K,P=P,valid_tokens=total_tokens,updates=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-json', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(); raw = args.request_json.encode()
    require(hashlib.sha256(raw).hexdigest() == args.request_sha256, 'Explicit request SHA differs')
    request = json.loads(raw)
    require(request['schema'] == 'opsd-first5-em-loss-extraction-request-v1' and
            [g['group'] for g in request['groups']] == list(GROUPS), 'Exact C0/C1/M request required')
    require(request['facts_reference']['sha256'] and request['facts_reference']['path'], 'Root accepted fact provenance required')
    reader = Reader(); emit = Emitter(); summaries = []; cohorts = {}; source_identities = set(); runtime_identities = set()
    started = time.time()
    for specification in request['groups']:
        config_ref = specification['config']; config = reader.json(config_ref); group = specification['group']
        require(config['variant'] == group and config['precision']['mode'] == 'bf16' and
                config['scientific']['learning_rate'] == (5e-6 if group=='C0' else 1e-6) and
                config['scientific']['component_clip'] == 10 and config['scientific']['outer_iterations'] == 10,
                'Original group configuration differs')
        require([o['outer'] for o in specification['outers']] == list(range(5)), 'Exactly outer0..4 required per group')
        source = reader.json(config['source_model_manifest']); source_files = source.get('files',source.get('source_files'))
        require(isinstance(source_files,list) and len(source_files)==12, 'Original source100 12-file identity required')
        source_identities.add(config['source_model_manifest']['sha256']);runtime_identities.add(config['runtime_manifest']['sha256'])
        core,qa = load_science(reader,config); group_ids = set()
        for fact in specification['outers']:
            ids,summary = extract_outer(reader,emit,config,config_ref,fact,core,qa)
            require(not(group_ids & ids), 'TRAIN first five cohorts unexpectedly repeat a question')
            group_ids.update(ids);key=fact['outer']
            if key in cohorts: require(cohorts[key]==ids,'Group same-outer TRAIN question sets differ')
            else:cohorts[key]=ids
            summaries.append(summary)
    require(len(source_identities)==len(runtime_identities)==1 and len(summaries)==15,'Source/runtime/outer identity differs')
    require(emit.counts['cohort_feature']==3*5*512 and emit.counts['pair_feature']==sum(o['K'] for o in summaries), 'Final feature count differs')
    reader.stable_end(); require('torch' not in sys.modules,'Unexpected Torch import')
    proof_rows = sorted(reader.proofs.values(),key=lambda r:r['path'])
    emit.emit('summary',dict(schema='opsd-first5-em-loss-extraction-summary-v1',accepted=True,
        request_sha256=args.request_sha256,root_facts_reference=request['facts_reference'],
        counts=dict(pairs=emit.counts['pair_feature'],valid_tokens=sum(o['valid_tokens'] for o in summaries),
                    cohort_rows=emit.counts['cohort_feature'],outers=len(summaries),
                    optimizer_updates=sum(o['updates'] for o in summaries)),
        started_unix=started,ended_unix=time.time(),outer_summaries=summaries,
        record_counts_before_summary=dict(emit.counts),record_SHA256_before_summary={k:h.hexdigest() for k,h in emit.hashes.items()},
        output_stream_sha256_before_summary=emit.stream_hash.hexdigest(),fresh_read_inputs=proof_rows,
        all_input_metadata_stable_at_end=True,all_token_files_fresh_complete_sha256=True,
        same_outer_three_group_question_sets_verified=True,selected_D_correct_iff_all_five_correct_verified=True,
        source12_payloads_not_rehashed=True,source_model_manifest_fresh_sha=True,
        no_model_or_optimizer_or_gpu_or_http_or_node_writes=True,no_new_tokenizer_or_logits=True,
        full_vocabulary_payloads_not_read_or_recomputed=True,parameter_gradients_not_recorded=True,
        fixed_heldout512_results_not_joined=True,training_entry_not_modified=True))


if __name__ == '__main__':
    main()
