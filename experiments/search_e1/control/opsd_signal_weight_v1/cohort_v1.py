"""One common frozen 25-outer TRAIN selection, original isolated seed42 shuffle."""
import hashlib
import copy
import json
import os
from pathlib import Path
import random
import re
import stat

COHORT_VERSION = 'search-e1-opsd-25-train-cohorts-seed42-v1'
TRAIN_ROWS = 169103
VALIDATION_ROWS = 512


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def freeze_cohorts(train_rows, validation_rows, train_identity, validation_identity):
    train = list(train_rows); validation = list(validation_rows)
    if len(train) != TRAIN_ROWS or len(validation) != VALIDATION_ROWS:
        raise ValueError('Require original TRAIN169103 and heldout512 populations')
    for rows, name in ((train, 'TRAIN'), (validation, 'validation')):
        if any(not isinstance(r.get('question_id'), str) or not r['question_id']
               or not isinstance(r.get('question'), str) or 'answers' not in r for r in rows):
            raise ValueError(name + ' question schema differs')
        if len({r['question_id'] for r in rows}) != len(rows):
            raise ValueError(name + ' has duplicate question IDs')
    for ref in (train_identity, validation_identity):
        if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not Path(ref['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')):
            raise ValueError('Explicit data path/SHA identities required')
    forbidden_ids = {r['question_id'] for r in validation}
    forbidden_text = {r['question'] for r in validation}
    permutation = list(range(TRAIN_ROWS)); random.Random(42).shuffle(permutation)
    selected = permutation[:25 * 512]
    overlap_ids = [train[i]['question_id'] for i in selected if train[i]['question_id'] in forbidden_ids]
    overlap_text_ids = [train[i]['question_id'] for i in selected if train[i]['question'] in forbidden_text]
    if overlap_ids or overlap_text_ids:
        raise ValueError('Fixed seed42 cohorts overlap heldout; no replacement/resampling: ' +
                         json.dumps(dict(id_overlap_question_ids=overlap_ids, exact_text_overlap_question_ids=overlap_text_ids), sort_keys=True))
    cohorts = []
    for outer in range(25):
        indices = selected[outer * 512:(outer + 1) * 512]
        questions = [copy.deepcopy(train[i]) for i in indices]
        cohorts.append(dict(outer_index=outer, train_indices=indices, questions=questions,
            before=dict(cycle=0, offset=outer * 512), after=dict(cycle=0, offset=(outer + 1) * 512),
            boundary_skips=[], question_ids_sha256=object_sha([r['question_id'] for r in questions]),
            row_semantic_sha256=[object_sha(r) for r in questions], questions_sha256=object_sha(questions)))
    return dict(schema='opsd-ablation-common-cohorts-v1', version=COHORT_VERSION, seed=42,
        outer_count=25, questions_per_outer=512, train_population=TRAIN_ROWS, validation_population=VALIDATION_ROWS,
        train_identity=copy.deepcopy(train_identity), validation_identity=copy.deepcopy(validation_identity), cohorts=cohorts,
        selected_train_indices_sha256=object_sha(selected), permutation_sha256=object_sha(permutation),
        selected_unique_question_count=len({train[i]['question_id'] for i in selected}),
        validation_id_overlap=0, validation_exact_question_text_overlap=0,
        selection_rule='Original Python random.Random(42).shuffle full TRAIN index permutation; first12800 without drops/replacements',
        private_data=True)


def verify_frozen(frozen, train_rows, validation_rows, train_identity, validation_identity):
    expected = freeze_cohorts(train_rows, validation_rows, train_identity, validation_identity)
    if frozen != expected:
        raise ValueError('Frozen cohorts/source/order/version differ from original seed42 TRAIN selection')
    return dict(accepted=True, version=COHORT_VERSION, outer_count=25, selected_questions=12800,
                frozen_sha256=object_sha(expected), validation_overlap=0)


def _read_bound_jsonl(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('Bound data path must be absolute regular input')
    before = path.stat(); fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Data input is not a regular file')
    rows = []; digest = hashlib.sha256()
    with path.open('rb') as handle:
        if any(getattr(os.fstat(handle.fileno()), k) != getattr(before, k) for k in fields):
            raise ValueError('Data inode changed before read')
        for line in handle:
            if not line.endswith(b'\n'):
                raise ValueError('Partial data JSONL line')
            digest.update(line); rows.append(json.loads(line))
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    if any(getattr(after, k) != getattr(before, k) or getattr(after_fd, k) != getattr(before, k) for k in fields):
        raise ValueError('Data bytes/stat changed during cohort freeze')
    if digest.hexdigest() != ref['sha256'] or ('bytes' in ref and ref['bytes'] != before.st_size):
        raise ValueError('Data SHA/bytes differ from bound identity')
    return rows


def freeze_from_files(train_identity, validation_identity):
    """CPU prepare only, after source publication; never invoked implicitly."""
    return freeze_cohorts(_read_bound_jsonl(train_identity), _read_bound_jsonl(validation_identity),
                          train_identity, validation_identity)


def verify_from_files(frozen, train_identity, validation_identity):
    expected = freeze_from_files(train_identity, validation_identity)
    if frozen != expected:
        raise ValueError('Saved cohort differs from freshly SHA-bound original data')
    return dict(accepted=True, version=COHORT_VERSION, outer_count=25, selected_questions=12800,
                frozen_sha256=object_sha(expected), validation_overlap=0)
