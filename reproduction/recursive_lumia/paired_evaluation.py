"""Persist native evaluation rows for same-question EM comparisons; no new rollout."""
import hashlib
import json
import math
import os
from pathlib import Path


def attach_question_ids(rows, identifiers):
    if len(rows) != len(identifiers):
        raise ValueError('Evaluation question IDs do not cover the scored batch')
    for row, identifier in zip(rows, identifiers):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Common holdout requires explicit string question IDs')
        row['question_id'] = identifier


def save_evaluation(rows, metrics, outer, expected_count):
    if expected_count not in (512, 51713) or len(rows) != expected_count:
        raise ValueError('Paired capture requires the complete fixed dev or final split')
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(r['em'] not in (0., 1.) for r in rows):
        raise ValueError('Repeated IDs or nonbinary native EM')
    em = sum(r['em'] for r in rows)/len(rows)
    if not math.isclose(em, metrics['val/em/all'], rel_tol=0., abs_tol=1e-12):
        raise ValueError('Paired rows disagree with the native aggregate EM')
    split = 'dev512' if expected_count == 512 else 'final51713'
    directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'evaluations'
    directory.mkdir(exist_ok=True)
    stem = f'{split}-outer-{outer:06d}'
    payload = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n' for r in rows)
    with (directory/(stem+'.jsonl')).open('x') as stream:
        stream.write(payload)
    record = dict(split=split, completed_outer=int(outer), count=len(rows), em=em,
        rows_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        ordered_ids_sha256=hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
        run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'], job=os.environ['SLURM_JOB_ID'],
        code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
        scope='Native evaluation metrics keyed by fixed question ID; no extra generation or selection')
    with (directory/(stem+'.json')).open('x') as stream:
        json.dump(record, stream, indent=2)
    print('PAIRED_EVALUATION_SAVED='+json.dumps(record, sort_keys=True), flush=True)
    return record
