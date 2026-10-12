"""Rebuild final ordered51713 binary EM/F1 and all seven source aggregates."""
import math
from collections import Counter
from pathlib import Path

from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256


def verify_final_rows(evaluation,run,canonical,ids,sources):
    assert evaluation['run_id'] == run['run_id'] and type(evaluation['step']) is int and evaluation['step'] == 200
    assert evaluation['final'] is True and evaluation['split'] == 'final'
    assert evaluation['validation_sha256'] == canonical['sha256']
    assert len(ids) == len(sources) == canonical['rows'] == 51713
    # Repeated normalized questions are canonical occurrences, not duplicates
    # to remove. Exact order and count must survive scoring.
    assert evaluation['question_ids'] == ids and evaluation['data_sources'] == sources
    em = evaluation['em']; f1 = evaluation['f1']
    assert len(em) == len(f1) == 51713
    assert all(type(value) in (int,float) and math.isfinite(value) and value in (0,1) for value in em)
    assert all(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 for value in f1)
    assert all(correct<=score for correct,score in zip(em,f1))
    counts = Counter(sources)
    assert len(counts) == 7 and counts['nq'] == 3610
    assert evaluation['counts'] == canonical['source_counts'] == dict(counts)
    correct = {source:0 for source in counts}; source_f1 = {source:[] for source in counts}
    for source,value,score in zip(sources,em,f1):
        correct[source] += int(value); source_f1[source].append(score)
    total = sum(correct.values()); weighted_em = total/51713; weighted_f1 = math.fsum(f1)/51713
    assert type(evaluation['correct']) in (int,float) and evaluation['correct'] == total
    assert type(evaluation['weighted_em']) in (int,float) and evaluation['weighted_em'] == weighted_em
    assert type(evaluation['weighted_f1']) in (int,float) and evaluation['weighted_f1'] == weighted_f1
    return dict(run_id=run['run_id'],step=200,question_count=51713,counts=dict(counts),correct=total,
        weighted_em=weighted_em,weighted_f1=weighted_f1,source_correct=correct,
        source_em={source:correct[source]/counts[source] for source in counts},
        source_f1={source:math.fsum(source_f1[source])/counts[source] for source in counts},
        ordered_ids_sha256=document_sha256(ids),ordered_sources_sha256=document_sha256(sources),
        ordered_em_sha256=document_sha256(em),ordered_f1_sha256=document_sha256(f1))


def read_final_evaluation(base,run):
    from reproduction.common8.actor_probe_rank_tail import _sealed_bytes
    from reproduction.common8.validation_identity import expected
    base = Path(base).resolve()
    raw,file = _sealed_bytes(base,'logs/final-identity.json',limit=32*1024**2)
    canonical,frame,ids = expected(base,'final')
    return dict(verify_final_rows(_strict_json(raw),run,canonical,ids,frame['data_source'].tolist()),
        actual_final_evaluation_file=file)
