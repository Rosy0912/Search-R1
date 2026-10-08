"""Bind actual evaluation rows to fixed64; retain IDs/EM privately, never text."""
import hashlib
import json
import re
import time
from pathlib import Path
from reproduction.a800.prepare_run import digest
from reproduction.a800.owned_processes import publish


def question_id(chat):
    if isinstance(chat,str):chat=json.loads(chat)
    text=chat[0]['content'];matches=list(re.finditer(r'Question:\s*',text))
    assert matches, 'Unknown fixed64 question schema'
    question=re.sub(r'\s+',' ',text[matches[-1].end():]).strip().casefold()
    return hashlib.sha256(question.encode()).hexdigest()


def bind_validation_dataframe(frame, base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert digest(base/'data/validation-64.parquet')==run['input_sha256']['validation-64.parquet']
    ids=[question_id(chat) for chat in frame['prompt']]
    assert ids==run['validation_question_ids'] and len(set(ids))==64
    assert frame['data_source'].value_counts().to_dict()==run['validation_source_counts']
    result=frame.copy();result['__g1_question_id']=ids
    return result


def record_validation(base,step,ids,sources,rewards,final=False):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    ids=list(ids);sources=list(sources);scores=[float(x) for x in rewards]
    assert ids==run['validation_question_ids'] and len(ids)==len(sources)==len(scores)==64
    assert len(set(ids))==64 and all(x in (0.,1.) for x in scores), 'Fixed64 must be actual binary EM'
    counts={s:sources.count(s) for s in set(sources)}
    assert counts==run['validation_source_counts']
    evidence=dict(run_id=run['run_id'],step=int(step),final=bool(final),question_ids=ids,
        data_sources=sources,em=scores,counts=counts,correct=sum(scores),weighted_em=sum(scores)/64,
        validation_sha256=run['input_sha256']['validation-64.parquet'],checked_unix=time.time())
    # Final proof is exclusive; no second evaluation may replace the first result.
    if final:
        path=base/'logs/final-fixed64-identity.json'
        with path.open('x') as f:json.dump(evidence,f)
    publish(base/'logs/latest-fixed64-identity.json',evidence)
    with (base/'logs/validation-identities.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')
    return evidence
