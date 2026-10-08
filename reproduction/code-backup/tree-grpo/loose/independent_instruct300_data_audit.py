import hashlib,json,random,time,itertools,collections
from pathlib import Path
import pandas as pd
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/common-data')
started=time.time();manifest_bytes=(root/'manifest.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
m=json.loads(manifest_bytes)
raws={}
for record in m['preparation']['raw_sources']:
    path=root/record['path'];h=hashlib.sha256();rows=[]
    with path.open('rb') as f:
        for i,line in enumerate(f):
            h.update(line);r=json.loads(line)
            rows.append(dict(index=f"{record['source']}:{record['split']}:{i}",question=r['question'].strip(),answers=r['golden_answers'],source=record['source']))
    assert h.hexdigest()==record['sha256'] and len(rows)==record['rows']
    raws[record['source'],record['split']]=rows
splits={name:[] for name in ['train','validation','stage','final']}
for source in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
    if source in ['nq','hotpotqa']:
        rows=list(raws[source,'train']);random.Random(42).shuffle(rows)
        splits['validation'].extend(rows[:256]);splits['stage'].extend(rows[256:288]);splits['train'].extend(rows[288:])
    rawsplit='test' if source in ['nq','triviaqa','popqa','bamboogle'] else 'dev'
    splits['final'].extend(raws[source,rawsplit])
canonical=lambda question:' '.join(question.casefold().rstrip('?').split())
held={k:{canonical(r['question']) for r in splits[k]} for k in ['validation','stage','final']}
retained=[];excluded=[]
for row in splits['train']:
    reasons=[name for name,questions in held.items() if canonical(row['question']) in questions]
    if reasons:excluded.append(dict(question_id=row['index'],canonical_question=canonical(row['question']),overlap_with=reasons))
    else:retained.append(row)
splits['train']=retained
checkfile=m['exclusion_audit'];audit_bytes=(root/checkfile['path']).read_bytes()
assert hashlib.sha256(audit_bytes).hexdigest()==checkfile['sha256']
audit=json.loads(audit_bytes);assert excluded==audit['excluded_train_rows'] and audit['initial_train_candidates']==169039
results={}
# Prescribed package text inserted below by local preparation, not imported from consumer.
INSTRUCTION='Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you lack knowledge, call a search engine by <search> query </search>; results appear between <information> and </information>. You may search at most four times. Provide the final answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.'
def plain(value):
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value
for name,expected in splits.items():
    actual=pd.read_parquet(root/m['splits'][name]['path']).to_dict('records')
    assert len(actual)==len(expected)==m['splits'][name]['rows']
    for i,(row,target) in enumerate(zip(actual,expected)):
        row=plain(row)
        assert row['data_source']==target['source'] and row['extra_info']==dict(index=target['index'],split=name),(name,i,'source/id/order')
        assert row['prompt']==[dict(role='user',content=INSTRUCTION+'\nQuestion: '+target['question'])],(name,i,'question/prompt')
        assert row['reward_model']==dict(style='rule',ground_truth=dict(target=target['answers'])),(name,i,'answers')
        assert row['ability']=='fact-reasoning'
    results[name]=dict(rows=len(actual),source_counts=dict(collections.Counter(r['source'] for r in expected)),all_rows_exact_source_question_answer_id_order_match=True)
assert [results[k]['rows'] for k in ['train','validation','stage','final']]==[169028,512,64,51713]
questions={name:{canonical(r['question']) for r in rows} for name,rows in splits.items()}
intersections={a+'|'+b:len(questions[a]&questions[b]) for a,b in itertools.combinations(splits,2)}
assert not any(intersections.values())
print(json.dumps(dict(status='RAW_TO_PREBUILT_ALL_ROWS_RECONSTRUCTED_PASS',manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),splits=results,excluded_rows=len(excluded),all_excluded_ids_and_reasons_exact_match=True,intersections=intersections,wall_seconds=time.time()-started),sort_keys=True),flush=True)
