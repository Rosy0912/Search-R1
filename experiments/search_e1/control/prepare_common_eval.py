import os,json,hashlib
from pathlib import Path
base=Path(os.environ['E1_BASE']);data=base/'data'
rows=[json.loads(line) for line in (data/'test.jsonl').open()]
selected=[]
for source in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
 pool=[r for r in rows if r['data_source']==source]
 pool.sort(key=lambda r:hashlib.sha256(('e1-config-common-64-v1:'+r['question_id']+r['question']).encode()).digest())
 selected.extend(pool[:10 if source=='nq' else 9])
assert len(selected)==64
(data/'test-common-64.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
(base/'receipts/common-evaluation.json').write_text(json.dumps({'holdout_questions':512,'test_questions':64,'ids':[r['question_id'] for r in selected],'note':'Identical IDs for both profiles; diagnostic test sample, not full seven-benchmark score'},indent=2))
