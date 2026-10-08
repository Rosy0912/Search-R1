import pathlib,json,time,collections
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');rows=[]
for p in (R/'output/validation-cost').glob('*.jsonl'):
 events=[json.loads(s) for s in p.read_text().splitlines()];c=collections.Counter(e['kind'] for e in events);rows.append({'path':str(p),'events':len(events),'kinds':dict(c),'sample_rows_submitted':sum(e['rows'] for e in events if e['kind']=='real_sample_batch'),'last_event':events[-1]})
print(json.dumps({'epoch':time.time(),'ledgers':rows}))
