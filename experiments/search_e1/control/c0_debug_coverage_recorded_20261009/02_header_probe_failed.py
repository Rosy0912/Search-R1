from pathlib import Path
import json
b=Path('work/search-e1-analysis-output/opsd-ablation-execution-20261008/raw-c0-closed0-7-direct-20261009/files/group-C0/runs/outers/outer-00000')
p=next((b/'training').glob('attempt-*/rank-*/update-*.jsonl'))
with p.open() as f:d=json.loads(next(f))
print(list(d));print(d.get('question_id'),d.get('identity'))
q=json.loads((b/'training/encoding.json').read_text());print('encodingkeys',list(q));print(str(q)[:600])
