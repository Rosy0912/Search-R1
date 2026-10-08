from pathlib import Path
import json
b=Path('work/search-e1-analysis-output/opsd-ablation-execution-20261008/raw-c0-closed0-7-direct-20261009/files/group-C0/runs/outers/outer-00000/teacher-diagnostics')
d=json.loads((b/'complete.json').read_text());print('exit',str(d['exit_proof'])[:1800]);print('childtype',type(d['child']),str(d['child'])[:300])
p=next((b/'cases').glob('case-00.json'));d=json.loads(p.read_text());print('casekeys',list(d));print('tracekeys',list(d['trace']));print('resultkeys',list(d['result']));print('selected',d['selected']);print('generationevents',[(x.get('type'),x.keys()) for x in d['trace'].get('events',[])][:3])
