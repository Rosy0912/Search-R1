from pathlib import Path
import json
b=Path('work/search-e1-analysis-output/opsd-ablation-execution-20261008/raw-c0-closed0-7-direct-20261009/files/group-C0/runs/outers/outer-00000')
for n in ['teacher-diagnostics/selection.json','teacher-diagnostics/child-complete.json','teacher-diagnostics/complete.json']:
 d=json.loads((b/n).read_text());print(n,list(d));print({k:v for k,v in d.items() if k not in ['cases','rows','selected','source_model','runtime','config'] and isinstance(v,(str,int,float,bool))});print('childlist',str(d.get('cases',d.get('selected')))[:700])
p=next((b/'training').glob('attempt-*/update-00000001.json'));d=json.loads(p.read_text());print('pairkeys',list(d['all_rank_actual_token_metrics'][0]['pairs'][0]));print('metric',d['all_rank_actual_token_metrics'][0].keys())
