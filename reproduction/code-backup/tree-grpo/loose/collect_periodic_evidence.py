import argparse,json,hashlib,time,re,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);a=ap.parse_args();p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-formal-2971');files={}
paths=[p/'resolved.yaml',p/'output/entry-audit/entry-config.yaml',p/'output/entry-audit/entry.jsonl',*sorted((p/'output/tree-rank-audit').glob(f'step-{a.step:06d}-rank-*.json'))]
for f in sorted((p/'output/validation-cost').glob('*.json')):
 x=json.loads(f.read_text())
 if x['global_step']==a.step:
  assert x['status']=='VALIDATION_COST_COMPLETE' and x['dataset_rows']==x['evaluated_rows']==64
  ledger=f.with_suffix('.jsonl');assert hashlib.sha256(ledger.read_bytes()).hexdigest()==x['ledger_sha256'];paths.extend([f,ledger])
lines=[l for l in (p/'stage.log').read_text().splitlines() if f'step:{a.step} - ' in l];assert len(lines)==1;metrics={}
for v in lines[0].split(f'step:{a.step} - ',1)[1].split(' - '):
 k,_,n=v.partition(':');metrics[k]=float(n)
sources=[k.split('/')[-1] for k in metrics if k.startswith('val/evaluated_count/')];assert len(sources)==7 and sum(metrics['val/evaluated_count/'+s] for s in sources)==metrics['val/qa_count/all']==64
for k in ['f1','em','format_rate','answer_format']:
 val=sum(metrics[f'val/{k}/{s}']*metrics['val/evaluated_count/'+s] for s in sources)/64;assert math.isclose(val,metrics[f'val/{k}/all'],abs_tol=1e-12)
for f in paths:
 b=f.read_bytes();files[str(f.relative_to(p))]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(dict(epoch=time.time(),run=str(p),step=a.step,selected_stage_log=lines,files=files,aggregate_check='64 rows, weighted F1/EM/format consistency and cost ledger SHA; not independent answer rescoring')))
