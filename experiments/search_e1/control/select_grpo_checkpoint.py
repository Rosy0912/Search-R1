"""Select GRPO handoff using training-held-out EM, never benchmark test rows."""
import argparse,json,re
from pathlib import Path
from collections import Counter
ANSI=re.compile(r'\x1b\[[0-9;]*m')
STEP=re.compile(r'\bstep\s*[:=]\s*(\d+)\b')
METRIC=re.compile(r'val/test_score/([A-Za-z0-9_.-]+):([0-9.eE+\-]+)')
def select(log,rows,initial,stage,steps):
 counts=Counter(r['data_source'] for r in rows)
 # Console metrics have three decimals. With fewer than1000 rows/source,
 # each binary-EM numerator can be uniquely recovered by rounding.
 if not counts or any(not 0<n<1000 for n in counts.values()):raise ValueError('Need exact metrics for this validation size')
 records={}
 for line in ANSI.sub('',log).splitlines():
  match=STEP.search(line);metrics=dict(METRIC.findall(line))
  if not match or not set(counts)<=set(metrics):continue
  logged_step=int(match[1]);step=steps if logged_step==steps+1 else logged_step
  if not (step==0 or (0<step<=steps and step%50==0)):continue
  path=Path(initial) if step==0 else Path(stage)/'actor'/f'global_step_{step}'
  if not (path/'config.json').exists():raise ValueError(f'Validated checkpoint missing: {path}')
  correct={}
  for source,n in counts.items():
   value=float(metrics[source]);k=round(value*n)
   if not 0<=k<=n or abs(value-k/n)>.0005001:raise ValueError('Metric incompatible with binary heldout EM')
   correct[source]=k
  records[step]=dict(step=step,logged_step=logged_step,path=str(path),correct=sum(correct.values()),count=sum(counts.values()),source_correct=correct,source_count=dict(counts),em=sum(correct.values())/sum(counts.values()))
 if 0 not in records:raise ValueError('Missing initial heldout baseline; refuse default-last selection')
 if not any(step>0 for step in records):raise ValueError('No validated trained checkpoints')
 best=max(records.values(),key=lambda r:(r['correct'],-r['step']))
 return dict(selected=best,candidates=[records[k] for k in sorted(records)],policy='Highest training-heldout binary EM; ties prefer baseline/earliest checkpoint',selection_data='train-heldout validation, never benchmark test64/test51713',binary_counts_recovered_from_three_decimal_console=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--validation',required=True);p.add_argument('--initial',required=True);p.add_argument('--stage',required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--report',required=True);a=p.parse_args()
 rows=[json.loads(x) for x in Path(a.validation).read_text().splitlines() if x.strip()]
 report=select(Path(a.log).read_text(),rows,a.initial,a.stage,a.steps)
 Path(a.report).write_text(json.dumps(report,indent=2));print(report['selected']['path'])
if __name__=='__main__':main()
