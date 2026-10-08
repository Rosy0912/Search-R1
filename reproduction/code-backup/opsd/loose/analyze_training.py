import json,math,re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text(errors='replace')
text=re.sub(r'\x1b\[[0-9;]*m','',text)
steps={};evaluations=[]
for line in text.splitlines():
 if 'BASELINE_EVALUATION_METRICS_JSON=' in line:
  evaluations.append(json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]))
 m=re.search(r'(?:^|\s)step:(\d+) - ',line)
 if not m:continue
 metrics={}
 for item in line[m.end():].split(' - '):
  key,sep,value=item.rpartition(':')
  if sep:
   try:metrics[key]=float(value)
   except ValueError:pass
 steps[int(m.group(1))]=metrics
required={'actor/samples':320,'actor/old_log_probs_present':1,'actor/old_log_probs_finite':1,'actor/loss_mask_valid':1,'actor/optimizer_steps':10,'actor/skipped_empty_minibatches':0,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0}
checks={}
for step,metrics in sorted(steps.items()):
 if step==0:continue
 problems=[]
 for rank in range(8):
  for name,expected in required.items():
   key=name+'/rank_'+str(rank)
   if key not in metrics or metrics[key]!=expected:problems.append(dict(key=key,actual=metrics.get(key),expected=expected))
  for name in ('actor/opsd_loss_mean','actor/grad_norm_mean','actor/supervised_tokens'):
   key=name+'/rank_'+str(rank)
   if key not in metrics or not math.isfinite(metrics[key]):problems.append(dict(key=key,actual=metrics.get(key),expected='present finite'))
 checks[step]=dict(accepted=not problems,problems=problems)
result=dict(completed_outer_steps=sorted(k for k in steps if k>0),rank_checks=checks,evaluations=evaluations,metrics=steps,console_number_precision='three decimal places; do not infer exact lr or tiny gradient/weight changes')
Path(sys.argv[2]).write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
