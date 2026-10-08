"""Pure verification of actual JSON/three-decimal logger pairs; no inference of results."""
import re,math
from training_evidence import loads,verify_validation_metrics
from strict_metric_parser import parse_metric_summary

def verify_periodic_validations(text,expected_steps=(0,50,100,150,200,250,300,350,400,450,500),require_final=True):
 text=re.sub(r'\x1b\[[0-9;]*m','',text)
 pending=None;seen=[];final_seen=False;final_summary=False
 for line in text.splitlines():
  marker='BASELINE_EVALUATION_METRICS_JSON='
  if marker in line:
   assert pending is None,'Evaluation JSON lacks paired completed summary'
   assert not final_summary,'Unexpected evaluation after final summary'
   pending=loads(line.split(marker,1)[1])
   count=pending.get('val/qa_count/all')
   if count==64:assert len(verify_validation_metrics(pending))==7
   else:
    assert require_final and count==51713 and seen==list(expected_steps),'Unexpected evaluation size/order'
   continue
  marker='SEARCH_R1_FINAL_METRICS_JSON='
  if marker in line:
   assert require_final and pending is not None and pending['val/qa_count/all']==51713
   assert not final_seen and loads(line.split(marker,1)[1])==pending,'Duplicate/mismatched final JSON'
   final_seen=True;continue
  m=re.search(r'(?:^|\s)step:(\d+) - ',line)
  if not m:continue
  step,metrics,events=parse_metric_summary(line)
  summary={key:value for key,value in metrics.items() if key.startswith('val/')}
  if not summary:
   assert pending is None,'Evaluation JSON followed by unpaired outer summary'
   continue
  assert pending is not None,'Validation summary lacks original evaluation JSON'
  assert set(summary)==set(pending),'JSON/summary key mismatch'
  for key,value in pending.items():
   assert type(value) in (int,float) and math.isfinite(value)
   # aggregate_logger uses f'{v:.3f}', while retained JSON is unrounded.
   assert summary[key]==float(f'{value:.3f}'),('JSON/summary value mismatch',step,key)
  if pending['val/qa_count/all']==64:
   assert step in expected_steps and step not in seen,'Unexpected/duplicate periodic validation step'
   assert step==expected_steps[len(seen)],'Missing/out-of-order periodic validation step'
   verify_validation_metrics(summary);seen.append(step)
  else:
   assert require_final and step==501 and final_seen and not final_summary,'Missing/duplicate final summary'
   final_summary=True
  pending=None
 assert pending is None and seen==list(expected_steps),'Incomplete periodic64 validation sequence'
 assert (not require_final) or (final_seen and final_summary),'Incomplete terminal final validation'
 return {'periodic_steps':seen,'questions_per_periodic':64,'terminal_final_questions':51713 if require_final else None}
