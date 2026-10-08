import hashlib,json,math,re
from pathlib import Path
def pairs(items):
 d={}
 for k,v in items:
  assert k not in d, ('Duplicate JSON key',k)
  d[k]=v
 return d
def loads(text):
 value=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(AssertionError('Nonfinite JSON '+s)))
 def finite(v):
  if isinstance(v,float):assert math.isfinite(v), 'Nonfinite JSON float'
  elif isinstance(v,dict):
   for item in v.values():finite(item)
  elif isinstance(v,list):
   for item in v:finite(item)
 finite(value);return value
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def own_file(value,base):
 p=Path(value)
 assert p.is_absolute() and p.is_file(), ('Missing absolute evidence',value)
 assert p.resolve().is_relative_to(base.resolve()), ('Evidence escapes own root',value)
 return p
ONE=('actor/old_log_probs_present','actor/old_log_probs_finite','actor/loss_mask_valid','actor/opsd_loss_finite','actor/grad_norm_finite','actor/allocator_is_native','actor/allocator_configuration_is_default','teacher/update_device_is_cuda_per_rank','teacher/storage_is_cpu_per_rank','teacher/all_parameter_dtype_is_fp32_per_rank','teacher/attention_is_eager_per_rank')
ZERO=('actor/skipped_empty_minibatches','teacher/trainable_parameters_per_rank','teacher/gradient_tensors_per_rank','teacher/student_optimizer_membership_per_rank')
def verify_outers(text,count):
 text=re.sub(r'\x1b\[[0-9;]*m','',text);steps=[]
 for line in text.splitlines():
  m=re.search(r'(?:^|\s)step:(\d+) - ',line)
  if not m or 'actor/optimizer_steps:' not in line:continue
  step=int(m[1]);steps.append(step);metrics={}
  for item in line[m.end():].split(' - '):
   key,sep,value=item.rpartition(':')
   if not sep:continue
   assert key not in metrics, ('Duplicate outer metric',key)
   value=float(value);assert math.isfinite(value), ('Nonfinite metric',key)
   metrics[key]=value
  for rank in range(8):
   def val(k):return metrics[f'{k}/rank_{rank}']
   for k in ONE:assert val(k)==1,(step,rank,k)
   for k in ZERO:assert val(k)==0,(step,rank,k)
   assert val('actor/samples')==320
   assert val('actor/optimizer_steps')==10
   assert val('teacher/cuda_update_entries_cumulative_per_rank')==step
   assert val('actor/supervised_tokens')>0 and val('actor/grad_norm_mean')>0
   assert math.isfinite(val('actor/opsd_loss_mean'))
 assert steps==list(range(1,count+1)), ('Actual outer sequence differs',steps)
 return {'completed_actual_outer_updates':count,'ranks':8,'samples_per_rank':320,'numeric_metrics_validated':True}
def verify_slurm(text,job):
 assert re.search(r'\bJobId='+re.escape(job)+r'\b',text)
 for key,value in (('JobState','COMPLETED'),('ExitCode','0:0')):
  found=re.findall(r'\b'+key+r'=([^\s]+)',text)
  assert found==[value], ('Slurm job is not successfully completed',job,key,found)
def verify_validation_metrics(metrics):
 assert metrics['val/qa_count/all']==64
 sources={k[len('val/qa_count/'):] for k in metrics if k.startswith('val/qa_count/') and k!='val/qa_count/all'}
 assert sources and sum(metrics['val/qa_count/'+name] for name in sources)==64
 for name in sources:
  count=metrics['val/qa_count/'+name]
  assert type(count) in (int,float) and count>0 and float(count).is_integer()
  assert metrics['val/evaluated_count/'+name]==count
 for name in sources|{'all'}:
  for kind in ('f1','em','answer_format','format_rate'):
   value=metrics[f'val/{kind}/{name}']
   assert type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
 # Do not permit missing or extraneous source-specific summaries.
 for kind in ('f1','em','answer_format','format_rate'):
  assert {k.split('/')[-1] for k in metrics if k.startswith('val/'+kind+'/')}==sources|{'all'}
 assert {k.split('/')[-1] for k in metrics if k.startswith('val/evaluated_count/')}==sources
 return sources
def verify_pilot_validations(text):
 text=re.sub(r'\x1b\[[0-9;]*m','',text)
 marker='BASELINE_EVALUATION_METRICS_JSON='
 records=[];summaries={};pending=None
 for line in text.splitlines():
  if marker in line:
   assert pending is None, 'Evaluation JSON lacks corresponding completed step summary'
   pending=loads(line.split(marker,1)[1]);verify_validation_metrics(pending);records.append(pending)
  m=re.search(r'(?:^|\s)step:(\d+) - ',line)
  if not m:continue
  step=int(m[1]);metrics={}
  for item in line[m.end():].split(' - '):
   key,sep,value=item.rpartition(':')
   if sep and key.startswith('val/'):
    assert key not in metrics;metrics[key]=float(value)
  if step in (0,5):
   assert step not in summaries and pending is not None, ('Missing/duplicate pilot validation',step)
   assert verify_validation_metrics(metrics)==verify_validation_metrics(pending)
   # Console rounds decimal scores; exact unrounded values are in the JSON.
   for key in metrics:
    if key.startswith(('val/qa_count/','val/evaluated_count/')):assert metrics[key]==pending[key]
   summaries[step]=metrics;pending=None
  else:assert not metrics, ('Unexpected intermediate pilot evaluation',step)
 assert set(summaries)=={0,5} and len(records)==2 and pending is None
 return True
