import importlib.util,json,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import json,hashlib,time,sys,os
from pathlib import Path
from collections import Counter
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-branpo-fresh200-3360';source='8b6f99d1b3353145200eb3fa7c16a579a79a987c';pkg=root/'instruct300/candidates'/('branpo-fresh200-'+source)
sys.path.insert(0,str(pkg));from production_contract import direct_rank_records
package=json.loads((pkg/'manifest.json').read_text());config='c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3';rank=direct_rank_records(run,package,config,200)
assert rank['completed_outer']==200 and rank['actual_writes_per_rank']=={str(i):4000 for i in range(8)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
paths=sorted((run/'output/training-rollout-cost').glob('*.json'));rollouts=[json.loads(p.read_text()) for p in paths]
assert len(rollouts)==200 and all(v['status']=='TRAINING_ROLLOUT_COST_COMPLETE' and v['job_id']=='3360' and v['code_commit']==source and v['config_sha256']==config for v in rollouts)
assert {v['candidate_outer'] for v in rollouts}==set(range(1,201))
totals=Counter()
for v in rollouts:totals.update(v['costs'])
validation_paths=sorted((run/'output/validation-cost').glob('*.json'));validations=[json.loads(p.read_text()) for p in validation_paths]
assert len(validations)==40 and all(v['status']=='VALIDATION_COST_COMPLETE' and v['evaluated_rows']==512 for v in validations)
peaks={};first=None;last=None;count=0;partial=0
with (run/'resource-samples.jsonl').open() as f:
 for line in f:
  try:v=json.loads(line)
  except json.JSONDecodeError:partial+=1;continue
  first=v['epoch'] if first is None else first;last=v['epoch'];count+=1
  for g in v['gpus']:
   p=peaks.setdefault(g['uuid'],dict(total_memory_bytes=g['total_memory_bytes'],total_used_memory_bytes=0,total_gpu_util=0,foreign_memory_fraction=0,foreign_sm_upper=0))
   for k in ('total_used_memory_bytes','total_gpu_util','foreign_memory_fraction','foreign_sm_upper'):p[k]=max(p[k],g[k])
process_max={};proc_samples=0;proc_partial=0;exited_without_ticks=0
with (run/'process-resources.jsonl').open() as f:
 for line in f:
  try:v=json.loads(line)
  except json.JSONDecodeError:proc_partial+=1;continue
  proc_samples+=1
  for p in v['processes']:
   if p.get('exited'):exited_without_ticks+=1;continue
   key=str(p['pid'])+':'+str(p.get('start_time_ticks',p.get('starttime_ticks','unrecorded')))
   seconds=(p['user_cpu_ticks']+p['system_cpu_ticks'])/p['hz'];process_max[key]=max(process_max.get(key,0),seconds)
cost=dict(observed_epoch=time.time(),job_id=3360,phase='ACTUAL_TRAIN200_COMPLETE_WITH_AUTHORIZED_PARTIAL_EVAL_HANDOFF',completed_outer=rank['completed_outer'],actual_successful_adam_per_rank=rank['actual_writes_per_rank'],actual_skips_per_rank=rank['skipped_opportunities_per_rank'],completed_rollout_cost_rows=len(rollouts),completed_rollout_totals=dict(totals),sampling_wall_seconds=sum(v['wall_seconds'] for v in rollouts),sampling_trainer_process_cpu_seconds=sum(v['trainer_process_cpu_seconds'] for v in rollouts),validation_reports=len(validations),validation_rows=sum(v['evaluated_rows'] for v in validations),validation_wall_seconds=sum(v['wall_seconds'] for v in validations),validation_cost_counters=dict(sum((Counter(v['counters']) for v in validations),Counter())),resource_samples=count,first_sample_epoch=first,last_sample_epoch=last,partial_resource_json_lines=partial,whole_card_peaks=peaks,resource_observed_gpu_hours=(last-first)*8/3600,resource_observed_allocated_cpu_core_hours=(last-first)*192/3600,resource_observed_requested_cpu_core_hours=(last-first)*32/3600,own_sampled_process_cpu_seconds_lower_bound=sum(process_max.values()),process_samples=proc_samples,partial_process_json_lines=proc_partial,process_exited_entries_without_final_ticks=exited_without_ticks,resource_cost_scope='Original8GPU reservation hours first-to-last sample including training/dev/slowpartialfinal; not actual GPU arithmetic usage. CPU process ticks lowerbound distinct from requested32/allocated192 reservation. Five-second whole-card peaks are lowerbounds on instantaneous maxima.',original_slow_final_metric_rows_preserved=30880,original_slow_final_compute_scope='Full final never completed; raw validation JSONL preserved remotely for separate wasted/partial cost accounting; do not count extra compute as saved scored rows.',code_commit=source,config_sha256=config,evidence_sha256={str(p):sha(p) for p in paths+validation_paths+[run/'resource-samples.jsonl',run/'process-resources.jsonl']},historical_failed_training_or_other_goal_runs_included=False)
out=run/'output/eval-handoff-acceptance-inputs/final-sampled-memory-and-cost.json';b=(json.dumps(cost,indent=2)+'\\n').encode()
if out.exists():raise RuntimeError('Existing final cost proof; inspect rather than overwrite')
with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(b).hexdigest(),cost=cost)))
'''
x=ssh.run('CUDA_VISIBLE_DEVICES= PYTHONOPTIMIZE=0 /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -c '+shlex.quote(code),timeout=120)
if x.returncode:raise RuntimeError(x.stderr+x.stdout)
Path('outputs/tree-eval-resume-20261013/FINAL_PREDECESSOR_COST.json').write_text(json.dumps(json.loads(x.stdout),indent=2)+'\n');s=json.loads(x.stdout);print(json.dumps(dict(path=s['path'],sha256=s['sha256'],completed_outer=s['cost']['completed_outer'],memory_peak_gib=max(v['total_used_memory_bytes'] for v in s['cost']['whole_card_peaks'].values())/1024**3,resource_samples=s['cost']['resource_samples'],gpu_reservation_hours=s['cost']['resource_observed_gpu_hours'])))
