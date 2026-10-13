"""Preserve actual separated cost scopes on H100; return only compact totals."""
import base64,importlib.util,json,shlex
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ssh',HERE/'ssh_retained18.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
observations={'failed-startup-observations.json':Path('outputs/tree-eval-resume-20261013/SHARED_FAILED_ATTEMPT_COSTS.json').read_bytes()}
for p in sorted(Path('outputs/tree-eval-resume-20261013').glob('HANDOFF_HF_POLL_*.json')):
 observations['cpu-observation-'+p.name]=p.read_bytes()
code='''import base64,hashlib,json,math,os,stat,subprocess,time
from collections import Counter
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=r/'runs/tree-branpo-fresh200-3360';out=run/'output/eval-handoff-hf-acceptance';inputs=run/'output/eval-handoff-acceptance-inputs';scores=run/'output/eval-resume/scores';shared=r/'runs/sst-tree-eval-shared-0e00998'
def digest(p):
 assert stat.S_ISREG(p.lstat().st_mode),str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  first=os.fstat(f.fileno())
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  last=os.fstat(f.fileno())
 assert (first.st_size,first.st_mtime_ns,first.st_ctime_ns)==(last.st_size,last.st_mtime_ns,last.st_ctime_ns)
 return h.hexdigest()
bound={}
def bind(p,expected=None):
 p=Path(p);assert p.resolve().is_relative_to(r.resolve());h=digest(p);assert expected is None or h==expected,str(p);bound[str(p)]=h;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text())
def write_once(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
composite=read(out/'identity-bound-handoff-hf.json')
assert composite['status']=='JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED' and composite['evaluated_count']==51713
raw=subprocess.check_output(['systemctl','show','sst-tree-handoff-hf-440b8c6.service','--property=ActiveState,SubState,MainPID,ExecMainStatus,CPUUsageNSec'],text=True)
state=dict(x.split('=',1) for x in raw.splitlines() if '=' in x)
assert state['ActiveState']=='inactive' and state['MainPID']=='0' and state['ExecMainStatus']=='0',state
cpu_terminal=dict(observed_epoch=time.time(),unit='sst-tree-handoff-hf-440b8c6.service',systemd=state,actual_cpu_cores_reserved=0,physical_gpus=0,slurm_reserved_gpus=0)
write_once(inputs/'cpu-terminal-cost-observation.json',cpu_terminal);bind(inputs/'cpu-terminal-cost-observation.json')
for name,b64 in OBSERVATIONS.items():
 p=inputs/name;data=base64.b64decode(b64)
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 bind(p)
training=read(inputs/'final-sampled-memory-and-cost.json','f39bcfa27ed30007931ce07be74248075154e3b465c9f750f4cd7bf2c8f3a2e3')
assert training['completed_outer']==200 and training['completed_rollout_cost_rows']==200
for name,h in training['evidence_sha256'].items():bind(name,h)
partial=Counter();first=None;last=None;partial_rows=0
p=bind(run/'output/validation-cost/validation-4011013-41.jsonl')
with p.open() as f:
 for line in f:
  v=json.loads(line);assert v['phase']=='final' and v['validation_sequence']==41
  epoch=v['epoch'];first=epoch if first is None else first;last=epoch
  partial[v['kind']+'_events']+=1
  for k in ('nonpad_model_token_events','prefill_token_events','queries','query_result_rows','padding_rows','submitted_rows'):
   if k in v:partial[k]+=v[k]
  if v['kind']=='real_sample_batch':partial_rows+=v['rows']
shared_raw=subprocess.check_output(['systemctl','show','sst-tree-eval-shared-0e00998.service','--property=ActiveState,SubState,MainPID,ExecMainStatus,CPUUsageNSec'],text=True)
shared_state=dict(x.split('=',1) for x in shared_raw.splitlines() if '=' in x)
assert shared_state['ActiveState']=='inactive' and shared_state['MainPID']=='0' and shared_state['ExecMainStatus']=='0'
write_once(inputs/'shared-terminal-cost-observation.json',dict(observed_epoch=time.time(),unit='sst-tree-eval-shared-0e00998.service',systemd=shared_state));bind(inputs/'shared-terminal-cost-observation.json')
shared_cpu=shared_state.get('CPUUsageNSec');shared_cpu_exact=int(shared_cpu)/1e9 if shared_cpu and shared_cpu.isdigit() else None
terminal=read(shared/'run-result.json');assert terminal['status']=='COMPLETE' and terminal['own_cleanup_residual']=={} and terminal['worker_exit_code']==0
batch_totals=Counter();scored=0;batches=0
for p in sorted((scores/'batches').glob('*.done.json')):
 done=read(p);payload=read(p.with_name(p.name.replace('.done.json','.json')),done['sha256']);assert done['count']==len(payload['rows'])
 scored+=done['count'];batches+=1
 assert set(payload['cost'])=={'context_mismatches','generated_tokens','padded_rows','prefill_tokens','process_cpu_seconds','raw_discarded_tail_tokens','raw_emitted_tokens','raw_inherited_leaf_tokens','raw_retained_tokens','raw_slots','retrieval_calls','semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds','trajectories','wall_seconds'}
 assert payload['cost']['trajectories']==done['count']
 for k,v in payload['cost'].items():
  assert type(v) in (int,float) and math.isfinite(v) and v>=0,(k,v)
  batch_totals[k]+=v
assert scored==20833 and batch_totals['context_mismatches']==0
samples=0;peaks={}
p=bind(shared/'resource-samples.jsonl')
with p.open() as f:
 for line in f:
  v=json.loads(line);samples+=1
  for gpu in v['gpus']:
   old=peaks.setdefault(gpu['uuid'],dict(max_used_bytes=0,max_own_bytes=0,min_free_bytes=gpu['total_bytes']))
   old['max_used_bytes']=max(old['max_used_bytes'],gpu['used_bytes']);old['max_own_bytes']=max(old['max_own_bytes'],gpu['own_memory_bytes']);old['min_free_bytes']=min(old['min_free_bytes'],gpu['free_bytes'])
failed=read(inputs/'failed-startup-observations.json')
failed_inputs={}
failed_observation=dict(path=str(inputs/'failed-startup-observations.json'),sha256=bound[str(inputs/'failed-startup-observations.json')])
for short in ('1c15c49','b316a77','993109d','13a5388','7d28b1b'):
 unit='sst-tree-eval-shared-'+short+'.service';failure=r/'runs'/unit.removesuffix('.service')/'run-result.json'
 failed_inputs[unit]=[failed_observation]
 if failure.exists():
  v=read(failure);assert v['status']!='COMPLETE' and v['worker_exit_code']!=0 and v['own_cleanup_residual']=={}
  failed_inputs[unit].append(dict(path=str(failure),sha256=bound[str(failure)]))
for p in sorted((run/'output/eval-resume').glob('shared*dispatch-once.json')):bind(p)
full=read(out/'full-HF-cold-reload.json',composite['full_hf_report_sha256'])
for name in ['cpu-dispatch-once.json','cpu-submission-result.json']:bind(inputs/name)
cpu_observed=[]
for p in inputs.glob('cpu-observation-HANDOFF_HF_POLL_*.json'):
 v=read(p);value=v['systemd'].get('CPUUsageNSec');
 if value and value.isdigit():cpu_observed.append(int(value)/1e9)
cold_inputs=[dict(path=name,sha256=h) for name,h in bound.items() if Path(name).name.startswith(('cpu-','full-HF-cold-reload'))]
cpu_final=state.get('CPUUsageNSec');cpu_exact=int(cpu_final)/1e9 if cpu_final and cpu_final.isdigit() else None
summary=dict(schema_version='tree_train200_eval_handoff_cost_v1',status='JOB3360_COMPOSITE_ACTUAL_SCOPE_COST_COMPLETE',training_job_id='3360',training_source_commit='8b6f99d1b3353145200eb3fa7c16a579a79a987c',evaluation_source_commit='0e00998fdc645281986250cd8632bb70c6e2fa9b',composite_receipt_sha256=bound[str(out/'identity-bound-handoff-hf.json')],input_sha256=bound,cold_hf_service_inputs=cold_inputs,failed_own_startup_inputs=failed_inputs,scope_flags={k:True for k in ('original_train200','original_dev40','original_partial_evaluation','successful_replacement_evaluation','failed_own_evaluation_startups','actual_cpu_hf_acceptance')},training200={k:training[k] for k in ('completed_rollout_totals','sampling_wall_seconds','sampling_trainer_process_cpu_seconds','validation_reports','validation_rows','validation_wall_seconds','validation_cost_counters','resource_samples','resource_observed_gpu_hours','resource_observed_allocated_cpu_core_hours','resource_observed_requested_cpu_core_hours','own_sampled_process_cpu_seconds_lower_bound','whole_card_peaks')},original_partial_evaluation=dict(preserved_metric_rows=30880,actual_real_batch_rows_submitted=partial_rows,observed_event_totals=dict(partial),first_epoch=first,last_observed_event_epoch=last,observed_event_wall_span_seconds=last-first,scope='Incomplete original final-eval compute, includes uncredited in-flight work; never add these counters to successful replacement cost or equate submitted rows to persisted metric rows'),successful_replacement_evaluation=dict(actual_metric_rows=scored,committed_batches=batches,counters=dict(batch_totals),wall_seconds=terminal['wall_seconds'],physical_gpu_capacity_hours=8*terminal['wall_seconds']/3600,slurm_gpu_reservation_hours=0,whole_card_samples=samples,whole_card_peaks=peaks,systemd_total_cpu_seconds=shared_cpu_exact,systemd_total_cpu_reason='Actual terminal CPUUsageNSec available' if shared_cpu_exact is not None else 'Actual terminal systemctl observation has unavailable CPUUsageNSec; summed worker batch CPU excludes startup/retriever/supervisor'),failed_own_evaluation_startups=failed,actual_cpu_hf_acceptance=dict(full_verifier_process_cpu_seconds=full['process_cpu_seconds'],full_verifier_wall_seconds=full['wall_seconds'],peak_verifier_self_rss_bytes=full['peak_self_rss_bytes'],wrapper_postverification_wall_seconds=composite['wall_seconds'],service_observed_cpu_seconds_lower_bound=max(cpu_observed) if cpu_observed else None,service_lower_bound_reason='Actual active-service CPUUsageNSec observations' if cpu_observed else 'No actual numeric CPU observation available',service_exact_cpu_seconds=cpu_exact,physical_gpus=0,slurm_reserved_gpus=0),limitations=['5-second NVML peaks are sampled lower bounds, not instantaneous peaks or available-capacity guarantees.','GPU capacity/reservation wall-hours are not arithmetic compute hours.','Original cost scope includes train/dev/slow partial final; sub-scopes are not additive to its total reservation hours.','Original raw PAD-filtered token counts differ from new native emitted/retained token counters; both conventions are retained.','Previous failed training runs and other experiment jobs are excluded.','Unknown complete transient-service CPU accounting is unknown, not zero.'])
write_once(out/'cost-summary.json',summary)
print(json.dumps(dict(path=str(out/'cost-summary.json'),sha256=digest(out/'cost-summary.json'),schema_version=summary['schema_version'],status=summary['status'],bound_inputs=len(bound),training200=summary['training200'],original_partial_evaluation=summary['original_partial_evaluation'],successful_replacement_evaluation=summary['successful_replacement_evaluation'],actual_cpu_hf_acceptance=summary['actual_cpu_hf_acceptance'],limitations=summary['limitations'])))
'''
code=code.replace('OBSERVATIONS',repr({n:base64.b64encode(v).decode() for n,v in observations.items()}))
r=ssh.run('python3 -c '+shlex.quote(code),timeout=180)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
d=json.loads(r.stdout)
Path('outputs/tree-eval-resume-20261013/COMPOSITE_COST_SUMMARY.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(path=d['path'],sha256=d['sha256'],bound_inputs=d['bound_inputs'])))
