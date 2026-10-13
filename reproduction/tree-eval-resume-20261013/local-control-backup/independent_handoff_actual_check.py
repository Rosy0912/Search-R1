"""Independent JSON/hash-only artifact review. No model load or remote mutation."""
import importlib.util,json,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('retained_route',HERE/'ssh_retained18.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code=r'''import hashlib,importlib.util,json,math,os,re,socket,stat,subprocess,sys,time
from collections import Counter
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-10'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sys.flags.optimize==0
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');RUN=ROOT/'runs/tree-branpo-fresh200-3360';OUT=RUN/'output/eval-handoff-hf-acceptance';INPUT=RUN/'output/eval-handoff-acceptance-inputs'
COMPOSITE_SHA='2ba00537b667b300cf4c0fd42328b2e080f3335ea2e1dff7925fdef42be14654'
COST_SHA='364a01760b9fdc786746d004b369d7d637218e6d43a87085d53ed2a60bb1e623'
WRAPPER_SHA='8cc378a52009a8a41740831aa2259f380004da94760b3e7e98c4c8243a9c79ff'
observed={};hashes={}
def require(x,message):
 if not x:raise ValueError(message)
def fingerprint(p):
 s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def sha(path,expected=None):
 p=Path(path);require(p.is_absolute() and p.resolve().is_relative_to(ROOT.resolve()) and stat.S_ISREG(p.lstat().st_mode),'Unsafe evidence '+str(p));key=str(p.resolve())
 if key not in hashes:
  with p.open('rb') as f:
   first=os.fstat(f.fileno());h=hashlib.sha256()
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   last=os.fstat(f.fileno())
  state=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  require(state(first)==state(last),'Changed during hash '+str(p));hashes[key]=h.hexdigest();observed[key]=state(last)
 require(fingerprint(p)==observed[key],'Changed after hash '+str(p));require(expected is None or hashes[key]==expected,'SHA differs '+str(p));return hashes[key]
def read(path,expected=None):
 h=sha(path,expected);v=json.loads(Path(path).read_text());require(fingerprint(Path(path))==observed[str(Path(path).resolve())],'Changed JSON '+str(path));return v,h
def phase(name):print('REVIEW_PHASE '+name,flush=True)
c,ch=read(OUT/'identity-bound-handoff-hf.json',COMPOSITE_SHA);cost,costh=read(OUT/'cost-summary-v2.json',COST_SHA)
require(c['schema_version']=='tree_train200_eval_handoff_composite_v1' and c['status']=='JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED','Composite schema/status')
require(c['actual_completed_outer']==200 and c['actual_writes_per_rank']=={str(i):4000 for i in range(8)},'Actual200/rank4000')
require(c['original_cancelled_state']=='CANCELLED' and c['original_slurm_exit_code']=='0:15' and c['original_training_body_exit_code'] is None and c['original_training_normal_exit_claimed'] is False and c['original_final_evaluation_completed'] is False and c['original_evaluation_stop_authorized'] is True,'Original truthful handoff')
require(c['replacement_evaluation_complete'] is True and c['evaluated_count']==51713 and c['preserved_count']==30880 and c['replacement_count']==20833 and c['replacement_worker_exit_code']==0 and c['replacement_own_cleanup_residual']=={},'Final coverage/cleanup')
require(c['training_or_optimizer_restored'] is False and c['independent_final_acceptance'] is False,'No fabricated restore/acceptance')
before=c['input_hashes_before'];require(before==c['input_hashes_after'] and before,'Immutable cold inputs')
phase('hashing_all_bound_composite_and_cost_inputs')
for item in before.values():sha(item['path'],item['sha256'])
for path,h in cost['input_sha256'].items():sha(path,h)
phase('bound_file_hashes_verified')
wrapper=Path(before['wrapper']['path']);sha(wrapper,WRAPPER_SHA)
spec=importlib.util.spec_from_file_location('reviewed_acceptance_source',wrapper);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
require(c['wrapper_sha256']==WRAPPER_SHA and c['acceptance_wrapper_files']=={'verify_handoff_hf.py':before['wrapper']},'Exact reviewed wrapper')
package,_=read(before['package_manifest']['path'],w.PACKAGE_SHA);require(package['code_commit']==w.SOURCE,'Original source')
require(c['training_source_commit']==w.SOURCE and c['evaluation_source_commit']==w.EVAL_SOURCE and c['config_sha256']==w.CONFIG,'Separate source/config identity')
sha(before['frozen_verifier']['path'],w.VERIFIER_SHA)
for name,h in package['files'].items():require(before['frozen_package/'+name]['sha256']==h,'Frozen inventory '+name)
release,_=read(before['eval_release']['path'],w.RELEASE_SHA);manifest,_=read(before['eval_asset_manifest']['path'],w.MANIFEST_SHA)
require(release['source_commit']==w.EVAL_SOURCE and manifest['config_sha256']==w.CONFIG and manifest['expected_source_counts']==w.EXPECTED_COUNTS and manifest['dataset_count']==51713,'Actual evaluation asset/config/source')
for entry in manifest['dataset_files']:sha(entry['path'],entry['sha256'])
original,_=read(before['original_job_terminal']['path'],'732e971af9260bff7afc3fabc18de6a7736a8d8e49f9fa3ba5892640cdf12221')
historical,_=read(before['original_scontrol_observation']['path'],'af0762ee0af4a94a03138480d03d9e7ec5c2eb8fe7a795a537e72cf8b7cdcef0');handoff,_=read(before['handoff_journal']['path'])
require(handoff==historical['journal'] and original['raw']==historical['old_slurm'] and original['observed_epoch']==historical['observed_epoch'],'Historical authority/raw identity')
require(original['state']=='CANCELLED' and original['exit_code']=='0:15' and handoff['cancel_returncode']==0 and handoff['scope']=='User-authorized evaluation-only handoff; no training restart','Authorized cancellation')
require(not (RUN/'run-result.json').exists() and not (RUN/'output/train200-terminal.json').exists(),'Absent original normal receipts')
queue=subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True);require(not set(queue.split()).intersection({'3360','3392'}),'Old own jobs absent')
states={}
for unit in ('sst-tree-eval-shared-0e00998.service','sst-tree-handoff-hf-440b8c6.service'):
 raw=subprocess.check_output(['systemctl','show',unit,'--property=ActiveState,SubState,MainPID,ExecMainStatus'],text=True);state=dict(x.split('=',1) for x in raw.splitlines() if '=' in x);require(state['ActiveState']=='inactive' and state['MainPID']=='0' and state['ExecMainStatus']=='0','Own terminal '+unit);states[unit]=state
terminal,th=read(before['replacement_terminal']['path'],c['replacement_terminal_sha256']);require(terminal['status']=='COMPLETE' and terminal['worker_exit_code']==0 and terminal['own_cleanup_residual']=={} and terminal['unit']==w.UNIT,'Replacement actual result')
allocation,_=read(before['replacement_allocation']['path']);require(allocation['release_sha256']==w.RELEASE_SHA and allocation['physical_gpus']==8 and allocation['slurm_reserved_gpus']==0,'Shared allocation')
sys.path.insert(0,str(Path(before['package_manifest']['path']).parent))
from production_contract import direct_rank_records,tree_and_safety_records,checkpoint_manifest
rank=direct_rank_records(RUN,package,w.CONFIG,200);require(rank['actual_writes_per_rank']==c['actual_writes_per_rank'] and rank['skipped_opportunities_per_rank']=={str(i):0 for i in range(8)},'Recomputed actual rank writes')
tree=tree_and_safety_records(RUN,200,package,w.CONFIG)
for path,h in dict(rank['record_sha256'],**tree['evidence_sha256']).items():require(before['training/'+path]['sha256']==h,'Recomputed training binding')
inventory=w.exact_attempt_inventory(RUN,rank['attempt_accounting']);published_inventory,ih=read(OUT/'exact-attempt-inventory.json',c['exact_attempt_inventory_sha256']);require(inventory==published_inventory and len(inventory)==200,'Full exact attempt inventory')
actual_numerics=[]
for event in rank['attempt_accounting']['credited']:
 for arank in range(8):
  p=RUN/'output/tree-rank-audit'/('step-%06d-attempt-%06d-rank-%02d.json'%(event['outer'],event['attempt'],arank));audit,_=read(p,before['training/'+str(p)]['sha256']);actual_numerics.append(w.check_rank_numerics(audit,event['outer'],arank))
num,nh=read(OUT/'training-numerical-json-proof.json',c['training_numerical_report_sha256']);require(num['records']==actual_numerics and num['rank_records']==1600 and num['finite_positive_gradient_norms']==32000 and num['zero_lr_outer1_rank_records']==8 and num['nonzero_lr_sample_movement_rank_records']==1592,'Actual1600 numerical proofs')
native,nsh=checkpoint_manifest(RUN,package,w.CONFIG,200);require(nsh==c['native_manifest_sha256'] and native['hf_identity']['sha256']==c['hf_identity_sha256'],'Native200/HF identity')
for record in native['worker_states']+[native['driver']]:sha(record['path'],record['sha256'])
def inventory_identity(path):
 files={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()};return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
base_identity=inventory_identity(w.BASE);hf_identity=inventory_identity(w.CHECKPOINT);require(base_identity==package['model_identity'] and hf_identity==native['hf_identity'],'Immutable actual base/HF inventory')
full,fh=read(OUT/'full-HF-cold-reload.json',c['full_hf_report_sha256']);tokenizer,tkh=read(OUT/'tokenizer-cold-reload.json',c['tokenizer_report_sha256'])
require(full['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and full['all_parameters_finite'] is True and full['cpu_forward_logits_finite'] is True and full['changed_parameters']>0 and full['total_parameters']>1000000000 and full['max_abs_parameter_delta']>0 and full['checkpoint_file_sha256']==hf_identity['files'] and not any(full['hf_loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Actual full cold model evidence')
require(Path(full['base']).resolve()==w.BASE.resolve() and Path(full['checkpoint']).resolve()==w.CHECKPOINT.resolve(),'Cold paths')
require(tokenizer['status']=='ACTUAL_COLD_TOKENIZER_IDENTITY_VERIFIED' and tokenizer['actual_encode_decode_probes']==3 and tokenizer['actual_chat_template_comparisons']==4 and all(tokenizer[k] is True for k in ('vocab_equal','special_pad_bos_eos_unk_equal','all_special_tokens_ids_equal','chat_template_equal','canonical_backend_equal','full_added_token_attributes_equal','full_special_roles_equal','normalized_semantic_config_equal')),'Actual tokenizer evidence')
phase('training_numerics_native_cold_reports_verified_recomputing51713')
ledger_bindings={};ledger_checked={};final=w.complete_ledger(manifest,ledger_bindings,ledger_checked);require(final['metrics']==c['final_metrics'] and final['evaluated_count']==51713 and c['final_per_dataset']==w.EXPECTED_COUNTS,'Independently recomputed final ledger metrics')
for name,path in ledger_bindings.items():require(before[name]['sha256']==ledger_checked[name] and Path(before[name]['path']).resolve()==path.resolve(),'Complete ledger descriptor '+name)
require(cost['schema_version']=='tree_train200_eval_handoff_cost_v1' and cost['status']=='JOB3360_COMPOSITE_ACTUAL_SCOPE_COST_COMPLETE' and cost['coverage_revision']==2 and cost['composite_receipt_sha256']==ch,'V2 cost identity')
require(cost['training_source_commit']==w.SOURCE and cost['evaluation_source_commit']==w.EVAL_SOURCE and all(cost['scope_flags'][k] is True for k in ('original_train200','original_dev40','original_partial_evaluation','successful_replacement_evaluation','failed_own_evaluation_startups','actual_cpu_hf_acceptance')),'Separate cost scope flags')
def cost_covers(path,h):
 matches=[v for p,v in cost['input_sha256'].items() if Path(p).resolve()==Path(path).resolve()];require(matches==[h],'Cost exact coverage '+str(path))
training,trh=read(INPUT/'final-sampled-memory-and-cost.json','f39bcfa27ed30007931ce07be74248075154e3b465c9f750f4cd7bf2c8f3a2e3');cost_covers(INPUT/'final-sampled-memory-and-cost.json',trh)
require(training['completed_outer']==200 and training['completed_rollout_cost_rows']==200 and training['validation_reports']==40 and training['validation_rows']==20480 and len(training['evidence_sha256'])==242,'Train200/dev40 underlying cost')
for path,h in training['evidence_sha256'].items():cost_covers(path,h)
for name,item in before.items():
 if name in ('replacement_terminal','replacement_allocation','final_metrics','final_metrics_done') or name.startswith(('batch/','batch_done/','original_partial_cost/')):cost_covers(item['path'],item['sha256'])
for p,h in [(OUT/'full-HF-cold-reload.json',fh),(OUT/'tokenizer-cold-reload.json',tkh),(OUT/'actual-terminal-observation.json',c['actual_terminal_observation_sha256'])]:cost_covers(p,h)
require(cost['training200']=={k:training[k] for k in cost['training200']},'Captured original cost scopes preserved')
rollouts=[];dev=[]
for path,h in training['evidence_sha256'].items():
 if '/training-rollout-cost/' in path:rollouts.append(read(path,h)[0])
 if '/validation-cost/' in path:dev.append(read(path,h)[0])
require(len(rollouts)==200 and {v['candidate_outer'] for v in rollouts}==set(range(1,201)) and all(v['status']=='TRAINING_ROLLOUT_COST_COMPLETE' and v['job_id']=='3360' and v['code_commit']==w.SOURCE and v['config_sha256']==w.CONFIG for v in rollouts),'Actual rollout200 cost rows')
tot=Counter()
for v in rollouts:tot.update(v['costs'])
require(dict(tot)==training['completed_rollout_totals'] and sum(v['wall_seconds'] for v in rollouts)==training['sampling_wall_seconds'] and sum(v['trainer_process_cpu_seconds'] for v in rollouts)==training['sampling_trainer_process_cpu_seconds'],'Recomputed rollout totals')
require(len(dev)==40 and all(v['status']=='VALIDATION_COST_COMPLETE' and v['evaluated_rows']==512 for v in dev),'Actual completed dev40')
require(sum(v['wall_seconds'] for v in dev)==training['validation_wall_seconds'] and dict(sum((Counter(v['counters']) for v in dev),Counter()))==training['validation_cost_counters'],'Recomputed dev cost')
resource_count=0;resource_first=None;resource_last=None;resource_peaks={}
for line in (RUN/'resource-samples.jsonl').open():
 v=json.loads(line);resource_count+=1;resource_first=v['epoch'] if resource_first is None else resource_first;resource_last=v['epoch']
 for gpu in v['gpus']:
  peak=resource_peaks.setdefault(gpu['uuid'],dict(total_memory_bytes=gpu['total_memory_bytes'],total_used_memory_bytes=0,total_gpu_util=0,foreign_memory_fraction=0,foreign_sm_upper=0))
  for key in ('total_used_memory_bytes','total_gpu_util','foreign_memory_fraction','foreign_sm_upper'):peak[key]=max(peak[key],gpu[key])
require(resource_count==training['resource_samples'] and resource_first==training['first_sample_epoch'] and resource_last==training['last_sample_epoch'] and resource_peaks==training['whole_card_peaks'],'Actual original resource sampled peaks')
require(training['resource_observed_gpu_hours']==(resource_last-resource_first)*8/3600 and training['resource_observed_allocated_cpu_core_hours']==(resource_last-resource_first)*192/3600 and training['resource_observed_requested_cpu_core_hours']==(resource_last-resource_first)*32/3600,'Original actual capacity/reservation hours')
process_max={};process_samples=0;exited=0
for line in (RUN/'process-resources.jsonl').open():
 v=json.loads(line);process_samples+=1
 for proc in v['processes']:
  if proc.get('exited'):exited+=1;continue
  key=str(proc['pid'])+':'+str(proc.get('start_time_ticks',proc.get('starttime_ticks','unrecorded')));seconds=(proc['user_cpu_ticks']+proc['system_cpu_ticks'])/proc['hz'];process_max[key]=max(process_max.get(key,0),seconds)
require(training['own_sampled_process_cpu_seconds_lower_bound']==sum(process_max.values()) and training['process_samples']==process_samples and training['process_exited_entries_without_final_ticks']==exited,'Original observed CPU lowerbound')
partial=Counter();rows=0;first=None;last=None
for line in (RUN/'output/validation-cost/validation-4011013-41.jsonl').open():
 v=json.loads(line);require(v['phase']=='final' and v['validation_sequence']==41,'Partial41 identity');first=v['epoch'] if first is None else first;last=v['epoch'];partial[v['kind']+'_events']+=1
 for k in ('nonpad_model_token_events','prefill_token_events','queries','query_result_rows','padding_rows','submitted_rows'):
  if k in v:partial[k]+=v[k]
 if v['kind']=='real_sample_batch':rows+=v['rows']
pc=cost['original_partial_evaluation'];require(pc['observed_event_totals']==dict(partial) and pc['actual_real_batch_rows_submitted']==rows and pc['preserved_metric_rows']==30880 and pc['first_epoch']==first and pc['last_observed_event_epoch']==last and pc['observed_event_wall_span_seconds']==last-first,'Separate incomplete41 cost recomputation')
require(not (RUN/'output/validation-cost/validation-4011013-41.json').exists(),'Incomplete original final remains incomplete')
batchcost=Counter();batchrows=0;batchcount=0
for p in sorted((w.SCORES/'batches').glob('*.done.json')):
 done,_=read(p);payload,_=read(p.with_name(p.name.replace('.done.json','.json')),done['sha256']);require(done['count']==len(payload['rows'])==payload['cost']['trajectories'],'Native batch actual rows');require(set(payload['cost'])=={'context_mismatches','generated_tokens','padded_rows','prefill_tokens','process_cpu_seconds','raw_discarded_tail_tokens','raw_emitted_tokens','raw_inherited_leaf_tokens','raw_retained_tokens','raw_slots','retrieval_calls','semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds','trajectories','wall_seconds'} and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in payload['cost'].values()),'Full native cost fields');batchrows+=done['count'];batchcount+=1;batchcost.update(payload['cost'])
sc=cost['successful_replacement_evaluation'];require(batchrows==20833 and sc['actual_metric_rows']==batchrows and sc['committed_batches']==batchcount and sc['counters']==dict(batchcost) and sc['wall_seconds']==terminal['wall_seconds'] and sc['physical_gpu_capacity_hours']==8*terminal['wall_seconds']/3600 and sc['slurm_gpu_reservation_hours']==0,'Separate successful20833 cost recomputation')
shared_samples=0;shared_peaks={}
for line in (ROOT/'runs/sst-tree-eval-shared-0e00998/resource-samples.jsonl').open():
 v=json.loads(line);shared_samples+=1
 for gpu in v['gpus']:
  peak=shared_peaks.setdefault(gpu['uuid'],dict(max_used_bytes=0,max_own_bytes=0,min_free_bytes=gpu['total_bytes']));peak['max_used_bytes']=max(peak['max_used_bytes'],gpu['used_bytes']);peak['max_own_bytes']=max(peak['max_own_bytes'],gpu['own_memory_bytes']);peak['min_free_bytes']=min(peak['min_free_bytes'],gpu['free_bytes'])
require(shared_samples==sc['whole_card_samples'] and shared_peaks==sc['whole_card_peaks'],'Actual shared resource sampled peaks')
for kind,unit in [('shared','sst-tree-eval-shared-0e00998.service'),('cpu','sst-tree-handoff-hf-440b8c6.service')]:
 obs,_=read(INPUT/(kind+'-terminal-cost-observation.json'));require(obs['unit']==unit and obs['systemd']['ActiveState']=='inactive' and obs['systemd']['MainPID']=='0' and obs['systemd']['ExecMainStatus']=='0','Actual cost terminal observation');value=obs['systemd'].get('CPUUsageNSec');seconds=int(value)/1e9 if value and value.isdigit() else None
 if kind=='shared':require(sc['systemd_total_cpu_seconds']==seconds,'Shared CPU unknown/numeric')
 else:require(cost['actual_cpu_hf_acceptance']['service_exact_cpu_seconds']==seconds,'CPU exact unknown/numeric')
cpu=cost['actual_cpu_hf_acceptance'];require(cpu['full_verifier_process_cpu_seconds']==full['process_cpu_seconds'] and cpu['full_verifier_wall_seconds']==full['wall_seconds'] and cpu['peak_verifier_self_rss_bytes']==full['peak_self_rss_bytes'] and cpu['wrapper_postverification_wall_seconds']==c['wall_seconds'] and cpu['physical_gpus']==cpu['slurm_reserved_gpus']==0,'Cold CPU actual costs')
lower=[]
for path,h in cost['input_sha256'].items():
 if Path(path).name.startswith('cpu-observation-HANDOFF_HF_POLL_'):
  value=read(path,h)[0]['systemd'].get('CPUUsageNSec')
  if value and value.isdigit():lower.append(int(value)/1e9)
require(cpu['service_observed_cpu_seconds_lower_bound']==(max(lower) if lower else None),'No unknown CPU zero')
failed=cost['failed_own_evaluation_startups'];failedmap={v['unit']:v for v in failed['attempts']}
require(set(failedmap)==set(cost['failed_own_startup_inputs']) and len(failedmap)==5,'Five actual failed own scopes')
for unit,items in cost['failed_own_startup_inputs'].items():
 require(len(items)>=2,'Actual failure observation+dispatch scope');journals=[]
 for item in items:
  cost_covers(item['path'],item['sha256'])
  if 'dispatch-once.json' in item['path']:journals.append(read(item['path'],item['sha256'])[0])
 require(any(v['unit']==unit for v in journals),'Actual failed own journal identity')
 v=failedmap[unit];s=dict(x.split('=',1) for x in v['actual_systemd'].splitlines() if '=' in x);require(s['MainPID']=='0' and s['ActiveState']=='failed' and s['ExecMainStatus']=='1','Actual failed terminal')
 if v['result'] is None:require(not (ROOT/'runs'/unit.removesuffix('.service')/'run-result.json').exists(),'Admission failure no synthetic result')
 else:
  actual,_=read(ROOT/'runs'/unit.removesuffix('.service')/'run-result.json');require(actual==v['result'] and actual['status']=='FAILED' and actual['worker_exit_code']==1 and actual['own_cleanup_residual']=={},'Actual failed runtime/result cleanup')
for item in cost['cold_hf_service_inputs']:cost_covers(item['path'],item['sha256'])
phase('all_actual_metrics_cost_and_failure_scopes_verified')
require(all(fingerprint(Path(p))==s for p,s in observed.items()),'Evidence changed during independent review')
require(w.exact_attempt_inventory(RUN,rank['attempt_accounting'])==inventory,'Final attempt inventory remains stable')
sha(OUT/'identity-bound-handoff-hf.json',COMPOSITE_SHA);sha(OUT/'cost-summary-v2.json',COST_SHA)
review=dict(schema_version='tree_train200_eval_handoff_composite_v1',status='INDEPENDENT_JOB3360_TRAIN200_EVAL_HANDOFF_ACCEPTED',fork_turns='none',inherited_parent_conversation=False,blocking_findings=[],reviewed_epoch=time.time(),composite_receipt_sha256=ch,training_source_commit=w.SOURCE,evaluation_source_commit=w.EVAL_SOURCE,reviewed_wrapper_files=c['acceptance_wrapper_files'],training200_accepted=True,final51713_accepted=True,full_hf_accepted=True,tokenizer_accepted=True,native_preservation_accepted=True,authorized_handoff_accepted=True,cleanup_accepted=True,cost_accepted=True,cost_summary=dict(path=str(OUT/'cost-summary-v2.json'),sha256=costh),actual_completed_outer=200,actual_writes_per_rank=c['actual_writes_per_rank'],actual_rank_numerical_records=1600,actual_finite_positive_gradient_norms=32000,outer1_zero_lr_rank_records=8,nonzero_lr_sample_movement_rank_records=1592,final_per_dataset=w.EXPECTED_COUNTS,final_metrics=final['metrics'],full_hf_report_sha256=fh,tokenizer_report_sha256=tkh,native_manifest_sha256=nsh,training_numerical_report_sha256=nh,exact_attempt_inventory_sha256=ih,actual_bound_unique_files=len(hashes),cost_bound_inputs=len(cost['input_sha256']),original_cancelled_state='CANCELLED',original_slurm_exit_code='0:15',original_training_normal_exit_claimed=False,original_training_body_exit_code=None,original_final_evaluation_completed=False,actual_terminal_states=states,weights_left_on_h100_10=True,training_or_optimizer_restored=False,review_method='Asset-local independent JSON/raw-file SHA verification, frozen direct training contracts recomputed, actual1600 audit proofs reread, exact51713 ordered question ledger/metrics recomputed, actual separated cost totals recomputed; no model/PT load or offnode tensor copy',limitations=['Fixed8192-coordinate movement proofs do not establish full-shard temporal finiteness; actual fullcoldHF proves final full parameter finiteness.','Resource peaks are sampled lowerbounds; reservation/capacity hours are not arithmetic usage.','Unavailable exact serviceCPU remains unknown; actual numeric historicalCPU observations are retained separately.'],numerical_caveat=c['numerical_caveat'])
print('REVIEW_RESULT '+json.dumps(review,allow_nan=False),flush=True)
'''
command='CUDA_VISIBLE_DEVICES= PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -c '+shlex.quote(code)
p=subprocess.Popen(ssh.route(command),stdout=subprocess.PIPE,stderr=sys.stderr,text=True)
result=None
for line in p.stdout:
 if line.startswith('REVIEW_RESULT '):result=json.loads(line.removeprefix('REVIEW_RESULT '))
 else:print(line.rstrip(),flush=True)
exit_code=p.wait()
if exit_code:raise SystemExit(exit_code)
if result is None:raise RuntimeError('Independent review result absent')
path=Path('outputs/tree-eval-resume-20261013/independent-composite-acceptance.json')
with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(path=str(path),status=result['status'],cost_summary=result['cost_summary'],actual_bound_unique_files=result['actual_bound_unique_files'])),flush=True)
