"""Fail closed on role-bound retained actual evidence. No GPU work or mutations."""
import hashlib,json,math,os,re,socket,subprocess
from pathlib import Path
COMMIT='8e60976d364ab7d7ae51714415813c1359dafe9a'
OWN=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
RUNTIME_NAMES=set('job.sh gpu_job_entry.sh train_supervisor.py semantic_device_gate.py gpu_teacher_probe-factor-dtype.py gpu_loss_checkpoint_equivalence.py capacity_supervisor.py microbatch_probe.py gpu_retrieval_bulk.py http_retrieval_check.py retrieval_compare.py resource_sample.py model_preflight.py checkpoint_reload.py'.split())
FORMAL_NAMES=set('formal_entry.sh formal_job.sh formal_supervisor.py verify_formal_prerequisites.py micro_evidence.py collect_receipt.py controller_exit_evidence.py submit_formal.sh strict_metric_parser.py'.split())
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
 steps=[];normalization_events=[]
 for line in text.splitlines():
  m=re.search(r'(?:^|\s)step:(\d+) - ',line)
  if not m or 'actor/optimizer_steps:' not in line:continue
  from strict_metric_parser import parse_metric_summary
  step,metrics,events=parse_metric_summary(line);steps.append(step)
  normalization_events.extend(events)
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
 return {'completed_actual_outer_updates':count,'ranks':8,'samples_per_rank':320,'numeric_metrics_validated':True,'recognized_trailing_ray_warning_frames':normalization_events}
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
def verify_reload(data,checkpoint,job='2996'):
 assert data['slurm_job_id']==job and data['node'].split('.')[0]=='h100-10'
 assert Path(data['checkpoint'])==checkpoint
 for key in ('config_computation_equal','tokenizer_equal','all_loaded_parameters_finite'):assert data[key] is True
 assert type(data['tensor_count']) is int and data['tensor_count']>0
 assert re.fullmatch('[0-9a-f]{64}',data['tokenizer_semantic_sha256'])
 for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):assert data['loading_info'][key]==[]
 assert 'H100' in data['gpu_name'] and len(data['cuda_visible_devices'].split(','))==1
 assert data['generation'] and data['seconds']>0
 assert data['cuda_peak_reserved_bytes']>=data['cuda_peak_allocated_bytes']>0
 expected={'model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.v_proj.weight','model.layers.0.mlp.down_proj.weight','model.embed_tokens.weight' if data['tied_embeddings'] else 'lm_head.weight'}
 assert set(data['student_change'])==expected
 assert sum(v['changed_elements'] for v in data['student_change'].values())>0
 for v in data['student_change'].values():
  assert type(v['changed_elements']) is int and v['changed_elements']>=0
  assert all(math.isfinite(v[k]) and v[k]>=0 for k in ('max_abs_delta','mean_abs_delta'))
 assert set(data['weights'])=={p.name for p in checkpoint.glob('*.safetensors')} and data['weights']
 for name,v in data['weights'].items():
  assert Path(name).name==name
  p=own_file(str(checkpoint/name),checkpoint)
  assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
 for name in ('config.json','tokenizer.json'):own_file(str(checkpoint/name),checkpoint)
 index=checkpoint/'model.safetensors.index.json'
 if index.exists():
  mapping=loads(index.read_text())['weight_map']
  assert len(mapping)==data['tensor_count'] and set(mapping.values())==set(data['weights'])
 else:assert len(data['weights'])==1
 return True
def verify_receipt(receipt,base=OWN):
 r=loads(own_file(str(receipt),base).read_text())
 assert r['source_commit']==COMMIT and r['pilot_job']=='2994' and r['reload_job']=='2996'
 pilot=base/'runs/opsd-new-acceptance-2994';reload=base/'runs/opsd-new-hf-reload-2996'
 roles={'pilot_raw_log':pilot,'pilot_commit':pilot,'pilot_slurm':pilot,'pilot_exit':pilot,'pilot_finished':pilot,'reload_result':reload,'reload_commit':reload,'reload_slurm':reload,'reload_exit':reload,'reload_cleanup':reload,'independent_review':base}
 assert set(r['evidence'])==set(roles)
 paths={}
 for role,scope in roles.items():
  item=r['evidence'][role];p=own_file(item['path'],scope)
  assert digest(p)==item['sha256'];paths[role]=p
 for stage,job in (('pilot','2994'),('reload','2996')):
  assert paths[stage+'_commit'].read_text().strip()==COMMIT
  slurm_text=paths[stage+'_slurm'].read_text()
  if stage=='pilot' and slurm_text.lstrip().startswith('{'):
   from controller_exit_evidence import verify_controller_exit
   verify_controller_exit(loads(slurm_text),job,COMMIT,pilot,own_file,digest,loads)
  else:verify_slurm(slurm_text,job)
  assert paths[stage+'_exit'].read_text().strip()=='0'
 assert paths['pilot_finished'].read_text().strip()=='TRAINING_FIVE_ITERATIONS_EXITED'
 raw=paths['pilot_raw_log'].read_text()
 verify_outers(raw,5);verify_pilot_validations(raw)
 verify_reload(loads(paths['reload_result'].read_text()),pilot/'output/checkpoints/opsd-new-gpu-acceptance-5/actor/global_step_5')
 c=loads(paths['reload_cleanup'].read_text())
 assert c['job_id']=='2996' and c['node_scope']=='h100-10' and c['previous_training_job']=='2994'
 assert c['previous_job_processes']==[] and c['previous_ports_listening'] is False
 review=loads(paths['independent_review'].read_text())
 assert review['source_commit']==COMMIT and review['pilot_job']=='2994' and review['reload_job']=='2996' and review['accepted'] is True
 assert review['reviewer'] and review['report']
 assert review['bound_evidence_sha256']=={k:r['evidence'][k]['sha256'] for k in roles if k!='independent_review'}
 assert r['picked_local_micro']==2 and r['picked_global_micro']==16
 from micro_evidence import verify_micro_evidence
 verify_micro_evidence(review,pilot,r['pilot_job'],COMMIT,2)
 return r
def load_reviewed_formal_manifest(base):
 expected=os.environ.get('OPSD_STAGE_MANIFEST_SHA256','')
 assert re.fullmatch('[0-9a-f]{64}',expected), 'Externally reviewed final formal manifest SHA256 required'
 path=own_file(str(base/'tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-8b5eeaf1694a/source-manifest.json'),base)
 assert digest(path)==expected, 'Formal manifest changed from externally reviewed final pin'
 return loads(path.read_text())
def verify_deployed(base=OWN):
 manifest=load_reviewed_formal_manifest(base)
 assert manifest['source_commit']==COMMIT
 for field,names,directory in (('runtime_sources',RUNTIME_NAMES,'pilot-runtime-8e60976d364a-reviewed-4246006a5605'),('formal_sources',FORMAL_NAMES,'formal-8e60976d364a-pilot-2994-hf-2996-reviewed-8b5eeaf1694a')):
  assert set(manifest[field])==names
  for name,sha in manifest[field].items():
   p=own_file(str(base/'tools'/directory/name),base)
   assert digest(p)==sha, ('Deployed source mismatch',directory,name)
 assert set(manifest['pilot_additional_sources'])=={'pilot_audit.py','prior_cleanup.py','source_bundle_check.py'}
 for name,expected in manifest['pilot_additional_sources'].items():assert digest(own_file(str(base/'tools'/manifest['pilot_runtime_directory_name']/name),base))==expected
 assert set(manifest['hf_sources'])=={'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py'}
 for name,expected in manifest['hf_sources'].items():assert digest(own_file(str(base/'tools'/manifest['hf_runtime_directory_name']/name),base))==expected
 for field,folder in (('pilot_manifest_sha256','pilot_runtime_directory_name'),('hf_manifest_sha256','hf_runtime_directory_name')):assert digest(own_file(str(base/'tools'/manifest[folder]/'source-manifest.json'),base))==manifest[field]
 assert subprocess.check_output(['git','-C',str(base/'repo'),'rev-parse','HEAD'],text=True).strip()==COMMIT
 subprocess.run(['git','-C',str(base/'repo'),'diff','--quiet','HEAD'],check=True)
 assert set(manifest['production_math_sources'])=={'search_r1/baselines/actor.py','search_r1/baselines/distillation.py'}
 for name,expected in manifest['production_math_sources'].items():assert digest(own_file(str(base/'repo'/name),base))==expected
 return digest(base/'tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-8b5eeaf1694a/source-manifest.json')
def verify_previous_cleanup():
 job=os.environ['SLURM_JOB_ID'];node=socket.gethostname().split('.')[0]
 assert node=='h100-10' and os.environ.get('SLURM_JOB_NODELIST')=='h100-10'
 assert re.search(r'job_'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())
 found=[]
 for p in Path('/proc').glob('[0-9]*/cgroup'):
  try:cg=p.read_text()
  except FileNotFoundError:continue
  if re.search(r'job_(?:2994|2996)(?:/|$)',cg):found.append({'pid':int(p.parent.name),'cgroup':cg})
 listeners=subprocess.check_output(['ss','-H','-ltnp'],text=True)
 ports=[line for line in listeners.splitlines() if re.search(r':(?:18084|16384)\s',line)]
 result={'job_id':job,'node_scope':node,'previous_jobs':['2994','2996'],'previous_job_processes':found,'previous_ports_listening':ports,'read_only_no_kill':True}
 assert not found and not ports, ('Previous jobs not cleaned up',result)
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--cleanup',action='store_true');a=p.parse_args()
 if a.cleanup:
  result=verify_previous_cleanup();Path(os.environ['OPSD_EVIDENCE'],'previous-pilot-reload-cleanup.json').write_text(json.dumps(result,indent=2))
 else:
  sha=verify_deployed();verify_receipt(os.environ['OPSD_ACCEPTANCE_RECEIPT']);print('ROLE_BOUND_ACTUAL_PILOT_RELOAD_AND_DEPLOYED_SOURCE_VERIFIED',sha,flush=True)
