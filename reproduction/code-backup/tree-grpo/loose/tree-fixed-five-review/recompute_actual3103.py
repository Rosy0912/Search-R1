import json,hashlib,collections,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'work/tree-fixed-five-review';P=OUT/'actual3103-files/runs/tree-fixed-five-3103';C=OUT/'actual3103-files/runs/tree-fixed-five-cpu-3103-3104';CAND=ROOT/'work/tree-numeric-candidate';COMMIT='296217a4b676a26a92fc729f5d9882b785dbfa96'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=sha(P/'resolved.yaml');assert cfg==sha(P/'output/entry-audit/entry-config.yaml')
launch=read(P/'launch-manifest.json');assert launch['config_sha256']==cfg and launch['code_commit']==COMMIT and launch['job_id']=='3103' and '/job_3103/step_batch' in launch['cgroup']
entry=[json.loads(x) for x in (P/'output/entry-audit/entry.jsonl').read_text().splitlines()];assert len(entry)==1 and entry[0]['actual_config_sha256']==cfg and entry[0]['entry_total_training_steps']==6 and entry[0]['commit']==COMMIT
state=read(P/'output/checkpoints/tree_grpo-qwen2.5-3b/algorithm/global_step_5/state.json');assert state['method']=='tree_grpo' and state['global_step']==5 and not state['resume_supported']
fields={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'};steps=[];trajectories=[]
for step in range(1,6):
 rfile=P/f'output/tree-audit/step_{step}_rollout_{step}.json';pfile=rfile.with_name(f'step_{step}_rollout_{step}_policy_batch.json');r=read(rfile);pol=read(pfile)
 assert r['global_step']==r['rollout_sequence']==pol['global_step']==pol['rollout_sequence']==step and r['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and pol['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE'
 assert len(r['branches'])==2048 and len(r['leaves'])==len(pol['rows'])==3072 and len(r['question_groups'])==512 and pol['empty_leaves']==0
 assert pol['rollout_file_sha256']==sha(rfile)
 source=collections.Counter(b['source'] for b in r['branches']);assert len(source)==1024 and set(source.values())=={2}
 for b in r['branches']:assert b['inherited'] and set(b['parent_fields'])==set(b['child_fields'])==fields and b['parent_fields']==b['child_fields']
 trees=collections.Counter(l['tree_hash'] for l in r['leaves']);questions=collections.defaultdict(list)
 for l in r['leaves']:
  questions[l['question_hash']].append(l);assert l['context_matches'] and l['generation_prefix_hash']==l['scoring_prefix_hash'] and l['action_count']+l['observation_count']==l['response_length'] and l['padding_count']>=0
 assert len(trees)==1024 and set(trees.values())=={3} and len(questions)==512 and all(len(leaves)==6 and len({l['tree_hash'] for l in leaves})==2 for leaves in questions.values())
 assert all(g['roots']==2 and g['leaves']==6 for g in r['question_groups'])
 ids={l['trajectory_hash']:l for l in r['leaves']};assert len(ids)==3072 and {row['trajectory_hash'] for row in pol['rows']}==set(ids);trajectories.append(set(ids))
 for row in pol['rows']:
  l=ids[row['trajectory_hash']];assert row['question_hash']==l['question_hash'] and row['tree_hash']==l['tree_hash'] and row['effective_tokens']==l['action_count'] and not row['empty_leaf'] and abs(row['policy_weight_sum']-1)<1e-6
 ranks=[read(f) for f in (P/'output/tree-rank-audit').glob(f'step-{step:06d}-rank-*.json')];assert len(ranks)==8 and {v['rank'] for v in ranks}==set(range(8)) and len({v['pid'] for v in ranks})==8
 assert all(v['existing_grad_norms']==ranks[0]['existing_grad_norms'] for v in ranks)
 for v in ranks:
  assert v['outer_step']==step and v['world_size']==8 and v['job_id']=='3103' and v['run_id']=='tree-fixed-five-3103' and v['code_commit']==COMMIT and v['config_sha256']==cfg and v['data_identity']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
  assert v['actor_source_sha256']==sha(CAND/'verl/workers/actor/dp_actor.py') and v['auditor_source_sha256']==sha(CAND/'search_r1/baselines/tree_rank_audit.py')
  assert (v['local_rows'],v['local_minibatch'],v['local_microbatch'],v['epochs'])==(384,32,8,1)
  assert v['successful_optimizer_calls']==v['finite_grad_norms']==v['nonzero_grad_norms']==len(v['existing_grad_norms'])==12 and v['skipped_empty_minibatches']==0 and all(math.isfinite(n) and n>0 for n in v['existing_grad_norms'])
  d=v['local_shard_sample_delta'];assert v['local_shard_sample_layout_unchanged'] and d['count']==d['finite']==8192
  if step==1:assert d['nonzero']==0 and v['local_shard_sample_before_sha256']==v['local_shard_sample_after_sha256']
  else:assert d['nonzero']>0 and v['local_shard_sample_before_sha256']!=v['local_shard_sample_after_sha256']
 steps.append({'step':step,'question_groups':512,'trees':1024,'leaves':3072,'source_exact2branches':True,'rollout_sha256':sha(rfile),'policy_sha256':sha(pfile),'policy_mass_min':min(row['policy_weight_sum'] for row in pol['rows']),'policy_mass_max':max(row['policy_weight_sum'] for row in pol['rows']),'branch_response_nonempty':sum(b['parent_fields']['response']!=hashlib.sha256(b'[]').hexdigest() for b in r['branches']),'prefix_token_events_prompt_action_observation':sum(b['prefix_length'] for b in r['branches']),'rank_sample_deltas':{v['rank']:v['local_shard_sample_delta'] for v in ranks},'diagnostic_cost':{'rollout':r['diagnostic_check_cost'],'policy':pol['diagnostic_check_cost']}})
assert len(set.union(*trajectories))==3072*5
text=re.sub(r'\x1b\[[0-9;]*m','',(P/'stage.log').read_text());metrics={}
for line in text.splitlines():
 match=re.search(r'\bstep:(\d+) - (.*)',line)
 if not match:continue
 step=int(match.group(1));assert step not in metrics;values={}
 for value in match.group(2).split(' - '):
  name,number=value.rsplit(':',1);values[name]=float(number);assert math.isfinite(values[name]),(name,number)
 metrics[step]=values
assert set(metrics)==set(range(1,6))
for step,m in metrics.items():assert m['training/actual_outer_updates']==step and m['actor/optimizer_steps']==12 and m['actor/skipped_empty_minibatches']==0 and m['rollout/trajectories']==3072 and m['rollout/context_mismatches']==0
installed=json.loads(next(line.split('TREE_VALIDATION_COST_INSTALLED_JSON=',1)[1] for line in text.splitlines() if 'TREE_VALIDATION_COST_INSTALLED_JSON=' in line));assert installed['job_id']=='3103' and installed['trainer_source_sha256']=='b3c9d740a3a4d7963eb48922551f0e6ff643325b58bea0c408392af478c01bf5' and installed['observer_source_sha256']=='6b60822b92f9e26b5c683198772ff2b50c3465780be55f0126660d889f6cd3b5' and installed['output'].endswith('/runs/tree-fixed-five-3103/output')
result=read(P/'run-result.json');cost=read(P/'cost.json');assert result['status']=='ACTUAL_STAGE_COMPLETED_UNREVIEWED' and result['training_body_exit_code']==0 and result['entry_config_verified'] and result['cleanup_residual']=={} and result['final_guard_healthy'] and result['gpu_kl_proof_passed'] and cost['exit_code']==0
assert result['actual_training_accepted'] is False # Acceptance is solely this independent record.
pids={int(x.split(',')[1]) for x in result['final_gpu']['processes']};assert pids=={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446} and all(x.split(',')[2].strip()=='1131' and 0<=int(x.split(',')[-1])<=15 for x in result['final_gpu']['gpus'])
resources=[json.loads(x) for x in (P/'resource-samples.jsonl').read_text().splitlines()];assert resources and all(x['job_id']=='3103' and not x.get('event') and len(x['gpus'])==8 for x in resources)
for x in resources:
 for g in x['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
cpu=read(C/'result.json');hf=read(C/'hf5-full-reload.json');assert cpu['status']=='HF5_FULL_RELOAD_COMPLETE_PENDING_REVIEW' and cpu['exit_code']==0 and cpu['parent_job_id']=='3103' and cpu['cpu_job_id']=='3104' and cpu['allocated_gpus']==0 and cpu['allocated_cpus']==8
assert hf['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and hf['checkpoint'].endswith('/tree-fixed-five-3103/output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_5') and all(not v for v in hf['hf_loading_info'].values()) and hf['all_parameters_finite'] and hf['cpu_forward_logits_finite'] and hf['tokenizer_vocab_equal'] and hf['bos_eos_equal'] and 0<hf['changed_parameters']<=hf['total_parameters'] and hf['changed_tensors']>0 and hf['max_abs_parameter_delta']>0
terminal=(C/'slurm-parent-terminal.txt').read_text();assert 'JobId=3103' in terminal and 'JobState=COMPLETED' in terminal and 'ExitCode=0:0' in terminal and 'NumCPUs=192' in terminal and 'TresPerNode=gres:gpu:8' in terminal
cpterminal=(C/'slurm-cpu-terminal.txt').read_text();assert 'JobId=3104' in cpterminal and 'COMPLETED' in cpterminal and 'ExitCode=0:0' in cpterminal and 'NumCPUs=8' in cpterminal and 'TresPerNode=gres:gpu:0' in cpterminal
def duration(t):
 v=re.search(r'RunTime=(\d+):(\d+):(\d+)',t);return int(v[1])*3600+int(v[2])*60+int(v[3])
seconds=duration(terminal);cpseconds=duration(cpterminal)
out={'status':'PASS_ACTUAL_FIVE_SCIENCE_AND_OPERATION_ONLY','stage':'five','candidate_commit':COMMIT,'job_id':'3103','actual_outer_updates':5,'direct_rank_outer_records':40,'tree_audit_files':10,'memory_profile_files':40,'config_sha256':cfg,'installed_cost_observer':installed,'steps':steps,'metrics':metrics,'training_nonpad_token_ledger':{k:sum(m[k] for m in metrics.values()) for k in ['rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls','rollout/padded_rows']},'warmup_limit':'Firstouter successful optimizer calls/finite nonzero norms butLR0 => no sampled weight changes; outer2..5 all8samples changed. Logged firstouterLR is post-scheduler next rate.','result':result,'scheduler_gpu_seconds':seconds,'scheduler_gpu_hours':seconds*8/3600,'scheduler_cpu_hours':seconds*192/3600,'cpu_hf_job':'3104','cpu_hf_seconds':cpseconds,'cpu_hf_allocation_hours':cpseconds*8/3600,'guard_samples':len(resources),'sampled_NVML_total_memory_peak_GiB':max(g['total_used_memory_bytes'] for x in resources for g in x['gpus'])/2**30,'foreign_memory_fraction_peak':max(g['foreign_memory_fraction'] for x in resources for g in x['gpus']),'foreign_SM_upper_peak':max(g['foreign_sm_upper'] for x in resources for g in x['gpus']),'hf':hf,'limits':['Only repaired-protocol fresh5; not500/final51713/effect acceptance or oldformal308cause','Tree diagnostics record hashes/counts and pinned runtime assertions, not savedfull response/mask tensors; branch prefix counts include prompt/action/observation and repeatedevents, notunique reusedactiontokens','HF is actual CPU source-pinned checker report/terminal; raw3Bweights not reread locally; exactresume unsupported','No64-question before/periodic evaluation in thissmoke; cost observer actualinstalled butvalidationnotrun']}
(OUT/'ACTUAL3103_FIVE_RECOMPUTED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['steps','metrics','result','hf']},indent=2))
