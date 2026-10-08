from pathlib import Path
import json,hashlib,math,re,time,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');R=ROOT/'runs/tree-instruct300-3154';C=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161';proof=[]
def read(p,json_value=True):
 b=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b) if json_value else b.decode()
m=read(C/'manifest.json');cfg=read(R/'resolved.yaml',False);assert hashlib.sha256(cfg.encode()).hexdigest()=='63296d0602c5978d1ff67d258b401054ce587e3a9a2c55052c35fbeac626c198'
for path in ['source/repo/verl/workers/actor/dp_actor.py','source/repo/search_r1/baselines/gradient_norm.py']:
 text=read(C/path,False);assert hashlib.sha256(text.encode()).hexdigest()==m['files'][path]
assert re.search(r'^\s+grad_clip: 1(?:\.0)?\s*$',cfg,re.M)
records={};summary=[]
for step in (78,79,80):
 if len(list((R/'output/tree-rank-audit').glob(f'step-{step:06d}-rank-*.json')))!=8:
  continue
 for rank in range(8):
  p=R/f'output/tree-rank-audit/step-{step:06d}-rank-{rank:02d}.json'
  if not p.exists():continue
  a=read(p);assert a['outer_step']==step and a['rank']==rank and a['world_size']==8 and a['job_id']=='3154' and a['run_id']==R.name and a['code_commit']==m['code_commit'] and a['config_sha256']==hashlib.sha256(cfg.encode()).hexdigest() and a['data_identity']==m['common_data_identity']
  assert a['actor_source_sha256']==m['files']['source/repo/verl/workers/actor/dp_actor.py'] and a['auditor_source_sha256']==m['files']['source/repo/search_r1/baselines/tree_rank_audit.py']
  assert a['successful_optimizer_calls']==a['finite_grad_norms']==a['nonzero_grad_norms']==len(a['existing_grad_norms'])==24 and a['skipped_empty_minibatches']==0 and all(math.isfinite(v) and v>0 for v in a['existing_grad_norms'])
  d=a['local_shard_sample_delta'];assert d['count']==d['finite']==8192 and d['nonzero']>0 and a['local_shard_sample_layout_unchanged'];records[(step,rank)]=a
 assert sum(s==step for s,rank in records) in (0,8)
 if (step,0) in records:
  norms=records[(step,0)]['existing_grad_norms'];assert all(records[(step,rank)]['existing_grad_norms']==norms for rank in range(8))
  summary.append(dict(step=step,rank_count=8,mean_preclip_norm=sum(norms)/len(norms),max_preclip_norm=max(norms),max_optimizer_call_1based=norms.index(max(norms))+1,norms=norms,sample_nonzero={rank:records[(step,rank)]['local_shard_sample_delta']['nonzero'] for rank in range(8)}))
for step in (79,80):
 for rank in range(8):
  if (step,rank) in records and (step-1,rank) in records:
   assert records[(step,rank)]['pid']==records[(step-1,rank)]['pid'] and records[(step,rank)]['local_shard_sample_before_sha256']==records[(step-1,rank)]['local_shard_sample_after_sha256']
log=read(R/'stage.log',False);clean=re.sub(r'\x1b\[[0-9;]*m','',log);lines={};metrics={}
for step in (78,79,80):
 found=[l for l in clean.splitlines() if re.search(r'(?<![\w/])step:'+str(step)+r' - ',l)]
 if not found:continue
 assert len(found)==1;lines[step]=found[0];v={k:float(x) for k,x in re.findall(r'(?:^| - )([^\s]+):([-+0-9.eE]+)(?= - |$)',found[0])};assert all(math.isfinite(x) for x in v.values()) and v['training/actual_outer_updates']==step and v['actor/optimizer_steps']==24
 norms=records[(step,0)]['existing_grad_norms'];assert math.isclose(v['actor/grad_norm'],sum(norms)/24,rel_tol=1e-12,abs_tol=1e-10);metrics[step]=v
tree_checks=[]
import collections
fields={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'}
for item in summary:
 step=item['step'];rf=R/f'output/tree-audit/step_{step}_rollout_{step}.json';pf=rf.with_name(f'step_{step}_rollout_{step}_policy_batch.json');roll=read(rf);policy=read(pf)
 assert roll['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and policy['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE' and roll['global_step']==policy['global_step']==roll['rollout_sequence']==policy['rollout_sequence']==step
 assert policy['rollout_file_sha256']==hashlib.sha256(rf.read_bytes()).hexdigest() and len(roll['branches'])==2048 and len(roll['leaves'])==len(policy['rows'])==3072 and len(roll['question_groups'])==512 and policy['empty_leaves']==0
 counts=collections.Counter(b['source'] for b in roll['branches']);assert len(counts)==1024 and set(counts.values())=={2}
 assert all(b['inherited'] and b['parent_fields']==b['child_fields'] and set(b['parent_fields'])==fields for b in roll['branches'])
 leaves=roll['leaves'];ids={l['trajectory_hash']:l for l in leaves};assert len(ids)==3072 and {z['trajectory_hash'] for z in policy['rows']}==set(ids)
 trees=collections.Counter(l['tree_hash'] for l in leaves);questions=collections.defaultdict(list)
 for leaf in leaves:
  questions[leaf['question_hash']].append(leaf);assert leaf['context_matches'] and leaf['generation_prefix_hash']==leaf['scoring_prefix_hash'] and leaf['action_count']+leaf['observation_count']==leaf['response_length'] and leaf['padding_count']>=0
 assert len(trees)==1024 and set(trees.values())=={3} and len(questions)==512 and all(len(v)==6 and len({z['tree_hash'] for z in v})==2 for v in questions.values()) and all(g['roots']==2 and g['leaves']==6 for g in roll['question_groups'])
 for row in policy['rows']:
  leaf=ids[row['trajectory_hash']];assert row['question_hash']==leaf['question_hash'] and row['tree_hash']==leaf['tree_hash'] and row['effective_tokens']==leaf['action_count'] and not row['empty_leaf'] and abs(row['policy_weight_sum']-1)<1e-6
 tree_checks.append(dict(step=step,question_groups=512,trees=1024,leaves=3072,branches=2048,exact2branches_per_source=True,inherited16fields_equal=True,generation_scoring_prefix_match=True,rollout_policy_file_sha_bound=True,response_attention_modelmask_steps_trajectory_binding='producer checks passed',uniform_effective_token_weights='producer tensor checks passed',finite_effective_oldlogprob_advantages='producer tensor checks passed',policy_mass_min=min(z['policy_weight_sum'] for z in policy['rows']),policy_mass_max=max(z['policy_weight_sum'] for z in policy['rows']),policy_effective_tokens=sum(z['effective_tokens'] for z in policy['rows']),algorithm_source_commit=roll.get('algorithm_source_commit'),base_commit=roll.get('base_commit')))
for name in ['source/repo/search_r1/baselines/controller.py','source/repo/search_r1/baselines/tree_audit.py','source/repo/search_r1/baselines/advantages.py','source/repo/search_r1/baselines/objectives.py','source/repo/verl/trainer/ppo/ray_trainer.py']:
 text=read(C/name,False);assert hashlib.sha256(text.encode()).hexdigest()==m['files'][name]
slurm=subprocess.check_output(['scontrol','show','job','3154'],text=True)
print(json.dumps(dict(status='REPEATED_FINITE_PRECLIP_SPIKE_READ_ONLY_VERIFIED',epoch=time.time(),job_id=3154,slurm=slurm,grad_clip=1,summary=summary,records={f'{s}:{rank}':a for (s,rank),a in records.items()},metrics=metrics,metric_lines=lines,tree_checks=tree_checks,proof=proof,scope='Direct adjacent rank records, finite preclip norms and8192-coordinate parameter samples; no failure tensors, allparameter scan or rootcause claim')))
