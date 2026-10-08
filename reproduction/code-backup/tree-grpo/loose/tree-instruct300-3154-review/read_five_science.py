from pathlib import Path
import json,hashlib,collections,math,re,sys,time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');R=ROOT/'runs/tree-instruct300-3154';C=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161';M=json.loads((C/'manifest.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=[]
def read(p):
 b=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b)
cfg=sha(R/'resolved.yaml');assert cfg==sha(R/'output/entry-audit/entry-config.yaml');steps=[];trajectorysets=[];profiles=[]
fields={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'}
for step in range(1,6):
 rf=R/f'output/tree-audit/step_{step}_rollout_{step}.json';pf=rf.with_name(f'step_{step}_rollout_{step}_policy_batch.json');r=read(rf);p=read(pf)
 assert r['global_step']==r['rollout_sequence']==p['global_step']==p['rollout_sequence']==step and r['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and p['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE'
 assert len(r['branches'])==2048 and len(r['leaves'])==len(p['rows'])==3072 and len(r['question_groups'])==512 and p['empty_leaves']==0 and p['rollout_file_sha256']==sha(rf)
 cnt=collections.Counter(b['source'] for b in r['branches']);assert len(cnt)==1024 and set(cnt.values())=={2}
 assert all(b['inherited'] and set(b['parent_fields'])==set(b['child_fields'])==fields and b['parent_fields']==b['child_fields'] for b in r['branches'])
 trees=collections.Counter(l['tree_hash'] for l in r['leaves']);questions=collections.defaultdict(list)
 for l in r['leaves']:
  questions[l['question_hash']].append(l);assert l['context_matches'] and l['generation_prefix_hash']==l['scoring_prefix_hash'] and l['action_count']+l['observation_count']==l['response_length'] and l['padding_count']>=0
 assert len(trees)==1024 and set(trees.values())=={3} and len(questions)==512 and all(len(v)==6 and len({z['tree_hash'] for z in v})==2 for v in questions.values())
 assert all(g['roots']==2 and g['leaves']==6 for g in r['question_groups']);ids={l['trajectory_hash']:l for l in r['leaves']};assert len(ids)==3072 and {z['trajectory_hash'] for z in p['rows']}==set(ids);trajectorysets.append(set(ids))
 for row in p['rows']:
  l=ids[row['trajectory_hash']];assert row['question_hash']==l['question_hash'] and row['tree_hash']==l['tree_hash'] and row['effective_tokens']==l['action_count'] and not row['empty_leaf'] and abs(row['policy_weight_sum']-1)<1e-6
 ranks=[]
 for rank in range(8):
  f=R/f'output/tree-rank-audit/step-{step:06d}-rank-{rank:02d}.json';a=read(f);ranks.append(a)
  assert a['outer_step']==step and a['rank']==rank and a['world_size']==8 and a['job_id']=='3154' and a['run_id']=='tree-instruct300-3154' and a['code_commit']==M['code_commit'] and a['config_sha256']==cfg and a['data_identity']==M['common_data_identity']
  assert a['actor_source_sha256']==M['files']['source/repo/verl/workers/actor/dp_actor.py'] and a['auditor_source_sha256']==M['files']['source/repo/search_r1/baselines/tree_rank_audit.py']
  assert (a['local_rows'],a['local_minibatch'],a['local_microbatch'],a['epochs'])==(384,16,2,1)
  assert a['successful_optimizer_calls']==a['finite_grad_norms']==a['nonzero_grad_norms']==len(a['existing_grad_norms'])==24 and a['skipped_empty_minibatches']==0 and all(math.isfinite(n) and n>0 for n in a['existing_grad_norms'])
  d=a['local_shard_sample_delta'];assert a['local_shard_sample_layout_unchanged'] and d['count']==d['finite']==8192
  if step==1:assert d['nonzero']==0 and a['local_shard_sample_before_sha256']==a['local_shard_sample_after_sha256']
  else:assert d['nonzero']>0 and a['local_shard_sample_before_sha256']!=a['local_shard_sample_after_sha256']
  j=read(R/f'output/tree-rank-audit/memory-profile/step-{step:06d}-rank-{rank:02d}.json')
  assert j['outer_step']==step and j['rank']==rank and j['world_size']==8 and j['job_id']=='3154' and j['pid']==a['pid'] and j['config_sha256']==cfg and j['code_commit']==M['code_commit'] and j['observer_sha256']==M['files']['sitecustomize.py']
  assert len(j['micro_records'])==192 and len(j['optimizer_records'])==24 and [z['pending_microbatches'] for z in j['micro_records']]==list(range(1,9))*24 and all(z['pending_after_clear']==0 for z in j['optimizer_records'])
  mems=[j['start'],j['end']]+[z['memory'] for z in j['micro_records']]+[z[k] for z in j['optimizer_records'] for k in ['before_clear','after_clear']]
  for z in mems:assert z['cuda_available'] and all(math.isfinite(z[k]) and z[k]>=0 for k in ['allocated','reserved','lifetime_peak_allocated','lifetime_peak_reserved']) and z['reserved']>=z['allocated'] and z['lifetime_peak_allocated']>=z['allocated']
  profiles.append(dict(step=step,rank=rank,pid=a['pid'],unique_retained_max=max(z['unique_retained_cuda_bytes'] for z in j['micro_records']),logical_retained_max=max(z['logical_cuda_tensor_bytes'] for z in j['micro_records']),lifetime_torch_peak_bytes=max(z['lifetime_peak_allocated'] for z in mems),recorded_micro_observer_wall_seconds=j['observer_wall_seconds']))
 assert len({a['pid'] for a in ranks})==8 and all(a['existing_grad_norms']==ranks[0]['existing_grad_norms'] for a in ranks)
 steps.append(dict(step=step,groups=512,trees=1024,leaves=3072,source_exact2branches=True,rollout_sha256=sha(rf),policy_sha256=sha(pf),policy_mass_min=min(z['policy_weight_sum'] for z in p['rows']),policy_mass_max=max(z['policy_weight_sum'] for z in p['rows']),prefix_reused_token_events_including_prompt_action_observation=sum(b['prefix_length'] for b in r['branches']),rank_sample_deltas={a['rank']:a['local_shard_sample_delta'] for a in ranks}))
assert len(set.union(*trajectorysets))==15360
sys.path.insert(0,str(ROOT/'tools'));from validate_tree_stage import parse_console
records=parse_console(R/'stage.log',5);metrics={v['step']:v['metrics'] for v in records}
for step,m in metrics.items():
 assert m['training/actual_outer_updates']==step and m['actor/optimizer_steps']==24 and m['actor/skipped_empty_minibatches']==0 and m['rollout/trajectories']==3072 and m['rollout/context_mismatches']==0 and abs(m['actor/lr']-step*1e-6/57)<1e-21
for file in [R/'stage.log',R/'resolved.yaml',R/'output/entry-audit/entry-config.yaml',R/'output/entry-audit/entry.jsonl',R/'resource-samples.jsonl',R/'retriever-coexistence.json']:
 if file.exists():proof.append(dict(path=str(file),sha256=sha(file),bytes=file.stat().st_size))
g=[json.loads(z) for z in (R/'resource-samples.jsonl').read_text().splitlines()];assert g and all(z['job_id']=='3154' and not z.get('event') and len(z['gpus'])==8 for z in g) and all(v['foreign_memory_fraction']<=.15 and v['foreign_sm_upper']<=15 for z in g for v in z['gpus'])
print(json.dumps(dict(status='ACTUAL_FIVE_RANK_TREE_PROFILE_CONSOLE_GUARD_RECOMPUTED_PASS',epoch=time.time(),code_commit=M['code_commit'],config_sha256=cfg,steps=steps,metrics=metrics,profiles=profiles,guard_sample_count=len(g),whole_gpu_sampled_peak_bytes=max(v['total_used_memory_bytes'] for z in g for v in z['gpus']),proof=proof,scope='Read-only existing actual5 evidence; no approval; fullstate/HF acceptance separate')))
