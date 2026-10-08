import json,hashlib,collections,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'work/tree-fixed-formal-3112-initial-review';P=OUT/'actual3112-files/runs/tree-fixed-formal-3112';C=OUT/'actual3112-files/runs/tree-fixed-formal-cpu-3112-3113';CAND=ROOT/'work/tree-numeric-candidate';COMMIT='296217a4b676a26a92fc729f5d9882b785dbfa96'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=sha(P/'resolved.yaml');assert cfg==sha(P/'output/entry-audit/entry-config.yaml')
launch=read(P/'launch-manifest.json');assert launch['config_sha256']==cfg and launch['code_commit']==COMMIT and launch['job_id']=='3112' and '/job_3112/step_batch' in launch['cgroup']
entry=[json.loads(x) for x in (P/'output/entry-audit/entry.jsonl').read_text().splitlines()];assert len(entry)==1 and entry[0]['actual_config_sha256']==cfg and entry[0]['entry_total_training_steps']==501 and entry[0]['commit']==COMMIT
#state=read(P/'output/checkpoints/tree_grpo-qwen2.5-3b/algorithm/global_step_5/state.json');assert state['method']=='tree_grpo' and state['global_step']==5 and not state['resume_supported']
fields={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'};steps=[];trajectories=[]
for step in [1,2]:
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
  assert v['outer_step']==step and v['world_size']==8 and v['job_id']=='3112' and v['run_id']=='tree-fixed-formal-3112' and v['code_commit']==COMMIT and v['config_sha256']==cfg and v['data_identity']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
  assert v['actor_source_sha256']==sha(CAND/'verl/workers/actor/dp_actor.py') and v['auditor_source_sha256']==sha(CAND/'search_r1/baselines/tree_rank_audit.py')
  assert (v['local_rows'],v['local_minibatch'],v['local_microbatch'],v['epochs'])==(384,32,8,1)
  assert v['successful_optimizer_calls']==v['finite_grad_norms']==v['nonzero_grad_norms']==len(v['existing_grad_norms'])==12 and v['skipped_empty_minibatches']==0 and all(math.isfinite(n) and n>0 for n in v['existing_grad_norms'])
  d=v['local_shard_sample_delta'];assert v['local_shard_sample_layout_unchanged'] and d['count']==d['finite']==8192
  if step==1:assert d['nonzero']==0 and v['local_shard_sample_before_sha256']==v['local_shard_sample_after_sha256']
  else:assert d['nonzero']>0 and v['local_shard_sample_before_sha256']!=v['local_shard_sample_after_sha256']
 steps.append({'step':step,'question_groups':512,'trees':1024,'leaves':3072,'source_exact2branches':True,'rollout_sha256':sha(rfile),'policy_sha256':sha(pfile),'policy_mass_min':min(row['policy_weight_sum'] for row in pol['rows']),'policy_mass_max':max(row['policy_weight_sum'] for row in pol['rows']),'branch_response_nonempty':sum(b['parent_fields']['response']!=hashlib.sha256(b'[]').hexdigest() for b in r['branches']),'prefix_token_events_prompt_action_observation':sum(b['prefix_length'] for b in r['branches']),'rank_sample_deltas':{v['rank']:v['local_shard_sample_delta'] for v in ranks},'diagnostic_cost':{'rollout':r['diagnostic_check_cost'],'policy':pol['diagnostic_check_cost']}})
assert len(set.union(*trajectories))==3072*2

import sys
sys.path.insert(0,str(ROOT/'work'));from validate_tree_stage import parse_console
records=parse_console(P/'stage.log',2);metrics={v['step']:v['metrics'] for v in records}
for step,m in metrics.items():assert m['training/actual_outer_updates']==step and m['actor/optimizer_steps']==12 and m['actor/skipped_empty_minibatches']==0 and m['rollout/trajectories']==3072 and m['rollout/context_mismatches']==0
assert abs(metrics[1]['actor/lr']-1e-6/142)<1e-22 and abs(metrics[2]['actor/lr']-2e-6/142)<1e-22
import yaml
fullcfg=yaml.safe_load((P/'resolved.yaml').read_text())
for st in [1,2]:
 rr=read(P/f'output/tree-audit/step_{st}_rollout_{st}.json');assert all(rr[k]==fullcfg['baselines'][k] for k in ['base_commit','algorithm_source_commit'])
raw=read(OUT/'TWO_PREFIX_GZIP_DECODED.json');resources=[json.loads(z) for z in (P/'resource-samples.jsonl').read_text().splitlines()];assert all(v['job_id']=='3112' and not v.get('event') and len(v['gpus'])==8 for v in resources) and all(g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15 for v in resources for g in v['gpus'])
assert 'JobState=RUNNING' in raw['slurm']['3112']['text']
out={'status':'PASS_ACTUAL_FORMAL_TWO_OUTER_PREFIX_ONLY','job_id':'3112','code_commit':COMMIT,'config_sha256':cfg,'actual_updates_in_prefix':2,'rank_records':16,'Tree_audit_files':4,'steps':steps,'metrics':metrics,'warmup_steps':142,'first_actual_parameter_update_LR':0,'second_actual_parameter_update_LR':1e-6/142,'prefix_training_nonpad_token_ledger':{k:sum(m[k] for m in metrics.values()) for k in ['rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls','rollout/padded_rows']},'guard_samples':len(resources),'whole_device_sampled_peak_GiB':max(g['total_used_memory_bytes'] for v in resources for g in v['gpus'])/2**30,'slurm_status':'RUNNING','actual500_accepted':False,'limitations':['Only2 formalouterprefix; not500/final51713/fullHF500/wholejobexit','Firstouter finite gradients+12calls do not mean nonzero weight change; LR0 expected','Shard8192 sampled parameters only, not all3B parameters perouter','Audit hashes/counts and runtime assertions, not savedfulltensor replay; prefix counts contain repeated prompt/action/observation events','Generated nonpad excludesEOS withPAD==EOS; submittedprefill is notunique/cached tokens orFLOPs']}
(OUT/'TWO_PREFIX_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print('all2Tree structures and16directrank finite successfulnorms and2ndouter all8nonzero deltas passed')
