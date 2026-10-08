import collections,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'work/shared-formal-2971-two-prefix-review/embedded';a=p/'output/tree-audit';rankdir=p/'output/tree-rank-audit';cfgsha=hashlib.sha256((p/'resolved.yaml').read_bytes()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p/'resolved.yaml')==sha(p/'output/entry-audit/entry-config.yaml')
entry=[json.loads(x) for x in (p/'output/entry-audit/entry.jsonl').read_text().splitlines()];assert len(entry)==1 and entry[0]['actual_config_sha256']==cfgsha and entry[0]['entry_total_training_steps']==501 and entry[0]['commit']=='4d133efeb1f675139d2558640f1e21d50780f133'

steps=[];fieldset={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'}
for step in [1,2]:
 rf=a/f'step_{step}_rollout_{step}.json';pf=a/f'step_{step}_rollout_{step}_policy_batch.json';r=json.loads(rf.read_text());pol=json.loads(pf.read_text());rs=[json.loads(f.read_text()) for f in rankdir.glob(f'step-{step:06d}-rank-*.json')]
 assert r['global_step']==pol['global_step']==r['rollout_sequence']==pol['rollout_sequence']==step and r['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and pol['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE'
 assert len(r['branches'])==2048 and len(r['leaves'])==len(pol['rows'])==3072 and len(r['question_groups'])==512 and pol['empty_leaves']==0
 assert pol['rollout_file_sha256']==sha(rf)
 bc=collections.Counter(b['source'] for b in r['branches']);assert len(bc)==1024 and set(bc.values())=={2}
 for b in r['branches']:assert b['inherited'] and set(b['parent_fields'])==set(b['child_fields'])==fieldset and b['parent_fields']==b['child_fields']
 trees=collections.Counter(l['tree_hash'] for l in r['leaves']);questions=collections.defaultdict(list)
 for l in r['leaves']:
  questions[l['question_hash']].append(l);assert l['context_matches'] and l['generation_prefix_hash']==l['scoring_prefix_hash'] and l['action_count']+l['observation_count']==l['response_length'] and l['padding_count']>=0
 assert len(trees)==1024 and set(trees.values())=={3} and len(questions)==512
 assert all(len(ls)==6 and len({l['tree_hash'] for l in ls})==2 for ls in questions.values()) and all(g['roots']==2 and g['leaves']==6 for g in r['question_groups'])
 ids={l['trajectory_hash']:l for l in r['leaves']};assert len(ids)==3072 and {row['trajectory_hash'] for row in pol['rows']}==set(ids)
 for row in pol['rows']:
  l=ids[row['trajectory_hash']];assert row['question_hash']==l['question_hash'] and row['tree_hash']==l['tree_hash'] and row['effective_tokens']==l['action_count'] and abs(row['policy_weight_sum']-1)<1e-6
 assert len(rs)==8 and {v['rank'] for v in rs}==set(range(8)) and len({v['pid'] for v in rs})==8
 deltas={}
 for v in rs:
  assert v['outer_step']==step and v['world_size']==8 and v['job_id']=='2971' and v['run_id']=='tree-formal-2971' and v['code_commit']=='4d133efeb1f675139d2558640f1e21d50780f133' and v['config_sha256']==cfgsha and v['data_identity']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
  assert v['actor_source_sha256']==sha(ROOT/'work/tree-grpo/verl/workers/actor/dp_actor.py') and v['auditor_source_sha256']==sha(ROOT/'work/tree-grpo/search_r1/baselines/tree_rank_audit.py')
  assert (v['local_rows'],v['local_minibatch'],v['local_microbatch'],v['epochs'])==(384,32,8,1)
  assert v['successful_optimizer_calls']==v['finite_grad_norms']==v['nonzero_grad_norms']==len(v['existing_grad_norms'])==12 and v['skipped_empty_minibatches']==0 and all(math.isfinite(n) and n>0 for n in v['existing_grad_norms']) and v['existing_grad_norms']==rs[0]['existing_grad_norms']
  d=v['local_shard_sample_delta'];assert v['local_shard_sample_layout_unchanged'] and d['count']==d['finite']==8192
  if step==1:assert d['nonzero']==0 and v['local_shard_sample_before_sha256']==v['local_shard_sample_after_sha256']
  else:assert d['nonzero']>0 and v['local_shard_sample_before_sha256']!=v['local_shard_sample_after_sha256']
  deltas[v['rank']]=d
 steps.append({'step':step,'groups':512,'roots':1024,'leaves':3072,'each_source_branches':2,'rollout_sha256':sha(rf),'policy_sha256':sha(pf),'mass_min':min(x['policy_weight_sum'] for x in pol['rows']),'mass_max':max(x['policy_weight_sum'] for x in pol['rows']),'branch_nonempty_response':sum(b['parent_fields']['response']!=hashlib.sha256(b'[]').hexdigest() for b in r['branches']),'prefix_token_events_including_prompt_action_observation':sum(b['prefix_length'] for b in r['branches']),'rank_sample_deltas':deltas})
text=re.sub(r'\x1b\[[0-9;]*m','',(p/'stage.log').read_text());metrics={}
for line in text.splitlines():
 match=re.search(r'\bstep:(\d+) - (.*)',line)
 if not match:continue
 s=int(match.group(1));assert s not in metrics
 values={}
 for v in match.group(2).split(' - '):
  k,n=v.rsplit(':',1);values[k]=float(n);assert math.isfinite(values[k]),(k,n)
 metrics[s]=values
assert set(metrics)=={1,2}
for step,v in metrics.items():assert v['training/actual_outer_updates']==step and v['actor/optimizer_steps']==12 and v['actor/skipped_empty_minibatches']==0 and v['rollout/trajectories']==3072 and v['rollout/context_mismatches']==0

import yaml
cfg=yaml.safe_load((p/'resolved.yaml').read_text()); assert cfg['trainer']['total_training_steps']==501 and cfg['trainer']['val_before_train'] is True and cfg['trainer']['test_freq']==50 and cfg['trainer']['n_gpus_per_node']==8
assert cfg['data']['val_batch_size']==64 and cfg['data']['val_files'].endswith('validation-64.parquet') and cfg['data']['final_val_files'].endswith('/test.parquet')
assert cfg['retriever']['url']=='http://127.0.0.1:18082/retrieve' and cfg['retriever']['topk']==3 and cfg['baselines']['local_runtime_only'] and cfg['baselines']['local_runtime_host']=='h100-9' and cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size']==64
assert int(501*cfg['actor_rollout_ref']['actor']['optim']['lr_warmup_steps_ratio'])==142
manifest=json.loads((p/'launch-manifest.json').read_text());assert manifest['job_id']=='2971' and manifest['config_sha256']==cfgsha and manifest['supervisor_source_sha256']==sha(ROOT/'work/tree_shared_supervisor_formal_cost_identity.py') and manifest['guard_source_sha256']==sha(ROOT/'work/resource_guard_identity.py') and len(manifest['gpu_leases'])==8
snapshot=ROOT/'outputs/evidence/shared-formal-2971-two-prefix.json'; snap=json.loads(snapshot.read_text());assert len(snap['files'])==27
for name,v in snap['files'].items():assert sha(p/name)==v['sha256']
for step,v in metrics.items():assert abs(v['actor/lr']-step*1e-6/142)<1e-22
out=dict(status='FORMAL_TWO_PREFIX_VERIFIED_NOT_TERMINAL_ACCEPTANCE',snapshot_sha256=sha(snapshot),embedded_file_count=27,actual_outer_updates_observed=2,rank_outer_records=16,config_sha256=cfgsha,steps=steps,metrics=metrics,source_manifest=manifest,config_checks=dict(total_training_steps=501,actual_outer_budget=500,warmup_steps=142,val64=True,periodic_freq=50,final_file=cfg['data']['final_val_files'],same_training_node='h100-9',retriever_loopback=True,global_microbatch=64),limitations=['Final file path is confirmed but actual51713rows/final evaluation await terminal evidence; no final count is invented.','Prefix successful8rank samples do not replace fullHF/checkpoint/final500acceptance, intentionally not required here.','Raw audits record hashed invariants/counts and runtime checks, not raw masks/tensors for local rerun.','Future later-step guard/resource and terminal cleanup remain pending.'])
(ROOT/'work/shared-formal-2971-two-prefix-review/TWO_PREFIX_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['steps','metrics','source_manifest']},indent=2))
