import pathlib,json,hashlib,collections,math,importlib.util,yaml,tempfile,shutil
root=pathlib.Path(__file__).resolve().parents[2];d=root/'work/aux-gate-review';e=root/'outputs/evidence';rp=e/'tree-two-2879-step_2_rollout_2.json';pp=e/'tree-two-2879-step_2_rollout_2_policy_batch.json';r=json.loads(rp.read_text());p=json.loads(pp.read_text());digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();fp=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();cfg=yaml.safe_load((e/'formal-protocol-resolved.yaml').read_text())
assert r['status']=='ROLLOUT_DIAGNOSTIC_COMPLETE' and p['status']=='POLICY_BATCH_DIAGNOSTIC_COMPLETE' and r['global_step']==p['global_step']==r['rollout_sequence']==p['rollout_sequence']==2
assert digest(rp)==p['rollout_file_sha256']
for k in ('base_commit','algorithm_source_commit'):assert r[k]==cfg['baselines'][k]
leaves=r['leaves'];rows=p['rows'];branches=r['branches'];groups=r['question_groups'];assert len(leaves)==len(rows)==r['leaf_count']==p['leaf_count']==3072 and len(branches)==r['branch_count']==2048 and len(groups)==512
by_source=collections.defaultdict(list);by_question=collections.defaultdict(list);by_tree=collections.defaultdict(list)
for l in leaves:by_source[l['source']].append(l);by_question[l['question_hash']].append(l);by_tree[l['tree_hash']].append(l)
assert set(by_source)==set(range(1024)) and len(by_tree)==1024 and len(by_question)==512
for source,ls in by_source.items():assert len(ls)==3 and len({x['tree_hash'] for x in ls})==len({x['question_hash'] for x in ls})==1
for question,ls in by_question.items():assert len(ls)==6 and len({x['tree_hash'] for x in ls})==len({x['source'] for x in ls})==2
assert len({x['trajectory_hash'] for x in leaves})==3072 and len({x['trajectory_hash'] for x in rows})==3072
known={x['trajectory_hash']:x for x in leaves};assert set(known)=={x['trajectory_hash'] for x in rows};widths=set();sum_actions=sum_obs=sum_padding=0
for l in leaves:
 assert all(type(l[k]) is int and l[k]>=0 for k in ('source','response_length','action_count','observation_count','padding_count','anchor_count','retained_pad_token_count'))
 assert l['action_count']+l['observation_count']==l['response_length'];widths.add(l['response_length']+l['padding_count']);sum_actions+=l['action_count'];sum_obs+=l['observation_count'];sum_padding+=l['padding_count']
 for k in ('question_hash','tree_hash','trajectory_hash','response_mask_hash','response_hash','step_ids_hash','attention_response_hash','generation_prefix_hash','scoring_prefix_hash'):assert isinstance(l[k],str) and len(l[k])==64 and all(c in '0123456789abcdef' for c in l[k])
 assert l['context_matches']==(l['generation_prefix_hash']==l['scoring_prefix_hash'])
assert len(widths)==1
masses=[];all_effective_match_raw=True
for row in rows:
 l=known[row['trajectory_hash']];assert row['question_hash']==l['question_hash'] and row['tree_hash']==l['tree_hash'] and row['raw_model_tokens']==l['action_count'] and 0<row['effective_tokens']<=row['raw_model_tokens'] and row['empty_leaf'] is False
 assert math.isfinite(row['policy_weight_sum']) and abs(row['policy_weight_sum']-1)<=1e-6;masses.append(row['policy_weight_sum']);all_effective_match_raw &= row['effective_tokens']==row['raw_model_tokens']
assert p['empty_leaves']==0
sources=collections.Counter(b['source'] for b in branches);assert sources==collections.Counter({i:2 for i in range(1024)}),'Strict two expansions per source failed'
fields={'prompt','rolling','response','model_mask','steps','anchors','records','observation','source','tree','node','turns','done','final_used','valid_actions','valid_searches'};nonempty=0;prefix=0
for b in branches:
 assert b['inherited'] is True and b['parent_fields']==b['child_fields'] and set(b['parent_fields'])==fields and b['parent_trajectory_hash']!=b['child_trajectory_hash']
 assert b['tree_hash']==by_source[b['source']][0]['tree_hash']==b['parent_fields']['tree'] and b['parent_fields']['source']==fp(b['source'])
 assert b['parent_node_hash']==b['parent_fields']['node'] and b['parent_trajectory_hash'] in known and b['child_trajectory_hash'] in known
 assert type(b['prefix_length']) is int and b['prefix_length']>0;prefix+=b['prefix_length'];nonempty+=b['parent_fields']['response']!=fp([])
assert nonempty>0 and prefix>0
for source in range(1024):
 edges=[b for b in branches if b['source']==source];trajectories={l['trajectory_hash'] for l in by_source[source]};assert len({b['child_trajectory_hash'] for b in edges})==2 and {b['child_trajectory_hash'] for b in edges}<=trajectories
 tree={x['tree_hash'] for x in by_source[source]};assert len(tree)==1 and all(b['tree_hash'] in tree for b in edges)
 # Both expansion edges connect all three retained leaf trajectory identities.
 assert {v for b in edges for v in (b['parent_trajectory_hash'],b['child_trajectory_hash'])}==trajectories
qgroups={g['question_hash']:g for g in groups};assert set(qgroups)==set(by_question) and all(g['roots']==2 and g['leaves']==6 for g in groups)
spec=importlib.util.spec_from_file_location('validator',root/'work/validate_tree_stage.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
with tempfile.TemporaryDirectory() as temp:
 run=pathlib.Path(temp);ad=run/'output/tree-audit';ad.mkdir(parents=True);
 for st in (1,2):
  for suf in ('','_policy_batch'):
   shutil.copyfile(e/('tree-two-2879-step_%d_rollout_%d%s.json'%(st,st,suf)),ad/('step_%d_rollout_%d%s.json'%(st,st,suf)))
 assert len(v.inspect_audits(run,2,cfg))==2
summary={'status':'SECOND_REAL_ROLLOUT_POLICY_STRUCTURE_REVIEW_COMPLETE_NOT_STAGE_ACCEPTANCE','job':2879,'step':2,'questions':512,'root_trees':1024,'leaves':3072,'branches':2048,'strict_branch_count_per_source':2,'nonempty_response_prefix_branches':nonempty,'inherited_rolling_prefix_token_events':prefix,'prefix_scope':'sum across branch events inclprompt/actions/observations; not unique tokens or pureaction reuse','uniform_composed_response_width':next(iter(widths)),'total_action_token_events':sum_actions,'total_observation_token_events':sum_obs,'total_padding_token_events':sum_padding,'all_effective_tokens_equal_raw_action_counts':all_effective_match_raw,'empty_leaves':0,'mass_min':min(masses),'mass_max':max(masses),'all_generation_scoring_context_hashes_equal':all(l['context_matches'] for l in leaves),'retained_pad_token_events':sum(l['retained_pad_token_count'] for l in leaves),'source_identity':{k:r[k] for k in ('base_commit','algorithm_source_commit')},'evidence_sha256':{str(rp):digest(rp),str(pp):digest(pp)},'diagnostic_check_cost':{'rollout':r['diagnostic_check_cost'],'policy':p['diagnostic_check_cost']},'limitation':'JSON contains hash/counts not raw mask/weights/logprobs/advantages; elementwise invariants are source-reviewed runtime assertions, not independently reconstructed tensors. No optimizer update/metrics/checkpoint/stage acceptance yet.'}
(d/'tree-step2-2879-independent-structure.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
