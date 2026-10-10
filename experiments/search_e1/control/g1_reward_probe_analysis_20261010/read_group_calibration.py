"""CPU-only exact five-row searched calibration reward reconstruction.

This does not reconstruct historical live TRAIN rewards. Arrays and raw rows stay
remote; only equal-question aggregate statistics are emitted.
"""
from pathlib import Path
import datetime, hashlib, json, os, socket, sys
import numpy as np
import joblib
BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
run = json.loads((BASE/'logs/run-manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def summary(a):
 a=np.asarray(a,dtype=float).ravel()
 if not len(a):return dict(count=0)
 assert np.isfinite(a).all()
 return dict(count=len(a),mean=float(a.mean()),std_population=float(a.std()),min=float(a.min()),max=float(a.max()),quantile_p=[0,.05,.25,.5,.75,.95,1],quantiles=np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist())
def aggregate(groups):
 all_r=np.concatenate([g['r'] for g in groups]) if groups else []
 zeros=[g for g in groups if g['r'].std(ddof=1)==0]
 mixed=[g for g in groups if 0<g['em'].sum()<5]
 ac=[g for g in groups if g['em'].sum()==5]
 aw=[g for g in groups if g['em'].sum()==0]
 wrong_a=[x for g in mixed for x in g['a'][g['em']==0]]
 delta_wrong=[x for g in mixed for x in (g['a']-g['a_em'])[g['em']==0]]
 correct_a=[x for g in mixed for x in g['a'][g['em']==1]]
 diffs_correct=[]; paircostdiff=0; rankpairs=0
 for g in groups:
  ix=np.flatnonzero(g['em'])
  for i in range(len(ix)):
   for j in range(i+1,len(ix)):
    u,v=ix[i],ix[j];rankpairs+=int(g['r'][u]!=g['r'][v]); paircostdiff+=int(g['cost'][u]!=g['cost'][v]);diffs_correct.append(abs(g['r'][u]-g['r'][v]))
 edges=[-.1000001,0,.0500001,.5,.9999999,1.0000001,1.1,1.2,1.3,1.4,1.5,1.6000001]
 return dict(question_groups=len(groups),nominal_rollouts=5*len(groups),correct_count_per_group_histogram={str(i):sum(int(g['em'].sum())==i for g in groups) for i in range(6)},all_correct=len(ac),all_wrong=len(aw),mixed=len(mixed),binary_em_zero_variance_groups=len(ac)+len(aw),reconstructed_reward_zero_variance_groups=len(zeros),all_correct_activated_nonzero_variance=sum(g['r'].std(ddof=1)>0 for g in ac),all_wrong_nonzero_variance=sum(g['r'].std(ddof=1)>0 for g in aw),raw_pre_normalization_reward=summary(all_r),raw_reward_zero_count=sum(int(x==0) for x in all_r),raw_reward_floor_count=sum(int(abs(x-.05)<1e-7) for x in all_r),raw_reward_correct_count=sum(int(x>=.98-1e-7) for x in all_r),reward_group_mean=summary([g['r'].mean() for g in groups]),raw_pre_normalization_reward_histogram=dict(edges=edges,counts=np.histogram(all_r,bins=edges)[0].tolist()),reward_group_sample_std=summary([g['r'].std(ddof=1) for g in groups]),reward_group_population_variance=summary([g['r'].var() for g in groups]),correct_correct_pair_count=sum(int(g['em'].sum())*(int(g['em'].sum())-1)//2 for g in groups),correct_correct_pairs_binary_em_tied_reward_separated=rankpairs,correct_correct_pairs_search_cost_differs=paircostdiff,correct_correct_absolute_reward_difference=summary(diffs_correct),mixed_group_wrong_advantage=summary(wrong_a),mixed_group_wrong_advantage_delta_vs_binary_em=summary(delta_wrong),mixed_group_correct_advantage=summary(correct_a),mixed_groups_any_correct_advantage_negative=sum(any(g['a'][g['em']==1]<0) for g in mixed),correct_rollouts_with_negative_advantage_in_mixed=sum(int((g['a'][g['em']==1]<0).sum()) for g in mixed),all_correct_trajectory_advantage=summary([a for g in ac for a in g['a']]),all_wrong_trajectory_advantage=summary([a for g in aw for a in g['a']]),sum_group_advantage_max_abs=max([abs(g['a'].sum()) for g in groups],default=0),author_vs_actual_search_count_difference_trajectories=sum(int(x!=y) for g in groups for x,y in zip(g['search'],g['actual_search'])),search_count_definition='proof.authored_search_tag_counts from identical frozen model_search_count function; not actual retrieval call count',search_count_histogram={str(i):sum(int(x==i) for g in groups for x in g['search']) for i in range(5)},shared_boundary_b_max_within_group_range=max([float(np.ptp(g['b'])) for g in groups],default=0))
out=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),run_id=run['run_id'],source_commit=run['source_commit'],readonly=True,cpu_only=True,scope_rounds=[0,50,100],interpretation='Retrospective matching fitted-head scores on saved calibration features, not historical live TRAIN rewards; exact complete legal-searched five-rollout groups only.',formula='R=.05 for valid wrong; R=1+.6*d*(1-b)-.02*min(1,max(n_search-1,0)) for correct searched. A=(R-mean(R))/(sample_std(R)+1e-6).',rounds=[],source=[],missing_full_group_reason='NPZ/jsonl store only legal searched calibration trajectories; missing unsearched or invalid trajectories lack full response / strict EM and cannot be fabricated.')
for rel in ['ts_probe_dump.py','verl/trainer/main_ppo_eco.py','verl/trainer/ppo/core_algos.py','search_g1/signals.py']:
 p=BASE/'repo'/rel; digest=sha(p);assert digest==run['source_sha256'][rel];out['source'].append(dict(path=str(p),sha256=digest))
all_groups={};question_orders=[]
for directory in sorted((BASE/'artifacts').glob('round_*')):
 mp=directory/'version-manifest.json'
 if not mp.is_file():continue
 m=json.loads(mp.read_text())
 if m['round'] not in [0,50,100]:continue
 dump=next((directory/'dumps').glob('*.npz'));assert sha(dump)==m['dump_sha256'][dump.name]
 proofp=directory/'dumps/calibration-label-proof.json';proof=json.loads(proofp.read_text());assert proof['group_size']==5 and proof['question_count']==300
 recordp=dump.with_suffix('.jsonl'); recs=[json.loads(l) for l in recordp.open()]
 hp={k:Path(m[k]) for k in ['boundary','commit']}
 for k,p in hp.items():assert sha(p)==m['head_sha256'][k+'_probe.pkl']
 heads={k:joblib.load(p) for k,p in hp.items()}
 with np.load(dump,allow_pickle=False) as z:
  gid=z['gid'].astype(int); em=z['y_em'].astype(int);qids=z['question_id'];bids=z['boundary_question_id'];question_orders.append(bids.tolist());dhead=heads['commit'];bhead=heads['boundary']
  assert len(recs)==len(gid)==len(em)==proof['searched_valid_trajectory_count'];assert np.all(np.diff(gid)>=0)
  assert all(r['qi']==int(gid[k]) and r['em']==int(em[k]) for k,r in enumerate(recs))
  assert all(qids[k]==bids[gid[k]] for k in range(len(gid)))
  d=dhead['clf'].predict_proba(z[dhead['position']+'_'+str(dhead['layer'])])[:,1]
  b=bhead['clf'].predict_proba(z['Xp_'+str(bhead['layer'])])[:,1]
  bq=bhead['clf'].predict_proba(z['Xp_question_'+str(bhead['layer'])])[:,1]
  assert np.isfinite(d).all() and np.isfinite(b).all();assert np.max(np.abs(b-bq[gid]))<1e-6
  groups=[]; missing=[]
  for qi in range(300):
   ix=np.flatnonzero(gid==qi)
   if len(ix)!=5:missing.append(qi);continue
   # Source flat_prompts is question-major, valid rows retain ascending flat index.
   # Five saved rows therefore are exactly original indices qi*5..qi*5+4.
   ns=np.asarray(proof['authored_search_tag_counts'][qi*5:qi*5+5],dtype=int);actual_ns=np.asarray(proof['actual_retrieval_counts'][qi*5:qi*5+5],dtype=int);assert np.all(actual_ns>0) and np.all(ns>0)
   cost=np.minimum(1,np.maximum(ns-1,0)); rr=np.where(em[ix]==1,1+.6*d[ix]*(1-b[ix])-.02*cost,.05)
   # Actual reward tensor is FP32; CPU reconstruction follows scalar reward cast.
   rr=rr.astype(np.float32).astype(np.float64); ee=em[ix].astype(np.float32).astype(np.float64)
   a=(rr-rr.mean())/(rr.std(ddof=1)+1e-6); ae=(ee-ee.mean())/(ee.std(ddof=1)+1e-6)
   groups.append(dict(qi=qi,em=em[ix],r=rr,a=a,a_em=ae,b=b[ix],d=d[ix],search=ns,actual_search=actual_ns,cost=cost))
 all_groups[m['round']]=groups
 out['rounds'].append(dict(round=m['round'],version=m['version'],manifest_sha256=sha(mp),dump_path=str(dump),dump_sha256=sha(dump),records_path=str(recordp),records_sha256=sha(recordp),proof_path=str(proofp),proof_sha256=sha(proofp),heads_sha256=m['head_sha256'],valid_searched_rows=len(gid),valid_searched_group_size_histogram={str(i):int(sum(np.sum(gid==qi)==i for qi in range(300))) for i in range(6)},complete_group_coverage=len(groups)/300,all300_exact_reward_reconstruction_possible=len(groups)==300,missing_incomplete_groups=len(missing),complete_group_stats=aggregate(groups)))
assert all(x==question_orders[0] for x in question_orders)
out['same_reserved_300_question_ids_and_order']=True
common=set(g['qi'] for g in all_groups[0])&set(g['qi'] for g in all_groups[50])&set(g['qi'] for g in all_groups[100])
out['common_complete_group_count']=len(common);out['common_complete_question_id_set_sha256']=hashlib.sha256(json.dumps(sorted(question_orders[0][qi] for qi in common)).encode()).hexdigest()
out['common_complete_groups_by_round']={str(k):aggregate([g for g in gs if g['qi'] in common]) for k,gs in all_groups.items()}
out['torch_imported']='torch' in sys.modules;assert not out['torch_imported'];out['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES');out['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else None))
