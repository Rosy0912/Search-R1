"""Archive pre-GRPO-normalization binary marginal distribution, steps0–150 only.

Derives compact distributions from already captured rounded native means.
Never reconstructs question-level five-sibling assignments.
"""
from pathlib import Path
import csv,hashlib,json,math
P=Path(__file__).resolve().parent
source=P/'curve-1-199.json';capture=json.loads(source.read_text());rows=capture['curve_rows'][:150]
assert len(rows)==150 and [r['iteration']for r in rows]==list(range(1,151))
N=2560
# Half a console ULP plus small FP32 arithmetic/representation slack. The bounds
# are deliberately inclusive/conservative; these are not raw exact counts.
PRINT_HALF_ULP_PLUS_FP_SLACK=.0005001
out=[{'iteration':0,'raw_reward_support':[0.,1.],'p_reward_1':None,'p_reward_0':None,'count_1_integer_candidates':None,'count_0_integer_candidates':None,'sum_reward_candidates':None,'marginal_reward_std_from_logged_mean':None,'status':'no_initial_liveTRAINrolloutbatch;initialdev512notTRAINdistribution'}]
for r in rows:
 m=r['metrics'];assert m['critic/rewards/max']==1. and m['critic/rewards/min']==0.
 assert m['critic/score/max']==1. and m['critic/score/min']==0.
 p=m['critic/rewards/mean'];strict=m['audit/strict_em']
 # Both native means use the same binary per-response scores, but NumPy/torch
 # rounding can differ at a .xxx5 boundary; retain conservative shared bounds.
 reward_counts=[k for k in range(1,N)if abs(k/N-p)<=PRINT_HALF_ULP_PLUS_FP_SLACK]
 audit_counts=[k for k in range(1,N)if abs(k/N-strict)<=PRINT_HALF_ULP_PLUS_FP_SLACK]
 possible=sorted(set(reward_counts)&set(audit_counts));assert possible
 lo,hi=possible[0],possible[-1]
 out.append({'iteration':r['iteration'],'raw_reward_support':[0.,1.],'p_reward_1':p,'p_reward_0':1-p,'native_logged_reward_mean':p,'native_logged_strict_EM':strict,'p1_true_conservative_bounds':[lo/N,hi/N],'p0_true_conservative_bounds':[1-hi/N,1-lo/N],'count_1_integer_candidate_bounds':[lo,hi],'count_1_integer_candidates':possible,'count_0_integer_candidate_bounds':[N-hi,N-lo],'count_0_integer_candidates':sorted(N-k for k in possible),'sum_reward_candidates':possible,'unique_count_inferred_not_raw':len(possible)==1,'marginal_reward_variance_from_logged_mean':p*(1-p),'marginal_reward_std_from_logged_mean':math.sqrt(p*(1-p)),'marginal_variance_is_not_within_question_five_sibling_variance':True,'source_line':r['line'],'source_line_sha256':r['line_sha256']})
actual=json.load(open(P/'observation-second.json'));old=json.load(open('outputs/e1-validation-reward-origin-remote-20261008.json'))['files'];qa=next(v for k,v in old.items()if'/runtime/archive/'in k and k.endswith('qa_em.py'));freshqa=next(v for v in actual['sources']if'/runtime/archive/'in v['path']and v['path'].endswith('qa_em.py'));assert qa['sha256']==freshqa['sha256']
qa_lines=qa['text'].splitlines()
data={'read_only_local_derivation':True,'scope_iterations':[0,150],'distribution_kind':'pre-group-normalization marginal per-response TRAINreward, notwithin-questiongroupdistribution','raw_reward_support_exact':[0.,1.],'equation':'P_t=(1-p_t)*delta_0+p_t*delta_1, p_t=mean(sequence_reward)','N_responses_per_iteration':N,'mean_precision':'nativeconsole3decimal; p_hat±.0005 plus.0000001 conservativeFPslack; countcandidatesareinference notrawhistogram','sum_reward_equation':'sumR=count(R=1)=N*p; onlycandidateboundsretained','candidate_inference':['enumerateintegerk1..N−1consistentwithprintedrewardmean andprintedstrictEMwithinhalfULPplusFPslack','all150native min0/max1 confirms bothsupportpoints actuallyoccur','sameper-response binary scores yielddifferentprintedmeansattieboundary;intersectioninclusive','notinventindividualtrajectoryrewardsorquestionassignment'], 'source':{'curve_path':str(source),'curve_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'node_log':capture['log'],'qa_source_path':freshqa['path'],'qa_source_sha256':freshqa['sha256'],'generated_em_binary_function_lines':[{'line':i,'text':qa_lines[i-1]}for i in range(140,147)],'select_generated_only':{'path':'runtime/archive/upstream/Search-R1/verl/trainer/main_ppo.py','sha256':'eddd4e68c45d1ecd1c0ab5e9815f49c55c2d847c6a958ccf792967424661fc3d','lines':[27,29,73,92,109,112]},'format_score_actual':0.,'correct_score_default':1.,'use_kl_loss_actual':True,'rewards_equal_scores':True,'reference_KL_only_inactor_loss_notR':True,'legacyEM_is_auditonly_notR':True},'rows':out,'unique_count_inferred_steps':[r['iteration']for r in out[1:]if r['unique_count_inferred_not_raw']],'limits':['ThisBernoullimarginaldistributionISrecoverableup_to3decimalmassuncertainty, unlikeunobservedwithin5jointdistribution','No groupk histogram/std/allcorrect/allwrong/mixed proportions inferred','MarginalBernoulli stdsqrt[p(1-p)] isnotconditionalwithin5rewardstd','Iteration0initial512validationnotimputedasTRAINdistribution','No new eval/train/GPU/Nodewrites/rawoutputtransfers']}
p=P/'marginal-pre-normal-reward-distribution-0-150.json';p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (P/'marginal-pre-normal-reward-distribution-0-150.csv').open('w',newline='')as f:
 names=['iteration','p_reward_1','p_reward_0','count_1_min','count_1_max','sumR_min','sumR_max','marginal_std_from_logged_mean','source_line'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
 for r in out:
  k=r.get('count_1_integer_candidate_bounds') or[None,None];w.writerow({'iteration':r['iteration'],'p_reward_1':r['p_reward_1'],'p_reward_0':r['p_reward_0'],'count_1_min':k[0],'count_1_max':k[1],'sumR_min':k[0],'sumR_max':k[1],'marginal_std_from_logged_mean':r['marginal_reward_std_from_logged_mean'],'source_line':r.get('source_line')})
manifest={'files':[{'path':str(z.resolve()),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest()}for z in sorted(P.iterdir())if z.is_file()and z.name!='manifest.json']};(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'uniquecountinferredsteps':data['unique_count_inferred_steps'],'examples':[{k:r.get(k)for k in['iteration','p_reward_1','p_reward_0','count_1_integer_candidate_bounds','count_1_integer_candidates']}for r in out if r['iteration']in[0,1,15,50,100,150]]},ensure_ascii=False))
