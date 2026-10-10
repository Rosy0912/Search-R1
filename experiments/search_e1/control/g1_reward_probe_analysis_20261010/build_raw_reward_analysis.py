"""Bound actual pre-normalization reward masses; plot separate calibration snapshots."""
import hashlib
import json
import math
from pathlib import Path

p = Path(__file__).parent
def read(name): return json.loads((p/name).read_text())
old=read('observation-2.json')
exact={r['step']:r['metrics']['critic/rewards/mean'] for r in old['actual_updates']['complete_rows']}
last=read('observation-5.json')['native_progress']['last_record']
exact[last['step']]=last['metrics']['critic/rewards/mean']
pred=read('reward-storage-and-predicates.json')
cal=read('group-calibration.json')
arc=read('archive/marginal-pre-normal-reward-distribution-0-150.json')
rows=[]
for r in pred['scalar_rows']:
    t=r['step']
    if t>150 or t not in exact: continue
    m=r['metrics']; pc=m['train/perf/em']; mu=exact[t]; mn=m['critic/rewards/min']
    assert mn in (-.1,.05)
    lo=max(0,pc-.0005001);hi=min(1,pc+.0005001)
    no_invalid=mn==.05
    valid_wrong_mass_interval=[1-hi,1-lo] if no_invalid else None
    cm=[.05+(mu-.05)/hi,.05+(mu-.05)/lo] if no_invalid and lo>0 else None
    if cm: assert .98-1e-4<=cm[0]<=cm[1]<=1.6+1e-4
    rows.append({'iteration':t,'n_rollouts':2560,'raw_reward_mean_native':mu,'correct_fraction_console':pc,
                 'correct_fraction_interval':[lo,hi], 'raw_correct_support':[.98,1.6],
                 'protocol_invalid_present':not no_invalid,'protocol_invalid_mass':0 if no_invalid else None,
                 'raw_reward_point05_mass_interval':valid_wrong_mass_interval,
                 'conditional_correct_reward_mean_interval':cm,
                 'fine_histogram':None,'within_question_raw_reward_vectors':None})
assert rows[-1]['iteration']==141
common=[]
for t in [0,50,100]:
    s=cal['common_complete_groups_by_round'][str(t)]
    n=s['nominal_rollouts'];hist=s['raw_pre_normalization_reward_histogram'];c=hist['counts']
    assert sum(c)==n==305 and s['question_groups']==61
    # Categorical bins, not densities: near-one interval merged to avoid microscopic bins.
    categories=['0.05','0.98–1.00','1.00–1.10','1.10–1.20','1.20–1.30','1.30–1.40','1.40–1.50','1.50–1.60']
    cnt=[c[1],c[3]+c[4]]+c[5:]
    assert c[0]==c[2]==0 and sum(cnt)==n
    conditional_correct_mean=(s['raw_pre_normalization_reward']['mean']*n-.05*s['raw_reward_floor_count'])/s['raw_reward_correct_count']
    common.append({'round':t,'n_questions':61,'n_rollouts':n,'categories':categories,'counts':cnt,
                   'raw_mean':s['raw_pre_normalization_reward']['mean'],
                   'marginal_std':s['raw_pre_normalization_reward']['std_population'],
                   'reward_point05_mass':s['raw_reward_floor_count']/n,
                   'correct_fraction':s['raw_reward_correct_count']/n,
                   'conditional_correct_mean':conditional_correct_mean,
                   'mean_within_group_sample_std':s['reward_group_sample_std']['mean'],
                   'zero_std_groups':s['reconstructed_reward_zero_variance_groups'],
                   'all_correct_groups':s['all_correct'],'mixed_groups':s['mixed'],'all_wrong_groups':s['all_wrong'],
                   'correct_correct_mean_abs_reward_difference':s['correct_correct_absolute_reward_difference']['mean']})
inputs=['observation-2.json','observation-5.json','reward-storage-and-predicates.json','group-calibration.json','archive/marginal-pre-normal-reward-distribution-0-150.json']
out={'scope':[0,150],'actual_training_g1':rows,'actual_training_archive':arc['rows'],
     'calibration_common61':common,'interpretation':'Actual TRAIN coarse mass bounds and binary marginal; calibration reconstructed complete groups are explicitly separate.',
     'missing_historical_g1_fine_histograms':True,'full512_group_distribution_recorded':False,
     'source_shas':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in inputs}}
(p/'raw-reward-distributions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
colors=['#2563eb','#ea580c','#0f766e']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13.3,8.4))
a=[r for r in arc['rows'] if r['iteration']>0]
x=[r['iteration'] for r in a];y=[r['p_reward_1'] for r in a]
axs[0,0].stackplot(x,y,[1-v for v in y],labels=['reward = 1','reward = 0'],colors=[colors[2],'#dce7e5'],alpha=.9)
axs[0,0].set(title='Archive: every-iteration raw 0 / 1 mass',xlim=(0,150),ylim=(0,1),xlabel='Outer iteration')
valid=[r for r in rows if not r['protocol_invalid_present']]
missing=[r['iteration'] for r in rows if r['protocol_invalid_present']]
axs[0,1].scatter([r['iteration'] for r in valid],[r['correct_fraction_console'] for r in valid],s=12,color=colors[0],label='correct: shaped R in [0.98, 1.60]')
axs[0,1].scatter([r['iteration'] for r in valid],[1-r['correct_fraction_console'] for r in valid],s=12,color=colors[1],label='valid wrong: shaped R = 0.05')
axs[0,1].scatter(missing,[-.025]*len(missing),s=12,marker='x',color='#94a3b8',label='invalid present: split unavailable')
axs[0,1].set(title=f'G1: recoverable coarse masses ({len(valid)}/{len(rows)} iterations)',xlim=(0,150),ylim=(-.055,1),xlabel='Outer iteration')
categories=common[0]['categories'];pos=list(range(len(categories)))
for j,r in enumerate(common):
    axs[1,0].bar([x+(j-1)*.25 for x in pos],[n/305 for n in r['counts']],width=.24,color=colors[j],label=f"calibration round{r['round']}")
axs[1,0].set(title='CALIBRATION shaped reward: SAME 61 questions',xticks=pos,xticklabels=categories,ylim=(0,.75),xlabel='Shaped R category (not uniform-width density)')
axs[1,0].tick_params(axis='x',labelrotation=30)
for j,r in enumerate(common):
    correct=r['correct_fraction']
    cc=[n/(305*correct) for n in r['counts'][1:]]
    axs[1,1].plot(range(7),cc,color=colors[j],marker=['o','s','^'][j],label=f"round{r['round']}, correct n={305-r['counts'][0]}")
axs[1,1].set(title='CALIBRATION: reward distribution conditioned on correct',xticks=list(range(7)),xticklabels=categories[1:],ylim=(0,.65),xlabel='Correct reward category')
axs[1,1].tick_params(axis='x',labelrotation=30)
for ax in axs.flat: ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Before GRPO normalization: actual TRAIN evidence and separate calibration reconstruction',fontsize=13,fontweight='bold')
fig.text(.5,.002,'0–150 only. Historical G1 fine histograms were NOT saved. Lower panels are fitted-head CPU reconstruction on a selected calibration cohort, NOT live TRAIN.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.025,1,.965));fig.savefig(p/'pre-normalization-rewards.png',dpi=180,bbox_inches='tight')
print(json.dumps({'qualifying_iterations':len(valid),'actual_iterations':len(rows),'calibration_common61':common},ensure_ascii=False))
