"""Local stdlib-only reader. Does not connect, mutate production, or run GPU work."""
import argparse, json, re, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--current',required=True);p.add_argument('--baseline',required=True);p.add_argument('--capacity');p.add_argument('--output',required=True);a=p.parse_args()
def rows(path):
 d=json.loads(Path(path).read_text())
 if 'outer_rows' in d:return d['outer_rows']
 out=[]
 for line in d.get('metric_lines',[]):
  m={}
  for part in re.sub(r'\x1b\[[0-9;]*m','',line).split(' - '):
   if ':' not in part:continue
   k,v=part.rsplit(':',1)
   try:m[k.strip()]=float(v)
   except ValueError:pass
  if m.get('step',0)>=1:out.append({'step':int(m['step']),'metrics':m})
 return out
micro=None
if a.capacity:
 c=json.loads(Path(a.capacity).read_text());assert c['passed'] is True and str(c['job_id'])=='2994';micro=c['max_safe_local_micro'];assert micro in(1,2,4,8)
def summarize(items):
 out=[];previous={}
 for row in items:
  m=row['metrics']; t={k:v for k,v in m.items() if k.startswith('timing_s/')};ranks={}
  for rank in range(8):
   suffix='/rank_'+str(rank);v={k[:-len(suffix)]:v for k,v in m.items() if k.endswith(suffix)}
   delta={k:v[k]-previous.get(rank,{}).get(k,0) for k in v if 'cumulative' in k and k in previous.get(rank,v)}
   ranks[str(rank)]={'observed':v,'cumulative_deltas':delta,'teacher_head_and_empty_chunk_counts':'UNMEASURED: production source has no projection/skip counters'};previous[rank]=v
  trajectories=m.get('rollout/trajectories');width=m.get('response_length/max');physical=trajectories*width if trajectories and width else None
  supervised=m.get('state_tokens/total')
  # This is an algebraic residual, not a causal/exclusive old-logprob timer.
  accounted=sum(t.get(k,0) for k in ('timing_s/gen','timing_s/adv','timing_s/update_actor','timing_s/stage_testing','timing_s/testing','timing_s/save_checkpoint'))
  out.append({'step':row['step'],'timing_seconds':t,'unlisted_outer_wall_residual_seconds':None if 'timing_s/step' not in t else t['timing_s/step']-accounted,'old_logprob_seconds':t.get('timing_s/compute_log_prob','UNMEASURED separate timer'), 'response_width_derived_from_logged_max':width,'width_derivation_scope':'compose pads raw response to max; max metric is attention-derived, equivalence assumes no valid response pad IDs; not direct tensor-shape instrumentation','physical_response_positions_derived':physical,'physical_width_confirmed_by_logged_clip_ratio':m.get('response_length/clip_ratio',0)>0,'observed_attention_response_positions_derived_from_rounded_mean':trajectories*m['response_length/mean'] if trajectories and 'response_length/mean' in m else None,'supervised_fraction_of_observed_attention_response':supervised/(trajectories*m['response_length/mean']) if supervised is not None and trajectories and m.get('response_length/mean',0)>0 else None,'supervised_tokens':supervised,'supervised_fraction_of_derived_physical_positions':supervised/physical if physical and supervised is not None else None,'rollout_budget':{k:v for k,v in m.items() if k.startswith('rollout/')},'all_eight_rank_observations':ranks})
 return out
new=summarize(rows(a.current));old=summarize(rows(a.baseline));keys=('timing_s/gen','timing_s/update_actor','timing_s/step')
def means(rs):return {k:statistics.mean([r['timing_seconds'][k] for r in rs if k in r['timing_seconds']]) for k in keys if any(k in r['timing_seconds'] for r in rs)}
bm=means(old);nm=means(new);h20={'timing_s/gen':69.41,'timing_s/update_actor':55.59,'timing_s/step':153.44}
d={'job_id':'2994','source':'8e60976d364ab7d7ae51714415813c1359dafe9a','status':'WAIT_ACTUAL_OUTER' if not new else 'PARTIAL_ACTUAL_OUTERS' if len(new)<5 else 'FIVE_ACTUAL_OUTERS_ANALYZED_NOT_ACCEPTANCE','actual_outer_count':len(new),'actual_selected_local_micro':micro or 'NOT_YET_OBSERVED','actual_selected_global_micro':micro*8 if micro else 'NOT_YET_OBSERVED','source_budget_when_allrank_audits_match':{'global_rollout_rows':2560,'samples_per_rank':320,'optimizer_updates_per_rank':10,'student_micro_forwards_per_rank':320//micro if micro else 'UNKNOWN'},'current':new,'baseline2960':old,'mean_seconds':{'2994':nm,'2960_five_actual':bm,'H20_user_average':h20},'ratios_2994_over_2960':{k:nm[k]/bm[k] for k in nm},'ratios_2994_over_H20':{k:nm[k]/h20[k] for k in nm},'scope_limits':['Teacher stream/host/transfer intervals overlap outer update and cannot be added or causally subtracted','Residual includes unlisted handling/old-policy LP and boundary overhead; not isolated measured old-policy LP','No production head/empty-chunk counters: actual avoided calls UNMEASURED','Partial new run versus old five average is diagnostic only; no five-outer speed claim before five','Different rollout lengths/search counts and actual chosen micro change work; preserve and report those budgets','H20 GRPO update differs from full-vocabulary FKL plus FP32 eager frozen teacher; H20 also computes student vocabulary head','Timing logs rounded three decimals; A strict paired equivalence gate does not establish wall-time savings','No GPU work or production changes by this reader']}
Path(a.output).write_text(json.dumps(d,indent=2)+'\n');print(d['status'],d['actual_outer_count'],nm)
