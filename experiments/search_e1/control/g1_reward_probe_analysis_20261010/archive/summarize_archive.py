"""Derive small Archive4 curve/terminal summaries from actual read-only captures."""
import csv, hashlib, json, math
from pathlib import Path
P=Path(__file__).resolve().parent
first=json.loads((P/'observation-first.json').read_text());second=json.loads((P/'observation-second.json').read_text());extra=json.loads((P/'extra-scalar-source-evidence.json').read_text())
assert first['log']==second['log']
assert first['curve']==second['curve']
assert len(second['curve'])==199 and [r['iteration'] for r in second['curve']]==list(range(1,200))
assert extra['log_sha256']==second['log']['sha256']
extra_by={r['iteration']:r for r in extra['row_extra']}
rows=[]
for r in second['curve']:
 z=extra_by[r['iteration']];assert z['line']==r['line'] and z['line_sha256']==r['line_sha256']
 merged=dict(r['metrics']);merged.update(z['metrics']);assert all(math.isfinite(x) for x in merged.values())
 rows.append({**r,'metrics':merged})
curve={'algorithm':'Archive4 GRPO (Search-E1 first GRPO stage, not OPSD)','b':None,'d':None,'probe_status':'not_collected_in_this_run_and_not_reconstructed','base':second['base'],'run_id':second['run_id'],'hostname':second['hostname'],'observations_cst':[first['observed_cst'],second['observed_cst']],'log':second['log'],'curve_rows':rows,'curve_complete_for_observed_iterations':True,'expected_budget':200,'observed_complete_iterations':199,'missing_iteration':200,'printed_precision':'native console scalar has three decimal digits; retain as logged, do not invent exact training counts','true_lr_from_console':'unavailable at printed precision; all1e-6-scale values printed0.000','validation_protocol':'original archive vLLM validation do_sample=False; different backend/generation budget from later native OPSD HF greedy512','fullraw_logs_copied':False,'new_eval_or_train':False}
(P/'curve-1-199.json').write_text(json.dumps(curve,ensure_ascii=False,indent=2)+'\n')
keys=['audit/strict_em','audit/legacy_em','critic/rewards/mean','actor/entropy_loss','actor/pg_loss','actor/grad_norm','actor/ppo_kl','actor/kl_loss','actor/pg_clipfrac','env/finish_ratio','env/number_of_actions/mean','env/number_of_valid_search','env/ratio_of_valid_action','response_length/mean','critic/score/max','critic/score/min','actor/lr']
with (P/'curve-1-199.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['iteration','source_line','source_line_sha256']+keys);w.writeheader()
 for r in rows:w.writerow({'iteration':r['iteration'],'source_line':r['line'],'source_line_sha256':r['line_sha256'],**{k:r['metrics'].get(k) for k in keys}})
valid=[]
for r in rows:
 if 'val/test_score/nq' not in r['metrics']:continue
 groups={}
 for k in ['nq','hotpotqa']:
  displayed=r['metrics']['val/test_score/'+k]
  candidates=[n for n in range(257) if abs(n/256-displayed)<0.0005000001]
  assert len(candidates)==1
  groups[k]={'n':256,'displayed_em':displayed,'correct_uniquely_inferred':candidates[0],'exact_em_from_unique_count':candidates[0]/256,'integer_candidates':candidates}
 valid.append({'iteration':r['iteration'],'n':512,'correct_uniquely_inferred':sum(z['correct_uniquely_inferred'] for z in groups.values()),'exact_em_from_unique_count':sum(z['correct_uniquely_inferred'] for z in groups.values())/512,'groups':groups,'line':r['line'],'line_sha256':r['line_sha256'],'F1':None,'F1_status':'not_logged','count_status':'uniquely_inferred_from_three_decimal_binarymean_256_per_source_not_raw_question_outputs'})
stop=next(e for e in second['terminal_evidence'] if 'human-stop' in e['path']); stopdata=stop['data']
windows=[]
for a,b in [(1,25),(26,50),(51,100),(101,150),(151,175),(176,180),(181,183),(184,185),(186,199)]:
 selected=[r for r in rows if a<=r['iteration']<=b]
 windows.append({'iterations':[a,b],'count':len(selected),'means_logged':{k:sum(r['metrics'][k] for r in selected)/len(selected) for k in ['audit/strict_em','critic/rewards/mean','actor/entropy_loss','env/finish_ratio','env/number_of_valid_search','response_length/mean']}})
zero_rows=[r['iteration'] for r in rows if r['metrics'].get('critic/score/max')==0]
summary={'scope':'read-only historical completed iteration accounting and actual stopped-run revalidation, no restart','observations_cst':[first['observed_cst'],second['observed_cst']],'actual_live_progress_interpretation':'stable completed history and stopped identities verified twice; no new iteration expected or launched','run_id':second['run_id'],'completed_iterations':199,'observed_iteration_200':False,'unique_contiguous_iterations':True,'all_observed_metric_scalars_finite':True,'b_d_probe_status':'not_collected; never substitute0 or compare invented b/d','latest_iteration':rows[-1],'current_original_pids':second['original_pids'],'terminal':{'stopped_preserved_receipt_path':stop['path'],'receipt_sha256':stop['sha256'],'receipt_accepted':stopdata['accepted'],'status':stopdata['status'],'original_status_unchanged':stopdata['original_status_unchanged'],'complete_Adam_resume_state':stopdata['complete_Adam_resume_state'],'record_observed_unix':stopdata['observed_unix'],'fresh_historical_process_checks_absent':[p['pid'] for p in second['original_pids'][:3] if not p['exists']],'pipeline_exit0_text':'2026-10-09T08:57:00+00:00 EXIT_0','pipeline_exit0_not_evidence_of_200_complete':True,'stale_running_receipt_not_current':True},'recent_fatal_log_matches':second['recent_fatal_lines'],'native_fixed512':valid,'native_fixed512_200':'not_produced; checkpoint200absent','initial_fixed512':{'nq_correct':45,'hotpot_correct':52,'total_correct':97,'n':512,'EM':97/512,'evidence':second['validation_log_records'],'source':'initial exact logged binarysource means'},'checkpoints':second['checkpoint_inventory'],'windows':windows,'collapse':{'before':[{k:r[k] for k in ['iteration','line','line_sha256','metrics']} for r in rows if 178<=r['iteration']<=186],'statement':'major observed collapse at184; prior finish/gradient instability observable181–183; no proven root cause','printed_train_em_183':.419,'printed_train_em_184':.019,'printed_train_em_185':.003,'printed_train_em_186_199':0.,'per_response_score_max_zero_iterations':zero_rows,'all_scores_zero_since_iteration':187 if all(i in zero_rows for i in range(187,200)) else None,'OPSD_stage_not_in_curve':True},'scoring_definition':'strict terminal generated-only normalizedEM; prompt/tool text masked; legacy full-transcript score audit only; use_kl_loss=True keeps strict scores as rewards; referenceKL separateactor penalty','scoring_sources':[{'path':s['path'],'sha256':s['sha256'],'bytes':s['bytes']} for s in second['sources']],'source_extra':extra['source_extra'],'limits':['Train scalar means printed3decimals; exact raw correct counts unavailable except scoremax proves allzero group','actor/lr console0.000 does not establish zero actualLR; frozen script lr1e-6/warmup ratio.285','grad_norm is preclip norm with clip_grad_norm config1, not norm ofactualupdate','Fixed512 countsinferreduniquelyfromroundedper-source256binarymeans, not per-question raw predictions; F1missing','No newb/d inference or counterfactual probes inArchive; comparisonG1b/dmustonlywithinG1','Rewardcollapse beforeany OPSD inthis GRPOcurve doesnotprove mathematical causalroot']}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
manifest={'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(P.iterdir()) if p.is_file() and p.name!='manifest.json']}
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'summary':str(P/'summary.json'),'summarysha':hashlib.sha256((P/'summary.json').read_bytes()).hexdigest(),'curvesha':hashlib.sha256((P/'curve-1-199.json').read_bytes()).hexdigest(),'all_scores_zero_since':summary['collapse']['all_scores_zero_since_iteration'],'fixed512':[{'iteration':z['iteration'],'correct':z['correct_uniquely_inferred'],'n':512} for z in valid],'windowmeans':windows},ensure_ascii=False))
