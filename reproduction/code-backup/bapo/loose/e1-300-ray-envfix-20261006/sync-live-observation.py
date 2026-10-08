"""Local reporting only. Never operates on remote jobs or signs final acceptance."""
import json,math,datetime,shutil,hashlib,sys
from pathlib import Path
h=Path(sys.argv[1]);x=json.loads(h.read_text());assert x['job_id']==3292 and x['source_commit']=='a2568d0df306ead6e059760e28b5bd53ee3f6876'
n=x['actual_learning_path_outer'];m=x['last_driver_metrics']['metrics'];checks=[]
assert x['completed_new_driver_steps']==list(range(26,n+1))
for r in x['driver_metrics_history']:
 a=r['metrics'];i=r['step'];z={k:a[k] for k in ['bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite']};bad=[k for k,v in a.items() if isinstance(v,(int,float))and not math.isfinite(v)];lr=math.isclose(a['actor/lr'],min(i,114)/114*1e-6,rel_tol=1e-10);assert not bad and lr and not any(z.values())and a['actor/optimizer_steps']==20 and a['bapo/stage_evaluated_count']==64;checks.append(dict(step=i,nonfinite_keys=bad,zero_counters=z,lr114_matches=lr,optimizer_steps=20,stage_count=64))
audit=h.with_name(h.stem+'-driver-audit.json');assert not audit.exists();audit.write_text(json.dumps(dict(scope=f'DRIVER_OBSERVED26_{n}_NOT_ALLRANK_OR_FINAL300',original_health_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),checks=checks,errors=[]),indent=2)+'\n')
p=Path('outputs/STATUS.json');s=json.loads(p.read_text());prior=s['current_training']['actual_completed_outer_updates'];assert n>=prior
s['current_training'].update(state=f'RUNNING_OUTER{n+1}_COMPLETED{n}',actual_completed_outer_updates=n,next_outer=n+1)
e=x['resource_events'];done=[a for a in e if a['event']=='recovery_finished'];starts=[a for a in e if a['event']=='recovery_started'];pending=len(starts)>len(done);reason=x['last_resource_sample']['reason']
s['current_operation'].update(continuous_completed_outer=n,next_outer=n+1,current_step=f'Outer{n+1} in progress; complete driver marker ends at{n}',health_evidence=str(h),new_completed_driver_steps=x['completed_new_driver_steps'],latest_step_wall_seconds=m['timing_s/step'],stage_correct=round(m['bapo/stage_validation']*64),latest_guard_reason=reason,latest_guard_state='RECOVERY_IN_PROGRESS'if pending else 'WITHIN_THRESHOLD'if reason=='within_threshold'else 'RESUMED_AFTER_VALIDATION',guardian_unfinished_recovery=pending,recovery_cycles=len(done),recovery_wall_seconds=sum(a.get('elapsed_seconds',0)for a in done),observed_metrics_audit=str(audit))
s['formal3292_driver_metrics_history']=x['driver_metrics_history'];s['current_reward_health'].update(scope=f'OBSERVED_OUTER26_{n}_ONLY',latest_reward=m['critic/rewards/mean'],latest_pg_loss=m['actor/pg_loss'],latest_grad_norm_max=m['bapo/grad_norm_max'],finite_metric_and_mask_audit_errors=0,effect_improvement_proven=False);s['last_actual_check_shanghai']=datetime.datetime.fromtimestamp(x['time_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat();p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
d=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/bapo');shutil.copy2(p,d/'STATUS.json')
for f in [h,audit]:shutil.copy2(f,d/'instruct300/ray-envfix-20261006'/f.name)
if n>prior:
 note=f'\n最新实际完成{n}/300/当前{n+1}；reward{m["critic/rewards/mean"]:.6f}/PG{m["actor/pg_loss"]:.9f}/gradmax{m["bapo/grad_norm_max"]:.6f}/stage{round(m["bapo/stage_validation"]*64)}/64/wall{m["timing_s/step"]:.3f}s。26..{n}driver有限/mask/logprob/LR114/20Adam/stage64错误0；资源reason={reason}/恢复pending={pending}，仅当前范围检查，非最终验收或效果提升证明，goalACTIVE。\n'
 for name in ['REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:
  pp=Path('outputs')/name
  with pp.open('a')as f:f.write(note)
  shutil.copy2(pp,d/name)
print(json.dumps(dict(completed=n,current=n+1,reward=m['critic/rewards/mean'],audit_errors=[],recovery_pending=pending)))
