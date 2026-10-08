import json,datetime,shutil,re
from pathlib import Path
p=Path('outputs/STATUS.json');s=json.loads(p.read_text());e=json.loads(Path('outputs/evidence/tree-fixed-five-3103-current-progress.json').read_text());n=e['rank_records']//8;now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat();wall=re.findall(r'timing_s/step:([0-9.]+)',e.get('stage.log_tail',''));state=re.search(r'JobState=(\S+)',e['slurm']).group(1)
s.update(status='fixed_five_3103_'+state+'_NOT_ACCEPTED',last_checked_at=now,updated_at=now,allocation_held=state=='RUNNING',active_retrieval_job=3103 if state=='RUNNING' else None,current_job_actual_outer_updates=n,current_job_checkpoint_steps=[],current_job_latest_step_wall_seconds=float(wall[-1]) if wall else None)
s['fixed_numeric_five'].update(status=state+'_ACTUAL_REVIEW_PENDING',actual_outer_updates=n,rank_records=e['rank_records'],guard_samples=e.get('guard_samples'),guard_events=e.get('guard_events'),progress_evidence='outputs/evidence/tree-fixed-five-3103-current-progress.json',slurm_state=state,gpu_kl_pass_records=sum(len(r['dtypes']) for r in e['kl']))
s['fixed_numeric_two_v3']['independent_actual_review']='PASS_SCIENTIFIC_AND_OPERATIONAL_TWO_AND_FULL_HF2'
s['pending']=['Fresh fixed five/fullHF5 independent acceptance','Fresh formal500 pre-dispatch review and actual completion','Fixed periodic64 and full final51713','FullHF500 CPU reload and terminal cleanup','Fair newGRPO final comparison if eligible control available']
s['hourly_automation']={'id':'tree-grpo','status':'ACTIVE','interval_hours':1,'prompt_updated_for_fixed_repair_stages':True}
s['fixed_formal_draft']={'local_directory':'work/tree-fixed-formal-296217a-draft','status':'LOCAL_DRAFT_NOT_DISPATCHABLE','actual_five_required':True,'independent_pre_dispatch_required':True,'submitted':False}
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
text=f'\n\n{now}：修复源码296217a下fresh5作业3103实际{n}/5（{e["rank_records"]}直接rank记录），Slurm {state}。最新完成步约{float(wall[-1])/60:.2f}分钟。24项GPU KL与实际Ray observer安装已通过，guard暂无事件，五步/HF5独立验收尚未成立。fresh2/HF2已有独立科学与运行验收PASS。修复协议新正式仍0/500，尚无64题新评分或51713最终评估；旧2971实际307后失败、旧300步F1 39.0625%/EM31.25%仅修复前历史。小时问询保持ACTIVE，每小时一次，过期2971运行提示已更新。新的500步调度及HF500取证只准备本地不可执行草稿，等待实际five和启动前独审。\n'
with Path('outputs/REPORT.md').open('a') as f:f.write(text)
roots=[Path('work/tree-grpo/reproduction/baselines/tree-validation-20261002'),Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')]
files=[p,Path('outputs/REPORT.md'),Path('outputs/evidence/tree-fixed-five-3103-current-progress.json')]
for q in [Path('work/tree-fixed-five-review/PRE_DISPATCH_RECEIPT.json'),Path('work/tree-fixed-five-review/REVIEW.md'),Path('work/tree-fixed-five-review/ACTUAL3103_KL_PREFIX_REVIEW.json'),Path('work/tree-fixed-five-review/ACTUAL3103_OBSERVER_INSTALL_PREFIX.json')]:
 if q.exists():files.append(q)
for root in roots:
 root.mkdir(parents=True,exist_ok=True)
 for q in files:shutil.copy2(q,root/q.name)
print(json.dumps({'state':state,'actual_outer':n,'rank_records':e['rank_records'],'checked_at':now},ensure_ascii=False))
