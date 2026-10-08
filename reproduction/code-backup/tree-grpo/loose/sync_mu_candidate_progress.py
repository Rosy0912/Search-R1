from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
OUT=ROOT/'outputs/tree-instruct-300'; NOW=datetime.now(timezone.utc).isoformat()
def save(p,value):
 b=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(b);tmp.replace(p)
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
review=ROOT/'work/tree-vllm-window-final-review/actual3296'; receipt=review/'MEASUREMENT_RECEIPT.json'; accepted=json.loads(receipt.read_text());assert accepted['actual_measurement_execution_verified'] is True and accepted['actual_optimizer_writes']==0 and not accepted['training_accepted']
for name in ['MEASUREMENT_RECEIPT.json','MEASUREMENT_REVIEW.md','MEASUREMENT_ADDITIONAL_PROOF.json']:
 shutil.copy2(review/name,OUT/('PPO_WINDOW_GPU3296_INDEPENDENT_'+name))
rs=ref(OUT/'PPO_WINDOW_GPU3296_INDEPENDENT_MEASUREMENT_RECEIPT.json')
owner_path=OUT/'PPO_WINDOW_GPU3296_OWNER_SUMMARY.json'; owner=json.loads(owner_path.read_text());owner.update(status='ACTUAL3296_MEASUREMENT_INDEPENDENTLY_ACCEPTED_PROBABILITY_STILL_UNACCEPTED',updated_at=NOW,independent_actual_measurement_accepted=True,independent_receipt=rs);save(owner_path,owner)
p=ROOT/'outputs/STATUS.json';s=json.loads(p.read_text())
s.update(updated_at=NOW,last_checked_at=NOW,current_phase='BEHAVIOR_MU_CORRECTION_SOURCE_INTEGRATION_BEFORE_INDEPENDENT_FREEZE',status='PPO_GUARDS_CPU_SYNTHETIC_ACCEPTED_BEHAVIOR_MU_SOURCE_CANDIDATE_REAL_TRAINING_PENDING',allocation_held=False,active_training=False,active_training_job=None,active_calibration_job=None,active_probability_probe_job=None,active_cpu_analysis_job=None,previous_goal_turn_classification='progress')
s['current_job_scheduler_state']='COMPLETED';s['current_counter_scope']='Historical3234 completed128; healthy parent110. Last3296 completed zero-write measurement. New production outer0/optimizer0; mu correction source-only candidate not submitted.'
s['passive_boundary_candidate'].update(status='ACTUAL3296_MEASUREMENT_INDEPENDENTLY_ACCEPTED_NO_PROBABILITY_OR_TRAINING_ACCEPTANCE',actual_runtime_accepted=True,independent_actual_receipt=rs,all2560_prompts_and808994_selected_token_probability_records_equal3295_verified=True)
s['last_completed_single_window_probe'].update(status=owner['status'],updated_at=NOW,independent_actual_measurement_accepted=True,independent_receipt=rs,scope=accepted['raw_statistics_scope']+' Nine anchor boundary vectors localize forward intervals only; not unique kernel attribution.')
s['behavior_mu_correction_candidate']=dict(status='SOURCE_INTEGRATION_CPU_ONLY_NOT_FROZEN_NOT_DEPLOYED',worktree=str(ROOT/'work/tree-behavior-mu-ppo'),base_source_commit='927ab58fdd8240d61aa52745700b2469f5406027',protocol='tree_behavior_mu_centered_ppo_v1',objective='D(pi/mu,A) actual sampled FP32 old probabilities; full beta keeps original reward/reference; monitor beta checks same partition repeat',original_sampler_actor_probability_gate_accepted=False,defaults_enabled=False,original_probability_gate_enforce_unchanged=True,source_frozen=False,independent_design_receipt=ref(ROOT/'work/tree-behavior-mu-ppo-design-review/RECEIPT.json'),owner_decision=ref(ROOT/'work/tree-same-backend-probability-diagnosis/OWNER_BEHAVIOR_MU_IMPLEMENTATION_DECISION.md'),actual_3b_zero_write_accepted=False,healthy_five_accepted=False,production_new_outer=0,production_new_optimizer_calls=0,updated_at=NOW)
s['ppo_three_guards'].update(updated_at=NOW,next_stage='Freeze and independently review explicit actual-mu correction source/CPU contracts; then actual3B zero-write/threshold observation and healthy110 five/HF/dev512 before300. Original sampler-actor probability gate remains failed; no tolerance loosening.',production_training_active=False,long_training_ready=False)
s['instruct300']['active_job_id']=None;s['instruct300']['phase']='healthy110_mu_source_candidate_no_active_training';save(p,s)
progress=f'''# 当前状态：三项防护已接入，采样概率校正候选正在验收

更新时间：{NOW}。恢复至300轮目标保持 active。历史失败分支3234完成128轮；选定健康完整checkpoint110，固定512题 **F1 49.8591%、EM 41.0156%**。新防护生产分支新增0轮、optimizer写入0；无活动训练或GPU诊断，评分未更新。

三项防护代码和实际8rank合成GPU验收已完成；真实3B概率/短训/生产阈值未通过。3296已COMPLETED0:0，419秒，实际8次生成、808994 emitted tokens、0更新；完整28份原记录、666对观察边界与资源/清理已独审接受。全部2560输入及808994采样token/概率记录与3295相同。原actor/sampler最大logprob差1.3392278仍存在；边界比较仅定位差异区间，不证明唯一kernel根因，也不证明训练容量。

依据实施规格§4，单列实际采样概率校正候选 D(π/μ,A)：固定每个原生token/前缀/温度/位置的真实FP32 μ作为PPO旧概率；独立保留fullbatch β作原参考奖励、同分区β-monitor作重复计算检查。Tree优势/等叶权重/分母、LR1e-6、train640/mini384/globalmicro16、完整outer10个optimizer调用、四轮及所有长度/奖励/split保持。原概率门禁未标通过，也未放宽容差。候选默认关闭、尚未冻结或部署；CPU源码测试不能替代实际3B零更新和健康110→115五轮/HF/512题验收。

零更新证据将分别保存μ/fullβ/monitorβ及Tree来源绑定；观测/padding的NaN按有效掩码处理。全部资产和服务继续限定h100-9，原资源门禁/8个背景PID保护不变。真实验收通过后继续300轮；最终51713及完整公平GRPO对照尚未完成。

[三项防护交付与证据](tree-instruct-300/PPO_THREE_GUARDS_DELIVERY.md)。

以下为保留的历史快照，不覆盖上述状态。

'''
p=ROOT/'outputs/REPORT.md';p.write_text(progress+p.read_text())
for name in ['PPO_THREE_GUARDS_DELIVERY.md','PPO_THREE_GUARDS_IMPLEMENTATION.md']:
 p=OUT/name
 p.write_text(f'当前补充（{NOW}）：3296完整实际测量已独审接受，原概率一致性仍失败。单列真实μ中心PPO源码候选整合中，默认关闭、尚未冻结/部署；原三项防护、Tree/LR/batch/奖励/10优化预算保持。实际3B零更新、健康五轮/HF/dev512、阈值和300轮均未完成，历史128、健康110及49.8591%/41.0156%不变。\n\n'+p.read_text())
p=OUT/'PPO_THREE_GUARDS_DELIVERY_EVIDENCE.json';e=json.loads(p.read_text());e.update(current_diagnostic_runtime_accepted=True,current_diagnostic_runtime_acceptance_scope=accepted['status'],current_diagnostic_measurement_receipt_sha256=rs['sha256'],current_diagnostic_full_raw_receipt_sha256=rs['sha256'],current_diagnostic_owner_summary=ref(owner_path),behavior_mu_correction_candidate=s['behavior_mu_correction_candidate'],updated_at=NOW)
for record in e['source_and_diff_records']:
 path=Path(record['path'])
 if path.exists():record.update(ref(path))
save(p,e)
for name in ['OWNER_BEHAVIOR_MU_IMPLEMENTATION_DECISION.md','BEHAVIOR_MU_PPO_DECISION_CANDIDATE.md']:
 p=ROOT/'work/tree-same-backend-probability-diagnosis'/name
 if p.exists():shutil.copy2(p,OUT/('PPO_'+name))
for name in ['RECEIPT.json','REVIEW.md']:
 shutil.copy2(ROOT/'work/tree-behavior-mu-ppo-design-review'/name,OUT/('PPO_BEHAVIOR_MU_DESIGN_'+name))
# Copy only new/changed small records; the completed383MB raw is already preserved.
newnames=['PPO_WINDOW_GPU3296_INDEPENDENT_MEASUREMENT_RECEIPT.json','PPO_WINDOW_GPU3296_INDEPENDENT_MEASUREMENT_REVIEW.md','PPO_WINDOW_GPU3296_INDEPENDENT_MEASUREMENT_ADDITIONAL_PROOF.json','PPO_WINDOW_GPU3296_OWNER_SUMMARY.json','PPO_THREE_GUARDS_DELIVERY.md','PPO_THREE_GUARDS_IMPLEMENTATION.md','PPO_THREE_GUARDS_DELIVERY_EVIDENCE.json','PPO_OWNER_BEHAVIOR_MU_IMPLEMENTATION_DECISION.md','PPO_BEHAVIOR_MU_PPO_DECISION_CANDIDATE.md','PPO_BEHAVIOR_MU_DESIGN_RECEIPT.json','PPO_BEHAVIOR_MU_DESIGN_REVIEW.md']
for dest,sub in [(Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),'tree-instruct-300'),(ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002','tree-instruct300')]:
 (dest/sub).mkdir(parents=True,exist_ok=True)
 for name in ['STATUS.json','REPORT.md']:shutil.copy2(ROOT/'outputs'/name,dest/name)
 for name in newnames:
  if (OUT/name).exists():shutil.copy2(OUT/name,dest/sub/name)
print(json.dumps(dict(updated_at=NOW,classification='progress',independent3296_receipt=rs,mu_source_only=True,active_gpu_training=False),ensure_ascii=False))
