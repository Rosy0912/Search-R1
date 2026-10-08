# -*- coding: utf-8 -*-
from pathlib import Path
import json,datetime,shutil
p=Path('outputs/STATUS.json');s=json.loads(p.read_text());s['time']=datetime.datetime.now(datetime.timezone.utc).isoformat();s['status']='B097_SMOKE2_ACCEPTED_PUBLIC_NUMERICAL_FIX_REQUIRED_BEFORE_FIVE_FORMAL';c=s['current_training'];c.update(state='COMPLETED_SMOKE2_INDEPENDENT_ACCEPTED',current_step='no GPU job running; isolated numerical repair preparation',smoke_accepted=True,independent_smoke_review={'receipt':'outputs/bapo-rewardfix-smoke3069-acceptance.json','receipt_sha256':'1032e45002e97679225a439ea75e6ccee116c2358bfe661a7483c2dd79823faf','report':'outputs/bapo-rewardfix-smoke3069-independent-review.md','report_sha256':'e10b03687a4f727d3cd271f74948adfcc1b34e8777c0d40194b786108aa43c91','bindings':111,'actual_deployed_receipt_gate_passed':True,'scope':'SMOKE2_ONLY_NOT_FIVE_OR_LONG_RUN_ACCEPTANCE'})
s['corrected_current_training']=c
s['numerical_fix_required']={'status':'MUST_FIX_ACTUAL_B097_FUNCTION_COUNTEREXAMPLES','source':'b09736d9d1b487a5c134e7a358761e04613cea32','report':'outputs/bapo-b097-numerical-risk-independent-review.md','proof':'work/evidence/bapo-b097-actual-numerical-boundaries.json','bugs':['finite legal logprob delta89/99 -> finite KL10 but NaN backward','finite FP32 gradients1e20 -> overflow FP32 clip norm, mutate zeros then refuse'], 'not_observed_in_smoke3069':True,'source_candidate_local':'work/bapo-numerics','source_candidate_state':'UNCOMMITTED_MINIMAL_PATCH_UNDER_INDEPENDENT_REVIEW','proposed_root':'/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric','unchanged_source_snapshots':['old b20 ROOT/bapo','completed b097 ROOT/bapo-rewardfix-b097'],'new_training_job':None}
s['next_required']='Finish minimal numerical candidate local/source independent review; new commit/root nativeLinux+8rank FSDP/KL+local retrieval gates -> fresh smoke2/five/HF/independent -> fresh500/final51713; do not submit b097 five/formal or reuse source-bound receipts'
s['new_grpo_control_effect_metrics']='Agent1 REPORT latest says full GRPO control stopped by user and incomplete; no matched corrected control. Do not restart other task or claim BAPO superiority.'
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
note='''

修复源3069两轮独立接受PASS（111直接证据/current108文件hash），receipt SHA1032e45002e97679225a439ea75e6ccee116c2358bfe661a7483c2dd79823faf；部署actual新smoke gate已PASS。但真实targetTorch2.4实际函数边界核查证明b097公共数值必须修：合法有限logprob差89/99的clipped KL前向10而反向NaN；有限FP32梯度1e20范数溢出并先变零后报错。3069未触发这些故障，保留已接受2轮产物；不以小样本PASS掩盖已证实长程风险，不提交b097five/formal。
采用最小不改数学/预算方案：借4dc公共gradient_norm FP64全局L2及KL饱和输入界、只改原有2函数调用；不引入GiGPO runtime_audit或目标变化。独立clone work/bapo-numerics新candidate正在local/full/源码独审，old b20、b097目标source/tools/receipts保持。新ROOT计划bapo-rewardfix-numeric，必须固定新commit并重签本次nativeLinux+8rank原生FSDP/KL+本机GPU检索、真实2/5/HF与独立门禁，不能借3064/3069为新源授权。
匹配控制只读核查：Agent1结果gigpo/REPORT.md末次报告GRPO正式对照因用户停止未完成；用户停止授权不能由本任务擅自恢复，现没有匹配修复版GRPO结果。GiGPO自身3026继续不等于GRPO控制；本任务不操作它，不宣称公平算法优越。
'''
for n in ['REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:
 with (Path('outputs')/n).open('a') as f:f.write(note)
d=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/bapo')
for n in ['STATUS.json','REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:shutil.copy2(Path('outputs')/n,d/n)
for p in Path('outputs').iterdir():
 if p.is_file() and (p.name.startswith('bapo-rewardfix-smoke3069') or p.name.startswith('bapo-b097-numerical') or p.name.startswith('bapo-numerical')):shutil.copy2(p,d/p.name)
for p in Path('work/evidence').iterdir():
 if p.is_file() and (p.name.startswith('bapo-b097-') or p.name.startswith('bapo-numerics-') or p.name.startswith('rewardfix-smoke3069')):shutil.copy2(p,d/p.name)
