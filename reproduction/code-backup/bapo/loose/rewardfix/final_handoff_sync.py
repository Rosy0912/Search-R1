# -*- coding: utf-8 -*-
from pathlib import Path
import json,datetime,shutil
p=Path('outputs/STATUS.json');s=json.loads(p.read_text());s['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
s['pre_fix_formal_training']=s['current_training'];s['current_training']=s['corrected_current_training']
s['current_job']=3069;s['current_run_root']=s['current_training']['run_root'];s['source_commit']=s['current_training']['source_commit'];s['actual_outer_updates']=0
s['pre_fix_five_iter_acceptance_outer_updates']=s['five_iter_acceptance_outer_updates'];s['five_iter_acceptance_outer_updates']=0
s['pre_fix_hardware_current']=s['hardware_current'];s['hardware_current']=s['corrected_hardware']
s['pre_fix_cpu_tests']=s['cpu_tests'];s['cpu_tests']={'passed':174,'skipped':0,'failed':0,'wall_seconds':13.49,'source_commit':s['source_commit']}
s['corrected_evidence_tools']={'native_tests':44,'skipped':0,'wall_seconds':.450,'scope':'CPU_TOOL_IMPLEMENTATION_ONLY_NOT_HF_TRAINING_ACCEPTANCE','root':s['current_training']['root']+'/tools/rewardfix-evidence','independent_review':'work/rewardfix/evidence-tools/INDEPENDENT-TOOL-REVIEW.md','independent_review_sha256':'ec4681b384acc30097d36238efb1eac8dcbf004d72eca2dbd15180576648a4b2','native_evidence':'work/evidence/rewardfix-evidence-tools-native-gate.log'}
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
note='''

最新交接：当前作业是修复版smoke3069（06:31:18启动），不要按历史2948字段重复运行。STATUS.current_training/current_job/source/actual_outer已切换b097/3069/0；旧500证据在pre_fix_formal_training，旧5轮在历史字段。3069回环检索已真实GET/POST200，GPU服务/数据集初始化正常，当前尚无完成outer/stage64成绩；原始log与实际500/HF验收必须另证。
新HF提取工具已独立复审并部署到newROOT/tools/rewardfix-evidence，目标Linux实际44tests PASS/0skip/.450s（39validator+5tied math）；仅工具门禁。工具包括bapo-smoke2-evidence-gpu64-tied.py / bapo-accept5-evidence-gpu64-tied.py / bapo-formal500-evidence-gpu64-tied.py。完成后以单一--log RUN/train.log、--job-id实际job、--slurm-record本次完整终态record、--base-model newROOT/assets-root/assets/qwen2.5-3b、--compare-weights --hf-load --hash-checkpoint，输出newROOT/logs；禁止假HF或用old检查点。提取器强制source-start marker和完整stage score replay，不允许缺/旧source或earlyTrue/stalled420伪状态。
3069终态watch session26242每15s后台只读保留，work/evidence/rewardfix-smoke3069-terminal-watch.log；不要重启/重复提交。完成后先保存terminal至worker新logs，归档run顶层测量/训练日志（明确排cache/runtime/tmp/output大文件，checkpoint另完整hash实际HF），下载安全校验，独立subagent签新smoke receipt部署并实际verify_smoke_receipt-gpu64.py 3069，再唯一fresh5。head D=/dev/shm/bapo-rewardfix-b097-scheduler-20261004，已签训练提交工具/3SBATCH部署；SMOKE2_JOB_ID=3069 bash D/submit-training.sh accept5，必须新smoke gate通过；随后ACCEPT5_JOB_ID实际fresh5job bash D/submit-training.sh formal500，必须新5gate通过。跨phase全局锁及intent-before-sbatch保留，永不盲重提。
3064固定hardware/stage/tools已有57bound新receipt，增加新evidence工具和training verifiers未改其原绑定文件；stage SHA1f8b59a9e525f03d2117b118235ae5b077dc41ed92f562193dca3289b80ee39c。算法源码b097不hotpatch。小时自动跟进保持ACTIVE，直到修复版500/final51713/独审与匹配GRPO效果结论完成。
'''
for n in ['REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:
 with (Path('outputs')/n).open('a') as f:f.write(note)
d=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/bapo')
for n in ['STATUS.json','REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:shutil.copy2(Path('outputs')/n,d/n)
for f in Path('outputs').glob('bapo-rewardfix*'):
 if f.is_file():shutil.copy2(f,d/f.name)
for f in Path('work/evidence').glob('rewardfix-*'):
 if f.is_file():shutil.copy2(f,d/f.name)
et=d/'rewardfix-evidence-tools';et.mkdir(exist_ok=True)
for f in Path('work/rewardfix/evidence-tools').iterdir():
 if f.is_file():shutil.copy2(f,et/f.name)
