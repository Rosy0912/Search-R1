# -*- coding: utf-8 -*-
import json,datetime,shutil
from pathlib import Path
r=Path('outputs');p=r/'STATUS.json';s=json.loads(p.read_text())
s['time']=datetime.datetime.now(datetime.timezone.utc).isoformat();s['last_heartbeat']='2026-10-03T21:59:42.422Z'
s['status']='CORRECTED_SOURCE_LINUX_CPU_STAGE64_PASS_HARDWARE_LAUNCH_REVIEW_PENDING'
s['training_accepted']=False
s['current_training']['current_phase']='COMPLETED_OLD_PROTOCOL_ARTIFACT_ACCEPTED'
s['current_training']['slurm_accounting']='disabled; terminal COMPLETED0:0 captured independently'
s['reward_protocol_fix']['corrected_protocol_experiment']='isolated b097 source deployed; actual targetLinux174PASS0skip and identical stage64 proved; fresh GPU and corrected training pending'
s['corrected_current_training']={'source_commit':'b09736d9d1b487a5c134e7a358761e04613cea32','root':'/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097','job_id':None,'state':'HARDWARE_LAUNCH_REVIEW_PREPARATION','actual_outer_updates':0,'planned_actual_outer_updates':500,'target_linux_cpu_tests':{'passed':174,'skipped':0,'test_wall_seconds':13.49,'gate_wall_seconds':19.102035760879517,'evidence':'work/evidence/rewardfix-linux-cpu-gate.log'},'stage64':{'rows':64,'same_train166222_and_stage64_hashes':True,'disjoint_training_periodic_final':True,'evidence':'work/evidence/rewardfix-stage64-proof.log'},'setup_evidence':'work/evidence/rewardfix-isolation-setup.json','hardware_accepted':False,'training_accepted':False}
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
note='\n\n2026-10-04 修复版推进：独立目录 `bapo-rewardfix-b097` 已固定 b097 源；目标 Linux 真实完整测试 174 passed、0 skipped（13.49s，整个CPU门禁19.10s）。新数据副本重新制备所得训练166222与阶段64题SHA和内容身份完全匹配原定split，64题与训练/周期/最终集不相交。独立子代理正在审查新提交工具和共享输入身份门禁；尚无修复版GPU训练作业，actual outer=0。旧2948保留为旧协议500轮产物，不能代表修复后的效果。\n'
with (r/'REPORT.md').open('a') as f:f.write(note)
with (r/'ACTIVE-FORMAL2948-HANDOFF.md').open('a') as f:f.write(note)
d=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/bapo');d.mkdir(exist_ok=True)
for name in ['STATUS.json','REPORT.md','ACTIVE-FORMAL2948-HANDOFF.md']:shutil.copy2(r/name,d/name)
for f in Path('work/evidence').glob('rewardfix-*'):
 if f.is_file():shutil.copy2(f,d/f.name)
