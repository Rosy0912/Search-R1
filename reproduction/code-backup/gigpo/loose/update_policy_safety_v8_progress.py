# -*- coding: utf-8 -*-
from pathlib import Path
import json, datetime, os

p = Path('work/gigpo-e1-instruct300-state.json')
s = json.loads(p.read_text()); q = s['policy_safety']
q.update(calibration_repair_status='V8_LOCAL_AND_TARGET_TORCH24_DATAROW_CPU_PASS_DEPLOYMENT_PREPARING',
         calibration_repair_actual_target_cpu_sha256='08c7336c8f0980b42a22a49690104ac3b5af027a26c5c07779167a24a3b933ba',
         calibration_repair_target_cpu_raw_sha256='a5bfaef28006d8c87ed48ceb055f91509601f992cbe8d30515c99d75494cdaa0',
         calibration_repair_target_cpu_review_sha256='cf083cbacfb199fd7046814e1b2aaf921e5ed6c730a09a8f62433dc4620c8711',
         calibration_repair_target_cpu_tests=9)
s['last_checked_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n'); os.replace(tmp,p)
p = Path('outputs/gigpo-e1-policy-safety/IMPLEMENTATION-STATUS.json')
status = json.loads(p.read_text())
status.update({key: value for key, value in q.items() if key.startswith('calibration_')})
status.update(updated_utc=s['last_checked_utc'], status='SOURCE52_NATIVE3267_PASS_CALIBRATION3281_FAILED_V8_DATAROW_TARGET_CPU_PASS_DEPLOYMENT_PENDING',
              actual_retained_iterations=25, actual_new_outer_updates=0, new_formal_iterations=0,
              production_thresholds=None, calibration_gpu_accepted=False, calibration_complete=False)
p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
p = Path('outputs/gigpo-e1-policy-safety/DESIGN.md')
t = p.read_text().replace('目标Torch2.4 CPU验证推进中','目标Torch2.4实际CPU回归9项已通过（只涉及记录与函数边界，非GPU训练验收）')
p.write_text(t)
report = Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct300/REPORT.md')
t = report.read_text()
addition = '''
### 2026-10-06 三项防护校准记录格式修复

3281失败原件及清理已独立核验：在完整概率校准和第一个Adam更新前，由轨迹行编号的NumPy数组类型触发DataProto契约错误。训练backward、Adam、新增outer均为0；正式保留25/300。限定失败独审SHA `db640db23cd3496eb19eaca17f673655547215a642c9ea83158594ad8a21c368`。

v8只把外部driver中这一字段的dtype由int64改成object，编号整数值、顺序及其他13运行文件保持一致。当地真实DataProto/select_rows及完整one_outer函数CPU回归9项通过，独审SHA `aafa54b745b84523c0d511238c932219e612b4f9246454dffddd48cabd4a9023`。h100-8原Torch2.4.0+cu121环境同9项CPU测试也实际通过，原JSON SHA `08c7336c8f0980b42a22a49690104ac3b5af027a26c5c07779167a24a3b933ba`。GPU未用于这些CPU测试；不能替代下一次实际概率及健康更新校准。原3281、旧v7与健康25产物均保留。v8部署和唯一校准提交继续推进，生产阈值尚未接受。
'''
if '### 2026-10-06 三项防护校准记录格式修复' not in t:
    report.write_text(t+addition)
print('State/design/report updated: actual target CPU9 PASS; formal25/300 new0')
