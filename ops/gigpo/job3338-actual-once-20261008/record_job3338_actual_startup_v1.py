# -*- coding: utf-8 -*-
from pathlib import Path
import json, hashlib

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'outputs/gigpo-e1-instruct200'
p=OUT/'RETAINED18-JOB3338-ACTUAL-STATUS-v1-actual-allocation.json'
o=json.loads(json.loads(json.loads(p.read_text())['stdout'])['stdout'])
now=o['utc']
s_path=BASE/'work/gigpo-e1-instruct200-state.json'
s=json.loads(s_path.read_text())
s.update(current_actual_state='RUNNING', active_training_job_status='RUNNING',
    phase='FORMAL200_JOB3338_RUNNING_STARTUP', status='FORMAL_RUNNING_STARTUP',
    latest_actual_observation_record=str(p), latest_actual_observation_path=str(p),
    latest_actual_observation_utc=now, latest_actual_observation_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    scheduler_latest=o['scheduler'], updated_utc=now)
s['last_progress']=dict(job=3338,state='RUNNING',node='h100-8',completed_outer_updates=0,
    actual_adam_writes_per_rank=0,startup=True,run_not_yet_created_at_observation=True,utc=now)
s_path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
RESULTS=Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200')
prefix=f'> 当前现场 {now}：新作业3338实际RUNNING于h100-8，8卡独占，正在启动，此快照run尚未建立；0/200、Adam0。基线跳过，额外测试0，本次静态审查按最新人工指令豁免。旧3337失败2轮/40写入保留，不拼入新轮数。v19同步CUDA执行是定位手段，尚未证明修复。代码已推送b83d46fea143670c6729726af6ade795c9cc0f80并独立Git读回57文件SHA一致；正式只提交一次。\n\n'
for doc in [RESULTS/'REPORT.md', OUT/'CONFIG-COMPATIBILITY.md']:
    doc.write_text(prefix+doc.read_text())
f=RESULTS/'FAILURES-AND-FIXES-20261008.md'
t=f.read_text()
t=t.replace('**最新状态：作业3337完成2轮训练、累计40次实际成功参数更新后，在第3轮生成响应时发生GPU非法内存访问并退出。当前没有训练在运行，也没有启动后继作业。**',
    '**当前新作业3338已在武汉H100集群h100-8启动（'+now+'），正在加载，0/200轮；基线跳过，本次独立静态审查按用户最新指令豁免。**\n\n上一作业3337完成2轮训练、累计40次实际成功参数更新后，在第3轮生成响应时发生GPU非法内存访问并退出。')
t=t.replace('**尚未解决，后继尚未提交。**','**根因尚未定位；后继3338已启动，用同步执行定位真实错误。**')
t=t.replace('该方案尚在准备，尚未执行；不能称为已确认修复。','该入口已由正式作业3338执行，全部工作进程的开关需实际日志核对；不能称为已确认修复。')
t=t.replace('失败运行使用runtime18，新增同步执行入口尚未执行。','失败运行3337使用runtime18；新作业3338已使用runtime19同步执行入口，科学源码保持不变。')
f.write_text(t)
print(json.dumps(dict(job=3338,actual_state='RUNNING',observation_utc=now)))
