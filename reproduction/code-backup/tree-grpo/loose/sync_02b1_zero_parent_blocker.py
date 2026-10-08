import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
status_path = root / 'outputs/STATUS.json'
report_path = root / 'outputs/REPORT.md'
audit_path = root / 'work/tree-fresh200-02b1-v10-v10-context-free-review/PREPARATION_STATIC_READINESS_AUDIT.json'
audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
if audit_sha != 'fd8e6b7c1a6d99fad8dff036e6e9b112cf89d733a41834264630720cc739388a':
    raise ValueError('Static audit byte identity changed')
data = json.loads(status_path.read_text())
fresh = data['fresh200']
if fresh['new_completed_outer'] != 0 or fresh['new_gpu_job_submitted']:
    raise ValueError('Do not overwrite a later running phase')
now = datetime.now(timezone.utc).isoformat()
data['updated_at'] = now
data['status'] = 'FRESH200_ACTUAL_CPU_PREP_PASSED_5120_RUNNING_ZERO_PARENT_REGISTRATION_FIX'
data['phase'] = 'OFFICIAL5120_CPU_RUNNING_ZERO_V11_PARENT_REGISTRATION_FIX_PENDING'
fresh['zero_parent_registry_static_audit'] = {
    'path': str(audit_path), 'sha256': audit_sha,
    'reviewer': '/root/utf8_v2_exact_none_review', 'fork_turns': 'none',
    'actual_cpu_preparation_accepted': True,
    'current_zero_dispatch_static_accepted': False,
    'blocking_finding': 'V10 calibration supervisor parent process lacks live tokenizer registration before mu validation',
    'isolated_fix': 'V11 pending owner completion and independent review',
    'source_02b1_unchanged': True, 'official5120_running_process_unchanged': True,
    'actual_gpu_or_slurm_execution': False,
}
fresh['dispatch_candidate']['blocker'] = 'Actual official5120 report pending; V10 zero parent tokenizer registration missing, isolated V11 required'
fresh['dispatch_candidate']['actual_remote_executed'] = True
fresh['dispatch_candidate']['actual_remote_execution_scope'] = 'V10 code/metadata staging reconciled, CPU preparation actually passed, single official5120 CPU replay running; no GPU/Slurm'
fresh['source_commit_scope'] = '02b1 source CPU accepted; actual same-node configuration prepared; official tokenizer5120 report pending; no GPU acceptance'
fresh['production_entry']['status'] = 'FORMAL_V5_SOURCE_CPU_ACCEPTED_ONLY_PENDING_ACTUAL_ZERO_CALIBRATION'
fresh['production_entry']['path'] = str(root / 'work/tree-fresh200-train-runtime-v5')
fresh['production_entry']['source_binding_scope'] = '02b1/V5 exact source/CPU combination accepted by fork:none review; no GPU execution or threshold acceptance'
fresh['new_exact_local_packet']['status'] = 'ACTUAL_V10_CODE_ONLY_STAGE_RECONCILED_CPU_PREPARATION_PASSED_OFFICIAL5120_RUNNING'
fresh['new_exact_local_packet']['actual_remote_execution_scope'] = 'Accidental reviewer code-only stage preserved without retrospective approval; remaining CPU prepare/replay prospectively accepted, prepare passed and one replay running'
status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
with report_path.open('a') as stream:
    stream.write('\n\n更新 ' + now + '\n\n')
    stream.write('新正式训练仍为 0/200、实际 Adam 写入 0，没有新评分。用户确认的计数保持：每轮 20 次机会全部处理且至少一次成功写入才计完整 outer；机会、跳过、实际写入分开报告；整轮跳过或中途放弃不计 outer/warmup，已发生的部分写入保留真实记录。\n\n')
    stream.write('独立无主上下文审查确认实际 CPU 准备报告及 89 个配置字段通过；同时发现 V10 零更新校准父进程未登记 tokenizer，µ 验证必然拒绝。因此 V10 当前不能派发 GPU，修复隔离到 V11；冻结源 02b1 和正在运行的唯一 5120 条完整 CPU 重验不改动、不重启。正式 V5 的源码/CPU 独审接受不代表实际零更新校准、阈值或训练通过。审计 SHA：' + audit_sha + '。\n')
destinations = [
    Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),
    root / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002',
]
for destination in destinations:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(status_path, destination / 'STATUS.json')
    shutil.copy2(report_path, destination / 'REPORT.md')
    audit_dest = destination / 'tree-fresh200/zero-parent-registry-audit'
    audit_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_path, audit_dest / audit_path.name)
print(json.dumps({'updated_at': now, 'status': data['status'], 'new_outer': 0, 'actual_adam_writes': 0, 'static_audit_sha': audit_sha}, ensure_ascii=False))
