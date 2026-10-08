"""Reconcile a saved read-only observation; does not control training."""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs/gigpo-e1-instruct200'


def main():
    path = Path(sys.argv[1]).resolve()
    outer = json.loads(path.read_bytes())
    assert outer['returncode'] == 0
    management = json.loads(outer['stdout'])
    assert management['returncode'] == 0
    actual = json.loads(management['stdout'])
    assert actual['job_id'] == '3340' and actual['hostname'].split('.')[0] == 'h100-8'
    match = re.search(r'\bJobState=(\S+)', actual['scheduler']['stdout'])
    scheduler_state = match.group(1) if actual['scheduler']['returncode'] == 0 and match else 'UNKNOWN'
    metadata = actual['metadata']
    live = metadata.get('live-progress.json', {})
    completed = live.get('completed_outer_updates', 0)
    writes = live.get('actual_adam_total', 0)
    policy = actual.get('actual_partial_policy_last_by_rank', [])
    if len(policy) == 8:
        counts = {row.get('actual_optimizer_writes') for row in policy}
        if len(counts) == 1 and isinstance(next(iter(counts)), int):
            writes = max(writes, next(iter(counts)))
    backend = metadata.get('actual-vllm-backend-identity.json')
    zero = metadata.get('actual-official-zero-initialization.json')
    state_path = BASE / 'work/gigpo-e1-instruct200-state.json'
    state = json.loads(state_path.read_bytes())
    assert state['job_id'] == 3340
    for key in ['run', 'active_run', 'current_run_directory', 'latest_run_absolute_path']:
        state[key] = actual['run']
    for key in ['actual_completed_iterations', 'formal_completed_outer', 'completed_outer_updates',
                'actual_current_completed_iterations', 'actual_completed_outer_updates']:
        state[key] = completed
    for key in ['actual_formal_adam_writes', 'actual_adam_writes', 'actual_adam_total']:
        state[key] = writes
    state['phase'] = 'FORMAL200_XFORMERS_' + scheduler_state + ('_ACTUAL_MODEL_INITIALIZED' if zero else '_STARTUP')
    state['active_training_job_status'] = scheduler_state
    state['last_updated_utc'] = actual['utc']
    state['last_progress'] = dict(job=3340, state=scheduler_state, node='h100-8',
        completed_outer_updates=completed, actual_adam_writes_per_rank=writes,
        run=actual['run'], utc=actual['utc'], original_observation=str(path),
        official_zero_initialization_observed=bool(zero), actual_allrank_backend_observed=bool(backend))
    state['run_progress_fields'] = state['last_progress']
    state['current_job_readonly_helper'] = 'work/read_gigpo_job3340_retained18_actual_template.sh'
    state['xformers_runtime20'].update(status='ACTUAL_' + scheduler_state,
        actual_backend_identity_observation=str(path) if backend else None)
    if backend:
        state['job3340_actual_attention_backend'] = backend
    if zero:
        state['job3340_actual_official_zero_initialization'] = zero
    originals = actual.get('fixed_original_main_worker_logs', [])
    if len(originals) == 1:
        state['job3340_fixed_original_main_worker'] = {k: v for k, v in originals[0].items() if k != 'tail'}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    header = (f'> 实际现场 {actual["utc"]}：XFORMERS正式作业3340在h100-8为{scheduler_state}；'
        f'完成{completed}/200轮，累计实际同步Adam写入{writes}次/rank。'
        f'当前run：{actual["run"] or "尚未建立"}。'
        f'官方零初始化已观察：{bool(zero)}；全部rank后端记录已观察：{bool(backend)}。'
        '跳过训练前基线，无额外测试。\n\n')
    for report in [OUT / 'CONFIG-COMPATIBILITY.md', Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/REPORT.md')]:
        report.write_text(header + report.read_text())
    print(json.dumps(state['last_progress'], ensure_ascii=False))


if __name__ == '__main__':
    main()
