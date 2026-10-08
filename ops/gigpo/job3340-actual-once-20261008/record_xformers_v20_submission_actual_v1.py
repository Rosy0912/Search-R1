"""Record an already consumed real submission; never starts or stops a job."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs/gigpo-e1-instruct200'


def main():
    transport_path = OUT / 'FRESH200-XFORMERS-V20-REPAIR3338-SUBMIT-TRANSPORT-v1.json'
    transport = json.loads(transport_path.read_bytes())
    assert transport['returncode'] == 0
    receipt = json.loads(transport['stdout'])
    assert receipt['status'] == 'SUBMITTED_EXACTLY_ONCE'
    job = int(receipt['job_id'])
    binding = receipt['binding']
    assert binding['sha256'] == '5f65aa458f3607f546608f38368da48321874f33b526fcc15e3764271d48199c'
    publication_path = OUT / 'FRESH200-XFORMERS-V20-PUBLICATION-v1.json'
    publication = json.loads(publication_path.read_bytes())
    assert publication['status'] == 'PUSHED_AND_INDEPENDENT_GIT_DOWNLOAD_VERIFIED'
    assert publication['new_commit'] == publication['remote_sha']
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state_path = BASE / 'work/gigpo-e1-instruct200-state.json'
    state = json.loads(state_path.read_bytes())
    assert state['historical_human_stopped_job3338']['actual_completed_outer_updates'] == 1
    for key in ['active_training_job', 'active_training_job_id', 'active_job_id', 'job_id', 'latest_job']:
        state[key] = job
    for key in ['actual_completed_iterations', 'formal_completed_outer', 'completed_outer_updates',
                'actual_current_completed_iterations', 'actual_completed_outer_updates',
                'actual_formal_adam_writes', 'actual_adam_writes', 'actual_adam_total',
                'actual_partial_current_attempt_adam_writes_per_rank']:
        state[key] = 0
    for key in ['run', 'active_run', 'current_run_directory', 'latest_run_absolute_path',
                'latest_complete_checkpoint', 'actual_job_start_utc']:
        state[key] = None
    for key in ['current_binding_path', 'binding', 'latest_binding']:
        state[key] = binding['path']
    for key in ['current_binding_sha256', 'binding_sha256', 'latest_binding_sha256']:
        state[key] = binding['sha256']
    state['current_binding'] = binding
    state['phase'] = 'FORMAL200_XFORMERS_SUBMITTED_ACTUAL_STARTUP_NOT_YET_OBSERVED'
    state['active_training_job_status'] = 'SUBMITTED_STARTUP_NOT_YET_OBSERVED'
    state['current_operation'] = dict(status='SUBMITTED_EXACTLY_ONCE', job_id=job,
        adapter='work/deploy_submit_xformers3338_v20_once_v1.py',
        local_intent_consumed=True, remote_intent_consumed=True, replay_allowed=False)
    state['last_progress'] = dict(job=job, state='SUBMITTED_STARTUP_NOT_YET_OBSERVED',
        node='h100-8', completed_outer_updates=0, actual_adam_writes_per_rank=0,
        run=None, utc=utc, gpu_training_observed=False)
    state['run_progress_fields'] = state['last_progress']
    state['xformers_runtime20'].update(status='SUBMITTED_EXACTLY_ONCE', job_id=job,
        publication_commit=publication['new_commit'], publication_record=str(publication_path))
    state['last_updated_utc'] = utc
    state['original_driver_pid'] = None
    state['original_worker_pids'] = []
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    header = (f'> 当前提交 {utc}：XFORMERS正式作业{job}已真实仅提交一次，'
        '启动和GPU训练状态尚待现场确认；0/200、实际Adam0，run尚未观察。'
        '跳过训练前基线、无额外测试。3338按人工要求停止，历史1轮/20次写入保留，'
        '不拼入本次官方fresh200。科学源码ebe91不变；XFORMERS须按实际全部8rank读回。\n\n')
    for path in [OUT / 'CONFIG-COMPATIBILITY.md', Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/REPORT.md')]:
        path.write_text(header + path.read_text())
    actual = dict(utc=utc, status='SUBMITTED_EXACTLY_ONCE', job_id=job, receipt=receipt,
        source_commit='ebe91bfebd56c3bd445219dcd94342d815f899f0',
        publication_commit=publication['new_commit'], observed_gpu_training=False,
        transport=dict(path=str(transport_path), bytes=transport_path.stat().st_size,
            sha256=hashlib.sha256(transport_path.read_bytes()).hexdigest()))
    actual_path = OUT / f'JOB{job}-XFORMERS-ACTUAL-SUBMISSION-v1.json'
    with actual_path.open('x') as f:
        json.dump(actual, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(job=job, binding=binding, publication=publication['new_commit'], actual_record=str(actual_path))))


if __name__ == '__main__':
    main()
