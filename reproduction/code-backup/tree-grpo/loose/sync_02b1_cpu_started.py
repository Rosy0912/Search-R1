import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

out = Path('work/tree-fresh200-freeze-02b1-v10/actual-cpu')
fetched = json.loads((out / 'PREPARATION_METADATA_FETCH.stdout').read_text())
assert not (out / 'PREPARATION_METADATA_FETCH.stderr').read_bytes()
for name, record in fetched.items():
    raw = record['text'].encode()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    with (out / name).open('xb') as stream:
        stream.write(raw)
report = json.loads((out / 'PREPARATION.json').read_text())
assert fetched['PREPARATION.json']['sha256'] == '03e273373cb2abaa4f8eb724632a5184d1d4d92c76452451c64703611b7e2bd4'
assert report['code_commit'] == '02b1ea2688ff3fb032ee6699003a7e09d651f166'
assert not report['submitted'] and not report['cuda_initialized'] and report['optimizer_writes'] == 0
assert report['resolved_config_sha256'] == fetched['resolved.yaml']['sha256']
assert report['budget']['optimizer_opportunities_per_rank_per_complete_outer'] == 20
assert report['budget']['minimum_successful_optimizer_writes_per_complete_outer'] == 1
status_path = Path('outputs/STATUS.json')
status = json.loads(status_path.read_text())
now = datetime.now(timezone.utc).isoformat()
status['updated_at'] = now
status['status'] = 'FRESH200_02B1_V10_ACTUAL_CPU_PREP_PASSED_OFFICIAL5120_RUNNING_NO_GPU'
status['phase'] = 'OFFICIAL_TOKENIZER_HISTORICAL5120_CPU_REPLAY_RUNNING_FORMAL_V4_REVIEW'
fresh = status['fresh200']
fresh['new02b1_actual_cpu_preparation'] = {
    'status': report['status'], 'source_commit': report['code_commit'],
    'report_sha256': fetched['PREPARATION.json']['sha256'],
    'resolved_config_sha256': report['resolved_config_sha256'],
    'checked_resolved_values_count': len(report['checked_resolved_values']),
    'budget': report['budget'], 'model_identity': report['model_identity']['sha256'],
    'data_identity': report['common_data_identity'], 'actual_hydra_resolved': True,
    'cuda_initialized': False, 'optimizer_writes': 0, 'submitted': False,
    'actual_stage_retried': False,
}
fresh['new02b1_actual_tokenizer_cpu'] = {
    'status': 'ONE_ACTUAL_EXACT_CPU_REPLAY_RUNNING_UNFILTERED',
    'intent': json.loads((out / 'TOKENIZER_ACTUAL_INTENT.json').read_text()),
    'exec_session_id': 27284, 'expected_rows': 5120, 'completed_rows_observed': None,
    'scope': 'Official tokenizer historical rows only; not probability/native24/freshTree/3Bupdate acceptance',
}
fresh['new_exact_local_packet']['execution_binding_review'] = {
    'status': 'EXACT_02B1_V10_REMAINING_CPU_PREPARATION_AND_TOKENIZER_REPLAY_ACCEPTED',
    'sha256': 'aaa20df4a5acea08834053be00ec18c19bd57e7e64ba249f82c08b75ff1479cf',
    'scope': 'Remaining CPU commands prospective only; staged artifact verified; priorstageapprovalnotretrospective',
}
fresh['production_entry']['v4_owner_inventory_sha256'] = 'ba118bdd4af566205c7c776ce8d76e8d6d5bd16e7c15cb135ad086cfd7be4408'
fresh['production_entry']['v4_owner_cpu_receipt_sha256'] = '2e9372462bf4bbc8024a381a592ac0a894d82a867a94183f6dfdb10c760dfec6'
fresh['production_entry']['v4_none_task'] = '/root/formal_v4_02b1_exact_none_review'
fresh['production_entry']['v4_independent_acceptance_pending'] = True
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
with Path('outputs/REPORT.md').open('a') as stream:
    stream.write('\n\n### ' + now + '：新02b1/V10实际本机CPU准备通过，5120全量重验运行中\n\n')
    stream.write('经独立执行前绑定接受后，未重stage，prepare实际返回0、20.34秒、stderr为空。')
    stream.write('本机模型身份744d9…与共同数据0b002…核对通过；Hydra解析' + str(len(report['checked_resolved_values'])) + '字段检查。')
    stream.write('预算明确20机会/rank/完整outer且至少1成功写入，实际写入单报；CUDA未初始化，Adam0，未提交Slurm。\n\n')
    stream.write('官方tokenizer＋历史3295/3296共5120条本版本CPU重验已单实例启动、不过滤，最终计数尚无产物；')
    stream.write('不把历史best_of/_real_n None或CPU解析当native概率门禁、真实新训练通过。新协议仍0/200、实际Adam0、暂无新评分。')
    stream.write('正式wrapper独立审查发现启动bootstrap/module CLI防护边界待最小修复，正式GPU许可保持false。\n\n')
    stream.write('route隔离事故的完整stage10文件已只读核对，首次返回日志丢失、第二次no-clobber拒绝永久保留；不追溯第一次执行前已获得批准。\n')
for destination in [
    Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),
    Path('work/tree-grpo/reproduction/baselines/tree-validation-20261002'),
]:
    for name in ['STATUS.json', 'REPORT.md']:
        shutil.copy2(Path('outputs') / name, destination / name)
    metadata = destination / 'tree-fresh200'
    metadata.mkdir(exist_ok=True)
    for name in ['PREPARATION.json', 'resolved.yaml', 'PREPARE_ACTUAL_RESULT.json', 'PREPARE_ACTUAL_INTENT.json', 'TOKENIZER_ACTUAL_INTENT.json']:
        shutil.copy2(out / name, metadata / ('02B1_V10_' + name))
    for name in ['CPU_EXECUTION_BINDING_REVIEW.json', 'CPU_ROUTE_SCOPE_INCIDENT_RECEIPT.json']:
        shutil.copy2(Path('work/tree-fresh200-02b1-v10-v10-context-free-review') / name, metadata / name)
print(json.dumps({'updated_at': now, 'cpu_preparation_passed': True, 'config_fields': len(report['checked_resolved_values']), 'tokenizer_running': True, 'fresh_outer': 0, 'actual_adam_writes': 0, 'synced': 2}))
