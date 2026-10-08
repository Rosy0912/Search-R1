import datetime
import hashlib
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
OWNER = ROOT / 'work/tree-fresh200-adam-offload-owner'
NATIVE = ROOT / 'work/tree-fresh200-freeze-02b1-v11/actual-zero-dispatch/terminal-native'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bound(path):
    return {'path': str(path), 'sha256': sha(path)}

review_path = OWNER / 'INDEPENDENT_ADAM_OFFLOAD_REVIEW.json'
owner_path = OWNER / 'ADAM_OFFLOAD_SOURCE_INTERFACE_CPU_RECEIPT.json'
assert sha(review_path) == 'fb05fc7647327987334a32782315f44c0375108779af0cdbc67a24e8797b9996'
assert sha(owner_path) == 'd1b75da26ec2fe608e8be248fa3858f3cc963b263ae722bc27f57ffad5ce5bab'
review = json.loads(review_path.read_text())
assert review['decision'] == 'accepted' and review['blocking_findings'] == []
assert review['code_commit'] == '53082e1667a6ab131a036ca2ef66e004989ed1a9'
assert review['actual_gpu_executed'] is False
counter_path = OWNER / 'BASE_02B1_COUNTEREXAMPLE.json'
counter = json.loads(counter_path.read_text())
assert len(counter['records']) == 2
assert all(x['optimizer_step_calls'] == 0 and x['state_entries_before'] == 0 and x['state_entries_after'] == 2 for x in counter['records'])

samples = [json.loads(x) for x in (NATIVE / 'resource-samples.jsonl').read_text().splitlines()]
procs = [json.loads(x) for x in (NATIVE / 'process-resources.jsonl').read_text().splitlines()]
peaks = {}
for sample in samples:
    for gpu in sample['gpus']:
        uid = gpu['uuid']
        record = peaks.setdefault(uid, {'whole_card_peak_used_bytes_sampled': 0, 'capacity_bytes': gpu['total_memory_bytes'], 'foreign_memory_fraction_max_sampled': 0, 'foreign_sm_upper_max_sampled': 0})
        record['whole_card_peak_used_bytes_sampled'] = max(record['whole_card_peak_used_bytes_sampled'], gpu['total_used_memory_bytes'])
        record['foreign_memory_fraction_max_sampled'] = max(record['foreign_memory_fraction_max_sampled'], gpu['foreign_memory_fraction'])
        record['foreign_sm_upper_max_sampled'] = max(record['foreign_sm_upper_max_sampled'], gpu['foreign_sm_upper'])
        assert record['capacity_bytes'] == gpu['total_memory_bytes']
process_max = {}
exited_without_ticks = 0
for sample in procs:
    for process in sample['processes']:
        if process.get('exited') is True:
            exited_without_ticks += 1
            continue
        pid = str(process['pid'])
        seconds = (process['user_cpu_ticks'] + process['system_cpu_ticks']) / process['hz']
        process_max[pid] = max(process_max.get(pid, 0), seconds)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
facts = {
    'created_utc': now,
    'job_id': 3314,
    'terminal_state': 'FAILED',
    'root_cause_cpu_confirmed': True,
    'diagnosis_scope': 'Actual h100-9 startup traceback plus exact original source CPU AdamW counterexample; source-only repaired behavior accepted independently. New real GPU behavior is still unobserved.',
    'base_counterexample': bound(counter_path),
    'repair_source_commit': review['code_commit'],
    'source_review': bound(review_path),
    'source_owner': bound(owner_path),
    'resource_samples': bound(NATIVE / 'resource-samples.jsonl'),
    'process_samples': bound(NATIVE / 'process-resources.jsonl'),
    'resource_sample_count': len(samples),
    'process_sample_count': len(procs),
    'process_exited_entries_without_final_ticks': exited_without_ticks,
    'whole_card_peaks': peaks,
    'all_sampled_foreign_memory_below_15_percent': all(x['foreign_memory_fraction_max_sampled'] < .15 for x in peaks.values()),
    'all_sampled_foreign_sm_below_15_percent': all(x['foreign_sm_upper_max_sampled'] < 15 for x in peaks.values()),
    'own_sampled_process_cpu_seconds_lower_bound': sum(process_max.values()),
    'cpu_consumption_scope': 'Sum of largest observed cumulative CPU ticks per own PID; sampling ends before all processes terminate. A lower bound, not final consumption; reservation cost is separate.',
    'peak_scope': '47 approximately five-second whole-card NVML samples during failed startup. This is not a real rollout/training peak or evidence of production headroom.',
    'actual_new_training_outer': 0,
    'actual_new_training_adam_writes': 0,
    'actual_native_zero_worker_optimizer_field': None,
    'actual_native_zero_worker_scheduler_field': None,
    'training_release': False,
}
facts_path = NATIVE / 'ADAM_ROOT_CAUSE_RESOURCE_ADDENDUM.json'
with facts_path.open('x') as handle:
    json.dump(facts, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush(); os.fsync(handle.fileno())

status_path = ROOT / 'outputs/STATUS.json'
status = json.loads(status_path.read_text())
assert status['current_job']['job_id'] == 3314 and status['current_job']['state'] == 'FAILED'
fresh = status['fresh200']
fresh['fresh_adam_offload_repair'] = {
    'status': 'MINIMAL_SOURCE_NONE_ACCEPTED_NEW_WRAPPER_HELPER_BINDING_PENDING',
    'source_commit': review['code_commit'],
    'source_directory': str(ROOT / 'work/tree-fresh200-adam-offload-source'),
    'source_owner': bound(owner_path),
    'source_independent_review': bound(review_path),
    'root_cause_and_resources': bound(facts_path),
    'strict_fresh_adam_and_scheduler_inspector_unchanged': True,
    'actual_owner_cpu_tests': 5,
    'actual_independent_cpu_tests': 5,
    'actual_tiny_cpu_adam_step_per_suite': 1,
    'actual_new_3b_adam_writes': 0,
    'new_source_interface_count': 20,
    'historical_5120_report_reusable_by_unchanged_raw_helper_and_tokenizer_sha': True,
    'actual_remote_or_gpu_execution_released': False,
    'training_release': False,
}
status['status'] = 'FRESH200_ADAM_SOURCE_NONE_ACCEPTED_ZERO_V12_BINDING_PENDING'
status['phase'] = 'MINIMAL_ADAM_SOURCE_ACCEPTED_NEW_ZERO_PACKAGE_PREPARATION'
status['current_phase'] = status['phase']
status['updated_at'] = now
status['last_checked_at'] = now
status['pending'] = [
    'Freeze minimally repinned zeroV12/helperV12 against clean53082e source and obtain exact-combination independent review.',
    'Stage code-only same-h1009 package and execute short CPU configuration preparation; reuse exact accepted5120 report bytes with unchanged helper/tokenizer identity.',
    'Independently bind prospective once-only zero dispatch, observe actual new8rank/fullTree/native24/freshAdam0/calibration and cleanup, then independently accept actual evidence.',
    'Continue authorized same-live-instance first5 through200, complete51713/HF/cost/cleanup; count20 processed opportunities and >=1 successful write for credited outer, report actual writes separately.',
]
temp = status_path.with_suffix('.json.adam-sync-tmp')
with temp.open('w') as handle:
    json.dump(status, handle, ensure_ascii=False, indent=2)
    handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(temp, status_path)
with (ROOT / 'outputs/REPORT.md').open('a') as handle:
    handle.write('\n\n### ' + now + ' — Adam startup cause confirmed; minimal source independently accepted\n\n')
    handle.write('Actual3314 remains FAILED and fully cleaned; credited fresh outer0/200 and actual3B Adam writes0. Original native failed-worker optimizer/scheduler fields remain null. Exact old helper CPU AdamW probes reproduced creation of two empty state entries with zero writes. New53082e adds only two membership guards (four lines) and an explanatory worker docstring; strict empty-state/scheduler0 checks remain unchanged. Actual owner5/5 and independent no-context5/5 CPU tests passed; each suite used one tiny CPU AdamW.step, not3B training. Source-only independent review accepted with no blocking findings. ZeroV12/helperV12 repinning and new exact execution review remain pending; no new job submitted and no actual zero/native24/fullTree/production acceptance.\n\n')
    handle.write('47 whole-card resource samples and20 own-process samples are retained. Foreign MEM/SM sample thresholds passed. Peaks are failed-startup samples, not rollout or training headroom. Actual partial CPU consumption is a sampled lower bound; allocated resource-hour costs remain separately reported. Evidence: ' + str(facts_path) + '.\n')
    handle.flush(); os.fsync(handle.fileno())
destinations = [
    pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),
    ROOT / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002',
]
for destination in destinations:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('STATUS.json', 'REPORT.md'):
        shutil.copy2(ROOT / 'outputs' / name, destination / name)
    shutil.copytree(NATIVE, destination / 'tree-fresh200/v11-zero3314-terminal/terminal-native', dirs_exist_ok=True)
    shutil.copytree(OWNER, destination / 'tree-fresh200/adam-offload-source-review', dirs_exist_ok=True)
print(json.dumps({'status': status['status'], 'source_commit': review['code_commit'], 'source_review': sha(review_path), 'resource_samples': len(samples), 'sampled_process_cpu_seconds_lower_bound': facts['own_sampled_process_cpu_seconds_lower_bound'], 'resource_peaks_mib': [x['whole_card_peak_used_bytes_sampled'] / 1048576 for x in peaks.values()], 'new_training_outer': 0, 'new_3b_adam_writes': 0, 'synced_destinations': len(destinations)}))
