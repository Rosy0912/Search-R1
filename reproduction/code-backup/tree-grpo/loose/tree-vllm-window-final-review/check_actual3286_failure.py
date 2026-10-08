"""Independent saved-source/metadata review; no remote, model, or tensor assets."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[1]
OUT = WORK.parent / 'outputs/tree-instruct-300'
REVIEW = Path(__file__).resolve().parent
PACKAGE = WORK / 'single-tp1-window-ca1f2af0-466df7c2'
def sha(data): return hashlib.sha256(data).hexdigest()
snapshot_path = OUT / 'PPO_WINDOW_GPU3286_METADATA_20261006T011001847867Z.json'
snapshot_bytes = snapshot_path.read_bytes()
assert sha(snapshot_bytes) == 'd32c8987c89d9b8b1ee938518da975a4a6f72d1b6d765213546c68a3732b960c'
snapshot = json.loads(snapshot_bytes)
raw_path = OUT / 'PPO_WINDOW_GPU3286_FAILURE_RAW.json'
raw_bytes = raw_path.read_bytes()
assert sha(raw_bytes) == 'bf777b74fd1d491635241552554cd40c1cad401b30d5c4461d79fd44374e1edc'
raw = json.loads(raw_bytes)
assert raw['job_id'] == snapshot['job_id'] == '3286'
assert raw['host'] == snapshot['host'] == 'h100-9'
assert raw['mutation'] is snapshot['mutation'] is False and raw['tensor_assets_read'] is False
records = {Path(row['path']).name: row for row in raw['source_records']}
assert len(records) == len(raw['source_records']) == 11
for row in records.values():
    data = row['text'].encode()
    assert len(data) == row['bytes'] and sha(data) == row['sha256']
probe = records['probe.log']['text']
pattern = r'\[rank(\d+)\]: .*Global diagnostic refusal at actual native engine settings/source metadata; local evidence: "AttributeError: \'SchedulerConfig\' object has no attribute \'enable_chunked_prefill\'"'
ranks = sorted(set(map(int, re.findall(pattern, probe))))
assert ranks == list(range(8))
assert 'rank      : 7 (local_rank: 7)' in probe and 'exitcode  : 1' in probe
assert snapshot['ready'] == []
result = json.loads(records['result.json']['text'])
assert result == snapshot['result.json']['record']
assert result['status'] == 'FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'
assert result['package_manifest_sha256'] == '36bfac94a3bc6fa5ed344d21d0110ab08a01cc098adebbccc42c538147a1f66c'
assert result['runner_commit'] == 'ca1f2af03c501862885cdfb55a2b79f6f4fae55f'
assert result['release_sha256'] == 'a405e16fb79209e960860b178479468a771ee8ddee54db537f0868fe74e5ba05'
for field in ('actual_optimizer_writes', 'actual_generation_invocations', 'actual_actor_scoring_native_forwards', 'actual_new_emitted_tokens'):
    assert result[field] is None
assert result['cleanup_residual'] == {} and result['original_background_pid_start_ticks_preserved'] is True
protected = {1680782:1502352181, 1680783:1502352181, 1680784:1502352181, 1680785:1502352181,
             1696443:1502398762, 1696444:1502398762, 1696445:1502398762, 1696446:1502398762}
initial = json.loads(records['initial-allocation.json']['text'])
assert {int(pid): tick for pid, tick in initial['background_pid_start_ticks'].items()} == protected
assert {int(line.split(',')[1]) for line in result['final_gpu']['processes']} == set(protected)
assert result['allocated_cpus'] == 192 and result['requested_cpus'] == 32 and result['allocated_gpus'] == 8
schedule = snapshot['scheduler']['stdout']
assert snapshot['scheduler']['returncode'] == 0
for literal in ('JobId=3286', 'JobState=FAILED', 'ExitCode=1:0', 'RunTime=00:03:35', 'NodeList=h100-9',
                'NumCPUs=192', 'CPUs/Task=32', 'Requeue=0'):
    assert literal in schedule
assert 'StartTime=2026-10-06T09:05:19 EndTime=2026-10-06T09:08:54' in schedule
terminal = [json.loads(line) for line in records['probe-scheduler-observation-3286.jsonl']['text'].splitlines() if line.strip()]
assert terminal[-1]['terminal_state_observed'] is True and terminal[-1]['scheduler_state'] == 'FAILED'
assert terminal[-1]['stdout'] == schedule
assert abs(result['wall_seconds'] - (result['end_epoch'] - result['start_epoch'])) < 1e-6
assert abs(result['gpu_hours'] - 8 * result['wall_seconds'] / 3600) < 1e-5
assert abs(result['allocated_cpu_hours'] - 192 * result['wall_seconds'] / 3600) < 1e-5
guard = [json.loads(line) for line in records['resources.jsonl']['text'].splitlines() if line.strip()]
assert len(guard) == snapshot['resources']['samples'] == 42
uuids = set(snapshot['launch.json']['record']['gpu_uuids'])
for row in guard:
    assert row['job_id'] == '3286' and not row.get('event')
    assert len(row['gpus']) == 8 and {g['uuid'] for g in row['gpus']} == uuids
    assert all(0 <= g['foreign_memory_fraction'] <= .15 and 0 <= g['foreign_sm_upper'] <= 15 for g in row['gpus'])
resources = [json.loads(line) for line in records['process-resources.jsonl']['text'].splitlines() if line.strip()]
maxima = {}
for row in resources:
    assert row['adapter']['job_id'] == '3286' and row['adapter']['original_background_processes_preserved'] is True
    for proc in row['processes']:
        observed = (proc['user_cpu_ticks'] + proc['system_cpu_ticks']) / proc['hz']
        maxima[proc['pid']] = max(maxima.get(proc['pid'], 0), observed)

bundle = json.loads(records['bundle.json']['text'])
config_path = WORK / 'tree-vllm-installed-source-extra-20261006/config.py'
config_bytes = config_path.read_bytes()
assert sha(config_bytes) == bundle['vllm_installed_sources']['config.py']['sha256'] == '3a9d50709aa2f7ca3b14741b507232e2cd954936780d4a0365cabea12e6cc8b0'
config_tree = ast.parse(config_bytes)
scheduler = next(n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SchedulerConfig')
constructor = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
assert 'enable_chunked_prefill' in {arg.arg for arg in constructor.args.args}
assignments = [n for n in ast.walk(constructor) if isinstance(n, ast.Assign)]
actual = [n for n in assignments if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self' and t.attr == 'chunked_prefill_enabled' for t in n.targets)]
assert len(actual) == 1 and isinstance(actual[0].value, ast.Name) and actual[0].value.id == 'enable_chunked_prefill'
assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self' and n.attr == 'enable_chunked_prefill' for n in ast.walk(scheduler))
runtime_path = PACKAGE / 'repo/single_window_runner.py'
runtime_bytes = runtime_path.read_bytes()
runtime_tree = ast.parse(runtime_bytes)
metadata = next(n for n in runtime_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'actual_engine_metadata')
assert any(isinstance(n, ast.Constant) and n.value == 'enable_chunked_prefill' for n in ast.walk(metadata))
runner_path = PACKAGE / 'repo/single_window_runner.py'
runner = runner_path.read_text()
stages = ["'source-only warmup first original adjacent4'", "'warmup actual FSDP native/effectiveweights'", "rollout,manager = build_rollout(self)", "runtime = coordinator.local_stage('actual native engine settings/source metadata'", "with manager:", 'rollout.generate_sequences(']
positions = [runner.index(stage) for stage in stages]
assert positions == sorted(positions)

receipt = dict(
    status='ACTUAL3286_NATIVE_ENGINE_METADATA_REFUSAL_TERMINAL_COST_AND_CLEANUP_INDEPENDENTLY_VERIFIED',
    reviewer_task='/root/tree_window_final_review', reviewed_at_utc=datetime.now(timezone.utc).isoformat(), job_id='3286',
    actual_failure_execution_verified=True, actual_full_window_measurement_verified=False,
    actual_terminal_verified=True, slurm_state='FAILED', exit_code='1:0', slurm_runtime_seconds=215,
    exact_failure="AttributeError: 'SchedulerConfig' object has no attribute 'enable_chunked_prefill'",
    explicit_same_refusal_rank_ids=ranks, actual_ready_count=0,
    actual_optimizer_writes=None, actual_generation_invocations=None, actual_actor_scoring_native_forwards=None,
    actual_new_emitted_tokens=None,
    counts_scope='Original failure result nulls retained; no successful all-rank measurement counters credited',
    reached_stage_scope='All-rank refusal at metadata stage occurs after original native constructor and warmup returned by bound control flow; failure precedes manager sync and new generation. No complete per-rank ready receipt exists.',
    diagnosis=dict(constructor_parameter='enable_chunked_prefill', actual_instance_field='chunked_prefill_enabled',
                   exact_installed_source_AST_mapping_verified=True, observation_requested_nonexistent_field=True,
                   previous_source_review_missed_actual_metadata_API_attribute_mapping=True),
    package_manifest_sha256=result['package_manifest_sha256'], runner_commit=result['runner_commit'],
    original_package_or_intent_retry_accepted=False, old_package_changed=False,
    cleanup_residual={}, original8_background_pid_start_ticks_preserved=True,
    guard_samples=len(guard), max_foreign_memory_fraction=snapshot['resources']['max_foreign_memory_fraction'],
    max_foreign_sm_upper=snapshot['resources']['max_foreign_sm_upper'],
    whole_card_5s_peak_used_bytes=snapshot['resources']['whole_card_5s_peak_used_bytes'],
    costs=dict(scheduler_allocated_GPU_hours=8*215/3600, scheduler_allocated_CPU_hours=192*215/3600,
               supervisor_wall_seconds=result['wall_seconds'], supervisor_GPU_hours=result['gpu_hours'],
               supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],
               sampled_process_CPU_maxima_sum_seconds=sum(maxima.values()), sampled_process_CPU_samples=len(resources),
               complete_consumed_CPU_hours=None, sampled_CPU_scope='PID maxima from process samples only; no complete exit totals or launcher/preflight/descendant accounting'),
    evidence=dict(snapshot_file=dict(path=str(snapshot_path.resolve()), sha256=sha(snapshot_bytes)),
                  full_raw_file=dict(path=str(raw_path.resolve()), sha256=sha(raw_bytes)),
                  exact_installed_config_source=dict(path=str(config_path.resolve()), sha256=sha(config_bytes)),
                  frozen_runtime_source=dict(path=str(runtime_path.resolve()), sha256=sha(runtime_bytes)),
                  original_probe_log_sha256=records['probe.log']['sha256'], original_probe_log_bytes=records['probe.log']['bytes'],
                  snapshot_later_probe_bytes=snapshot['probe.log']['bytes'],
                  probe_capture_scope='Raw captured 37706 bytes; later snapshot observed 38066 bytes. Identities not conflated; raw includes all eight refusal traces and terminal child failure.',
                  source_records=[{key: row[key] for key in ('path', 'bytes', 'sha256')} for row in records.values()]),
    reviewer_remote_access=False, reviewer_tensor_asset_read=False, reviewer_GPU_or_model_execution=False,
    probability_accepted=False, training_accepted=False, GPU_capacity_proven=False,
    actual_loaded_binary_identity_proven=False, threshold_calibration_accepted=False,
    scope='Failure evidence only. Constructor/warmup resource peak cannot establish generation or training capacity. Exact passive metadata repair needs separate source delta review and new fixed dispatch intent.')
target = REVIEW / 'ACTUAL3286_FAILURE_RECEIPT.json'
assert not target.exists()
target.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
print(json.dumps(dict(receipt_path=str(target), receipt_sha256=sha(target.read_bytes()), ranks=ranks, costs=receipt['costs']), sort_keys=True))
