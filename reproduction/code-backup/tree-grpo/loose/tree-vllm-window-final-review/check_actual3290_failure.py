"""Independent failure review from captured JSON/text only; no remote/assets/GPU."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

WORK=Path(__file__).resolve().parents[1]
OUT=WORK.parent/'outputs/tree-instruct-300'
REVIEW=Path(__file__).resolve().parent
PACKAGE=WORK/'single-tp1-window-e29b27d1-466df7c2'
def sha(data):return hashlib.sha256(data).hexdigest()
snapshot_path=OUT/'PPO_WINDOW_GPU3290_METADATA_20261006T013356606094Z.json'
snapshot_bytes=snapshot_path.read_bytes()
assert sha(snapshot_bytes)=='51822fd3f97d00ed15df2b88571b59eb99f1b6dea1da97e0e7d8f72507eb566b'
snapshot=json.loads(snapshot_bytes)
raw_path=OUT/'PPO_WINDOW_GPU3290_FAILURE_RAW.json'
raw_bytes=raw_path.read_bytes()
assert sha(raw_bytes)=='3a804473c1a6a4aaa879d0a3915b54e76fe980cfe5da661b3db2998d588ccf81'
raw=json.loads(raw_bytes)
assert raw['job_id']==snapshot['job_id']=='3290' and raw['host']==snapshot['host']=='h100-9'
assert raw['mutation'] is snapshot['mutation'] is False and raw['tensor_assets_read'] is False
records={Path(row['path']).name:row for row in raw['source_records']}
assert len(records)==len(raw['source_records'])==11
for row in records.values():
    data=row['text'].encode()
    assert len(data)==row['bytes'] and sha(data)==row['sha256']
probe=records['probe.log']['text']
pattern=r"\[rank(\d+)\]: .*Global diagnostic refusal at passive native hook source binding; local evidence: 'ValueError: Actual native method source path differs from bound source'"
ranks=sorted(set(map(int,re.findall(pattern,probe))))
assert ranks==list(range(8))
assert 'rank      : 1 (local_rank: 1)' in probe and 'exitcode  : 1' in probe
assert snapshot['ready']==[]
assert snapshot['probe.log']['bytes']==records['probe.log']['bytes']==37614
result=json.loads(records['result.json']['text'])
assert result==snapshot['result.json']['record']
assert result['status']=='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'
assert result['package_manifest_sha256']=='d55e3836496fd6439087ee17611f0b65eed18bb07fc722b8f5bf9c087e372084'
assert result['runner_commit']=='e29b27d1b8b2b869349fc1c6366d81b89ae62fe0'
assert result['release_sha256']=='4a89af2629a07e4f4c5ef4379ac82b35609f3c3e93457a875aa1671a0e8480cd'
for field in ('actual_optimizer_writes','actual_generation_invocations','actual_actor_scoring_native_forwards','actual_new_emitted_tokens'):
    assert result[field] is None
assert result['cleanup_residual']=={} and result['original_background_pid_start_ticks_preserved'] is True
protected={1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
           1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
initial=json.loads(records['initial-allocation.json']['text'])
assert {int(pid):tick for pid,tick in initial['background_pid_start_ticks'].items()}==protected
assert {int(line.split(',')[1]) for line in result['final_gpu']['processes']}==set(protected)
assert result['allocated_cpus']==192 and result['requested_cpus']==32 and result['allocated_gpus']==8
schedule=snapshot['scheduler']['stdout']
assert snapshot['scheduler']['returncode']==0
for literal in ('JobId=3290','JobState=FAILED','ExitCode=1:0','RunTime=00:03:39','NodeList=h100-9','NumCPUs=192','CPUs/Task=32','Requeue=0'):
    assert literal in schedule
assert 'StartTime=2026-10-06T09:29:34 EndTime=2026-10-06T09:33:13' in schedule
terminal=[json.loads(line) for line in records['probe-scheduler-observation-3290.jsonl']['text'].splitlines() if line.strip()]
assert terminal[-1]['terminal_state_observed'] is True and terminal[-1]['scheduler_state']=='FAILED'
assert terminal[-1]['stdout']==schedule
assert abs(result['wall_seconds']-(result['end_epoch']-result['start_epoch']))<1e-6
assert abs(result['gpu_hours']-8*result['wall_seconds']/3600)<1e-5
assert abs(result['allocated_cpu_hours']-192*result['wall_seconds']/3600)<1e-5
guard=[json.loads(line) for line in records['resources.jsonl']['text'].splitlines() if line.strip()]
assert len(guard)==snapshot['resources']['samples']==43
uuids=set(snapshot['launch.json']['record']['gpu_uuids'])
for row in guard:
    assert row['job_id']=='3290' and not row.get('event')
    assert len(row['gpus'])==8 and {g['uuid'] for g in row['gpus']}==uuids
    assert all(0<=g['foreign_memory_fraction']<=.15 and 0<=g['foreign_sm_upper']<=15 for g in row['gpus'])
resources=[json.loads(line) for line in records['process-resources.jsonl']['text'].splitlines() if line.strip()]
maxima={}
for row in resources:
    assert row['adapter']['job_id']=='3290' and row['adapter']['original_background_processes_preserved'] is True
    for proc in row['processes']:
        observed=(proc['user_cpu_ticks']+proc['system_cpu_ticks'])/proc['hz']
        maxima[proc['pid']]=max(maxima.get(proc['pid'],0),observed)
runner_path=PACKAGE/'repo/single_window_runner.py'
runner_bytes=runner_path.read_bytes();runner=runner_bytes.decode()
stage_positions=[runner.index(text) for text in ["'actual native engine settings/source metadata'",
    "'actual native active method source chain'","'passive native hook source binding'",'with manager:',
    'rollout.generate_sequences(']]
assert stage_positions==sorted(stage_positions)
runtime_path=PACKAGE/'repo/window_runtime.py';runtime_bytes=runtime_path.read_bytes()
tree=ast.parse(runtime_bytes)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeWindowWrappers')
enter=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__enter__')
first=enter.body[0]
assert isinstance(first,ast.Assign) and isinstance(first.value,ast.Dict)
assert first.targets[0].attr=='method_receipts'
assert {k.value for k in first.value.keys}=={'register','execute','output','sync'}
assert len([n for n in ast.walk(first) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='bind_method'])==4
assert all(not(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='patch') for n in ast.walk(first))
bundle=json.loads(records['bundle.json']['text'])
installed_runner_path=WORK/'tree-vllm-installed-source-20261006/worker/model_runner.py'
installed_runner_bytes=installed_runner_path.read_bytes()
assert sha(installed_runner_bytes)==bundle['vllm_installed_sources']['worker/model_runner.py']['sha256']
native_tree=ast.parse(installed_runner_bytes)
native_cls=next(n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
execute=next(n for n in native_cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
decorators=[ast.unparse(n) for n in execute.decorator_list]
assert decorators==["torch.inference_mode()", "dump_input_when_exception(exclude_args=[0], exclude_kwargs=['self'])"]
assert any(isinstance(n,ast.ImportFrom) and n.module=='vllm.worker.model_runner_base' and
           any(alias.name=='dump_input_when_exception' for alias in n.names) for n in native_tree.body)

receipt=dict(status='ACTUAL3290_PASSIVE_NATIVE_SOURCE_BINDING_REFUSAL_TERMINAL_COST_AND_CLEANUP_INDEPENDENTLY_VERIFIED',
    reviewer_task='/root/tree_window_final_review',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),job_id='3290',
    actual_failure_execution_verified=True,actual_full_window_measurement_verified=False,actual_terminal_verified=True,
    slurm_state='FAILED',exit_code='1:0',slurm_runtime_seconds=219,
    exact_failure='ValueError: Actual native method source path differs from bound source',
    explicit_same_refusal_rank_ids=ranks,actual_ready_count=0,actual_optimizer_writes=None,
    actual_generation_invocations=None,actual_actor_scoring_native_forwards=None,actual_new_emitted_tokens=None,
    counts_scope='Original failure result nulls retained; no successful all-rank measurement counters credited',
    reached_stage_scope='Bound control flow places failure after actual engine metadata and six active method bindings, before any four facade/runner hooks patched, manager sync or new generation. Original native constructor/warmup had returned; no full ready receipt.',
    diagnostic_scope='Actual generic binding error is verified. Specific failed method/extra wrapper path is not logged here and requires captured chain evidence; do not promote likely decorator cause to actual proof.',
    source_clue=dict(actual_bound_execute_model_decorators=decorators,
                     dump_decorator_import='vllm.worker.model_runner_base.dump_input_when_exception',
                     source_clue_not_specific_runtime_failure_identity=True,
                     previous_source_review_did_not_fully_verify_actual_wrapper_chain=True),
    package_manifest_sha256=result['package_manifest_sha256'],runner_commit=result['runner_commit'],
    original_package_or_intent_retry_accepted=False,old_package_changed=False,
    cleanup_residual={},original8_background_pid_start_ticks_preserved=True,
    guard_samples=len(guard),max_foreign_memory_fraction=snapshot['resources']['max_foreign_memory_fraction'],
    max_foreign_sm_upper=snapshot['resources']['max_foreign_sm_upper'],
    whole_card_5s_peak_used_bytes=snapshot['resources']['whole_card_5s_peak_used_bytes'],
    costs=dict(scheduler_allocated_GPU_hours=8*219/3600,scheduler_allocated_CPU_hours=192*219/3600,
               supervisor_wall_seconds=result['wall_seconds'],supervisor_GPU_hours=result['gpu_hours'],
               supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],
               sampled_process_CPU_maxima_sum_seconds=sum(maxima.values()),sampled_process_CPU_samples=len(resources),
               complete_consumed_CPU_hours=None,sampled_CPU_scope='PID maxima from198 process samples only; no complete exit totals or launcher/preflight/descendant accounting'),
    evidence=dict(snapshot_file=dict(path=str(snapshot_path.resolve()),sha256=sha(snapshot_bytes)),
                  full_raw_file=dict(path=str(raw_path.resolve()),sha256=sha(raw_bytes)),
                  exact_installed_model_runner_source=dict(path=str(installed_runner_path.resolve()),sha256=sha(installed_runner_bytes)),
                  frozen_runner_source=dict(path=str(runner_path.resolve()),sha256=sha(runner_bytes)),
                  frozen_wrapper_source=dict(path=str(runtime_path.resolve()),sha256=sha(runtime_bytes)),
                  original_probe_log_sha256=records['probe.log']['sha256'],original_probe_log_bytes=records['probe.log']['bytes'],
                  source_records=[{key:row[key] for key in ('path','bytes','sha256')} for row in records.values()]),
    reviewer_remote_access=False,reviewer_tensor_asset_read=False,reviewer_GPU_or_model_execution=False,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,
    actual_loaded_binary_identity_proven=False,threshold_calibration_accepted=False,
    scope='Failure evidence only. Whole-card constructor/warmup peak is not generation or training capacity. Wrapper provenance revision needs exact actual source-chain evidence, independent delta-only review and new fixed intent; no path/hash/unwrap relaxation.')
target=REVIEW/'ACTUAL3290_FAILURE_RECEIPT.json';assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(receipt_path=str(target),receipt_sha256=sha(target.read_bytes()),ranks=ranks,costs=receipt['costs']),sort_keys=True))
