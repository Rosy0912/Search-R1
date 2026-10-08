"""Independent saved-metadata failure review; no remote/assets/model work."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

WORK=Path(__file__).resolve().parents[1]
OUT=WORK.parent/'outputs/tree-instruct-300'
REVIEW=Path(__file__).resolve().parent
PACKAGE=WORK/'single-tp1-window-3d8c6f6d-466df7c2'
def sha(data):return hashlib.sha256(data).hexdigest()

snapshot_path=OUT/'PPO_WINDOW_GPU3285_METADATA_20261006T005647708949Z.json'
snapshot_bytes=snapshot_path.read_bytes()
assert sha(snapshot_bytes)=='95842e960b47bef2db6d4b98a1301dcb74eca1246edb1f26227b1bb0928191a2'
snapshot=json.loads(snapshot_bytes)
raw_path=OUT/'PPO_WINDOW_GPU3285_FAILURE_RAW.json'
raw_bytes=raw_path.read_bytes()
assert sha(raw_bytes)=='644be435926717336ea3a8a031ec242d0b095f88321d7f1995cc20b008622871'
raw=json.loads(raw_bytes)
assert raw['job_id']==snapshot['job_id']=='3285' and raw['host']==snapshot['host']=='h100-9'
assert raw['mutation'] is snapshot['mutation'] is False and raw['tensor_assets_read'] is False
records={Path(x['path']).name:x for x in raw['source_records']}
assert len(records)==len(raw['source_records'])==11
for record in records.values():
    data=record['text'].encode()
    assert len(data)==record['bytes'] and sha(data)==record['sha256']
probe=records['probe.log']['text']
pattern=r'\[rank(\d+)\]: .*Global diagnostic refusal at actual complete call5 CPU tensors before CUDA; local evidence: \'ValueError: Bound parent HF token ID fields missing\''
ranks=sorted(set(map(int,re.findall(pattern,probe))))
assert ranks==list(range(8))
assert 'rank      : 0 (local_rank: 0)' in probe and 'exitcode  : 1' in probe
assert snapshot['ready']==[]
result=json.loads(records['result.json']['text'])
assert result==snapshot['result.json']['record']
assert result['status']=='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'
assert result['package_manifest_sha256']=='f3e43a57f7164cc987d1ad211d617126d7bc5ece5450e4e13d365f405f15d6e1'
assert result['runner_commit']=='3d8c6f6d6cd82c23c4f2092f78a17d37126bc4a0'
assert result['release_sha256']=='32aa5f2a19638c2031955e42aa381fd578f0116e795f00c2623e49a5886d339f'
for field in ('actual_optimizer_writes','actual_generation_invocations','actual_actor_scoring_native_forwards','actual_new_emitted_tokens'):
    assert result[field] is None
assert result['cleanup_residual']=={} and result['original_background_pid_start_ticks_preserved'] is True
protected={1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
           1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
initial=json.loads(records['initial-allocation.json']['text'])
assert {int(pid):tick for pid,tick in initial['background_pid_start_ticks'].items()}==protected
final_pids={int(line.split(',')[1]) for line in result['final_gpu']['processes']}
assert final_pids==set(protected)
assert result['allocated_cpus']==192 and result['requested_cpus']==32 and result['allocated_gpus']==8
schedule=snapshot['scheduler']['stdout']
assert snapshot['scheduler']['returncode']==0
for literal in ('JobId=3285','JobState=FAILED','ExitCode=1:0','RunTime=00:00:41','NodeList=h100-9','NumCPUs=192','CPUs/Task=32','Requeue=0'):
    assert literal in schedule
assert 'StartTime=2026-10-06T08:55:24 EndTime=2026-10-06T08:56:05' in schedule
terminal=[json.loads(x) for x in records['probe-scheduler-observation-3285.jsonl']['text'].splitlines() if x.strip()]
assert terminal[-1]['terminal_state_observed'] is True and terminal[-1]['scheduler_state']=='FAILED'
assert terminal[-1]['stdout']==schedule
assert abs(result['wall_seconds']-(result['end_epoch']-result['start_epoch']))<1e-6
assert abs(result['gpu_hours']-8*result['wall_seconds']/3600)<1e-5
assert abs(result['allocated_cpu_hours']-192*result['wall_seconds']/3600)<1e-5
guard=[json.loads(x) for x in records['resources.jsonl']['text'].splitlines() if x.strip()]
assert len(guard)==snapshot['resources']['samples']==8
uuids=set(snapshot['launch.json']['record']['gpu_uuids'])
for row in guard:
    assert row['job_id']=='3285' and not row.get('event')
    assert len(row['gpus'])==8 and {g['uuid'] for g in row['gpus']}==uuids
    assert all(0<=g['foreign_memory_fraction']<=.15 and 0<=g['foreign_sm_upper']<=15 for g in row['gpus'])
resources=[json.loads(x) for x in records['process-resources.jsonl']['text'].splitlines() if x.strip()]
maxima={}
for row in resources:
    assert row['adapter']['job_id']=='3285' and row['adapter']['original_background_processes_preserved'] is True
    for proc in row['processes']:
        observed=(proc['user_cpu_ticks']+proc['system_cpu_ticks'])/proc['hz']
        maxima[proc['pid']]=max(maxima.get(proc['pid'],0),observed)

config_path=OUT/'PPO_HF110_ACTUAL_CONFIG_METADATA.json'
config_bytes=config_path.read_bytes()
assert sha(config_bytes)=='879f8585cd1807a067222aa146212f59537cc498f8f9e9d3adc8c791739a7fd0'
config=json.loads(config_bytes)
assert 'bos_token_id' not in config and config['pad_token_id']==151643 and config['eos_token_id']==151645
source=(PACKAGE/'repo/prepare_window_inputs.py').read_bytes()
assert sha(source)=='a3f8752f0d58924b3bfb782925d1eb27b47bfa8f4b6e55e84882ed82685f81f9'
tree=ast.parse(source)
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='original_tokenizer_ids')
def require(value,message):
    if not value:raise ValueError(message)
namespace={'require':require}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(PACKAGE/'repo/prepare_window_inputs.py'),'exec'),namespace)
try:namespace['original_tokenizer_ids'](config)
except ValueError as error:assert str(error)=='Bound parent HF token ID fields missing'
else:raise AssertionError('Expected exact actual metadata refusal')
assert namespace['original_tokenizer_ids'](dict(config,bos_token_id=None))==dict(pad_token_id=151643,eos_token_id=151645,bos_token_id=None)

runner=(PACKAGE/'repo/single_window_runner.py').read_text()
assert runner.index("cpu_coordinator.local_stage('actual complete call5 CPU tensors before CUDA'") < runner.index('torch.cuda.set_device(local_rank)')
assert runner.index('torch.cuda.set_device(local_rank)') < runner.index("dist.init_process_group('nccl'")
# Failure is before torch.load source tensor call in materialize_rank_inputs.
assert source.decode().index('ids = original_tokenizer_ids(hf_config)') < source.decode().index('payload = load_tensor_verified(')

receipt=dict(status='ACTUAL3285_PRE_CUDA_INPUT_REFUSAL_TERMINAL_COST_AND_CLEANUP_INDEPENDENTLY_VERIFIED',
    reviewer_task='/root/tree_window_final_review',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),job_id='3285',
    actual_failure_execution_verified=True,actual_full_window_measurement_verified=False,
    actual_terminal_verified=True,slurm_state='FAILED',exit_code='1:0',slurm_runtime_seconds=41,
    exact_failure='ValueError: Bound parent HF token ID fields missing',
    explicit_same_refusal_rank_ids=ranks,actual_ready_count=0,
    actual_optimizer_writes=None,actual_generation_invocations=None,actual_actor_scoring_native_forwards=None,
    counts_scope='Original failure result nulls retained; no successful all-rank measurement counters credited',
    pre_cuda_stage_proven_by_bound_control_flow=True,
    initialized_NCCL_or_model_path_reached=False,actual_model_forward_execution_verified=False,
    diagnosis=dict(bound_HF_config_bos_key_present=False,pad_token_id=151643,eos_token_id=151645,
                   old_function_refusal_on_exact_config_reproduced=True,explicit_null_fixture_would_pass=True,
                   previous_source_review_missed_actual_omitted_BOS_integration=True),
    package_manifest_sha256=result['package_manifest_sha256'],runner_commit=result['runner_commit'],
    original_package_or_intent_retry_accepted=False,old_package_changed=False,
    cleanup_residual={},original8_background_pid_start_ticks_preserved=True,
    guard_samples=8,max_foreign_memory_fraction=snapshot['resources']['max_foreign_memory_fraction'],
    max_foreign_sm_upper=snapshot['resources']['max_foreign_sm_upper'],
    whole_card_5s_peak_used_bytes=snapshot['resources']['whole_card_5s_peak_used_bytes'],
    costs=dict(scheduler_allocated_GPU_hours=8*41/3600,scheduler_allocated_CPU_hours=192*41/3600,
               supervisor_wall_seconds=result['wall_seconds'],supervisor_GPU_hours=result['gpu_hours'],
               supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],
               sampled_process_CPU_maxima_sum_seconds=sum(maxima.values()),sampled_process_CPU_samples=len(resources),
               complete_consumed_CPU_hours=None,sampled_CPU_scope='20 supervisor samples, PID maxima only; no exit totals or complete supervisor/launcher/descendant accounting'),
    evidence=dict(snapshot_file=dict(path=str(snapshot_path.resolve()),sha256=sha(snapshot_bytes)),
                  full_raw_file=dict(path=str(raw_path.resolve()),sha256=sha(raw_bytes)),
                  actual_HF_config_metadata=dict(path=str(config_path.resolve()),sha256=sha(config_bytes)),
                  original_probe_log_sha256=records['probe.log']['sha256'],original_probe_log_bytes=13710,
                  source_records=[{key:row[key] for key in ('path','bytes','sha256')} for row in records.values()]),
    reviewer_remote_access=False,reviewer_tensor_asset_read=False,reviewer_GPU_or_model_execution=False,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,
    actual_loaded_binary_identity_proven=False,threshold_calibration_accepted=False,
    scope='Failure evidence only; source CPU acceptance did not prove actual full generation compatibility. New minimal revision requires separate exact delta review and new fixed intent.')
target=REVIEW/'ACTUAL3285_FAILURE_RECEIPT.json'
assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(receipt_path=str(target),receipt_sha256=sha(target.read_bytes()),ranks=ranks,costs=receipt['costs']),sort_keys=True))
