"""Actual failed post-generation observation from saved local metadata only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
PACKAGE=ROOT/'work/single-tp1-window-3583419c-466df7c2'
OUT=ROOT/'outputs/tree-instruct-300'
def sha(data):return hashlib.sha256(data).hexdigest()
snapshot_path=OUT/'PPO_WINDOW_GPU3291_METADATA_20261006T020125942670Z.json'
snapshot_bytes=snapshot_path.read_bytes()
assert sha(snapshot_bytes)=='5c5475cb71e364791ccfe811a06858ace38def50ed52e0f69f82eb44677db95e'
snapshot=json.loads(snapshot_bytes)
raw_path=OUT/'PPO_WINDOW_GPU3291_FAILURE_RAW.json'
raw_bytes=raw_path.read_bytes()
assert len(raw_bytes)==9677138 and sha(raw_bytes)=='799331dd21b82092c21e0f2b9666bfc669e2373708180e8f86a1d3cd8eca89c6'
raw=json.loads(raw_bytes)
assert raw['job_id']==snapshot['job_id']=='3291' and raw['host']==snapshot['host']=='h100-9'
assert raw['mutation'] is snapshot['mutation'] is False and raw['tensor_assets_read'] is False
records={Path(row['path']).name:row for row in raw['source_records']}
assert len(records)==len(raw['source_records'])==11
for row in records.values():
    data=row['text'].encode()
    assert len(data)==row['bytes'] and sha(data)==row['sha256']
probe=records['probe.log']['text']
pattern=r"\[rank(\d+)\]: .*Global diagnostic refusal at original actual sampling parameters unchanged; local evidence: 'ValueError: Native window changed original SamplingParams'"
ranks=sorted(set(map(int,re.findall(pattern,probe))))
assert ranks==list(range(8))
assert snapshot['ready']==[] and snapshot['probe.log']['bytes']==records['probe.log']['bytes']==37594
result=json.loads(records['result.json']['text'])
assert result==snapshot['result.json']['record'] and result['status']=='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'
manifest_sha='20d42ab8f48ab390af6088990661d09f408d30cccb2946b407da007b8da8d48e'
release_sha='afe3cdf0f9c85bfdfe0e16824d39707daa9e3e61054e41afeccc022093f7c48f'
assert result['package_manifest_sha256']==manifest_sha
assert result['runner_commit']=='3583419cecee4691e72131863a61a0f6925cd7b5'
assert result['release_sha256']==release_sha
for field in ('actual_optimizer_writes','actual_generation_invocations','actual_actor_scoring_native_forwards','actual_new_emitted_tokens'):
    assert result[field] is None
assert result['cleanup_residual']=={} and result['original_background_pid_start_ticks_preserved'] is True
protected={1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
           1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
initial=json.loads(records['initial-allocation.json']['text'])
launch=json.loads(records['launch.json']['text'])
for row in (initial,launch):
    assert {int(pid):tick for pid,tick in row['background_pid_start_ticks'].items()}==protected
    assert row['package_manifest_sha256']==manifest_sha and row['release_sha256']==release_sha
assert {int(line.split(',')[1]) for line in result['final_gpu']['processes']}==set(protected)
assert result['allocated_cpus']==192 and result['requested_cpus']==32 and result['allocated_gpus']==8
manifest_bytes=(PACKAGE/'package-manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
assert sha(manifest_bytes)==manifest_sha and launch['source_manifest_files']==manifest['files']
assert len(launch['source_manifest_files'])==755
bundle=json.loads(records['bundle.json']['text'])
assert bundle['runner_commit']==result['runner_commit'] and bundle['window_contract']==result['window_contract']
assert bundle['vllm_installed_sources']['worker/model_runner_base.py']['sha256']=='49fe0eedd1667889daa63d5aa572d00f5eefbb794ea4368497f5243b02a54173'
assert len(launch['leases'])==len(launch['gpu_uuids'])==len(set(launch['gpu_uuids']))==8
assert {row['uuid'] for row in launch['leases']}==set(launch['gpu_uuids'])
dispatch_path=ROOT/'work/tree-vllm-window-wrapper-repair3290/DISPATCH_REMOTE_RESULT.json'
dispatch_bytes=dispatch_path.read_bytes();dispatch=json.loads(dispatch_bytes)
assert dispatch['job_id']==dispatch['scheduler_response']=='3291' and dispatch['automatic_retry_allowed'] is False
assert dispatch['identity']['package_manifest_sha256']==manifest_sha and dispatch['identity']['release_sha256']==release_sha
assert dispatch['identity']['runner_commit']==result['runner_commit']
prepared_path=ROOT/'work/tree-vllm-window-wrapper-repair3290/DISPATCH_PREPARED.json'
prepared_bytes=prepared_path.read_bytes();prepared=json.loads(prepared_bytes)
assert prepared['fixed_intent']=='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/SINGLE_TP1_WINDOW_WRAPPER_REPAIR3290_ONCE.json'
assert prepared['automatic_retry_allowed'] is False
assert prepared['package_manifest_sha256']==manifest_sha and prepared['release_sha256']==release_sha
schedule=snapshot['scheduler']['stdout'];assert snapshot['scheduler']['returncode']==0
for literal in ('JobId=3291','JobState=FAILED','ExitCode=1:0','RunTime=00:06:24','NodeList=h100-9',
                'NumCPUs=192','CPUs/Task=32','Requeue=0','OverSubscribe=NO','TresPerNode=gres:gpu:8'):
    assert literal in schedule
assert 'StartTime=2026-10-06T09:54:00 EndTime=2026-10-06T10:00:24' in schedule
terminal=[json.loads(line) for line in records['probe-scheduler-observation-3291.jsonl']['text'].splitlines() if line.strip()]
assert terminal[-1]['terminal_state_observed'] is True and terminal[-1]['scheduler_state']=='FAILED'
assert terminal[-1]['stdout']==schedule
assert abs(result['wall_seconds']-(result['end_epoch']-result['start_epoch']))<1e-6
assert abs(result['wall_seconds']-381.9918704032898)<1e-6
assert abs(result['gpu_hours']-8*result['wall_seconds']/3600)<1e-5
assert abs(result['allocated_cpu_hours']-192*result['wall_seconds']/3600)<1e-5
guard=[json.loads(line) for line in records['resources.jsonl']['text'].splitlines() if line.strip()]
assert len(guard)==snapshot['resources']['samples']==76
uuids=set(launch['gpu_uuids'])
for row in guard:
    assert row['job_id']=='3291' and not row.get('event')
    assert len(row['gpus'])==8 and {g['uuid'] for g in row['gpus']}==uuids
    assert all(0<=g['foreign_memory_fraction']<=.15 and 0<=g['foreign_sm_upper']<=15 for g in row['gpus'])
resources=[json.loads(line) for line in records['process-resources.jsonl']['text'].splitlines() if line.strip()]
maxima={}
for row in resources:
    assert row['adapter']['job_id']=='3291' and row['adapter']['original_background_processes_preserved'] is True
    for proc in row['processes']:
        observed=(proc['user_cpu_ticks']+proc['system_cpu_ticks'])/proc['hz']
        maxima[proc['pid']]=max(maxima.get(proc['pid'],0),observed)
assert len(resources)==359 and abs(sum(maxima.values())-2829.08)<1e-6
runner_path=PACKAGE/'repo/single_window_runner.py';runner_bytes=runner_path.read_bytes();runner=runner_bytes.decode()
markers=["'passive native hook source binding'", "'passive logits/sampler hooks'", 'with manager:',
         'raw = rollout.generate_sequences(prompts=prompts)', 'raw = manager.postprocess_data(raw)',
         "'live generated state before offload'", 'after_params = sampling_params_metadata(rollout.sampling_params)',
         "'original actual sampling parameters unchanged'", 'finally:',
         'observer.__exit__(*sys.exc_info()); wrappers.__exit__(*sys.exc_info())',
         "'actual original unpadded output/context mapping'", 'actor_logprobs = actor.compute_log_prob(raw)']
positions=[runner.index(marker) for marker in markers];assert positions==sorted(positions)
assert "before_params == after_params,'Native window changed original SamplingParams'" in runner
# Complete observer/restoration/counter receipts are published only after this refusal.
assert runner.index("wrapper_receipt = coordinator.local_stage('passive original method once/restoration'")>positions[7]
assert runner.index("weight_receipt = coordinator.local_stage('both live weight boundaries'")>positions[7]
assert sha(runner_bytes)==manifest['files']['repo/single_window_runner.py']
startup_path=REVIEW/'STARTUP_RECEIPT.json';startup_bytes=startup_path.read_bytes()
assert sha(startup_bytes)=='29fbf98c43c4e27d0839570d49df381e950bdb036a322418ae71c8d18d164409'
receipt=dict(status='ACTUAL3291_POST_NATIVE_GENERATION_SAMPLING_METADATA_REFUSAL_TERMINAL_COST_AND_CLEANUP_INDEPENDENTLY_VERIFIED',
    reviewer_task='/root/tree_window_final_review',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),job_id='3291',
    actual_failure_execution_verified=True,actual_full_window_measurement_verified=False,actual_terminal_verified=True,
    slurm_state='FAILED',exit_code='1:0',slurm_runtime_seconds=384,
    exact_failure='ValueError: Native window changed original SamplingParams',explicit_same_refusal_rank_ids=ranks,
    actual_ready_count=0,actual_optimizer_writes=None,actual_generation_invocations=None,
    actual_actor_scoring_native_forwards=None,actual_new_emitted_tokens=None,
    counts_scope='Original failure result nulls retained; no successful ready, output mapping, actor comparison or measurement counters credited',
    reached_stage_scope='All-rank bound control flow reaches comparison after original native generation call and manager postprocess return and live before_offload stage succeeds; before observer/method-restoration/weight final receipts, emitted alignment,80 actor comparisons and publish. This proves reached stage, not missing runtime counters or full measurement.',
    exact_changed_sampling_fields_saved=False,actual_concrete_sampling_change_cause_proven=False,
    diagnosis_scope='Only before_params != after_params is verified. Neither snapshots nor full logs preserved before/after fields. Native default EOS updates or any other specific mutation are hypotheses, not this failed run\'s observed cause.',
    actual_native_wrapper_source_bind_gate_reached_beyond=True,
    actual_native_hook_restoration_receipt_verified=False,
    restoration_scope='Bound finally source invokes observer/wrapper exit; complete restored native identity/counter receipt was not saved and cannot be accepted from control flow alone.',
    package_manifest_sha256=manifest_sha,runner_commit=result['runner_commit'],release_sha256=release_sha,
    source_manifest755_exact_matches_frozen_manifest=True,added_wrapper_pin_sha256=bundle['vllm_installed_sources']['worker/model_runner_base.py']['sha256'],
    fixed_intent=prepared['fixed_intent'],confirmed_unique_job_id_in_dispatch_record='3291',intent_spent_no_retry=True,
    intent_scope='Reviewed fixed-intent source/schema and recorded single3291 dispatch verified; durable intent file contents are not included in these11 runtime records.',
    original_package_or_intent_retry_accepted=False,old_package_changed=False,
    cleanup_residual={},original8_background_pid_start_ticks_preserved=True,
    guard_samples=len(guard),max_foreign_memory_fraction=snapshot['resources']['max_foreign_memory_fraction'],
    max_foreign_sm_upper=snapshot['resources']['max_foreign_sm_upper'],whole_card_5s_peak_used_bytes=snapshot['resources']['whole_card_5s_peak_used_bytes'],
    costs=dict(scheduler_allocated_GPU_hours=8*384/3600,scheduler_allocated_CPU_hours=192*384/3600,
               supervisor_wall_seconds=result['wall_seconds'],supervisor_GPU_hours=result['gpu_hours'],
               supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],
               sampled_process_CPU_maxima_sum_seconds=sum(maxima.values()),sampled_process_CPU_samples=len(resources),
               sampled_process_PID_count=len(maxima),complete_consumed_CPU_hours=None,
               sampled_CPU_scope='PID maxima from359 process samples only; no complete exit totals or launcher/preflight/descendant accounting; CPU allocation is not consumption'),
    evidence=dict(snapshot_file=dict(path=str(snapshot_path.resolve()),sha256=sha(snapshot_bytes)),
                  full_raw_file=dict(path=str(raw_path.resolve()),sha256=sha(raw_bytes),bytes=len(raw_bytes)),
                  frozen_runner_source=dict(path=str(runner_path.resolve()),sha256=sha(runner_bytes)),
                  dispatch_result=dict(path=str(dispatch_path.resolve()),sha256=sha(dispatch_bytes)),
                  dispatch_prepared=dict(path=str(prepared_path.resolve()),sha256=sha(prepared_bytes)),
                  startup_receipt=dict(path=str(startup_path.resolve()),sha256=sha(startup_bytes)),
                  original_probe_log_sha256=records['probe.log']['sha256'],original_probe_log_bytes=records['probe.log']['bytes'],
                  source_records=[{key:row[key] for key in ('path','bytes','sha256')} for row in records.values()]),
    reviewer_remote_access=False,reviewer_tensor_asset_read=False,reviewer_GPU_or_model_execution=False,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,actual_loaded_binary_identity_proven=False,
    threshold_calibration_accepted=False,source_or_GPU_suite_repeated=False,
    scope='Failed actual observation only. Native generation returning does not validate all prompts/scalars/context alignment, actor probability comparisons, exact native restoration, full production capacity or training. Original3291 fixed intent is spent; no new experiment or source revision authorized by this receipt.')
target=REVIEW/'ACTUAL3291_FAILURE_RECEIPT.json';assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target.read_bytes()),costs=receipt['costs']),sort_keys=True))
