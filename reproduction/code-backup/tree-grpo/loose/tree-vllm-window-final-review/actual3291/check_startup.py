"""Captured nonterminal startup only; no remote/tensors/GPU or suite tests."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
PACKAGE=ROOT/'work/single-tp1-window-3583419c-466df7c2'
SNAPSHOT=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3291_METADATA_20261006T015716635122Z.json'
MANIFEST='20d42ab8f48ab390af6088990661d09f408d30cccb2946b407da007b8da8d48e'
RELEASE_SHA='afe3cdf0f9c85bfdfe0e16824d39707daa9e3e61054e41afeccc022093f7c48f'
def sha(data):return hashlib.sha256(data).hexdigest()
blob=SNAPSHOT.read_bytes()
assert sha(blob)=='c3be68e79c69ccee8999582837084a3ee21959573976e5aaebdb82bd8beaf62c'
s=json.loads(blob)
assert s['job_id']=='3291' and s['host']=='h100-9' and s['mutation'] is False and s['tensor_assets_read'] is False
manifest_bytes=(PACKAGE/'package-manifest.json').read_bytes();m=json.loads(manifest_bytes)
assert sha(manifest_bytes)==MANIFEST
release_path=ROOT/'work/tree-vllm-window-final-review/wrapper-repair3290/RELEASE.json'
release_bytes=release_path.read_bytes();r=json.loads(release_bytes)
assert sha(release_bytes)==RELEASE_SHA
dispatch_path=ROOT/'work/tree-vllm-window-wrapper-repair3290/DISPATCH_REMOTE_RESULT.json'
dispatch_bytes=dispatch_path.read_bytes();d=json.loads(dispatch_bytes)
assert d['job_id']==d['scheduler_response']=='3291'
assert d['status']=='SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE' and d['automatic_retry_allowed'] is False
for key in ('package_manifest_sha256','bundle_payload_sha256','runner_commit','selection_sha256',
            'diagnostic_profile','window_inputs_sha256','window_contract'):
    assert d['identity'][key]==r[key]
assert d['identity']['release_sha256']==RELEASE_SHA
l=s['launch.json']['record'];i=s['initial-allocation.json']['record']
assert l['source_manifest_files']==m['files'] and len(l['source_manifest_files'])==755
for value in (l,i):assert value['package_manifest_sha256']==MANIFEST and value['release_sha256']==RELEASE_SHA
assert l['bundle_payload_sha256']==r['bundle_payload_sha256'] and l['window_inputs_sha256']==r['window_inputs_sha256']
protected={1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
           1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}
for value in (l,i):assert {int(pid):tick for pid,tick in value['background_pid_start_ticks'].items()}==protected
assert i['allocated_gpus']==8 and i['allocated_cpus']==192 and i['requested_cpus']==32
assert len(l['gpu_uuids'])==len(set(l['gpu_uuids']))==len(l['leases'])==8
assert {row['uuid'] for row in l['leases']}==set(l['gpu_uuids'])
assert len({row['fd'] for row in l['leases']})==8 and l['shared_lock_fd'] not in {row['fd'] for row in l['leases']}
assert l['actual_loaded_binary_identity_proven'] is False and l['cgroup_relative'].endswith('/job_3291/step_0')
cmd=s['torchrun-command.json']['record']
assert '--master_addr=127.0.0.1' in cmd['command'] and '--nproc_per_node=8' in cmd['command']
assert cmd['lease_fds_inherited_by_children'] is False
assert sha(json.dumps(cmd['command'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())==cmd['command_sha256']
schedule=s['scheduler']['stdout'];assert s['scheduler']['returncode']==0
for literal in ('JobId=3291','JobState=RUNNING','RunTime=00:03:20','Requeue=0','StartTime=2026-10-06T09:54:00',
                'NodeList=h100-9','NumCPUs=192','CPUs/Task=32','OverSubscribe=NO','TresPerNode=gres:gpu:8'):
    assert literal in schedule
assert s['ready']==[] and s.get('result.json') is None
assert s['resources']['samples']==40 and s['resources']['max_foreign_memory_fraction']<=.15 and s['resources']['max_foreign_sm_upper']<=15
receipt=dict(status='ACTUAL3291_SINGLE_DISPATCH_STARTUP_IDENTITY_AND_GUARD_METADATA_INDEPENDENTLY_VERIFIED_NOT_TERMINAL',
    reviewed_at_utc=datetime.now(timezone.utc).isoformat(),reviewer_task='/root/tree_window_final_review',job_id='3291',
    runner_commit=r['runner_commit'],package_manifest_sha256=MANIFEST,release_sha256=RELEASE_SHA,
    source_manifest755_exactly_matches_frozen_manifest=True,single_confirmed_dispatch=True,
    allocated_gpus=8,allocated_cpus=192,requested_cpus=32,loopback_torchrun_verified=True,
    unique8_UUID_lease_metadata_verified=True,original8_background_PID_start_ticks_metadata_verified=True,
    guard_samples_at_snapshot=40,max_foreign_memory_fraction_at_snapshot=s['resources']['max_foreign_memory_fraction'],
    max_foreign_sm_upper_at_snapshot=s['resources']['max_foreign_sm_upper'],scheduler_state_at_snapshot='RUNNING',
    scheduler_runtime_seconds_at_snapshot=200,actual_ready_count_at_snapshot=0,
    actual_optimizer_writes=None,actual_generation_invocations=None,actual_terminal_verified=False,
    cleanup_verified=False,complete_costs_verified=False,actual_measurement_verified=False,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,actual_loaded_binary_identity_proven=False,
    reviewer_remote_access=False,reviewer_tensor_asset_read=False,reviewer_GPU_or_model_execution=False,
    automatic_retry_allowed=False,intent_spent_no_retry=True,
    evidence=[dict(path=str(SNAPSHOT),sha256=sha(blob)),dict(path=str(dispatch_path),sha256=sha(dispatch_bytes))],
    scope='Captured startup metadata only. RUNNING and manifest/lease metadata prove startup identity; no terminal counters, actual method chains/restoration, numerical consistency or generation/training capacity accepted.')
target=REVIEW/'STARTUP_RECEIPT.json';assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target.read_bytes())),sort_keys=True))
