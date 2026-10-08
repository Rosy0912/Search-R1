import datetime
import hashlib
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKET = ROOT / 'work/tree-fresh200-freeze-53082e-v12'
JOINT = ROOT / 'work/tree-fresh200-53082e-v12-context-free-review'
FORMAL = ROOT / 'work/tree-fresh200-formal-dispatch-v6'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
bound = lambda path: {'path': str(path), 'sha256': sha(path)}
prep_path = PACKET / 'actual-cpu-v12/PREPARATION.json'
assert sha(prep_path) == 'bddc7dbd8fe73cd823340e31dea9bed97b55f0f415d291d102bb02e38818553f'
prep = json.loads(prep_path.read_text())
assert prep['code_commit'] == '53082e1667a6ab131a036ca2ef66e004989ed1a9'
assert prep['actual_hydra_resolved'] is True and prep['cuda_initialized'] is False and prep['optimizer_writes'] == 0
assert len(prep['checked_resolved_values']) == 89
stage = json.loads((PACKET / 'actual-cpu-v12/STAGE_RESULT.json').read_text())
prepare = json.loads((PACKET / 'actual-cpu-v12/PREPARE_RESULT.json').read_text())
assert stage['returncode'] == prepare['returncode'] == 0
assert sha(JOINT / 'RECEIPT.json') == 'ce3ef87928c463db6f47518ca547492b32204c540d709d13c98fef3646489f6d'
assert sha(PACKET / 'CPU_EXECUTION_BINDING_REVIEW_V2.json') == '11ae256dd1ed7513338de577d5315052d9858baed3258238666c6620c8c320f3'
assert sha(FORMAL / 'INDEPENDENT_FORMAL_V6_HELPER_STATIC_REVIEW.json') == '20038e67a47e3e42d4e0923957840aaa0ab85550a8e356c5cdb62d2fda4310f0'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
status_path = ROOT / 'outputs/STATUS.json'
status = json.loads(status_path.read_text())
fresh = status['fresh200']
assert fresh['actual_new_outer_updates'] == 0
fresh['source_previous_reviewed_02b1'] = fresh['source_commit']
fresh['source_commit'] = prep['code_commit']
fresh['source_state'] = 'CLEAN_53082E_ZERO_V12_HELPER_V12_SOURCE_CPU_ACCEPTED_ACTUAL_CPU_PREPARED'
fresh['source_commit_scope'] = 'New source53082e/zeroV12/helperV12 independently SOURCE reviewed; real same-h1009 short CPU config passed. No actual new GPU zero or production training accepted.'
fresh['new_v12_local_package'] = {
    'packet': bound(PACKET / 'CPU_EXECUTION_PACKET.json'),
    'manifest': bound(PACKET / 'package-code-only/manifest.json'),
    'archive': bound(PACKET / 'CODE_PACKAGE.tar.gz'),
    'code_only_regular_archive_members': 252,
    'source_interfaces': 20, 'zero_wrapper_files': 15, 'helper_files': 3,
    'source_combination_review': bound(JOINT / 'RECEIPT.json'),
    'original_bad_argv_rejection': bound(PACKET / 'CPU_EXECUTION_BINDING_REVIEW.json'),
    'corrected_explicit_stdin_v2_review': bound(PACKET / 'CPU_EXECUTION_BINDING_REVIEW_V2.json'),
    'old_payloads_and_rejection_retained': True,
}
fresh['new_v12_actual_cpu_preparation'] = {
    'status': prep['status'], 'stage_result': stage, 'prepare_result': prepare,
    'native_preparation': bound(prep_path),
    'native_resolved_config': bound(PACKET / 'actual-cpu-v12/resolved.yaml'),
    'checked_resolved_value_count': 89,
    'actual_gpu_accepted': False, 'actual_zero_accepted': False,
    'actual_3b_training_accepted': False, 'submitted': False,
    'official5120_reused_by_exact_helper_tokenizer_identity': True,
    'existing_report_sha256': '2cc2366e9e3473276e4386820fa28249db9e1bb28875ca5298889c48d09c0761',
    'second5120_executed': False,
}
fresh['new_v12_zero_dispatch'] = {
    'status': 'PROSPECTIVE_ACTUAL_CPU_BOUND_ONCE_ZERO_NONE_REVIEW_RUNNING',
    'reviewer': '/root/v12_actual_zero_dispatch_none_review', 'actual_fork_turns': 'none',
    'draft_inputs': bound(PACKET / 'ZERO_DISPATCH_INPUTS.json'),
    'job_id': None, 'actual_submitted': False, 'actual_gpu_executed': False,
    'actual_native24_or_fullTree_accepted': False, 'actual_zero_accepted': False,
    'training_release': False,
}
fresh['formal_v6_source_helper_prepared_only'] = {
    'source_review': bound(FORMAL / 'INDEPENDENT_FORMAL_V6_HELPER_STATIC_REVIEW.json'),
    'owner': bound(FORMAL / 'OWNER_FORMAL_V6_CPU_PREPARE_RECEIPT.json'),
    'closure': bound(FORMAL / 'FINAL_FORMAL_V6_LOCAL_CLOSURE.json'),
    'runtime_directory': str(ROOT / 'work/tree-fresh200-train-runtime-v6'),
    'actual_four_observation_descriptors': None,
    'actual_formal_hydra_preparation': False, 'actual_formal_builder_executed': False,
    'production_thresholds_accepted': False, 'actual_first5_accepted': False,
}
status['status'] = 'FRESH200_53082E_V12_ACTUAL_CPU_PREPARED_ONCE_ZERO_NONE_REVIEW_RUNNING'
status['phase'] = 'ACTUAL_CPU_PREPARED_PROSPECTIVE_ZERO_DISPATCH_REVIEW'
status['current_phase'] = status['phase']
status['updated_at'] = now
status['last_management_observation_at'] = '2026-10-06T16:17:59.901550+00:00'
status['goal_status'] = 'active'
status['active_training_job'] = None
status['active_job_id'] = None
status['allocation_held'] = False
status['current_job']['latest_controller_row_expired'] = True
status['current_job']['terminal_state_scope'] = 'Retained actual positive3314 FAILED terminal; latest controller3314 invalid-id after retention expiry, not new running or stopped evidence.'
status['pending'] = [
    'Complete fresh fork:none binding of actual V12 CPU artifacts and sole prospective zero receipt placement/once submit; submit one bounded zero only after acceptance.',
    'Observe actual8rank fresh empty states/native24/2560Tree/mu/dev512/zero Adam and terminal cost/cleanup/protected8; independent actual acceptance and explicit bootstrap thresholds required.',
    'Bind FormalV6 CPU preparation from four actual observations; first5 then independent same-live-instance continuation6..200,51713/HF/cost/cleanup.',
]
temp = status_path.with_suffix('.json.v12-cpu-tmp')
with temp.open('w') as handle:
    json.dump(status, handle, ensure_ascii=False, indent=2)
    handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(temp, status_path)
with (ROOT / 'outputs/REPORT.md').open('a') as handle:
    handle.write('\n\n### ' + now + ' — New53082e/V12 source accepted and actual CPU preparation passed\n\n')
    handle.write('Credited fresh outer remains0/200, actual3B Adam writes0 and no new score. Joint no-context source/zero/helper review accepted with12 targeted independent CPU counterexamples; old full suites and5120 not rerun. Original prospective stage argv omitted explicit stdin and was independently REJECTED before execution; original rejection400513 retained. V2 adds explicit stdin, new fork:none review11ae accepted, then exactly one actual same-h1009 code/audit-only STAGE0 (' + str(stage['wall_seconds']) + 's) and PREPARE0 (' + str(prepare['wall_seconds']) + 's) executed. Native PREPARATION bddc7d and resolved16f325 retrieved byte-exact;89 config values, model13files/data identity verified; CUDAfalse/write0/not submitted. Historical official5120 report2cc is reused by unchanged raw helper/tokenizer hashes; historical supported rows0 never becomes new native24/fullTree acceptance.\n\n')
    handle.write('New actual fork:none once-zero-dispatch review is running. No new GPU job submitted. The original3314 failure/cleanup/cost/protected8 proof remains preserved; its live scheduler record expired, and the new helper requires exact bound prior positive terminal evidence plus current native/own-queue checks when accounting is disabled. FormalV6 helper/source-only review20038 also passed; actual calibration/bootstrap descriptors remain null, so no formal preparation/first5/200 release exists. User counting stays20 processed opportunities plus at least1 actual successful Adam write per complete outer, no allskip/abandoned warmup credit, real partial writes separately retained.\n')
    handle.flush(); os.fsync(handle.fileno())
for destination in [pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'), ROOT / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002']:
    for name in ('STATUS.json', 'REPORT.md'):
        shutil.copy2(ROOT / 'outputs' / name, destination / name)
    shutil.copytree(PACKET, destination / 'tree-fresh200/v12-actual-cpu-packet', dirs_exist_ok=True)
    shutil.copytree(JOINT, destination / 'tree-fresh200/v12-source-combination-review', dirs_exist_ok=True)
    shutil.copytree(FORMAL, destination / 'tree-fresh200/formal-v6-source-helper', dirs_exist_ok=True)
print(json.dumps({'status': status['status'], 'new_outer': 0, 'actual3b_writes': 0, 'actual_CPU_stage': stage['returncode'], 'actual_CPU_prepare': prepare['returncode'], 'source': prep['code_commit'], 'new_GPU_submitted': False, 'synced_destinations': 2}))
