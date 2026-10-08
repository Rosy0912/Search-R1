"""Build a code-only V14 native6 continuation from a supplied clean source commit.

Never deploys, submits, initializes CUDA, loads checkpoints, or runs tests.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
TEMPLATE = HERE/'runtime-template'
OLD = HERE/'package-code-only-v13'
ROOT = '/dataset/shaoshitong/search-r1-rl/tree-grpo'
PARENT_SOURCE = '62271081ba1bb74d8fc3f4958d623758efa74fa5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    source = args.source_commit
    if not re.fullmatch('[0-9a-f]{40}', source) or source == PARENT_SOURCE:
        raise RuntimeError('Supply the actual new40-character frozen source commit')
    repo = HERE/'source'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != source:
        raise RuntimeError('Supplied source commit is not the actual isolated source HEAD')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Freeze source changes before building; no uncommitted source is packaged')
    output = HERE/'package-code-only-v14'
    if output.exists():
        raise RuntimeError('Existing immutable packet must be preserved, not overwritten')
    remote_package = ROOT+'/instruct300/candidates/current200-retained18-speed-v14-'+source[:7]
    authority_path = ROOT+'/instruct300/CURRENT200_RETAINED18_SPEED_HUMAN_AUTHORITY-V14-'+source[:7]+'.json'
    release_path = ROOT+'/instruct300/CURRENT200_RETAINED18_SPEED_DIRECT_RELEASE-V14-'+source[:7]+'.json'
    intent = ROOT+'/instruct300/SUBMISSION_ONCE-retained18-current200-speed-v14-'+source[:7]+'.json'
    terminal_path = WORKSPACE/'outputs/tree-fresh200/RETAINED18_FORMAL3339_MANUAL_STOP_TERMINAL_SUMMARY_20261008.json'
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise RuntimeError('Actual3339 CANCELLED terminal receipt is required before materializing V14')
    terminal = json.loads(terminal_path.read_text())
    if not (terminal['job_id'] == 3339 and terminal['scheduler_state'] == 'CANCELLED'
            and terminal['manual_human_stop'] is True and terminal['last_completed_outer'] == 6
            and terminal['source_commit'] == PARENT_SOURCE
            and terminal['manifest_sha256'] == 'cbca5e1de3188eaf071ce02c295d2607475ec003c53352110a4c6fbf72cbfa63'
            and terminal['last_rolling_manifest_sha256'] == '2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3'
            and terminal['own_job_cgroup_exists'] is False
            and terminal['own_cgroup_boundary_released_verified'] is True
            and terminal['stopped_outer7_partial_adam_writes'] is None):
        raise RuntimeError('Actual3339 terminal/source/native6/cgroup/unknown-partial identity differs')
    ended = datetime.fromisoformat(terminal['actual_end_time_beijing'])
    if ended.tzinfo is None or ended.timestamp() > datetime.now(timezone.utc).timestamp() or terminal['scheduler_exit_code'] != '0:15' or not terminal['stop_authority']['instruction'].strip():
        raise RuntimeError('Actual3339 terminal end time/exit/this stop authority missing')
    shutil.copytree(TEMPLATE, output, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(terminal_path, output/terminal_path.name)
    write(output/'RETAINED18_HUMAN_STOP_SLOW3339_RESTART_AUTHORITY_20261008.json',terminal['stop_authority'])
    shutil.copyfile(HERE/'submit_speed_once.py', output/'submit_once.py')
    contract = output/'production_contract.py'
    text = contract.read_text()
    if text.count('__SOURCE_COMMIT__') != 1:
        raise RuntimeError('Template source placeholder differs')
    contract.write_text(text.replace('__SOURCE_COMMIT__', source))
    source_out = output/'source/repo'; source_out.mkdir(parents=True)
    raw = subprocess.check_output(['git', 'archive', '--format=tar', source], cwd=repo)
    sys.path.insert(0, str(output))
    from build_package_draft import materialize_source_archive
    selection = materialize_source_archive(raw, source_out)
    from current_chain_runtime import spec
    resume = spec()
    if resume['parent_completed_outer'] != 6 or resume['next_outer_update'] != 7:
        raise RuntimeError('Template must restore actual6 and continue7')
    old = json.loads((OLD/'manifest.json').read_text())
    manifest = dict(old)
    for name in ('retained18_migration', 'current_chain_resume', 'parent_source_mapping'):
        if name in manifest:
            manifest['historical_v13_'+name] = manifest.pop(name)
    stop_name = 'RETAINED18_FORMAL3339_MANUAL_STOP_TERMINAL_SUMMARY_20261008.json'
    manifest.update(code_commit=source, packet_revision=14,
        status='LOCAL_V14_SPEED_CONTINUATION_MATERIALIZED_NOT_DEPLOYED',
        materialized_at_utc=datetime.now(timezone.utc).isoformat(),
        current_chain_resume=resume, expected_initialized_completed_outer=6,
        planned_new_completed_outer=194, fresh_independent_200=False, resume=True,
        actual_migration_executed=False, core_source_bytes_changed=True,
        new_cpu_tests_executed=False, new_tests_executed=False, new_reviews_executed=False,
        external_first5_approval_required=False,
        full_run_continuation_requires_same_instance_first5_threshold_assessment=False,
        pre_review_hold=False, context_free_review_parallel_to_normal_training=True,
        speed_target_normal_outer_wall_seconds=420,
        speed_target_observed_accepted=False,
        source_selection=dict(selection, code_commit=source, contains_model_data_index_checkpoint_assets=False),
        historical_hf5_native_manifest=dict(path=ROOT+'/runs/tree-current200-continue-3335/checkpoints/training_state/global_step_5/manifest.json',
            sha256='56832544db6e55cbd346575aca0d06238602141df2704309c1a7f36aa4a27260'),
        parent_native_migration_receipt=dict(path=ROOT+'/runs/tree-current200-continue-3339/output/current-chain-migrated.json',
            sha256='6384b8350550ef767c8d1e0b5c27705842913517c642d8556d59b37058a98c3e'),
        stopped_predecessor3339=dict(path=remote_package+'/'+stop_name, sha256=sha(output/stop_name)),
        parent_source_mapping=dict(parent_source=PARENT_SOURCE,
            parent_job_id='3339',parent_run_id='tree-current200-continue-3339',parent_completed_outer=6,
            child_initial_optimizer_calls_per_rank={str(rank):120 for rank in range(8)},child_initial_scheduler_step=6,
            required_actual_cold_file_and_logical_comparison=True,
            parent_config_sha256='98a88882c09d8bbba89908b04d1f72930b4fccccc2bbcb5b6eb6886584655f3b',
            child_source=source, scope='Explicit performance-fix source mapping; same saved current200 state and immutable user protocol. Child resolved config is natively observed at startup.'),
        retained18_migration=dict(destination_node='h100-10', destination_ip='10.33.0.18',
            same_cluster_assets=True, assets_local_on_destination=True,
            output_server='10.33.0.18', output_server_retained=True,
            preserved_parent_completed_outer=6,
            storage_receipt_sha256=old['retained18_migration']['storage_receipt_sha256'],
            extra_short_training_added=False, exact_executor_resume=False))
    manifest['historical_v13_protocol'] = dict(old['protocol'])
    manifest['protocol'] = dict(old['protocol'], local_runtime_host='h100-10',
        retriever='same retained18 E5 and complete8-GPU sharded FP16 FAISS',
        parent_completed_outer=6,planned_new_completed_outer=194)
    manifest['historical_code_commit_by_job'] = dict(old['historical_code_commit_by_job'],
        **{'3335':'6efb60466a233f70bf8f50de26df32012f49b4bc','3339':PARENT_SOURCE})
    manifest['dispatch'] = dict(old['dispatch'], nodes=['h100-10'], once_intent=intent, train_wall_hours=168)
    manifest['wrapper_inventory_scope'] = 'V14 external wrappers plus exact frozen new source; immutable V13 controls and package preserved'
    stop_authority = terminal['stop_authority']
    speed_authority = json.loads((output/'RETAINED18_SPEED420_AND_PARALLEL_REVIEW_AUTHORITY_20261008.json').read_text())
    authority = dict(status=manifest['execution_mode'], scope=manifest['scope'], packet_revision=14,
        code_commit=source, current_chain_resume=resume, same_instance_continuation=True,
        completed_outer_goal=200, final_evaluation_rows=51713,
        production_thresholds_accepted=False, independent_acceptance=False,
        training_accepted=False, original_sampler_actor_gate_accepted=False,
        configured_prospective_thresholds=manifest['configured_prospective_thresholds'],
        human_instruction=stop_authority['instruction'], latest_stop_and_restart=stop_authority,
        speed_and_parallel_review=speed_authority,
        original_human_time_known=stop_authority.get('raw_human_timestamp_available',False), preserved_completed_outer=6,
        actual_predecessor_terminal=manifest['stopped_predecessor3339'],
        predecessor_actual_end_time_beijing=terminal['actual_end_time_beijing'],
        normal_outer_target_seconds=420, initial_evaluation_added=False,
        extra0_to5_training_added=False, pre_review_hold=False, step5_approval_hold_added=False,
        retained18_migration=manifest['retained18_migration'])
    authority_file = HERE/'human-authority-v14.json'; write(authority_file, authority)
    manifest['human_training_authority'] = dict(path=authority_path, sha256=sha(authority_file))
    manifest['files'] = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p != output/'manifest.json'}
    manifest['wrapper_files'] = {name:digest for name,digest in manifest['files'].items()
                                 if not name.startswith('source/')}
    write(output/'manifest.json', manifest)
    release = dict(status=manifest['execution_mode'], execution_mode=manifest['execution_mode'],
        scope=manifest['scope'], packet_revision=14, code_commit=source,
        current_chain_resume=resume, manifest_sha256=sha(output/'manifest.json'),
        human_training_authority_sha256=sha(authority_file), cluster_name='wuhan-h100', node_list='h100-10',
        dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],
        dispatch_submitter_sha256=manifest['files']['submit_once.py'], dispatch_intent_path=intent,
        production_thresholds_accepted=False, independent_acceptance=False,
        training_accepted=False, original_sampler_actor_gate_accepted=False)
    release_file = HERE/'direct-release-v14.json'; write(release_file, release)
    from render_dispatch import render
    script_file = HERE/'formal-speed-v14.sbatch'
    with script_file.open('x') as stream:
        stream.write(render(remote_package, sha(output/'manifest.json'), release_path, sha(release_file), 168))
    deployment = dict(package=remote_package, manifest_sha=sha(output/'manifest.json'),
        release_path=release_path, release_sha=sha(release_file),
        authority_path=authority_path, authority_sha=sha(authority_file), once_intent=intent,
        source_commit=source, local_package=str(output), remote_submitted=False,
        script_path=ROOT+'/instruct300/FORMAL_RETAINED18_SPEED-V14-'+source[:7]+'.sbatch',
        script_sha=sha(script_file), parent_completed_outer=6, next_outer=7, train_wall_hours=168)
    write(HERE/'deployment-v14.json', deployment)
    print(json.dumps(deployment, ensure_ascii=False))


if __name__ == '__main__':
    main()
