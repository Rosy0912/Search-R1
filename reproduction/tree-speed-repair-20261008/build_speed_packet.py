"""Build a code-only V13 native5 continuation from a supplied clean source commit.

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
OLD = WORKSPACE/'work/tree-retained18-migration-20261008/package-code-only-v11'
ROOT = '/dataset/shaoshitong/search-r1-rl/tree-grpo'
PARENT_SOURCE = '6efb60466a233f70bf8f50de26df32012f49b4bc'


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
    output = HERE/'package-code-only-v13'
    if output.exists():
        raise RuntimeError('Existing immutable packet must be preserved, not overwritten')
    remote_package = ROOT+'/instruct300/candidates/current200-retained18-speed-v13-'+source[:7]
    authority_path = ROOT+'/instruct300/CURRENT200_RETAINED18_SPEED_HUMAN_AUTHORITY-V13-'+source[:7]+'.json'
    release_path = ROOT+'/instruct300/CURRENT200_RETAINED18_SPEED_DIRECT_RELEASE-V13-'+source[:7]+'.json'
    intent = ROOT+'/instruct300/SUBMISSION_ONCE-retained18-current200-speed-v13-'+source[:7]+'.json'
    shutil.copytree(TEMPLATE, output, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
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
    if resume['parent_completed_outer'] != 5 or resume['next_outer_update'] != 6:
        raise RuntimeError('Template must restore actual5 and continue6')
    old = json.loads((OLD/'manifest.json').read_text())
    manifest = dict(old)
    for name in ('failed_predecessor3334', 'retained18_migration', 'path_fix_scope', 'current_chain_resume'):
        if name in manifest:
            manifest['historical_v11_'+name] = manifest.pop(name)
    stop_name = 'RETAINED18_FORMAL3335_MANUAL_STOP_TERMINAL_SUMMARY_20261008.json'
    manifest.update(code_commit=source, packet_revision=13,
        status='LOCAL_V13_SPEED_CONTINUATION_MATERIALIZED_NOT_DEPLOYED',
        materialized_at_utc=datetime.now(timezone.utc).isoformat(),
        current_chain_resume=resume, expected_initialized_completed_outer=5,
        planned_new_completed_outer=195, fresh_independent_200=False, resume=True,
        actual_migration_executed=False, core_source_bytes_changed=True,
        new_cpu_tests_executed=False, new_tests_executed=False, new_reviews_executed=False,
        external_first5_approval_required=False,
        full_run_continuation_requires_same_instance_first5_threshold_assessment=False,
        pre_review_hold=False, context_free_review_parallel_to_normal_training=True,
        speed_target_normal_outer_wall_seconds=420,
        speed_target_observed_accepted=False,
        source_selection=dict(selection, code_commit=source, contains_model_data_index_checkpoint_assets=False),
        stopped_predecessor3335=dict(path=remote_package+'/'+stop_name, sha256=sha(output/stop_name)),
        parent_source_mapping=dict(parent_source=PARENT_SOURCE,
            parent_config_sha256='5e530fc597e26e113c42418ca54faa9bdd0a42ad40bc1998e22c9599be9668fb',
            child_source=source, scope='Explicit performance-fix source mapping; same saved current200 state and immutable user protocol. Child resolved config is natively observed at startup.'),
        retained18_migration=dict(destination_node='h100-10', destination_ip='10.33.0.18',
            same_cluster_assets=True, assets_local_on_destination=True,
            output_server='10.33.0.18', output_server_retained=True,
            preserved_parent_completed_outer=5,
            storage_receipt_sha256=old['retained18_migration']['storage_receipt_sha256'],
            extra_short_training_added=False, exact_executor_resume=False))
    manifest['dispatch'] = dict(old['dispatch'], nodes=['h100-10'], once_intent=intent, train_wall_hours=168)
    manifest['wrapper_inventory_scope'] = 'V13 external wrappers plus exact new source; oldV11 untouched'
    stop_authority = json.loads((output/'RETAINED18_HUMAN_STOP_SLOW3335_RESTART_AUTHORITY_20261008.json').read_text())
    speed_authority = json.loads((output/'RETAINED18_SPEED420_AND_PARALLEL_REVIEW_AUTHORITY_20261008.json').read_text())
    authority = dict(status=manifest['execution_mode'], scope=manifest['scope'], packet_revision=13,
        code_commit=source, current_chain_resume=resume, same_instance_continuation=True,
        completed_outer_goal=200, final_evaluation_rows=51713,
        production_thresholds_accepted=False, independent_acceptance=False,
        training_accepted=False, original_sampler_actor_gate_accepted=False,
        configured_prospective_thresholds=manifest['configured_prospective_thresholds'],
        human_instruction=stop_authority['instruction'], latest_stop_and_restart=stop_authority,
        speed_and_parallel_review=speed_authority,
        original_human_time_known=False, preserved_completed_outer=5,
        normal_outer_target_seconds=420, initial_evaluation_added=False,
        extra0_to5_training_added=False, pre_review_hold=False, step5_approval_hold_added=False,
        retained18_migration=manifest['retained18_migration'])
    authority_file = HERE/'human-authority-v13.json'; write(authority_file, authority)
    manifest['human_training_authority'] = dict(path=authority_path, sha256=sha(authority_file))
    manifest['files'] = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p != output/'manifest.json'}
    manifest['wrapper_files'] = {name:digest for name,digest in manifest['files'].items()
                                 if not name.startswith('source/')}
    write(output/'manifest.json', manifest)
    release = dict(status=manifest['execution_mode'], execution_mode=manifest['execution_mode'],
        scope=manifest['scope'], packet_revision=13, code_commit=source,
        current_chain_resume=resume, manifest_sha256=sha(output/'manifest.json'),
        human_training_authority_sha256=sha(authority_file), cluster_name='wuhan-h100', node_list='h100-10',
        dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],
        dispatch_submitter_sha256=manifest['files']['submit_once.py'], dispatch_intent_path=intent,
        production_thresholds_accepted=False, independent_acceptance=False,
        training_accepted=False, original_sampler_actor_gate_accepted=False)
    release_file = HERE/'direct-release-v13.json'; write(release_file, release)
    from render_dispatch import render
    script_file = HERE/'formal-speed-v13.sbatch'
    with script_file.open('x') as stream:
        stream.write(render(remote_package, sha(output/'manifest.json'), release_path, sha(release_file), 168))
    deployment = dict(package=remote_package, manifest_sha=sha(output/'manifest.json'),
        release_path=release_path, release_sha=sha(release_file),
        authority_path=authority_path, authority_sha=sha(authority_file), once_intent=intent,
        source_commit=source, local_package=str(output), remote_submitted=False,
        script_path=ROOT+'/instruct300/FORMAL_RETAINED18_SPEED-V13-'+source[:7]+'.sbatch',
        script_sha=sha(script_file), parent_completed_outer=5, next_outer=6, train_wall_hours=168)
    write(HERE/'deployment.json', deployment)
    print(json.dumps(deployment, ensure_ascii=False))


if __name__ == '__main__':
    main()
