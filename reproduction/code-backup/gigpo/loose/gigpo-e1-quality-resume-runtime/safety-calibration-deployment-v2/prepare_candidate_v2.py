"""Local immutable package preparation only; no network, deployment or submission."""
import copy
import gzip
import io
import json
import os
import tarfile
from pathlib import Path

from deployment_contract_v2 import ROOT, SOURCE, HELPERS, CONTROLS, SCOPE, NATIVE_SCOPE, MANIFEST, DATA, CHECKPOINT, require, sha, write_once

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
NATIVE_SHA = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
PHYSICAL_SHA = '6f15809b7ff5d2eb0e17716cadb579e6c871d298b241b0a8843b4acb29ba1d2a'
STATIC_SHA = '200380b128013451c25736b47ace916dd4f7bab0134ec571790b57b14d75c45b'
IMPLEMENTATION_SHA = '92db5c558103ef0e568cc22b97d2199ef0c155746373146bdb6bafec43441cc8'


def preview_config(candidate, config_directory=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    q = candidate['quality_selection_control']
    env = dict(GIGPO_E1_ROOT=str(ROOT), GIGPO_E1_RUN=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'),
        AGENT_ROOT=str(ROOT), SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',
        SEARCH_R1_BASELINE_OUTPUT=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'), GIGPO_QUALITY_GATE='30',
        GIGPO_QUALITY_CHECKPOINT=CHECKPOINT, GIGPO_QUALITY_MANIFEST_SHA256=MANIFEST,
        GIGPO_QUALITY_SELECTION=q['selection_path'], GIGPO_QUALITY_SELECTION_SHA256=q['selection_sha256'],
        GIGPO_QUALITY_REVIEW=q['selection_review_path'], GIGPO_QUALITY_REVIEW_SHA256=q['selection_review_sha256'])
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        directory = config_directory or LOCAL / 'work/gigpo-e1-quality-resume/verl/trainer/config'
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            config = compose(config_name='ppo_gigpo_instruct300_quality10_safe')
        config.actor_rollout_ref.actor.policy_safety.mode = 'calibration'
        config.trainer.experiment_name = 'gigpo-e1-policy-safety-calibration25'
        OmegaConf.resolve(config)
        return OmegaConf.to_container(config, resolve=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_archive(path, sources):
    """Byte reproducible gzip/tar, no weights/data or symlinks embedded."""
    with Path(path).open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name, source in sorted(sources.items()):
                payload = Path(source).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid, info.mode = len(payload), 0, 0, 0, 0o644
                archive.addfile(info, io.BytesIO(payload))


def main():
    native_path = OUT / 'NATIVE-CANDIDATE-52BEFE9-v2.json'
    physical_path = OUT / 'independent-policy-safety-native-physical-v2.json'
    static_path = OUT / 'CALIBRATION-STATIC-INDEPENDENT-REVIEW-v2.json'
    implementation_path = OUT / 'CALIBRATION-IMPLEMENTATION-v1.json'
    for path, expected in ((native_path, NATIVE_SHA), (physical_path, PHYSICAL_SHA),
                           (static_path, STATIC_SHA), (implementation_path, IMPLEMENTATION_SHA)):
        require(sha(path) == expected, 'Changed immutable preparation input: ' + str(path))
    inherited = json.loads(native_path.read_text())
    physical = json.loads(physical_path.read_text())
    sizes = {r['path']: r['bytes'] for r in physical['metadata']}
    require(len(sizes) == len(inherited['files']) == 42327, 'Inherited complete physical path coverage differs')
    candidate = copy.deepcopy(inherited)
    candidate.update(status='CANDIDATE_NOT_READY', validation_scope=NATIVE_SCOPE, calibration_scope=SCOPE,
        allowed_safety_modes=['calibration'], calibration_phase='probability_probe_then_wait',
        allowed_calibration_phases=['probability_probe_then_wait'], source_native_acceptance=None,
        calibration_actual_outer_budget=1, retained_training_updates_contributed=0, optimizer_steps_per_outer=10,
        calibration_submitted=False, training_submitted=False, target_old_policy_kl=None, negative_tail_limit=None,
        pad_token_id=None,
        actor_recompute_abs_tolerance=None, actor_vllm_abs_tolerance=None,
        actor_recompute_joint_abs_tolerance=None, actor_vllm_joint_abs_tolerance=None,
        probability_tolerance_evidence=[], probability_review_wait_seconds=43200,
        prefix_mismatch_limit=0, action_token_mismatch_limit=0,
        healthy_baseline_f1_floor=.3231979640745351 / 2,
        inherited_native_candidate=dict(path=str(ROOT / 'logs/policy-safety-native-52befe9-candidate-v2.json'), sha256=NATIVE_SHA),
        inherited_physical=dict(path=str(ROOT / 'logs/policy-safety-native-52befe9-physical-v2.json'), sha256=PHYSICAL_SHA),
        preparation_scope='Immutable local calibration draft; native acceptance null; physical/deployment review required',
        controls=str(CONTROLS))
    candidate.pop('native_test_threshold_sets', None)
    candidate['files'] = [dict(r, bytes=sizes[r['path']]) for r in inherited['files']]
    candidate['evidence'] = [dict(r, bytes=None) for r in inherited['evidence']]
    implementation = json.loads(implementation_path.read_text())
    require(len(implementation['files']) == 13, 'Reviewed helper inventory changed')
    sources = {}
    files = {r['path']: r for r in candidate['files']}
    for record in implementation['files']:
        path = Path(record['path'])
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Frozen helper changed')
        remote = str(HELPERS / path.name)
        new = dict(path=remote, bytes=record['bytes'], sha256=record['sha256'])
        require(remote not in files or files[remote] == new, 'Existing native shared helper would be overwritten')
        files[remote] = new
        sources['runtime/' + path.name] = path
    for path in sorted(HERE.glob('*.py')):
        if path.name.startswith('test_'):
            continue
        files[str(CONTROLS / path.name)] = dict(path=str(CONTROLS / path.name), bytes=path.stat().st_size, sha256=sha(path))
        sources['control/' + path.name] = path
    candidate['files'] = list(files.values())
    for path in (static_path, implementation_path, OUT / 'CALIBRATION-DESIGN-v1.md', OUT / 'CALIBRATION-CPU-CONTRACT-v1.log'):
        remote = str(ROOT / 'logs' / path.name)
        require(remote not in {r['path'] for r in candidate['evidence']}, 'Duplicate new evidence')
        candidate['evidence'].append(dict(path=remote, bytes=path.stat().st_size, sha256=sha(path)))
        sources['evidence/' + path.name] = path
    candidate['helper_static_review'] = dict(path=str(ROOT / 'logs' / static_path.name), sha256=STATIC_SHA)
    preview = preview_config(candidate)
    preview_path = OUT / 'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json'
    write_once(preview_path, dict(status='LOCAL_PREVIEW_NOT_ACTUAL_RUNTIME_CONFIG',
        volatile_run_placeholder=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'), resolved=preview))
    candidate['expected_config_preview'] = dict(path=str(ROOT / 'logs' / preview_path.name), bytes=preview_path.stat().st_size, sha256=sha(preview_path))
    candidate['evidence'].append(candidate['expected_config_preview'])
    sources['evidence/' + preview_path.name] = preview_path
    target = OUT / 'CALIBRATION-CANDIDATE-52BEFE9-v2.json'
    write_once(target, candidate)
    sources['candidate/' + target.name] = target
    archive = HERE / 'policy-safety-calibration-52befe9-v2-package.tar.gz'
    make_archive(archive, sources)
    write_once(OUT / 'CALIBRATION-PACKAGE-v2.json', dict(status='IMMUTABLE_LOCAL_PACKAGE_NOT_READY_NOT_DEPLOYED',
        candidate=dict(path=str(target), sha256=sha(target)), archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)),
        native_acceptance=None, submitted_job=None, retained_training_updates_contributed=0,
        helper_count=13, inherited_files=42327, file_count=len(candidate['files']), evidence_count=len(candidate['evidence']),
        native_shared_helpers_preserved_byte_identical=True, package_members=list(sorted(sources))))
    print(json.dumps(dict(candidate_sha256=sha(target), package_sha256=sha(archive), ready=False, submitted=False)))


if __name__ == '__main__':
    main()
