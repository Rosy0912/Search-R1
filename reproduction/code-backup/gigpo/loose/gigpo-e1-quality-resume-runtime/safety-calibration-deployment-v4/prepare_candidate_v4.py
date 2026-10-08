"""Prepare new isolated local v4 only; no network, deployment, READY or submission."""
import copy
import gzip
import io
import json
import os
import tarfile
from pathlib import Path

from deployment_contract_v4 import (ROOT, SOURCE, HELPERS, NATIVE_HELPERS, CONTROLS,
    MANIFEST, CHECKPOINT, require, sha, write_once, validate_candidate)

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
RUNTIME = HERE.with_name('safety-calibration-v4')
NATIVE_SHA = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
PREDECESSOR_SHA = '04c4119187d60122e8a01b3a2bdf0180628bfc50267938fc946159b01c964a9b'
IMPLEMENTATION_SHA = '92db5c558103ef0e568cc22b97d2199ef0c155746373146bdb6bafec43441cc8'
CD1_SHA = '25f25dd04eb4b47366bae49d319878dd719994ed0a246cdfc03de6d02e56f968'


def preview_config(candidate, config_directory=None):
    # Same reviewed configuration bytes/semantics; only the controller version changes.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    q = candidate['quality_selection_control']
    env = dict(GIGPO_E1_ROOT=str(ROOT), GIGPO_E1_RUN=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'),
        AGENT_ROOT=str(ROOT), SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',
        SEARCH_R1_BASELINE_OUTPUT=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'), GIGPO_QUALITY_GATE='30',
        GIGPO_QUALITY_CHECKPOINT=CHECKPOINT, GIGPO_QUALITY_MANIFEST_SHA256=MANIFEST,
        GIGPO_QUALITY_SELECTION=q['selection_path'], GIGPO_QUALITY_SELECTION_SHA256=q['selection_sha256'],
        GIGPO_QUALITY_REVIEW=q['selection_review_path'], GIGPO_QUALITY_REVIEW_SHA256=q['selection_review_sha256'])
    previous = {k: os.environ.get(k) for k in env}
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
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_archive(path, sources):
    with Path(path).open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name, source in sorted(sources.items()):
                payload = Path(source).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid, info.mode = len(payload), 0, 0, 0, 0o644
                archive.addfile(info, io.BytesIO(payload))


def main():
    native_path = OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json'
    predecessor_path = OUT/'CALIBRATION-CANDIDATE-52BEFE9-v3.json'
    runtime_manifest = OUT/'CALIBRATION-IMPLEMENTATION-v1.json'
    cd1_path = OUT/'CALIBRATION-DEPLOYMENT-CONTROL-STATIC-INDEPENDENT-REVIEW-v3.json'
    for path, expected in ((native_path,NATIVE_SHA),(predecessor_path,PREDECESSOR_SHA),
                           (runtime_manifest,IMPLEMENTATION_SHA),(cd1_path,CD1_SHA)):
        require(sha(path) == expected, 'Immutable predecessor/static input changed: ' + str(path))
    native = json.loads(native_path.read_text())
    predecessor = json.loads(predecessor_path.read_text())
    require(native['code_commit'] == SOURCE and native['helpers'] == str(NATIVE_HELPERS)
            and len(native['files']) == 42327, 'Wrong inherited native closure')
    candidate = copy.deepcopy(predecessor)
    candidate.update(helpers=str(HELPERS),controls=str(CONTROLS),source_native_acceptance=None,
        native_pending_job_acceptance=None,predecessor_calibration_candidate_sha256=PREDECESSOR_SHA,
        preparation_scope='New isolated v4 local draft; actual native terminal/physical/independent acceptance still null',
        native_terminal_contract='Actual JSON+gzip originals, same job h100-8 COMPLETED0:0 Requeue0 Restarts0, readable-ancestor ENOENT job+scope',
        native_physical_acceptance_required=True, native_directory_modified=False)
    old_files = {r['path']: r for r in predecessor['files']}
    files = {}
    for record in native['files']:
        observed = old_files[record['path']]
        require(observed['sha256'] == record['sha256'] and isinstance(observed['bytes'],int),
                'Inherited native file SHA/actual bytes differ')
        files[record['path']] = copy.deepcopy(observed)
    original_native_files = copy.deepcopy(files)
    runtime = json.loads(runtime_manifest.read_text())
    require(len(runtime['files']) == 13, 'Frozen runtime inventory differs')
    runtime_records, sources = [], {}
    for record in runtime['files']:
        original = Path(record['path'])
        local = RUNTIME/original.name
        require(sha(original) == sha(local) == record['sha256'] and local.stat().st_size == record['bytes'],
                'Frozen runtime original/copy bytes changed: ' + local.name)
        remote = str(HELPERS/local.name)
        require(remote not in files, 'Calibration runtime would touch inherited native closure')
        new = dict(path=remote,bytes=record['bytes'],sha256=record['sha256'])
        files[remote] = new
        runtime_records.append(new)
        sources['runtime/'+local.name] = local
    for local in sorted(HERE.glob('*.py')):
        if local.name.startswith('test_'):
            continue
        remote = str(CONTROLS/local.name)
        require(remote not in files, 'Control path already inherited')
        files[remote] = dict(path=remote,bytes=local.stat().st_size,sha256=sha(local))
        sources['control/'+local.name] = local
    require({path:files[path] for path in original_native_files} == original_native_files,
            'Native-v2 inventory bytes would change')
    candidate['files'] = list(files.values())
    candidate['runtime_helpers'] = runtime_records
    # Original inherited evidence already has actual bytes from authorized v3 read-only proof.
    require(all(isinstance(r['bytes'],int) for r in candidate['evidence']), 'Unestablished inherited evidence bytes')
    previous_evidence = {r['path']:r for r in predecessor['evidence']}
    native_evidence_paths = {r['path'] for r in native['evidence']}
    for record in candidate['evidence']:
        if record['path'] not in native_evidence_paths:
            local = OUT/Path(record['path']).name
            require(local.stat().st_size == record['bytes'] and sha(local) == record['sha256'],
                    'Preserved local calibration proof changed')
            sources['evidence/'+local.name] = local
    for local in (cd1_path,OUT/'CALIBRATION-DEPLOYMENT-v4.md',OUT/'CALIBRATION-DEPLOYMENT-CPU-v4.log'):
        remote = str(ROOT/'logs'/local.name)
        require(remote not in previous_evidence, 'Existing v4 evidence path: never overwrite')
        candidate['evidence'].append(dict(path=remote,bytes=local.stat().st_size,sha256=sha(local)))
        sources['evidence/'+local.name] = local
    expected = json.loads((OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json').read_text())
    require(preview_config(candidate) == expected['resolved'], 'Controller change modified reviewed config preview')
    validate_candidate(candidate)
    target = OUT/'CALIBRATION-CANDIDATE-52BEFE9-v4.json'
    write_once(target,candidate)
    sources['candidate/'+target.name] = target
    sources['cpu/test_deployment_v4.py'] = HERE/'test_deployment_v4.py'
    archive = HERE/'policy-safety-calibration-52befe9-v4-package.tar.gz'
    make_archive(archive,sources)
    receipt = dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',
        candidate=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),
        predecessor_candidate_sha256=PREDECESSOR_SHA,native_candidate_sha256=NATIVE_SHA,
        native_acceptance=None,submitted_job=None,retained_training_updates_contributed=0,
        runtime_helpers_path=str(HELPERS),controls_path=str(CONTROLS),resource_guard_path=str(NATIVE_HELPERS),
        runtime_13_helpers_unchanged=True,shared_helpers_copied_separately=True,
        inherited_native_42327_paths_unchanged=True,native_directory_modified=False,
        cpu_tests=82,actual_gpu_executed=False,all_file_sizes_established=True,all_evidence_sizes_established=True,
        file_count=len(candidate['files']),evidence_count=len(candidate['evidence']),package_members=sorted(sources))
    write_once(OUT/'CALIBRATION-PACKAGE-v4.json',receipt)
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),
        files=len(candidate['files']),evidence=len(candidate['evidence']),ready=False,submitted=False)))


if __name__ == '__main__':
    main()
