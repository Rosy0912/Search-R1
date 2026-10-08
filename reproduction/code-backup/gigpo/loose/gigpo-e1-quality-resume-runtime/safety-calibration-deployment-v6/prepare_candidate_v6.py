"""Prepare v6 only after explicit final v6-v3 runtime/failure proof closure; local only."""
import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

from deployment_contract_v6 import (ROOT, SOURCE, HELPERS, NATIVE_HELPERS, CONTROLS,
    MANIFEST, CHECKPOINT, STARTUP_SCOPE, STARTUP_PROOF_ROLES, require, sha, write_once, validate_candidate)

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
RUNTIME = HERE.with_name('safety-calibration-v6')
OLD_RUNTIME = HERE.with_name('safety-calibration-v4')
NATIVE_SHA = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
PREDECESSOR_SHA = '111b4f9fddf9dc7e2ec18f4a86daaaf93e987fe472d5ea018dd6835777bf0105'
BASE_RUNTIME_MANIFEST_SHA = '92db5c558103ef0e568cc22b97d2199ef0c155746373146bdb6bafec43441cc8'
FINAL_RUNTIME_MANIFEST_NAME = 'CALIBRATION-RUNTIME-IMPLEMENTATION-v6-v3.json'
CONTROL_NAMES = {'deployment_contract_v6.py','freeze_ready_v6.py','prepare_candidate_v6.py','submit_once_v6.py','verify_physical_v6.py'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


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


def bound_local(record):
    require(set(record) == {'path','bytes','sha256'}, 'Proof input must explicitly bind path/bytes/SHA')
    path=Path(record['path'])
    require(path.is_absolute() and path.resolve()==path and path.is_relative_to(OUT)
            and path.is_file() and not path.is_symlink(), 'Not a dedicated local original proof')
    require(type(record['bytes']) is int and record['bytes']>=0 and SHA_RE.fullmatch(record['sha256'])
            and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'Missing/changed proof original')
    return path


def validate_runtime_inputs(proofs):
    paths={role:bound_local(proofs[role]) for role in STARTUP_PROOF_ROLES}
    require(paths['runtime_manifest'].name == FINAL_RUNTIME_MANIFEST_NAME,
            'Only final v6-v3 runtime manifest may gate this draft, never superseded v6')
    manifest=json.loads(paths['runtime_manifest'].read_text())
    require(manifest.get('source_commit',manifest.get('code_commit'))==SOURCE
            and manifest['changed_filenames']==['calibration_run_v1.py'], 'Not the final narrow startup-only runtime')
    base_path=OUT/'CALIBRATION-IMPLEMENTATION-v1.json'
    require(sha(base_path)==BASE_RUNTIME_MANIFEST_SHA,'Frozen original runtime manifest changed')
    originals={Path(record['path']).name:record for record in json.loads(base_path.read_text())['files']}
    records=manifest['files'];new={Path(record['path']).name:record for record in records}
    require(len(records)==len(new)==13 and set(new)==set(originals),'Not exact thirteen runtime files')
    changed=[]
    for name,record in new.items():
        path=RUNTIME/name
        require(Path(record['path'])==path and path.is_file() and not path.is_symlink()
                and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'Runtime manifest differs from actual v6 bytes')
        old=OLD_RUNTIME/name
        require(old.stat().st_size==originals[name]['bytes'] and sha(old)==originals[name]['sha256'], 'Preserved v4 runtime changed')
        if path.read_bytes()!=old.read_bytes():changed.append(name)
    require(changed==['calibration_run_v1.py'],'Probability/driver/worker/mathematical runtime may not change')
    review=json.loads(paths['runtime_review'].read_text())
    validate_runtime_review(proofs,review,changed)
    require(cpu_success_count(paths['runtime_cpu'].read_text())>0,'Missing actual local cooldown CPU test result')
    return manifest,paths


def validate_runtime_review(proofs,review,changed):
    require(review['status']=='PASS' and review.get('code_commit',review.get('source_commit'))==SOURCE
            and review['runtime_manifest_sha256']==proofs['runtime_manifest']['sha256']
            and review['runtime_diff_sha256']==proofs['runtime_diff']['sha256']
            and review['runtime_cpu_sha256']==proofs['runtime_cpu']['sha256'], 'Independent runtime review not exact final manifest/diff/CPU')
    require(review['startup_change_scope']==STARTUP_SCOPE and review['runtime_changed_filenames']==changed
            and review['gpu_used'] is False and review['source_modified'] is False
            and review['production_thresholds_selected'] is False, 'Runtime static review exceeds startup-only scope')


def cpu_success_count(text):
    """Accept explicit successful pytest or unittest terminals, never a failure summary."""
    if re.search(r'\b(FAILED|FAILURES|ERRORS|ERROR|failed|error)\b',text):return 0
    pytest=re.search(r'(\d+) passed(?:[, ].*)?\s*$',text)
    unittest=re.search(r'Ran (\d+) tests? in [^\n]+\n\s*OK\s*$',text)
    match=pytest or unittest
    return int(match.group(1)) if match is not None else 0


def validate_failure_inputs(proofs, paths):
    payload=json.loads(paths['failure_payload'].read_text());review=json.loads(paths['failure_review'].read_text())
    release=json.loads(paths['failure_cgroup'].read_text())
    require(review['status']=='FAILED_CALIBRATION_STARTUP_NOT_ACCEPTED' and review['job_id']=='3272'
            and review['code_commit']==SOURCE and review['restore_manifest_sha256']==MANIFEST
            and review['review_version']==2 and review['actual_retained_training_updates']==0
            and review['actual_sampling'] is False and review['actual_adam_writes']==0
            and review['actual_Ray_driver_started'] is False,'Not actual failed3272 pre-sampling/startup evidence')
    require(review['consumed_payload']['path']==proofs['failure_payload']['path']
            and review['consumed_payload']['sha256']==proofs['failure_payload']['sha256']
            and review['resource']['cgroup_release_evidence']['path']==proofs['failure_cgroup']['path']
            and review['resource']['cgroup_release_evidence']['sha256']==proofs['failure_cgroup']['sha256'],
            'Failure independent review raw/release SHA does not close')
    require(payload['job']=='3272' and payload['node']=='h100-8'
            and payload['queue']['returncode']==payload['compute']['returncode']==0
            and not payload['queue']['stdout'].strip() and not payload['compute']['stdout'].strip()
            and payload['matching_cgroups']==[] and payload['cgroup_mount_readable'] is True,'Failed3272 current release evidence incomplete')
    resource=review['resource']
    require(resource['saved_cleanup_residual_states']=={} and resource['saved_finalGPUcompute']==[]
            and resource['actual_postterminal_queue_empty'] is True
            and resource['actual_postterminal_GPUcompute_empty'] is True
            and resource['matching_own_cgroups']==[] and resource['cgroup_mount_readable'] is True
            and resource['explicit_own_scope_and_jobroot_ENOENT'] is True,
            'Failed3272 independent release summary incomplete')
    require(release['job']=='3272' and release['node']=='h100-8','Wrong failed job cgroup release')
    expected={'/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3272',
              '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3272/step_0'}
    require(len(release['paths'])==2 and {record['path'] for record in release['paths']}==expected
            and all(record['exists'] is False and record['errno']==2 and record['ancestor_readable'] is True
                    and record['readable_ancestor']=='/sys/fs/cgroup/freezer/slurm_h100-8'
                    for record in release['paths']),'Original failed3272 exact job/scope ENOENT missing')
    originals=[]
    for path,record in payload['files'].items():
        raw=record['content'].encode('utf-8')
        require(Path(path).is_absolute() and '..' not in Path(path).parts
                and (Path(path).parent==ROOT/'logs' or Path(path).is_relative_to(payload['run']))
                and len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],'Failure raw bytes/SHA corrupt')
        originals.append(dict(path=path,bytes=record['bytes'],sha256=record['sha256']))
    by={record['path']:record for record in originals}
    require(len(by)==len(originals) and review['consumed_originals']==originals,'Failure consumed raw inventory differs')
    terminal=json.loads(payload['files'][review['terminal_path']]['content'])
    require(by[review['terminal_path']]['sha256']==review['terminal_sha256']
            and terminal['fields']['JobId']=='3272' and terminal['fields']['NodeList']=='h100-8'
            and terminal['fields']['JobState']=='FAILED' and terminal['fields']['ExitCode']=='1:0'
            and terminal['fields']['Requeue']==terminal['fields']['Restarts']=='0','Not actual preserved failed3272 terminal')
    result=json.loads(payload['files'][str(Path(payload['run'])/'run-result.json')]['content'])
    require(result['status']=='FAILED' and result['job_id']=='3272' and result['source_commit']==SOURCE
            and result['binding_sha256']==review['binding_sha256']
            and result['restore_checkpoint']==CHECKPOINT and result['restore_manifest_sha256']==MANIFEST
            and result['retained_training_updates_contributed']==0
            and result['cleanup_residual_states']=={} and result['final_gpu']['processes']==[],
            'Failed3272 original result contradicts restore/source/cleanup review')
    return originals


def seal_installer(candidate, sources, archive):
    """Seal only our new local unsealed installer, after immutable draft/package exist."""
    path=HERE.with_name('safety-calibration-install-v6')/'install_calibration_v6.py'
    source=path.read_text()
    manifest_record=candidate['startup_control_proofs']['runtime_manifest']
    constants=dict(PACKAGE_SHA=sha(archive),CANDIDATE_SHA=sha(OUT/'CALIBRATION-CANDIDATE-52BEFE9-v6.json'),
        IMPLEMENTATION_SHA=manifest_record['sha256'],RUNTIME_MANIFEST_NAME=Path(manifest_record['path']).name,
        EVIDENCE_NAMES={name.split('/',1)[1] for name in sources if name.startswith('evidence/')},
        EVIDENCE_COUNT=len(candidate['evidence']),REQUIRED_EXISTING_COUNT=len(candidate['evidence'])-
            len([name for name in sources if name.startswith('evidence/')]))
    for name,value in constants.items():
        marker=name+' = None';require(source.count(marker)==1,'Installer was already sealed or code changed: '+name)
        source=source.replace(marker,name+' = '+repr(value),1)
    ast.parse(source)
    path.write_text(source)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),status='LOCAL_SEALED_NOT_DEPLOYED')


def main(proof_inputs_path, proof_inputs_sha):
    target=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v6.json';archive=HERE/'policy-safety-calibration-52befe9-v6-package.tar.gz'
    receipt_path=OUT/'CALIBRATION-PACKAGE-v6.json'
    require(not any(path.exists() for path in (target,archive,receipt_path)), 'Draft already/partly exists; inspect and preserve, never rebuild')
    spec_path=Path(proof_inputs_path)
    require(spec_path.is_absolute() and spec_path.resolve()==spec_path and spec_path.is_relative_to(OUT)
            and spec_path.is_file() and not spec_path.is_symlink()
            and sha(spec_path)==proof_inputs_sha,'Explicit dedicated proof input SHA/path changed')
    spec=json.loads(spec_path.read_text());proofs=spec['proofs']
    require(spec['schema']==1 and spec['code_commit']==SOURCE and spec['runtime_version']=='v6-v3'
            and set(proofs)==set(STARTUP_PROOF_ROLES),'Wrong/missing final proof role specification')
    require(len({record['path'] for record in proofs.values()})==len(proofs),'Duplicate explicit proof role original')
    runtime,proof_paths=validate_runtime_inputs(proofs)
    failure_originals=validate_failure_inputs(proofs,proof_paths)
    native_path=OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json';predecessor_path=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v5.json'
    require(sha(native_path)==NATIVE_SHA and sha(predecessor_path)==PREDECESSOR_SHA,'Frozen predecessor/native candidate changed')
    native=json.loads(native_path.read_text());predecessor=json.loads(predecessor_path.read_text())
    require(native['code_commit']==SOURCE and native['helpers']==str(NATIVE_HELPERS) and len(native['files'])==42327,
            'Wrong inherited native source/assets closure')
    candidate=copy.deepcopy(predecessor)
    candidate.update(helpers=str(HELPERS),controls=str(CONTROLS),source_native_acceptance=None,
        native_pending_job_acceptance=None,predecessor_calibration_candidate_sha256=PREDECESSOR_SHA,
        preparation_scope='Final v6-v3 startup cooldown only; probability runtime12 unchanged; actual native/physical/READY still separate',
        startup_change_scope=STARTUP_SCOPE,runtime_changed_filenames=['calibration_run_v1.py'],
        native_directory_modified=False)
    old={record['path']:record for record in predecessor['files']};files={}
    for record in native['files']:
        observed=old[record['path']]
        require(observed['sha256']==record['sha256'] and type(observed['bytes']) is int,'Inherited native path/SHA/bytes differ')
        files[record['path']]=copy.deepcopy(observed)
    sources={};runtime_records=[]
    for record in runtime['files']:
        local=Path(record['path']);remote=str(HELPERS/local.name)
        require(remote not in files,'Runtime targets inherited native path')
        entry=dict(path=remote,bytes=record['bytes'],sha256=record['sha256']);files[remote]=entry;runtime_records.append(entry)
        sources['runtime/'+local.name]=local
    require({path.name for path in HERE.glob('*.py') if not path.name.startswith('test_')}==CONTROL_NAMES,'Not exact five versioned controllers')
    for name in sorted(CONTROL_NAMES):
        local=HERE/name;remote=str(CONTROLS/name);files[remote]=dict(path=remote,bytes=local.stat().st_size,sha256=sha(local));sources['control/'+name]=local
    require(len(files)==42345,'Full file count differs from native42327 plus13+5')
    candidate['files']=list(files.values());candidate['runtime_helpers']=runtime_records
    native_evidence={record['path'] for record in native['evidence']}
    for record in predecessor['evidence']:
        if record['path'] not in native_evidence:
            local=OUT/Path(record['path']).name
            require(local.stat().st_size==record['bytes'] and sha(local)==record['sha256'],'Original calibration evidence changed')
            sources['evidence/'+local.name]=local
    evidence={record['path']:record for record in candidate['evidence']}
    def add_local(local):
        local=Path(local);record=dict(path=str(ROOT/'logs'/local.name),bytes=local.stat().st_size,sha256=sha(local))
        require(record['path'] not in evidence,'New proof would overwrite an old evidence path')
        evidence[record['path']]=record;sources['evidence/'+local.name]=local
        return record
    candidate['startup_control_proofs']={role:add_local(path) for role,path in proof_paths.items()}
    candidate['startup_proof_input_spec']=add_local(spec_path)
    for local in (OUT/'CALIBRATION-DEPLOYMENT-v6.md',OUT/'CALIBRATION-DEPLOYMENT-CPU-v6.log'):add_local(local)
    for record in failure_originals:
        require(record['path'] not in evidence or evidence[record['path']]==record,'Conflicting preserved failure original')
        evidence[record['path']]=record
    candidate['failure3272_originals']=failure_originals;candidate['evidence']=list(evidence.values())
    preview=json.loads((OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json').read_text())
    require(preview_config(candidate)==preview['resolved'],'New startup/runtime paths altered reviewed optimization/config')
    validate_candidate(candidate)
    write_once(target,candidate);sources['candidate/'+target.name]=target;sources['cpu/test_deployment_v6.py']=HERE/'test_deployment_v6.py'
    make_archive(archive,sources);installer=seal_installer(candidate,sources,archive)
    receipt=dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',candidate=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),installer=installer,
        proof_inputs=dict(path=str(spec_path),sha256=proof_inputs_sha),predecessor_candidate_sha256=PREDECESSOR_SHA,
        native_candidate_sha256=NATIVE_SHA,native_acceptance=None,submitted_job=None,retained_training_updates_contributed=0,
        runtime_helpers_path=str(HELPERS),controls_path=str(CONTROLS),startup_change_scope=STARTUP_SCOPE,
        runtime_changed_filenames=['calibration_run_v1.py'],unchanged_runtime_files=12,
        inherited_native_42327_paths_sha_unchanged=True,native_directory_modified=False,actual_gpu_executed=False,
        file_count=len(candidate['files']),evidence_count=len(candidate['evidence']),package_members=sorted(sources))
    write_once(receipt_path,receipt)
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),installer_sha256=installer['sha256'],
        files=len(files),evidence=len(evidence),ready=False,submitted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-inputs',required=True);parser.add_argument('--proof-inputs-sha',required=True)
    args=parser.parse_args();main(args.proof_inputs,args.proof_inputs_sha)
