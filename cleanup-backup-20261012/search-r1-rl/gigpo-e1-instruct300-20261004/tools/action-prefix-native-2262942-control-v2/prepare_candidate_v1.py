"""Local immutable native candidate/package preparation; never target or GPU."""
import copy
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from deployment_contract_v1 import (ROOT, SOURCE, REPO, HELPERS, CONTROLS, SCOPE, REVISION,
    MODEL, RESTORE, DRIVER_SHA, BASE_CANDIDATE_SHA, BASE_PHYSICAL_SHA, CLONE, CLONE_SHA,
    NAMES, HELPER_NAMES, require, sha, closure, match_closure, validate_candidate, validate_source_review, validate_proof_input, write_once)


def item(local, remote):
    local = Path(local)
    require(local.is_file() and not local.is_symlink(), 'Nonregular local input')
    return dict(path=str(remote), bytes=local.stat().st_size, sha256=sha(local))


def prepare(parent_path, physical_path, source_path, clone_wrapper, runtime_directory,
            proof_spec, output, archive):
    require(sha(parent_path) == BASE_CANDIDATE_SHA and sha(physical_path) == BASE_PHYSICAL_SHA,
            'Changed inherited parent')
    parent = json.loads(Path(parent_path).read_text())
    clone = json.loads(Path(clone_wrapper).read_text())
    require(clone['receipt'] == dict(path=str(CLONE), bytes=128687, sha256=CLONE_SHA), 'Wrong actual clone receipt')
    actual = clone['actual']
    require(actual['status'] == 'ACTUAL_ISOLATED_SOURCE_FULL_SHA_PASS_NOT_DEPLOYMENT_READY'
            and actual['source_commit'] == SOURCE and actual['repo'] == str(REPO)
            and actual['clean'] is True and actual['source_file_count'] == 480
            and actual['gpu_used'] is False and actual['optimizer_writes'] == 0, 'Source clone not actual clean CPU identity')
    source = Path(source_path).resolve()
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
            and not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
            'Local source changed')
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')[:-1]
    source_files = [item(source / name, REPO / name) for name in names]
    require(len(source_files) == 480 and closure(source_files, 'source') == closure(actual['files'], 'clone source'),
            'Actual clone differs from local full tracked source')
    runtime = Path(runtime_directory).resolve()
    require({p.name for p in runtime.iterdir() if p.is_file()} == set(HELPER_NAMES), 'Unexpected native helper file')
    controls = Path(__file__).resolve().parent
    new_files = source_files + [item(runtime / n, HELPERS / n) for n in HELPER_NAMES]
    new_files += [item(controls / n, CONTROLS / n) for n in NAMES]
    remove = (parent['repo'], parent['helpers'], parent['controls'])
    files = [copy.deepcopy(x) for x in parent['files'] if not any(x['path'].startswith(p + '/') for p in remove)] + new_files
    proof_input = json.loads(Path(proof_spec).read_text())
    validate_proof_input(proof_input)
    roles = proof_input['roles']
    require(set(roles) == {'source_implementation', 'source_review', 'native_helper_manifest', 'native_helper_review'}, 'Proof spec roles incomplete')
    evidence = copy.deepcopy(parent['evidence'])
    parent_record = item(parent_path, ROOT / 'logs/CALIBRATION-CANDIDATE-52BEFE9-v9.json')
    physical_record = item(physical_path, ROOT / 'logs/policy-safety-calibration-52befe9-physical-v9.json')
    extras = [parent_record, physical_record, clone['receipt']]
    normalized_roles = {}
    for role, spec in roles.items():
        record = item(spec['local_path'], spec['path'])
        require(record['bytes'] == spec['bytes'] and record['sha256'] == spec['sha256'], 'Proof spec changed: ' + role)
        normalized_roles[role] = record
        extras.append(record)
    for spec in proof_input['additional_evidence']:
        require(str(spec['path']).startswith(str(ROOT / 'logs') + '/')
                and '..' not in Path(spec['path']).parts, 'New proof outside isolated logs')
        record = item(spec['local_path'], spec['path'])
        require(record['bytes'] == spec['bytes'] and record['sha256'] == spec['sha256'], 'Dependency proof changed')
        extras.append(record)
    closure(extras, 'new proof input closure')
    extras.append(item(proof_spec, ROOT / 'logs/ACTION-PREFIX-NATIVE-PROOF-INPUTS-v2.json'))
    helper_manifest = json.loads(Path(roles['native_helper_manifest']['local_path']).read_text())
    require(helper_manifest['source_commit'] == SOURCE and helper_manifest['gpu_used'] is False
            and helper_manifest['actual_gpu_result'] is None, 'Native draft scope inflated')
    manifest_helper_files = [dict(path=str(HELPERS / Path(entry['path']).name), bytes=entry['bytes'], sha256=entry['sha256'])
        for entry in helper_manifest['new_or_changed_files'] + helper_manifest['inherited_unchanged_files']]
    match_closure([record for record in new_files if record['path'].startswith(str(HELPERS) + '/')],
                  manifest_helper_files, 'actual new helper vs independently reviewed manifest')
    source_review = json.loads(Path(roles['source_review']['local_path']).read_text())
    helper_review = json.loads(Path(roles['native_helper_review']['local_path']).read_text())
    validate_source_review(source_review, normalized_roles['source_implementation']['sha256'])
    require(helper_review['status'] == 'PASS', 'New independent native helper static review missing')
    require(helper_review['native_scope'] == SCOPE and helper_review['implementation_sha256'] == normalized_roles['native_helper_manifest']['sha256'],
            'Helper review not this manifest')
    require(helper_review['actual_gpu_acceptance'] is False and helper_review['deployment_ready'] is False,
            'Helper static review inflated to GPU/deployment acceptance')
    for record in extras:
        same = [x for x in evidence if x['path'] == record['path']]
        require(not same or same == [record], 'Historical proof must not be overwritten')
        if not same:
            evidence.append(record)
    candidate = dict(status='CANDIDATE_NOT_READY', code_commit=SOURCE, repo=str(REPO), helpers=str(HELPERS),
        controls=str(CONTROLS), native_scope=SCOPE, validation_scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY',
        native_initialization='OFFICIAL_MODEL_INITIALIZATION', native_actual_restore=False,
        formal_updates_contributed=0, official_revision=REVISION, official_model_path=str(MODEL),
        native_config_driver=dict(path=str(RESTORE / 'training_state/driver.pt'), bytes=742936, sha256=DRIVER_SHA),
        restore_step=25, restore_checkpoint=str(RESTORE),
        restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
        restore_identity_scope='FUTURE_CALIBRATION_ORIGIN_AND_CONFIG_ONLY_NOT_NATIVE_RECOVERY',
        allowed_safety_modes=['native_fixture'], native_test_threshold_sets=[[100., 1.]],
        expected_optimizer_steps_per_outer=10, expected_gradient_micro_batches_per_rank=160,
        global_row_denominator=256, production_denominator_per_mini=32, gpu_count=8, node='h100-8',
        native_submitted=False, training_submitted=False, production_probability_tolerance=None,
        production_thresholds=None, source_native_acceptance=None, files=files, evidence=evidence,
        resource_guard=dict(path=str(HELPERS / 'resource_guard_v2.py'), sha256='9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
        resource_ownership=dict(path=str(HELPERS / 'resource_ownership_v2.py'), sha256='7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'),
        source_file_count=480, source_actual_clone=clone['receipt'], parent_candidate=parent_record,
        inherited_physical=physical_record, proof_roles=normalized_roles)
    validate_candidate(candidate)
    require(not Path(output).exists() and not Path(archive).exists(), 'Immutable output already exists')
    write_once(output, candidate)
    with tarfile.open(archive, 'x:gz') as package:
        for prefix, directory, selected in [('runtime', runtime, HELPER_NAMES), ('controls', controls, NAMES)]:
            for name in selected:
                package.add(directory / name, arcname=prefix + '/' + name, recursive=False)
    return dict(candidate=str(output), candidate_sha256=sha(output), package=str(archive), package_sha256=sha(archive),
                files=len(files), evidence=len(evidence), target_installed=False, gpu_used=False, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 9, 'Need parent/physical/local source/actual clone wrapper/runtime/proofspec/output/archive')
    print(json.dumps(prepare(*sys.argv[1:])))
