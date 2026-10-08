"""Native-only immutable physical/READY/once contract; no GPU imports."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SOURCE = '2262942142f9df8f1666ce3f1b9ebfe9c9239241'
REPO = ROOT / 'repo-raw-action-2262942'
ENV_PYTHON = ROOT / 'env/bin/python'
HELPERS = ROOT / 'tools/action-prefix-native-2262942-v2'
CONTROLS = ROOT / 'tools/action-prefix-native-2262942-control-v2'
SCOPE = 'RAW_ACTION_PREFIX_PACKING_OFFICIAL_3B_NATIVE_ONLY'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MODEL = ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'
RESTORE = ROOT / 'runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25'
DRIVER_SHA = 'ed14df80cd34e4cdf2d06bbfec71927f4c6127300002f787bdf9f879820be258'
BASE_CANDIDATE_SHA = '264fc8ebdf5f2c8d81c762a744a4ae0d25b39d8a4efdcd137c727e12c648e045'
BASE_PHYSICAL_SHA = '02c96a172561257afc96dcb2875fb69c51c8e63a9fd9017830b710a766b9f6b5'
CLONE_SHA = '6e044c4e9265c3bf01c69751432558b62157131ca6cf150bcaf8390fd6f2cd59'
CLONE = ROOT / 'logs/RAW-ACTION-SOURCE-ACTUAL-CLONE-2262942-v1.json'
NAMES = ('deployment_contract_v1.py', 'prepare_candidate_v1.py', 'verify_physical_v1.py',
         'freeze_ready_v1.py', 'submit_once_v1.py')
HELPER_NAMES = ('action_prefix_native_control_v1.py', 'native_action_prefix_gpu_v1.py',
                'native_run_safety_v1.py', 'native_safety_v1.sbatch',
                'test_action_prefix_native_control_v1.py', 'native_guard_binding_v2.py',
                'owned_runtime_resume_v1.py', 'resource_guard_v2.py', 'resource_ownership_v2.py',
                'safety_audit_v1.py', 'test_native_guard_binding_v2.py', 'test_resource_ownership_v2.py')
PREFIX = ROOT / 'logs/ACTION-PREFIX-NATIVE'
CANDIDATE = Path(str(PREFIX) + '-CANDIDATE-2262942-v2.json')
PHYSICAL = Path(str(PREFIX) + '-PHYSICAL-2262942-v2.json')
READY = Path(str(PREFIX) + '-READY-2262942-v2.json')
BINDING = Path(str(PREFIX) + '-BINDING-2262942-v2.json')
INTENT = Path(str(PREFIX) + '-SUBMIT-INTENT-2262942-v2.json')
RECEIPT = Path(str(PREFIX) + '-SUBMIT-RECEIPT-2262942-v2.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1048576)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def write_once(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_bound(record):
    require(sha(record['path']) == record['sha256'], 'Bound file changed: ' + record['path'])
    if 'bytes' in record:
        require(Path(record['path']).stat().st_size == record['bytes'], 'Bound bytes changed')
    return json.loads(Path(record['path']).read_text())


def closure(records, label):
    require(isinstance(records, list) and records, 'Missing ' + label)
    paths = set()
    for item in records:
        require(isinstance(item, dict) and set(item) >= {'path', 'bytes', 'sha256'}, 'Incomplete ' + label)
        path = item['path']
        require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts,
                'Invalid closure path')
        require(path not in paths, 'Duplicate closure path: ' + path)
        require(type(item['bytes']) is int and item['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', str(item['sha256'])), 'Invalid closure bytes/SHA')
        paths.add(path)
    return {item['path']: item for item in records}


def validate_source_review(review, implementation_sha):
    require(review.get('status') == 'PASS_SOURCE_AND_EXACT_CPU_EVIDENCE_SCOPE_ONLY'
            and review.get('scope') == 'RAW_SAMPLED_ACTION_IDENTITY_SOURCE_STATIC_REVIEW_NOT_GPU_OR_PROBABILITY_OR_DEPLOYMENT_READY'
            and review.get('source_commit') == SOURCE
            and review.get('source_manifest_sha256') == implementation_sha,
            'Source static review identity/scope differs')
    for key in ('gpu_used', 'deployment_ready', 'actual_gpu_path_accepted', 'probability_identity_accepted',
                'production_thresholds_selected'):
        require(review.get(key) is False, 'Static source review scope inflated: ' + key)
    return True


def validate_proof_input(value):
    require(isinstance(value, dict) and set(value) == {'roles', 'additional_evidence'}, 'Explicit proof input closure missing')
    require(set(value['roles']) == {'source_implementation', 'source_review', 'native_helper_manifest', 'native_helper_review'},
            'Proof input roles incomplete')
    require(isinstance(value['additional_evidence'], list) and value['additional_evidence'], 'Dependency proof list missing')
    all_specs = list(value['roles'].values()) + value['additional_evidence']
    for spec in all_specs:
        require(isinstance(spec, dict) and set(spec) == {'local_path', 'path', 'bytes', 'sha256'}, 'Incomplete proof input record')
        require(isinstance(spec['local_path'], str) and Path(spec['local_path']).is_absolute(), 'Local proof input must be absolute')
        require(str(spec['path']).startswith(str(ROOT / 'logs') + '/') and '..' not in Path(spec['path']).parts,
                'New proof outside isolated logs')
    closure([{key: spec[key] for key in ('path', 'bytes', 'sha256')} for spec in all_specs], 'proof input dependency closure')
    return True


def validate_candidate(value):
    require(value.get('status') == 'CANDIDATE_NOT_READY', 'Candidate status differs')
    require(value.get('code_commit') == SOURCE and value.get('repo') == str(REPO), 'Wrong source')
    require(value.get('helpers') == str(HELPERS) and value.get('controls') == str(CONTROLS), 'Wrong isolated directories')
    require(value.get('native_scope') == SCOPE and value.get('validation_scope') == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY',
            'Wrong native-only scope')
    require(value.get('native_initialization') == 'OFFICIAL_MODEL_INITIALIZATION'
            and value.get('native_actual_restore') is False and value.get('formal_updates_contributed') == 0,
            'Fresh native inflated to healthy restore/training')
    require(value.get('official_revision') == REVISION and value.get('official_model_path') == str(MODEL), 'Wrong official model')
    require(value.get('native_config_driver') == dict(path=str(RESTORE / 'training_state/driver.pt'),
            bytes=742936, sha256=DRIVER_SHA), 'Configuration-only driver differs')
    require(value.get('restore_step') == 25 and value.get('restore_checkpoint') == str(RESTORE)
            and value.get('restore_manifest_sha256') == '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
            'Future calibration origin differs')
    require(value.get('allowed_safety_modes') == ['native_fixture'] and value.get('native_test_threshold_sets') == [[100., 1.]],
            'Production or altered fixture mode')
    for name, expected in [('expected_optimizer_steps_per_outer', 10), ('expected_gradient_micro_batches_per_rank', 160),
            ('global_row_denominator', 256), ('production_denominator_per_mini', 32), ('gpu_count', 8), ('node', 'h100-8')]:
        require(value.get(name) == expected, 'Changed native budget/node: ' + name)
    require(value.get('native_submitted') is False and value.get('training_submitted') is False,
            'Candidate contains submission claim')
    require(value.get('production_probability_tolerance') is None and value.get('production_thresholds') is None
            and value.get('source_native_acceptance') is None, 'Native cannot borrow old acceptance or thresholds')
    files = closure(value.get('files'), 'files')
    evidence = closure(value.get('evidence'), 'evidence')
    for path in files.keys() & evidence.keys():
        require({key: files[path][key] for key in ('path', 'bytes', 'sha256')} ==
                {key: evidence[path][key] for key in ('path', 'bytes', 'sha256')},
                'Conflicting cross-list closure identity: ' + path)
    for directory, names in [(HELPERS, HELPER_NAMES), (CONTROLS, NAMES)]:
        require({p for p in files if p.startswith(str(directory) + '/')} == {str(directory / n) for n in names},
                'Helper/control full closure differs')
    source = [item for path, item in files.items() if path.startswith(str(REPO) + '/')]
    require(len(source) == value.get('source_file_count') == 480, 'Incomplete tracked source closure')
    require(str(RESTORE / 'training_state/driver.pt') in files
            and files[str(RESTORE / 'training_state/driver.pt')]['sha256'] == DRIVER_SHA, 'Unbound config driver')
    for key, filename, digest in [('resource_guard', 'resource_guard_v2.py',
            '9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
            ('resource_ownership', 'resource_ownership_v2.py',
            '7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed')]:
        require(value.get(key) == dict(path=str(HELPERS / filename), sha256=digest), 'Guard changed')
        require(files[str(HELPERS / filename)]['sha256'] == digest, 'Guard missing from full closure')
    require(value.get('source_actual_clone') == dict(path=str(CLONE), bytes=128687, sha256=CLONE_SHA), 'Actual clone identity differs')
    require(evidence.get(str(CLONE)) == value['source_actual_clone'], 'Clone not in evidence closure')
    roles = value.get('proof_roles', {})
    require(set(roles) == {'source_implementation', 'source_review', 'native_helper_manifest', 'native_helper_review'},
            'Native/source proof roles incomplete')
    require(all(evidence.get(record['path']) == record for record in roles.values()), 'Proof roles unbound')
    require(value['parent_candidate']['sha256'] == BASE_CANDIDATE_SHA
            and value['inherited_physical']['sha256'] == BASE_PHYSICAL_SHA, 'Inherited parent identity changed')
    return True


def match_closure(actual, expected, label):
    observed = closure(actual, label)
    wanted = closure(expected, label)
    require(set(observed) == set(wanted), 'Actual closure incomplete: ' + label)
    for path, item in wanted.items():
        require({key: observed[path][key] for key in ('path', 'bytes', 'sha256')} == item,
                'Actual candidate bytes/SHA differs: ' + path)


def validate_physical(value, candidate, candidate_sha):
    require(value.get('status') == 'PHYSICAL_NATIVE_SOURCE_HELPER_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU'
            and value.get('candidate_sha256') == candidate_sha and value.get('code_commit') == SOURCE,
            'Physical proof is not this candidate')
    require(value.get('node') == 'h100-8' and value.get('cuda_initialized') is False
            and value.get('torch') == '2.4.0+cu121' and value.get('executable') == str(ENV_PYTHON), 'Physical environment differs')
    require(value.get('source_clean') is True and value.get('source_file_count') == 480, 'Source physical closure incomplete')
    match_closure(value['metadata'], candidate['files'], 'files')
    match_closure(value['evidence'], candidate['evidence'], 'evidence')
    expected_imports = dict(action_prefix_module='search_r1/baselines/action_prefix_forward.py',
        actor_module='verl/workers/actor/dp_actor.py', recovery_module='search_r1/baselines/recovery.py',
        policy_safety_module='search_r1/baselines/policy_safety.py')
    require(value['imports'] == {role: str(REPO / path) for role, path in expected_imports.items()},
            'Actual imported source role/path differs')
    return True


def validate_ready(value):
    require(value.get('status') == 'REVIEWED_FOR_NATIVE_ONLY'
            and value.get('readiness') == 'READY_FOR_OFFICIAL3B_RAW_PREFIX_NATIVE_ONLY', 'No native READY')
    candidate = read_bound(value['candidate_record'])
    validate_candidate(candidate)
    physical = read_bound(value['physical_record'])
    validate_physical(physical, candidate, value['candidate_record']['sha256'])
    review = read_bound(value['deployment_review'])
    require(review['status'] == 'PASS' and review['scope'] == SCOPE
            and review['code_commit'] == SOURCE
            and review['candidate_sha256'] == value['candidate_record']['sha256']
            and review['physical_sha256'] == value['physical_record']['sha256'], 'Deployment review identity differs')
    for key in candidate:
        require(value.get(key) == candidate[key] or key == 'status', 'READY candidate field changed: ' + key)
    require(value.get('actual_gpu_acceptance') is False, 'READY cannot claim GPU result')
    require(review['freeze_helper_sha256'] == sha(CONTROLS / 'freeze_ready_v1.py')
            and review['submit_helper_sha256'] == sha(CONTROLS / 'submit_once_v1.py'), 'Reviewed controls changed')
    return candidate
