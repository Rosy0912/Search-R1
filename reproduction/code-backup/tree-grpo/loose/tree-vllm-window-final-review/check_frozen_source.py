"""Independent local metadata/source review. No remote, assets or CUDA work."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[1]
REVIEW = Path(__file__).resolve().parent
PACKAGE = WORK / 'single-tp1-window-3d8c6f6d-466df7c2'
MANIFEST_SHA = 'f3e43a57f7164cc987d1ad211d617126d7bc5ece5450e4e13d365f405f15d6e1'
PAYLOAD_SHA = 'b7f47ee839aa655f7af60beb279e494861f50a701395552ee9c0e4324a42953f'
COMMIT = '3d8c6f6d6cd82c23c4f2092f78a17d37126bc4a0'
CORE_SHA = '46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)

def production_functions(path, names):
    tree = ast.parse(Path(path).read_bytes())
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            result[node.name] = ast.dump(node, include_attributes=False)
    assert set(result) == set(names)
    return result

assert sha(PACKAGE / 'package-manifest.json') == MANIFEST_SHA
m = unique_json((PACKAGE / 'package-manifest.json').read_bytes())
b = unique_json((PACKAGE / 'bundle-template.json').read_bytes())
assert m['runner_commit'] == b['runner_commit'] == COMMIT
assert len(m['files']) == 745
actual = set()
for file in PACKAGE.rglob('*'):
    assert not file.is_symlink()
    assert '__pycache__' not in file.parts
    if file.is_file() and file.name != 'package-manifest.json':
        relative = file.relative_to(PACKAGE).as_posix()
        actual.add(relative)
        assert file.suffix not in {'.pt', '.safetensors', '.parquet', '.bin', '.so', '.pyc', '.pyo'}
        assert sha(file) == m['files'][relative], relative
assert actual == set(m['files'])

runner = WORK / 'tree-vllm-window-runner'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runner, text=True).strip() == COMMIT
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=runner, text=True) == ''
for relative in m['files']:
    if relative.startswith('repo/'):
        assert sha(runner / relative[5:]) == m['files'][relative]

sys.path.insert(0, str(PACKAGE))
import probe_package as P
assert P.digest(P.bundle_payload(b)) == PAYLOAD_SHA == m['bundle_payload_sha256']
assert P.digest(P.validate_window_inputs(b['window_inputs'])) == m['window_inputs_sha256'] == b['window_inputs_sha256']
assert b['window_inputs']['actual_tensor_rows_verified'] is False
assert b['window_inputs']['CPU_job_executed_by_this_plan'] is False
assert sha(PACKAGE / 'audits/metadata-only-input-plan.json') == sha(runner / 'WINDOW_INPUTS_PLAN.json')
assert unique_json((PACKAGE / 'audits/metadata-only-input-plan.json').read_bytes()) == b['window_inputs']
assert sha(PACKAGE / 'repo/vllm_window_weights.py') == b['weights_core']['sha256'] == CORE_SHA
assert len(b['vllm_installed_sources']) == 29 and len(b['torch_wrapper_sources']) == 1
assert b['installed_kernel_binaries'] == m['installed_kernel_binaries'] == P.KERNEL_BINARY_PINS
assert len(b['probe_source_files']) == 733
remote = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/' + PACKAGE.name
assert {e['path']:e['sha256'] for e in b['probe_source_files']} == {
    remote + '/' + relative:wanted for relative,wanted in m['files'].items() if relative.startswith('repo/')}
assert b['native_method_sources']['facade']['sha256'] == m['files']['repo/verl/third_party/vllm/vllm_v_0_6_3/llm.py']
assert b['native_method_sources']['engine_add_request'] == b['vllm_installed_sources']['engine/llm_engine.py']
assert b['native_method_sources']['model_runner'] == b['vllm_installed_sources']['worker/model_runner.py']

# Known resource functions are unchanged; new profile binding is reviewed above.
adapter_names = ['fdinfo_exclusive_owner', 'allocation', 'guard_record', 'check', 'process', 'installed_bindings']
assert production_functions(PACKAGE / 'resource_adapter_window.py', adapter_names) == production_functions(
    WORK / 'tree-probability-probe-owned/resource_adapter_v2.py', adapter_names)
supervisor_names = ['compiler_header_environment', 'wait_probe', 'process', 'protected', 'healthy_guard', 'owned_adapter']
assert production_functions(PACKAGE / 'supervisor.py', supervisor_names) == production_functions(
    WORK / 'tree-probability-probe-owned/supervisor-candidate-r3/supervisor.py', supervisor_names)
for filename in ('start.sh', 'start.py'):
    assert sha(PACKAGE / filename) == sha(WORK / 'tree-probability-probe-owned/supervisor-candidate-r3' / filename)

native_entries = {
    'verl/workers/actor/dp_actor.py', 'verl/workers/sharding_manager/fsdp_vllm.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'verl/third_party/vllm/vllm_v_0_6_3/llm.py',
    'verl/utils/torch_functional.py', 'verl/utils/fsdp_utils.py', 'search_r1/baselines/preflight.py'}
prior = unique_json((WORK / 'probability-probe-49a6b35-29822f9-r3/package-manifest.json').read_bytes())
for filename in native_entries:
    assert m['files']['repo/' + filename] == prior['files']['repo/' + filename]
for filename in ('single_window_runner.py','window_runtime.py','window_observer.py','prepare_window_inputs.py','window_contracts.py'):
    tree = ast.parse((PACKAGE / 'repo' / filename).read_bytes())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                   {'backward','step','Adam','AdamW','update_actor'} for n in ast.walk(tree)), filename

helpers = {
    'build_tree_vllm_window_package.py':'28790518c6a674383edce5f0dbf7386e097367ad94d960e67ef600c351f29174',
    'upload_tree_vllm_window_source.py':'d53586818a39e79884a39c4f79a8fb97f204ce0d82adc177b492ecdce37abe96'}
for filename, wanted in helpers.items():
    assert sha(WORK / filename) == wanted

release = dict(status=P.RELEASE_STATUS, schema_version=1,
    scope='INDEPENDENT_LOCAL_SOURCE_AND_PRE_DISPATCH_REVIEW_ONLY',
    reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    reviewer_task='/root/tree_window_final_review', reviewer_production_source_participation=False,
    package_manifest_sha256=MANIFEST_SHA, bundle_payload_sha256=PAYLOAD_SHA,
    runner_commit=COMMIT, selection_sha256=P.SELECTION_SHA,
    adapter_sha256=b['reviewed_resource_adapter']['sha256'], protocol=P.PROTOCOL,
    diagnostic_profile=P.WINDOW_PROFILE, window_contract=P.WINDOW_CONTRACT,
    runner_module=P.RUNNER_MODULE, window_inputs_sha256=m['window_inputs_sha256'],
    installed_kernel_binaries=P.KERNEL_BINARY_PINS, dispatch_intent_path=P.DISPATCH_INTENT,
    actual_optimizer_writes=0,
    actual_optimizer_writes_scope='Frozen zero-write source contract; no actual job execution is asserted',
    runtime_actual_optimizer_writes=None, actual_gpu_execution_verified=False,
    actual_remote_source_bytes_verified=False, probability_accepted=False, training_accepted=False,
    GPU_capacity_proven=False, actual_loaded_binary_identity_proven=False,
    historical_3268_sampler_identity_proven=False, real_assets_read_by_reviewer=False,
    conditional_execution_limit='Once only after root verifies exact same h100-9 source bytes, current queue and actual preflight/leases/guard; no production training')
assert P.validate_release(m, MANIFEST_SHA, b, release) is True

# Six precise substituted-envelope refusals, using metadata only.
counterexamples = []
mutations = [
    ('old native profile', lambda x:x.update(diagnostic_profile='native_original_fsdp')),
    ('old runner entry', lambda x:x.update(runner_module='search_r1.baselines.probability_probe_runner')),
    ('different inputs payload', lambda x:x.update(window_inputs_sha256='a'*64)),
    ('another intent', lambda x:x.update(dispatch_intent_path=str(P.ROOT / 'instruct300/HEADER_REPAIR_ONCE-3280.json'))),
    ('changed generation budget', lambda x:x['window_contract'].update(generation_invocations_per_rank=2)),
    ('different kernel metadata', lambda x:x['installed_kernel_binaries'][0].update(sha256='a'*64)),
]
for label, mutate in mutations:
    candidate = copy.deepcopy(release); mutate(candidate)
    try:
        P.validate_release(m, MANIFEST_SHA, b, candidate)
    except ValueError as error:
        counterexamples.append(dict(label=label, refused=True, reason=str(error)))
    else:
        raise AssertionError('Unsafe metadata substitution accepted: ' + label)

for filename, count in [('RUNNER_CPU.stderr',42),('LAUNCHER_CPU.stderr',94)]:
    output = (REVIEW / filename).read_text()
    assert 'Ran '+str(count)+' tests' in output and output.rstrip().endswith('OK')

proof = dict(status='FROZEN_SINGLE_WINDOW_SOURCE_METADATA_AND_PRE_DISPATCH_CONTRACT_CHECKS_PASSED',
    manifest_sha256=MANIFEST_SHA,bundle_payload_sha256=PAYLOAD_SHA,runner_commit=COMMIT,
    immutable_local_files_verified=745,repo_source_files_verified=733,
    unchanged_adapter_resource_functions=adapter_names,unchanged_supervisor_resource_functions=supervisor_names,
    unchanged_native_files=sorted(native_entries),helpers_sha256=helpers,
    metadata_counterexamples=counterexamples,source_core_sha256=CORE_SHA,
    local_cpu_runner_tests=42,local_cpu_launcher_tests=94,
    previous_core_44_tests_rerun=False,remote_access=False,actual_tensor_asset_read=False,
    actual_gpu_job_execution=False,actual_cuda_initialized=False,real_collectives_executed=False,
    actual_model_forwards=0,actual_optimizer_writes=None,planned_optimizer_writes=0,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,
    log_sha256={name:sha(REVIEW/name) for name in
                ('RUNNER_CPU.stdout','RUNNER_CPU.stderr','LAUNCHER_CPU.stdout','LAUNCHER_CPU.stderr')})

for filename, value in [('FROZEN_SOURCE_CPU_PROOF.json',proof),('RELEASE.json',release)]:
    path = REVIEW / filename
    assert not path.exists()
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(proof_sha256=sha(REVIEW/'FROZEN_SOURCE_CPU_PROOF.json'),
                      release_sha256=sha(REVIEW/'RELEASE.json'),checks_passed=True),sort_keys=True))
