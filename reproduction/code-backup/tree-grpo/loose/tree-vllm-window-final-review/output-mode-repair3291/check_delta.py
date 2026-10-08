"""Bound output-mode admission source delta; CPU metadata only, no assets."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
WORK = REVIEW.parents[1]
OLD = WORK/'single-tp1-window-3583419c-466df7c2'
REJECTED = WORK/'single-tp1-window-4ee0de33-466df7c2'
NEW = WORK/'single-tp1-window-292ed2f9-466df7c2'
COMMIT = '292ed2f97d69c221aa4edf625163df9b3d18d196'
MANIFEST = 'c400e691fe340d10ea11f83e2feaaae647063a171077fe1db95f5dbd448dec9d'
PAYLOAD = '0d1348a8cea3b3ec3da60f44d07e07bc55926213e945f9f25a61db3ef9fce883'
PINS = {'entrypoints/llm.py': '1393441b1b5cf4750b2e3a72ff30272a7e0db2220fa23ffa459f135265686582',
        'sampling_params.py': 'f379042fe14ca7adb2a74eeb1b746de4583481c2a10416f7fe1b05aba81952bc'}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write_new(name, value):
    path = REVIEW/name
    assert not path.exists(), 'Independent receipts never replace prior evidence'
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')

assert sha(NEW/'package-manifest.json') == MANIFEST
assert sha(OLD/'package-manifest.json') == '20d42ab8f48ab390af6088990661d09f408d30cccb2946b407da007b8da8d48e'
assert sha(REJECTED/'package-manifest.json') == '6084d45ce746bf16735cc27ff00f4de3620c92156466a941d940200f2aa61433'
m0, m = read(OLD/'package-manifest.json'), read(NEW/'package-manifest.json')
b0, b = read(OLD/'bundle-template.json'), read(NEW/'bundle-template.json')
assert len(m['files']) == 762
actual = set()
for path in NEW.rglob('*'):
    assert not path.is_symlink() and '__pycache__' not in path.parts
    if path.is_file() and path.name != 'package-manifest.json':
        name = path.relative_to(NEW).as_posix(); actual.add(name)
        assert sha(path) == m['files'][name]
        assert path.suffix not in {'.pt','.safetensors','.parquet','.so','.bin','.pyc','.pyo'}
assert actual == set(m['files'])
added = sorted(set(m['files'])-set(m0['files']))
changed = sorted(k for k in set(m['files']) & set(m0['files']) if m['files'][k] != m0['files'][k])
assert not (set(m0['files'])-set(m['files']))
assert added == ['repo/OUTPUT_MODE_REPAIR3291.md', 'repo/OUTPUT_MODE_REPAIR3291_CPU.error.txt',
                 'repo/OUTPUT_MODE_REPAIR3291_CPU.output.txt', 'repo/OUTPUT_MODE_REPAIR3291_CPU_RECEIPT.json',
                 'repo/fixtures/vllm-063-llm-output-source.txt',
                 'repo/fixtures/vllm-063-sampling-params-source.txt', 'repo/test_sampling_output_transition.py']
assert changed == ['bundle-template.json','probe_package.py','repo/single_window_runner.py','repo/window_runtime.py']
old_runner = (OLD/'repo/single_window_runner.py').read_text()
new_runner = (NEW/'repo/single_window_runner.py').read_text()
old_stage = """            coordinator.local_stage('original actual sampling parameters unchanged',lambda:require(
                before_params == after_params,'Native window changed original SamplingParams'))"""
new_stage = """            transition = coordinator.local_stage('complete actual sampling parameter snapshots',lambda:
                sampling_params_transition(before_params,after_params,bundle['vllm_installed_sources']))
            coordinator.local_stage('actual sampling transition evidence before admission',lambda:
                atomic_write_json_no_clobber(output/'sampling-params-transition.json',transition))
            coordinator.local_stage('original actual sampling distribution unchanged',lambda:require(
                transition['allowed'],'Native window SamplingParams refused: '+repr(transition['reasons'])))"""
expected_runner = old_runner.replace('NativeWindowWrappers, bind_method, class_source, plain_object, sampling_params_metadata)',
    'NativeWindowWrappers, bind_method, class_source, plain_object, sampling_params_metadata,\n    sampling_params_transition)')
assert old_stage in expected_runner
expected_runner = expected_runner.replace(old_stage,new_stage).replace(
    '            sampling_params_before=before_params,sampling_params_after=after_params,\n',
    '            sampling_params_before=before_params,sampling_params_after=after_params,\n            sampling_params_transition=transition,\n')
assert expected_runner == new_runner
runtime = (NEW/'repo/window_runtime.py').read_text()
tree = ast.parse(runtime)
fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'sampling_params_transition')
lines = runtime.splitlines(keepends=True)
assert ''.join(lines[:fn.lineno-1]+lines[fn.end_lineno+2:]) == (OLD/'repo/window_runtime.py').read_text()
# Everything except the diagnostic helper and evidence/admission remains byte-identical.
assert (NEW/'repo/window_observer.py').read_bytes() == (OLD/'repo/window_observer.py').read_bytes()
assert (NEW/'repo/window_contracts.py').read_bytes() == (OLD/'repo/window_contracts.py').read_bytes()
assert sha(NEW/'repo/vllm_window_weights.py') == '46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'
assert sha(NEW/'resource_adapter_window.py') == '466df7c26d45342069eb0bd9dc9f79e0cead9ea1a0a0b803fd52a67398f7922b'

for key, relative, captured in [
    ('entrypoints/llm.py','fixtures/vllm-063-llm-output-source.txt','tree-vllm-installed-source-20261006/entrypoints/llm.py'),
    ('sampling_params.py','fixtures/vllm-063-sampling-params-source.txt','tree-vllm-installed-sampling-source-20261006/sampling_params.py')]:
    assert sha(NEW/'repo'/relative) == sha(WORK/captured) == PINS[key]
    assert b['vllm_installed_sources'][key]['sha256'] == PINS[key]
assert len(b['vllm_installed_sources']) == 30
assert len(b['torch_wrapper_sources']) == 1
owner = read(NEW/'repo/OUTPUT_MODE_REPAIR3291_CPU_RECEIPT.json')
assert owner['tests_passed'] == 6 and owner['captured_source_sha256'] == PINS
assert owner['actual_GPU_executed'] is owner['probability_accepted'] is owner['training_accepted'] is False
assert owner['runtime3291_specific_changed_fields_observed'] is False
for name, expected in owner['logs_sha256'].items(): assert sha(NEW/'repo'/name) == expected
assert 'Ran 6 tests' in (NEW/'repo/OUTPUT_MODE_REPAIR3291_CPU.error.txt').read_text()
assert 'First4 PASS/1ERROR' in owner['initial_fixture_error']

old_intent = 'SINGLE_TP1_WINDOW_WRAPPER_REPAIR3290_ONCE.json'
new_intent = 'SINGLE_TP1_WINDOW_OUTPUT_MODE_REPAIR3291_ONCE.json'
assert (OLD/'probe_package.py').read_text().replace(old_intent,new_intent) == (NEW/'probe_package.py').read_text()
for name in ('start.sh','start.py','supervisor.py','dispatch_once.py','freeze_package.py','resource_adapter_window.py'):
    assert (OLD/name).read_bytes() == (NEW/name).read_bytes()
    assert (REJECTED/name).read_bytes() == (NEW/name).read_bytes()
assert (REJECTED/'probe_package.py').read_bytes() == (NEW/'probe_package.py').read_bytes()
helpers = {
    'build_tree_vllm_window_output_mode_repair3291_v3_package.py': '32ffc894c7a2eae6a8594b3931448643d649a73d759cce00759e417333dde28d',
    'upload_tree_vllm_window_output_mode_repair3291_v3_source.py': 'cf2e93479d39615b4a395afc379a28d27592c8628786bc9f2286ab561af83fc3'}
for name, expected in helpers.items(): assert sha(WORK/name) == expected
for stem, suffix in [('build','package'),('upload','source')]:
    assert (WORK/(stem+'_tree_vllm_window_wrapper_repair3290_'+suffix+'.py')).read_text().replace(
        'tree-vllm-window-wrapper-repair3290','tree-vllm-window-output-mode-repair3291-v3') == (
        WORK/(stem+'_tree_vllm_window_output_mode_repair3291_v3_'+suffix+'.py')).read_text()
repo = WORK/'tree-vllm-window-runner'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == COMMIT
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True) == ''
for name in m['files']:
    if name.startswith('repo/'): assert sha(repo/name[5:]) == m['files'][name]
prefix0 = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+OLD.name
prefix = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+NEW.name
for key in set(b0)|set(b):
    if key not in ('probe_source_files','runner_commit','weights_core','reviewed_resource_adapter','native_method_sources'):
        assert b[key] == b0[key],key
for key in ('weights_core','reviewed_resource_adapter','native_method_sources'):
    assert json.loads(json.dumps(b0[key]).replace(prefix0,prefix)) == b[key],key
assert m['protocol'] == m0['protocol'] and b['runner_commit'] == m['runner_commit'] == COMMIT
assert b['window_inputs_sha256'] == '45f2d6d688cc0a4555d6c89d5700f8014b96ee8f395b3aa00b86fb94849552d3'
failure = REVIEW.parent/'actual3291/ACTUAL3291_FAILURE_RECEIPT.json'
assert sha(failure) == '1c69c62a5ad134a3919f50fd0b704dfd19e55ed53ea9745945d719b1f8b0b575'
counterexamples = REVIEW/'BLOCKING_COUNTEREXAMPLES_4ee0.json'
assert sha(counterexamples) == '781c56ee3a7c4ec85081dd2ad8d7d3a192989166b1b045174487030e082156e2'
assert not (REVIEW/'RELEASE_4ee0.json').exists()
assert read(REVIEW/'DELTA_CPU_RESULT.json')['returncode'] == 0
assert 'Ran 6 tests' in (REVIEW/'DELTA_CPU.error.txt').read_text()
assert (REVIEW/'DELTA_CPU.error.txt').read_text().rstrip().endswith('OK')
readback = WORK/'tree-vllm-window-output-mode-repair3291-v3/SOURCE_UPLOAD_REMOTE_RESULT.json'
assert sha(readback) == '985b1dd8c368eb0912f2d5f6e14ecd78b4b0a5f23f0dac18caecfba851567aaa'
remote_record = read(readback)
assert remote_record['manifest_sha256'] == MANIFEST
assert remote_record['gpu_submitted'] is remote_record['tensor_assets_transferred'] is False
diagnosis = WORK/'tree-same-backend-probability-diagnosis/SOURCE_ONLY_SAMPLING_PARAMS_FAILURE3291.md'
diagnosis_pins = WORK/'tree-same-backend-probability-diagnosis/SAMPLING_PARAMS_FAILURE3291_SOURCE_BINDINGS.json'
assert sha(diagnosis) == '25cacc42ac168b090838c95835d25d51d46873781604e3e04609176ea534bc67'
assert sha(diagnosis_pins) == 'f5aecd2876fb30b2e6e752a216b204ba2fb3f59afbf24067423d6591c367d607'
dp = read(diagnosis_pins)
assert dp['actual3291_before_after_values_saved'] is False
# This historical source-only diagnosis binds the old4ee0 helper, not the new final helper.
assert dp['files']['work/tree-vllm-window-runner/window_runtime.py']['sha256'] == sha(REJECTED/'repo/window_runtime.py')
sys.path.insert(0,str(NEW))
import probe_package as P
assert P.digest(P.bundle_payload(b)) == m['bundle_payload_sha256'] == PAYLOAD
assert P.digest(P.validate_window_inputs(b['window_inputs'])) == b['window_inputs_sha256']
assert P.DISPATCH_INTENT.endswith('/'+new_intent)
previous_release = REVIEW.parent/'wrapper-repair3290/RELEASE.json'
assert sha(previous_release) == 'afe3cdf0f9c85bfdfe0e16824d39707daa9e3e61054e41afeccc022093f7c48f'
release = copy.deepcopy(read(previous_release))
release.pop('failed3290_runtime_method_MRO_or_chain_identity_proven',None)
release.update(package_manifest_sha256=MANIFEST,bundle_payload_sha256=PAYLOAD,runner_commit=COMMIT,
    dispatch_intent_path=P.DISPATCH_INTENT,reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='INDEPENDENT_EXACT_SOURCE_PINNED_OUTPUT_DELIVERY_AND_TYPE_STRICT_SNAPSHOT_ADMISSION_DELTA_ONLY',
    predecessor_failed_job_id='3291',predecessor_failure_receipt_sha256=sha(failure),
    predecessor_package_manifest_sha256=sha(OLD/'package-manifest.json'),predecessor_package_and_intent_unchanged=True,
    successor_reason='Preserve full sampling field snapshots before admission; only exact source-pinned native CUMULATIVE/FINAL_ONLY to FINAL_ONLY delivery mode is exempt, all other fields compared as type-sensitive canonical JSON',
    failed3291_specific_sampling_difference_proven=False,original_intent_retry_accepted=False,
    rejected_unsubmitted_source_commit='4ee0de33526e01b41cd999432021c5e1fb3477aa',
    rejected_source_counterexample_sha256=sha(counterexamples),delta_CPU_tests=6,
    conditional_execution_limit='One measurement using the unused new OUTPUT_MODE intent only after fresh same-node readback, queue/UUID/all8GPU guard; Python -B and PYTHONDONTWRITEBYTECODE=1 for outer and child; zero backward/optimizer, no production training',
    saved_remote_readback_metadata_sha256=sha(readback),actual_remote_source_bytes_verified=False,
    real_assets_read_by_reviewer=False,actual_gpu_execution_verified=False,actual_loaded_binary_identity_proven=False,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,runtime_actual_optimizer_writes=None)
assert P.validate_release(m,MANIFEST,b,release) is True
for label, candidate in [('old complete release',read(previous_release)),
    ('old fixed intent',dict(release,dispatch_intent_path=P.DISPATCH_INTENT.replace(new_intent,old_intent))),
    ('rejected4ee0 manifest substitution',dict(release,package_manifest_sha256=sha(REJECTED/'package-manifest.json'))),
    ('rejected4ee0 commit substitution',dict(release,runner_commit='4ee0de33526e01b41cd999432021c5e1fb3477aa'))]:
    try: P.validate_release(m,MANIFEST,b,candidate)
    except ValueError: pass
    else: raise AssertionError('Stale or foreign release accepted: '+label)
proof = dict(status='EXACT_SOURCE_PINNED_OUTPUT_DELIVERY_TYPE_STRICT_ADMISSION_DELTA_VERIFIED',
    manifest_sha256=MANIFEST,bundle_payload_sha256=PAYLOAD,runner_commit=COMMIT,
    local_frozen_files_verified=762,changed_files=changed,added_files=added,deleted_files=[],helpers_sha256=helpers,
    source_pins=PINS,source_only_diagnosis_sha256=sha(diagnosis),source_only_diagnosis_bindings_sha256=sha(diagnosis_pins),
    diagnosis_historical_helper_reference_not_current_freeze=True,actual3291_specific_changed_fields_proven=False,
    strict_type_alias_rejection_verified=True,full_snapshots_publication_precedes_admission=True,
    native_constructor_generation_math_and_budget_unchanged=True,weights_core_and_resource_adapter_unchanged=True,
    wrapper_source_binding_and_observer_unchanged=True,
    new_fixed_intent=P.DISPATCH_INTENT,old_release_intent_or_rejected_source_substitutions_refused=4,
    independent_delta_CPU_tests_run=6,independent_delta_CPU_tests_passed=6,
    CPU_scope='Exact frozen helper AST, actual captured enum and native entry method AST with explicit synthetic Params/helper request stubs; no Torch/vLLM imports or real runtime/model assets',
    base136_tests_repeated=False,core44_tests_repeated=False,all10_context_suite_repeated=False,
    saved_source_only_upload_metadata_sha256=sha(readback),reviewer_remote_access=False,real_tensor_assets_read=False,
    actual_gpu_execution_verified=False,actual_loaded_binary_identity_proven=False,runtime_actual_optimizer_writes=None,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,
    predecessor_actual_failure_receipt_sha256=sha(failure),rejected4ee0_counterexamples_sha256=sha(counterexamples),
    owner_fixture_first4pass1error_disclosed=True,owner_legacy_status_label_CPU5_not_count_evidence=True,
    logs_sha256={name:sha(REVIEW/name) for name in ('DELTA_CPU.output.txt','DELTA_CPU.error.txt','DELTA_CPU_RESULT.json')})
write_new('RELEASE.json',release)
write_new('DELTA_SOURCE_CPU_PROOF.json',proof)
write_new('REJECTED_4ee0_SOURCE_RECEIPT.json',dict(status='INDEPENDENT_SOURCE_REFUSED_NO_RELEASE_OR_DISPATCH_ACCEPTANCE',
    runner_commit='4ee0de33526e01b41cd999432021c5e1fb3477aa',
    package_manifest_sha256=sha(REJECTED/'package-manifest.json'),
    counterexamples_sha256=sha(counterexamples),reason='Python equality hides bool/int and float/int metadata changes; exact enum integer provenance not enforced',
    production_source_modified=False,actual_remote_access=False,GPU_executed=False,real_assets_read=False,release_accepted=False))
print(json.dumps(dict(release_sha256=sha(REVIEW/'RELEASE.json'),proof_sha256=sha(REVIEW/'DELTA_SOURCE_CPU_PROOF.json'),
    rejected_source_receipt_sha256=sha(REVIEW/'REJECTED_4ee0_SOURCE_RECEIPT.json')),sort_keys=True))
