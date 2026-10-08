"""Only the strict actual SchedulerConfig observation delta; local source/CPU."""
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
OLD = WORK / 'single-tp1-window-ca1f2af0-466df7c2'
NEW = WORK / 'single-tp1-window-e29b27d1-466df7c2'
MANIFEST_SHA = 'd55e3836496fd6439087ee17611f0b65eed18bb07fc722b8f5bf9c087e372084'
PAYLOAD_SHA = '2c10f05feabd915c03b6ae20b713905d966fc61a3acb5231a019f82378754cb3'
COMMIT = 'e29b27d1b8b2b869349fc1c6366d81b89ae62fe0'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
assert sha(NEW / 'package-manifest.json') == MANIFEST_SHA
assert sha(OLD / 'package-manifest.json') == '36bfac94a3bc6fa5ed344d21d0110ab08a01cc098adebbccc42c538147a1f66c'
m0 = read(OLD / 'package-manifest.json'); m = read(NEW / 'package-manifest.json')
b0 = read(OLD / 'bundle-template.json'); b = read(NEW / 'bundle-template.json')
assert len(m['files']) == 751
actual = set()
for path in NEW.rglob('*'):
    assert not path.is_symlink() and '__pycache__' not in path.parts
    if path.is_file() and path.name != 'package-manifest.json':
        name = path.relative_to(NEW).as_posix(); actual.add(name)
        assert sha(path) == m['files'][name]
        assert path.suffix not in {'.pt', '.safetensors', '.parquet', '.so', '.bin', '.pyc', '.pyo'}
assert actual == set(m['files'])
added = sorted(set(m['files']) - set(m0['files']))
changed = sorted(k for k in m['files'] if k in m0['files'] and m['files'][k] != m0['files'][k])
assert not (set(m0['files']) - set(m['files']))
assert added == ['repo/SCHEDULER_REPAIR3286.md', 'repo/SCHEDULER_REPAIR3286_CPU_RECEIPT.json',
                 'repo/fixtures/vllm-0.6.3-config-source.txt', 'repo/test_actual_engine_metadata.py']
assert changed == ['bundle-template.json', 'probe_package.py', 'repo/single_window_runner.py']

old_tree = ast.parse((OLD / 'repo/single_window_runner.py').read_bytes())
new_tree = ast.parse((NEW / 'repo/single_window_runner.py').read_bytes())
old_function = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'actual_engine_metadata')
old_constants = [n for n in ast.walk(old_function) if isinstance(n, ast.Constant) and n.value == 'enable_chunked_prefill']
assert len(old_constants) == 1
old_constants[0].value = 'chunked_prefill_enabled'
assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
# Whole production source is unchanged except the exact strict getter string.
assert (OLD / 'repo/single_window_runner.py').read_text().replace(
    "'max_num_seqs','enable_chunked_prefill','num_scheduler_steps'",
    "'max_num_seqs','chunked_prefill_enabled','num_scheduler_steps'") == (NEW / 'repo/single_window_runner.py').read_text()
fixture = NEW / 'repo/fixtures/vllm-0.6.3-config-source.txt'
captured = WORK / 'tree-vllm-installed-source-extra-20261006/config.py'
assert fixture.read_bytes() == captured.read_bytes()
assert sha(fixture) == b['vllm_installed_sources']['config.py']['sha256'] == '3a9d50709aa2f7ca3b14741b507232e2cd954936780d4a0365cabea12e6cc8b0'
new_function = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'actual_engine_metadata')
properties = ast.literal_eval(next(n.value for n in new_function.body if isinstance(n, ast.Assign) and
    any(isinstance(t, ast.Name) and t.id == 'properties' for t in n.targets)))
classes = {n.name: n for n in ast.parse(fixture.read_bytes()).body if isinstance(n, ast.ClassDef)}
field_count = 0
for name, fields in properties.items():
    cls = classes[''.join(word.title() for word in name.split('_'))]
    stored = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
              and isinstance(n.value, ast.Name) and n.value.id == 'self'}
    assert set(fields) <= stored
    field_count += len(fields)
assert field_count == 18
owner_receipt = read(NEW / 'repo/SCHEDULER_REPAIR3286_CPU_RECEIPT.json')
assert sha(NEW / 'repo/SCHEDULER_REPAIR3286_CPU_RECEIPT.json') == '10178e032d6c0315aabde8c5f369405a05239e25e48ac75cb37f5043c72d4e37'
assert owner_receipt['supersedes_unaccepted_receipt_commit'] == 'e4790d4e5392f5e3fe47f318418f97abb54c5789'
assert 'incorrectly included PASS receipt' in owner_receipt['initial_test_failure']
for name, wanted in owner_receipt['sources'].items(): assert sha(NEW / 'repo' / name) == wanted

old_intent = 'SINGLE_TP1_WINDOW_BOS_REPAIR3285_ONCE.json'
new_intent = 'SINGLE_TP1_WINDOW_SCHEDULER_REPAIR3286_ONCE.json'
assert (OLD / 'probe_package.py').read_text().replace(old_intent, new_intent) == (NEW / 'probe_package.py').read_text()
for name in ('start.sh', 'start.py', 'supervisor.py', 'dispatch_once.py', 'freeze_package.py', 'resource_adapter_window.py'):
    assert (OLD / name).read_bytes() == (NEW / name).read_bytes()
helpers = {
    'build_tree_vllm_window_scheduler_repair3286_package.py': '005d677eaa4fa6cd957f98dda93a69285701ae22c6f43de330fdf18f77a8e7c6',
    'upload_tree_vllm_window_scheduler_repair3286_source.py': '667f93dbecdd8f59657366b3720f3ac75a7b77236323ac4087bcb4764690953f'}
assert (WORK / 'build_tree_vllm_window_bos_repair3285_package.py').read_text().replace(
    'tree-vllm-window-bos-repair3285/launcher', 'tree-vllm-window-scheduler-repair3286/launcher').replace(
    'tree-vllm-window-bos-repair3285/PACKAGE_BUILD.json', 'tree-vllm-window-scheduler-repair3286/PACKAGE_BUILD.json') == (
    WORK / 'build_tree_vllm_window_scheduler_repair3286_package.py').read_text()
assert (WORK / 'upload_tree_vllm_window_bos_repair3285_source.py').read_text().replace(
    'tree-vllm-window-bos-repair3285/SOURCE_UPLOAD_PREPARED.json',
    'tree-vllm-window-scheduler-repair3286/SOURCE_UPLOAD_PREPARED.json') == (
    WORK / 'upload_tree_vllm_window_scheduler_repair3286_source.py').read_text()
for name, wanted in helpers.items(): assert sha(WORK / name) == wanted

repo = WORK / 'tree-vllm-window-runner'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == COMMIT
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True) == ''
for name in m['files']:
    if name.startswith('repo/'): assert sha(repo / name[5:]) == m['files'][name]
log = (REVIEW / 'DELTA_CPU.stderr').read_text()
assert 'Ran 3 tests' in log and log.rstrip().endswith('OK')
prefix0 = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/' + OLD.name
prefix = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/' + NEW.name
for key in ('window_inputs', 'window_inputs_sha256', 'window_contract', 'window_profile', 'diagnostic_profile',
            'config_text', 'config_path', 'identity', 'inventory', 'selection', 'selection_file', 'hf_identity', 'hf_index', 'hf_path',
            'native_logprob_sha256', 'installed_versions', 'installed_sources', 'vllm_installed_sources',
            'torch_wrapper_sources', 'installed_kernel_binaries', 'original_worker_path', 'original_worker_sha256', 'parent_state_manifest'):
    assert b[key] == b0[key], key
assert m['protocol'] == m0['protocol'] and b['runner_commit'] == m['runner_commit'] == COMMIT
for key in ('weights_core', 'reviewed_resource_adapter', 'native_method_sources'):
    assert json.loads(json.dumps(b0[key]).replace(prefix0, prefix)) == b[key], key
assert sha(NEW / 'repo/vllm_window_weights.py') == '46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'
assert sha(NEW / 'resource_adapter_window.py') == '466df7c26d45342069eb0bd9dc9f79e0cead9ea1a0a0b803fd52a67398f7922b'
sys.path.insert(0, str(NEW))
import probe_package as P
assert P.digest(P.bundle_payload(b)) == m['bundle_payload_sha256'] == PAYLOAD_SHA
assert P.digest(P.validate_window_inputs(b['window_inputs'])) == b['window_inputs_sha256']
assert P.DISPATCH_INTENT.endswith('/' + new_intent)
failure = WORK / 'tree-vllm-window-final-review/ACTUAL3286_FAILURE_RECEIPT.json'
assert sha(failure) == '2aa7f2dd9e67ce9393c452b052ef1b5c4aefe89a4eb9635add7544e59803423b'
old_release_path = WORK / 'tree-vllm-window-final-review/bos-repair3285/RELEASE.json'
assert sha(old_release_path) == 'a405e16fb79209e960860b178479468a771ee8ddee54db537f0868fe74e5ba05'
release = copy.deepcopy(read(old_release_path))
release.update(package_manifest_sha256=MANIFEST_SHA, bundle_payload_sha256=PAYLOAD_SHA, runner_commit=COMMIT,
    dispatch_intent_path=P.DISPATCH_INTENT, reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='INDEPENDENT_ACTUAL_SCHEDULER_STORED_FIELD_DELTA_SOURCE_AND_PRE_DISPATCH_REVIEW_ONLY',
    predecessor_failed_job_id='3286', predecessor_failure_receipt_sha256=sha(failure),
    successor_reason='Strict passive getter compatibility repair: SchedulerConfig.chunked_prefill_enabled is the actual stored attribute; original constructor keyword/kwargs unchanged',
    predecessor_package_manifest_sha256=sha(OLD / 'package-manifest.json'), predecessor_package_and_intent_unchanged=True,
    original_intent_retry_accepted=False, full_base_136_tests_repeated=False, core44_repeated=False, delta_CPU_tests=3,
    owner_interim_false_PASS_receipt_accepted=False, owner_interim_false_PASS_commit='e4790d4e5392f5e3fe47f318418f97abb54c5789')
assert P.validate_release(m, MANIFEST_SHA, b, release) is True
for label, candidate in [('old complete release', read(old_release_path)),
                         ('old fixed intent', dict(release, dispatch_intent_path=P.DISPATCH_INTENT.replace(new_intent, old_intent)))]:
    try: P.validate_release(m, MANIFEST_SHA, b, candidate)
    except ValueError: pass
    else: raise AssertionError('Wrong successor release accepted: ' + label)
proof = dict(status='EXACT_ACTUAL_SCHEDULER_STORED_FIELD_SUCCESSOR_SOURCE_DELTA_VERIFIED', manifest_sha256=MANIFEST_SHA,
    bundle_payload_sha256=PAYLOAD_SHA, runner_commit=COMMIT, local_frozen_files_verified=751,
    changed_files=changed, added_files=added, deleted_files=[], helpers_sha256=helpers,
    altered_production_AST_constants=1, actual_installed_config_fixture_sha256=sha(fixture), observed_config_fields=18,
    unchanged_allocation_resource_native_budgets=True, native_constructor_sampling_and_math_unchanged=True,
    new_fixed_intent=P.DISPATCH_INTENT, old_intent_or_release_substitutions_refused=2,
    independent_delta_CPU_tests_run=3, independent_delta_CPU_tests_passed=3, base136_tests_repeated=False, core44_tests_repeated=False,
    owner_interim_false_PASS_receipt_accepted=False, owner_disclosed_two_prior_fixture_failures=True,
    historical_failed_fixture_runs_independently_reexecuted=False,
    real_tensor_assets_read=False, remote_access=False, actual_gpu_execution_verified=False, actual_runtime_optimizer_writes=None,
    probability_accepted=False, training_accepted=False, GPU_capacity_proven=False, actual_loaded_binary_identity_proven=False,
    predecessor_actual_failure_receipt_sha256=sha(failure),
    logs_sha256={name:sha(REVIEW/name) for name in ('DELTA_CPU.stdout','DELTA_CPU.stderr')})
for name, value in [('RELEASE.json', release), ('DELTA_SOURCE_CPU_PROOF.json', proof)]:
    target = REVIEW/name; assert not target.exists()
    target.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
print(json.dumps(dict(release_sha256=sha(REVIEW/'RELEASE.json'), proof_sha256=sha(REVIEW/'DELTA_SOURCE_CPU_PROOF.json')), sort_keys=True))
