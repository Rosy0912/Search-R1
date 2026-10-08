"""Frozen R3 local source/metadata review. No actual asset/scheduler/GPU access."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = BASE / 'work/probability-probe-49a6b35-29822f9-r2'
NEW = BASE / 'work/probability-probe-49a6b35-29822f9-r3'
REMOTE_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
checks = []
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

m2, m3 = read(OLD / 'package-manifest.json'), read(NEW / 'package-manifest.json')
check('exact R3 manifest SHA', sha(NEW / 'package-manifest.json') ==
      '014751600fbc6336dca1fa2d78fcef2a1dcb69730e4f6e7dcbbf17fb38035660')
spec = importlib.util.spec_from_file_location('review_r3_exact_pure_metadata', NEW / 'probe_package.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
actual = {x.relative_to(NEW).as_posix() for x in NEW.rglob('*') if x.is_file() and
          x.name != 'package-manifest.json'}
check('full755 local source inventory no alias', actual == set(m3['files']) and
      not any(x.is_symlink() for x in NEW.rglob('*')))
check('full755 exact local source bytes SHA', all(sha(NEW / k) == v for k, v in m3['files'].items()))
check('expected755 count', len(m3['files']) == 755)
changed = [k for k in m2['files'] if k in m3['files'] and m2['files'][k] != m3['files'][k]]
check('only four existing files changed', set(changed) ==
      {'API_AND_SCOPE.md', 'bundle-template.json', 'probe_package.py', 'supervisor.py'})
check('no R2 file removed', set(m2['files']) <= set(m3['files']))
check('all719 repo files preserved byte for byte', sum(x.startswith('repo/') for x in m3['files']) == 719 and
      all(m2['files'][x] == value for x, value in m3['files'].items() if x.startswith('repo/')))
check('candidate supervisor exact source', sha(NEW / 'supervisor.py') ==
      '16398fef6935aedc7b4196fca9a34f185e7ab85b69005f9249493b9b79092e82')
check('candidate helper exact source', sha(NEW / 'probe_package.py') ==
      'b8ef6b7c3149c3db474101f74eb9be34fbc1d9e72b93f85e8416161969603c52')
old_supervisor = ast.parse((OLD / 'supervisor.py').read_bytes())
new_supervisor = ast.parse((NEW / 'supervisor.py').read_bytes())
header = [n for n in new_supervisor.body if isinstance(n, ast.FunctionDef) and
          n.name == 'compiler_header_environment']
check('one new pure header helper', len(header) == 1)
new_supervisor.body.remove(header[0])
main = next(n for n in new_supervisor.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
environment = [n for n in main.body if isinstance(n, ast.Expr) and ast.dump(n.value, include_attributes=False) ==
               ast.dump(ast.parse('os.environ.update(compiler_header_environment())', mode='eval').body,
                        include_attributes=False)]
check('one include environment mutation before guard/probe spawn', len(environment) == 1 and
      main.body.index(environment[0]) < next(i for i, n in enumerate(main.body) if isinstance(n, ast.Try)))
main.body.remove(environment[0])
field = [n for n in ast.walk(main) if isinstance(n, ast.keyword) and n.arg == 'compiler_environment']
check('one recorded actual include environment', len(field) == 1 and
      ast.dump(field[0].value, include_attributes=False) == ast.dump(ast.parse(
          "dict(C_INCLUDE_PATH=os.environ['C_INCLUDE_PATH'])", mode='eval').body, include_attributes=False))
holder = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and field[0] in n.keywords)
holder.keywords.remove(field[0])
check('all other supervisor AST unchanged', ast.dump(old_supervisor, include_attributes=False) ==
      ast.dump(new_supervisor, include_attributes=False))
new_helper = (NEW / 'probe_package.py').read_text().replace(
    'PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json', 'PROBABILITY_PROBE_ONCE-3268.json').replace(
    'Header repair must use its new fixed intent; old or arbitrary intent paths are refused',
    'This probe must keep its original fixed single-attempt intent path')
check('metadata helper only fixed new intent+description', new_helper == (OLD / 'probe_package.py').read_text())
for name in ['dispatch_once.py', 'start.py', 'start.sh', 'freeze_package.py', 'resource_adapter_v2.py',
             'test_candidate.py']:
    check('unchanged runtime ' + name, (OLD / name).read_bytes() == (NEW / name).read_bytes())
for name in ['supervisor.py', 'probe_package.py', 'dispatch_once.py', 'start.py', 'freeze_package.py',
             'test_header_environment.py', 'repo/search_r1/baselines/tree_probability_probe/runner.py']:
    path = NEW / name
    if path.exists():
        compile(path.read_bytes(), str(path), 'exec')
        check('actual frozen source compile ' + name, True)
t2, t3 = read(OLD / 'bundle-template.json'), read(NEW / 'bundle-template.json')
old_prefix = str(REMOTE_ROOT / 'instruct300/candidates' / OLD.name)
new_prefix = str(REMOTE_ROOT / 'instruct300/candidates' / NEW.name)

def remap(value):
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    if isinstance(value, dict):
        return {k: remap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v) for v in value]
    return value

check('full original bundle inputs only package root remapped', remap(t2) == t3)
check('exact normalized new payload', P.digest(P.bundle_payload(t3)) == m3['bundle_payload_sha256'] ==
      '0b9cb4f2bfab39c813e71cba0809331ca7d94b64e70ab107de031107771ce57f')
check('original protocol identical', m3['protocol'] == m2['protocol'] == P.PROTOCOL)
check('original full source/installed mapping unchanged',
      remap(m2['probe_source_files']) == m3['probe_source_files'] == t3['probe_source_files'] and
      m3['installed_sources'] == m2['installed_sources'] == t3['installed_sources'] and
      m3['installed_versions'] == m2['installed_versions'] == t3['installed_versions'])
check('native runner adapter/config/HF/selection unchanged',
      all(t3[k] == t2[k] for k in ('identity', 'selection', 'inventory', 'selection_file', 'config_path',
          'config_text', 'original_worker_path', 'original_worker_sha256', 'hf_path', 'hf_identity',
          'hf_index', 'parent_state_manifest', 'probe_mode', 'runner_commit', 'native_logprob_sha256')) and
      sha(NEW / 'resource_adapter_v2.py') == '29822f98f8749cdbede4f62c91a0745d241a71c4bd16ce7e52e2f58126c3e273')
for row in m3['probe_source_files']:
    path = Path(row['path'])
    relative = path.relative_to(Path(new_prefix))
    check('source root mapping ' + str(relative), m3['files'][str(relative)] == row['sha256'])
actual_meta = read(BASE / 'outputs/tree-instruct-300/PPO_PROBE_GPU3280_FAILURE_METADATA_RAW.json')
added_external = {k: v for k, v in m3['external_files'].items() if k not in m2['external_files']}
check('external previous3 unchanged', all(m3['external_files'][k] == v for k, v in m2['external_files'].items()))
headers = ['python-dev/usr/include/python3.10/Python.h', 'python-dev/usr/include/python3.10/pyconfig.h',
           'python-dev/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h']
expected_external = {str(REMOTE_ROOT / k): actual_meta['headers'][k]['sha256'] for k in headers}
expected_external[str(REMOTE_ROOT / 'runs/tree-probability-probe-3280/result.json')] = \
    sha(BASE / 'outputs/tree-instruct-300/gpu3280/result.json')
check('exact3 header+actual failure SHA external additions', added_external == expected_external)
for local_name, original in {
    'actual3280-failure-receipt.json': BASE / 'work/tree-probability-probe-owned/actual3280-review/ACTUAL_FAILURE_RECEIPT.json',
    'actual3280-gcc-and-header-cpu-proof.json': BASE / 'outputs/tree-instruct-300/PPO_PROBE_GPU3280_FAILURE_METADATA_RAW.json',
    'actual3280-original-result.json': BASE / 'outputs/tree-instruct-300/gpu3280/result.json',
    'actual3280-original-probe.log': BASE / 'outputs/tree-instruct-300/gpu3280/probe.log',
    'actual3280-original-launcher.out': BASE / 'outputs/tree-instruct-300/gpu3280/tree-probability-probe-3280.out',
    'prior-r2-release.json': BASE / 'work/tree-probability-probe-owned/pre-dispatch-r2/RELEASE.json',
}.items():
    check('exact audit copy ' + local_name, (NEW / 'audits' / local_name).read_bytes() == original.read_bytes())
check('previousfailure receipt exact scope', sha(NEW / 'audits/actual3280-failure-receipt.json') ==
      '72aaa83e18f4995d003a2be1d60f9010be466493bebae239b47a68cb3c05ad4a')
owner_proof = read(NEW / 'OWNER_SOURCE_CPU_PROOF_R3.json')
check('exact owner53 CPU source proof', sha(NEW / 'OWNER_SOURCE_CPU_PROOF_R3.json') ==
      'e921f4b004dddb9e4e49a58b99d291b2c239d9cb0c065858cfd98fbff3a6ff0a' and
      owner_proof['targeted_tests'] == 14 and owner_proof['regression_tests'] == 39 and
      owner_proof['total_unique_tests'] == 53)
for name, record in owner_proof['r3_files_sha256'].items():
    check('owner source proof byte pin ' + name, sha(NEW / name) == record)
test_log = HERE / 'INDEPENDENT_HEADER_TEST.stderr'
check('14 targeted independently executed on identical source', 'Ran 14 tests' in test_log.read_text() and
      test_log.read_text().endswith('\nOK\n') and
      (NEW / 'test_header_environment.py').read_bytes() ==
      (BASE / 'work/tree-probability-probe-owned/supervisor-candidate-r3/test_header_environment.py').read_bytes())
proof = {'status': 'R3_EXACT_PACKAGE_AND_HEADER_DELTA_SOURCE_CPU_REVIEWED',
    'package_manifest_sha256': sha(NEW / 'package-manifest.json'),
    'bundle_payload_sha256': m3['bundle_payload_sha256'], 'files': len(m3['files']),
    'checks_passed': len(checks), 'checks': checks,
    'independently_reexecuted_header_cpu_tests': 14, 'owner_regression_tests_saved_evidence': 39,
    'unchanged39_regression_independently_reexecuted_this_review': False,
    'header_test_log_sha256': sha(test_log), 'actual_asset_or_header_reads': 0,
    'actual_scheduler_calls': 0, 'remote_calls': 0, 'actual_gpu_calls': 0,
    'actual_optimizer_writes_by_reviewer': 0, 'actual_gpu_probability_or_training_accepted': False,
    'known_scope': 'Candidate targeted14 source tests on exact same frozen bytes; full755 source/mapping hashes; actual headers external SHA only',
}
(HERE / 'PACKAGE_AND_HEADER_CPU_PROOF.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
print(json.dumps({'checks_passed': len(checks), 'proof_sha256': sha(HERE / 'PACKAGE_AND_HEADER_CPU_PROOF.json')}))
