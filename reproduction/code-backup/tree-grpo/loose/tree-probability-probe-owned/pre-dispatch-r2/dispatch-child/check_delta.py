"""R2 queue-only CPU delta; all external scheduler APIs are local callables."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OLD = WORK / 'probability-probe-49a6b35-29822f9'
NEW = WORK / 'probability-probe-49a6b35-29822f9-r2'
sys.path.insert(0, str(NEW))
import dispatch_once as dispatch
import probe_package as contract

MANIFEST_SHA = '0a1a39ec5d9f86498d8e66a9f9291aefffba02f88776e2fa1aaeb200085e7dc8'
PAYLOAD_SHA = '37d5debdb09beaac3346fd2912724b84413ed54cb673bb8015990c3eaa467f5f'
checks = []
def check(name, value):
    if not value: raise AssertionError(name)
    checks.append(dict(name=name, passed=True))

manifest = json.loads((NEW / 'package-manifest.json').read_text())
template = json.loads((NEW / 'bundle-template.json').read_text())
check('exact new manifest bytes', contract.file_sha(NEW / 'package-manifest.json') == MANIFEST_SHA)
check('exact new full bundle payload', contract.digest(contract.bundle_payload(template)) == manifest['bundle_payload_sha256'] == PAYLOAD_SHA)
hashes = {}
for name in ['dispatch_once.py', 'start.py', 'start.sh', 'supervisor.py', 'probe_package.py', 'bundle-template.json']:
    hashes[name] = contract.file_sha(NEW / name)
    check('exact direct launcher hash ' + name, hashes[name] == manifest['files'][name])
    if name.endswith('.py'): compile((NEW / name).read_text(), str(NEW / name), 'exec')
for name in ['start.py', 'start.sh', 'supervisor.py', 'probe_package.py']:
    check('old source unchanged ' + name, (OLD / name).read_bytes() == (NEW / name).read_bytes())
old_text = (OLD / 'dispatch_once.py').read_text()
new_text = (NEW / 'dispatch_once.py').read_text()
old_ast = ast.parse(old_text); new_ast = ast.parse(new_text)
old_functions = {node.name: ast.dump(node, include_attributes=False) for node in old_ast.body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: ast.dump(node, include_attributes=False) for node in new_ast.body if isinstance(node, ast.FunctionDef)}
check('same dispatch function inventory', old_functions.keys() == new_functions.keys())
for name in old_functions:
    if name != 'queue_conflicts': check('unchanged AST ' + name, old_functions[name] == new_functions[name])
for node in old_ast.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'queue_conflicts': node.body = []
for node in new_ast.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'queue_conflicts': node.body = []
check('entire dispatch AST unchanged outside queue_conflicts body', ast.dump(old_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False))
old_snippet = 'node_sets.append(set(expand(match.group(1))))'
new_snippet = "expanded=expand(match.group(1))\n                require(type(expanded) is list and expanded and all(type(node) is str and node.strip() and node==node.strip() for node in expanded),\n                        'Unknown/empty/blank expanded node identity; refuse duplicate GPU dispatch')\n                node_sets.append(set(expanded))"
check('exact source delta only expanded identity validation', old_text.replace(old_snippet, new_snippet) == new_text)
(HERE / 'EXACT_DISPATCH_DIFF.txt').write_text(''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True), fromfile='r1/dispatch_once.py', tofile='r2/dispatch_once.py')))

# A copied synthetic release is schema verification, never an issued release.
release = dict(status=contract.RELEASE_STATUS, package_manifest_sha256=MANIFEST_SHA,
    bundle_payload_sha256=PAYLOAD_SHA, selection_sha256=manifest['selection_sha256'],
    runner_commit=manifest['runner_commit'], protocol=copy.deepcopy(contract.PROTOCOL), actual_optimizer_writes=0,
    adapter_sha256=manifest['reviewed_resource_adapter']['sha256'], dispatch_intent_path=contract.DISPATCH_INTENT)
check('new exact envelope binds launch pins/full bundle', contract.validate_release(manifest, MANIFEST_SHA, template, release))
for field, value in [('package_manifest_sha256', 'c542a698262d251a741afbd4958f37682e31dff97e6276fe387ef9f806f5bfb5'),
                     ('bundle_payload_sha256', '2c17f7dfca5cc655bf40a7fdb4e4399cb999ef626d37de20035b38eb723250e8')]:
    altered = copy.deepcopy(release); altered[field] = value
    try: contract.validate_release(manifest, MANIFEST_SHA, template, altered)
    except ValueError: check('reject stale R1 release ' + field, True)
    else: raise AssertionError('stale release admitted')

class ExpandedOutput:
    """Only for malformed splitlines() contract injection; not real tool stdout."""
    def __init__(self, result): self.result = result
    def splitlines(self): return self.result

cases = [('empty-list', []), ('empty-hostname', ['']), ('blank-hostname', [' ']), ('none-hostname', [None]),
         ('leading-space', [' h100-9']), ('trailing-space', ['h100-9 ']),
         ('wrong-tuple-type', ('h100-9',)), ('wrong-set-type', {'h100-9'}),
         ('wrong-none-type', None), ('wrong-string-type', 'h100-9'),
         ('matched-h1009', ['h100-9']), ('fixed-other-h10016', ['h100-16'])]
results = []
for name, expanded in cases:
    is_other = name == 'fixed-other-h10016'
    node = 'h100-16' if is_other else 'h100-9'
    row = '83|tree-pending|PENDING|(null)|gpu:8'
    job = 'JobId=83 NodeList=(null) ReqNodeList=' + node + ' ReqTRES=cpu=32,gres/gpu=8'
    with tempfile.TemporaryDirectory(dir=HERE, prefix='fixture-') as temporary:
        intent = Path(temporary) / 'intent.json'; calls = []; submissions = []
        def tool(argv, **kwargs):
            calls.append(argv)
            if argv[0] == 'squeue': return row + '\n'
            if argv[:3] == ['scontrol', 'show', 'job']: return job
            if argv[:3] == ['scontrol', 'show', 'hostnames']:
                if isinstance(expanded, list) and all(isinstance(value, str) for value in expanded):
                    return ''.join(value + '\n' for value in expanded)
                return ExpandedOutput(expanded)
            raise AssertionError('Unexpected tool call')
        def submit(argv):
            submissions.append(argv)
            return SimpleNamespace(returncode=0, stdout='999;synthetic', stderr='')
        error = None; result = None
        with patch.object(dispatch.subprocess, 'check_output', side_effect=tool):
            try:
                result = dispatch.submit_once(intent, ['fake-sbatch-never-executed'], {'delta_fixture': name}, submit,
                    pre_submit=lambda: dispatch.refreshed_queue_gate(intent))
            except ValueError as failure: error = str(failure)
        evidence = json.loads(next(Path(temporary).glob('intent.json.queue-*.json')).read_text())
        if is_other:
            check('positive fixed h100-16 remains excluded', error is None and len(submissions) == 1 and result['job_id'] == '999')
        else:
            check('full refreshed queue blocks before attempt/submit ' + name, error is not None and not intent.exists() and not submissions)
            if name == 'matched-h1009': check('matched node conflict kept in evidence', evidence['unknown_error'] is None and len(evidence['conflicts']) == 1)
            else: check('malformed expansion evidence unknown ' + name, evidence['unknown_error'] is not None and evidence['conflicts'] is None)
        results.append(dict(case=name, synthetic_expanded_result=repr(expanded), recorded_tool_argv=calls,
            real_string_splitlines=isinstance(expanded, list) and all(isinstance(value, str) for value in expanded),
            intent_created=intent.exists(), fake_submit_calls=len(submissions), error=error,
            queue_unknown_error=evidence['unknown_error'], queue_conflicts=evidence['conflicts']))

proof = dict(status='SOURCE_AND_DISPATCH_DELTA_REVIEWED_ZERO_UPDATE_PROBE_ONLY', package_manifest_sha256=MANIFEST_SHA,
    bundle_payload_sha256=PAYLOAD_SHA, source_hashes=hashes, checks=checks, cases=results, blocking_findings=[],
    prior_review_receipt_sha256='1ca539f43c5fe406738a99cf5827104996c43098bc8852551ef327bdfbc9943a',
    unchanged_scope='Once intent/UNKNOWN-no-retry/command/main/release/start/supervisor AST or bytes unchanged; prior 15 tests +46 checks retained without rerun.',
    real_scheduler_calls=0, remote_calls=0, real_gpu_calls=0, real_tensor_or_model_reads=0, frozen_package_modified=False,
    actual_gpu_probe_accepted=False, release_issued=False,
    limitations='Tiny synthetic splitlines fixtures include malformed return types; full immutable source mapping/live resource review is parent scope. Positive test invokes one local fake callable, never sbatch.')
(HERE / 'INDEPENDENT_DELTA_PROOF.json').write_text(json.dumps(proof, sort_keys=True, indent=2) + '\n')
print(json.dumps(dict(status=proof['status'], checks=len(checks), cases=len(cases), dispatch_sha=hashes['dispatch_once.py'])))
