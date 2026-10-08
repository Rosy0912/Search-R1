"""Independent synthetic CPU checks. Never execute a scheduler or remote command."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / 'probability-probe-49a6b35-29822f9'
# dispatch-child -> pre-dispatch -> tree-probability-probe-owned -> work
sys.path.insert(0, str(PACKAGE))
import probe_package as contract
import dispatch_once as dispatch
import supervisor

EXPECTED_MANIFEST = 'c542a698262d251a741afbd4958f37682e31dff97e6276fe387ef9f806f5bfb5'
MANIFEST = json.loads((PACKAGE / 'package-manifest.json').read_text())
TEMPLATE = json.loads((PACKAGE / 'bundle-template.json').read_text())
checks = []
findings = []


def check(name, value, detail=None):
    if not value:
        raise AssertionError(name)
    checks.append(dict(name=name, passed=True, detail=detail))


def refused(name, function):
    try:
        function()
    except (ValueError, OSError, KeyError) as error:
        check(name, True, str(error))
    else:
        raise AssertionError('Expected refusal: ' + name)


def release():
    return dict(status=contract.RELEASE_STATUS, package_manifest_sha256=EXPECTED_MANIFEST,
        bundle_payload_sha256=MANIFEST['bundle_payload_sha256'], selection_sha256=MANIFEST['selection_sha256'],
        runner_commit=MANIFEST['runner_commit'], protocol=copy.deepcopy(contract.PROTOCOL),
        actual_optimizer_writes=0, adapter_sha256=MANIFEST['reviewed_resource_adapter']['sha256'],
        dispatch_intent_path=contract.DISPATCH_INTENT)


check('exact fixed package manifest SHA', contract.file_sha(PACKAGE / 'package-manifest.json') == EXPECTED_MANIFEST)
source_hashes = {}
for name in ['dispatch_once.py', 'start.py', 'start.sh', 'supervisor.py', 'probe_package.py', 'bundle-template.json',
             'repo/search_r1/baselines/preflight.py']:
    source_hashes[name] = contract.file_sha(PACKAGE / name)
    check('frozen direct launch source hash: ' + name, source_hashes[name] == MANIFEST['files'][name])
    if name.endswith('.py'):
        compile((PACKAGE / name).read_text(), str(PACKAGE / name), 'exec')
check('direct launch Python compile in memory (no bytecode writes)', True)
shell = subprocess.run(['bash', '-n', str(PACKAGE / 'start.sh')], text=True, capture_output=True)
check('start.sh syntax only; script not executed', shell.returncode == 0, shell.stderr)
check('actual frozen bundle payload digest', contract.digest(contract.bundle_payload(TEMPLATE)) == MANIFEST['bundle_payload_sha256'])
check('actual manifest/template fixture release binding', contract.validate_release(MANIFEST, EXPECTED_MANIFEST, TEMPLATE, release()))
for field, value in [('package_manifest_sha256', '0' * 64), ('bundle_payload_sha256', '0' * 64),
                     ('selection_sha256', '0' * 64), ('runner_commit', '0' * 40),
                     ('adapter_sha256', '0' * 64), ('dispatch_intent_path', str(contract.ROOT / 'alternate-intent.json')),
                     ('actual_optimizer_writes', False), ('actual_optimizer_writes', 1), ('status', 'SOURCE_CPU_ONLY')]:
    altered = release(); altered[field] = value
    refused('release mismatch rejected: ' + field + '=' + repr(value),
        lambda altered=altered: contract.validate_release(MANIFEST, EXPECTED_MANIFEST, TEMPLATE, altered))
altered = release(); altered['protocol']['requested_cpus'] = 16
refused('release resource budget differs rejected', lambda: contract.validate_release(MANIFEST, EXPECTED_MANIFEST, TEMPLATE, altered))
altered = copy.deepcopy(TEMPLATE); altered['probe_source_files'][0]['sha256'] = '0' * 64
refused('full source entry tamper rejected through bundle digest', lambda: contract.validate_release(MANIFEST, EXPECTED_MANIFEST, altered, release()))
for path in ['/tmp/foreign-package', 'relative/package', str(contract.ROOT / '../foreign-package')]:
    refused('own-root confinement: ' + path, lambda path=path: contract.same_node_path(path))

# Invoke the real start.main with held metadata fixtures and intercepted exec.
# Package validation is explicitly mocked; full 736-file integrity is parent's scope.
spec = importlib.util.spec_from_file_location('child_start', PACKAGE / 'start.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
captured = []
events = []
def metadata(path, sha):
    events.append(('read_bound', str(path), sha))
    if Path(path).name == 'package-manifest.json': return copy.deepcopy(MANIFEST)
    if Path(path).name == 'bundle-template.json': return copy.deepcopy(TEMPLATE)
    if Path(path).name == 'child-release.json': return release()
    raise AssertionError('unexpected local metadata path')
def intercepted_exec(program, argv, env):
    captured.append((program, argv, env)); events.append(('exec_intercepted', program))
arguments = ['start.py', '--package', str(PACKAGE), '--manifest-sha256', EXPECTED_MANIFEST,
             '--release', str(contract.ROOT / 'child-release.json'), '--release-sha256', '6' * 64]
with patch.object(sys, 'argv', arguments), patch.object(entry, 'same_node_path', side_effect=lambda path: Path(path).resolve()), \
     patch.object(entry, 'read_bound', side_effect=metadata), patch.object(entry, 'validate_package_files', side_effect=lambda *args: events.append(('full_validation_mocked',))), \
     patch.object(entry.socket, 'gethostname', return_value='h100-9'), \
     patch.dict(os.environ, {'SLURM_JOB_NODELIST': 'h100-9', 'SLURM_JOB_GPUS': '0,1,2,3,4,5,6,7'}, clear=True), \
     patch.object(entry.os, 'chdir', side_effect=lambda path: events.append(('chdir_intercepted', str(path)))), \
     patch.object(entry.os, 'execvpe', side_effect=intercepted_exec):
    entry.main()
check('start.main captured exactly one exec; no srun executed', len(captured) == 1)
program, argv, env = captured[0]
python = str(contract.ROOT / 'env/bin/python')
expected = ['srun', '--ntasks=1', '--kill-on-bad-exit=1', python, '-m', 'search_r1.baselines.preflight', '--',
            python, str(PACKAGE / 'supervisor.py'), '--package', str(PACKAGE), '--manifest-sha256', EXPECTED_MANIFEST,
            '--release', str(contract.ROOT / 'child-release.json'), '--release-sha256', '6' * 64]
check('start exact preflight -> supervisor argv, no shell command string', program == 'srun' and argv == expected)
check('start passes exact manifest/release environment',
    env['TREE_PROBE_PACKAGE_MANIFEST_SHA256'] == EXPECTED_MANIFEST and env['TREE_PROBE_RELEASE_SHA256'] == '6' * 64)
check('start frozen imports and no bytecode', env['PYTHONPATH'] == ':'.join(map(str, [PACKAGE / 'repo', PACKAGE, contract.ROOT / 'tools'])) and env['PYTHONDONTWRITEBYTECODE'] == '1')
check('validation precedes captured launch', events[-1][0] == 'exec_intercepted' and events.index(('full_validation_mocked',)) < len(events) - 1)
for environment in [{'SLURM_JOB_NODELIST': 'h100-16', 'SLURM_JOB_GPUS': '0'}, {'SLURM_JOB_NODELIST': 'h100-9'}]:
    with patch.object(sys, 'argv', arguments), patch.object(entry, 'same_node_path', side_effect=lambda path: Path(path).resolve()), \
         patch.object(entry, 'read_bound', side_effect=metadata), patch.object(entry, 'validate_package_files'), \
         patch.object(entry.socket, 'gethostname', return_value='h100-9'), patch.dict(os.environ, environment, clear=True), \
         patch.object(entry.os, 'execvpe') as execute:
        refused('start missing/wrong Slurm allocation rejected ' + repr(environment), entry.main)
        check('rejected start invokes zero exec ' + repr(environment), execute.call_count == 0)

# Exact API names, paths and output receipts are source checks only.
source = (PACKAGE / 'supervisor.py').read_text()
runner = (PACKAGE / 'repo/search_r1/baselines/probability_probe_runner.py').read_text()
check('supervisor parser accepts forwarded flags', all(flag in source for flag in ['--package', '--manifest-sha256', '--release', '--release-sha256']))
check('guard readiness exact suffix API', "guard_path=run/'resources.jsonl'" in source and "guard_path.with_suffix('.ready')" in source,
      'resources.jsonl -> resources.ready; external guard implementation reviewed by parent')
check('supervisor adapter explicit parent/pre-rank/live API', all(value in source for value in ["('prepare_and_validate_bundle','check')", 'adapter.prepare_and_validate_bundle(bundle) is True', "adapter.check('supervisor_before_torchrun')", "adapter.check('supervisor_live')"]))
check('runner rank directory and ready receipt match consumer', "('rank-%02d'%rank)" in runner and "output/'ready.json'" in runner and "('rank-%02d'%rank)" in (PACKAGE / 'probe_package.py').read_text())
check('torchrun only loopback rendezvous API', "'--master_addr=127.0.0.1'" in source and "'--nproc_per_node=8'" in source)

# Minimal actual-source counterexample: successful empty hostname expansion.
row = '83|tree-pending|PENDING|(null)|gpu:8'
job = 'JobId=83 NodeList=(null) ReqNodeList=h100-9 ReqTRES=cpu=32,gres/gpu=8'
conflicts = dispatch.queue_conflicts([row], lambda _: job, lambda _: [])
check('BUG reproduced: nonempty h100-9 request + empty hostnames yields empty conflicts', conflicts == [])
with tempfile.TemporaryDirectory(dir=HERE, prefix='synthetic-queue-') as temporary:
    intent = Path(temporary) / 'intent.json'
    tools_called = []; submits = []
    def fake_tool(argv, **kwargs):
        tools_called.append(argv)
        if argv[0] == 'squeue': return row + '\n'
        if argv[:3] == ['scontrol', 'show', 'job']: return job
        if argv[:3] == ['scontrol', 'show', 'hostnames']: return ''
        raise AssertionError('unexpected subprocess; no real process')
    def fake_submit(argv):
        submits.append(argv)
        return SimpleNamespace(returncode=0, stdout='999;synthetic', stderr='')
    with patch.object(dispatch.subprocess, 'check_output', side_effect=fake_tool):
        result = dispatch.submit_once(intent, ['never-real-sbatch'], {'synthetic': True}, fake_submit,
                                     pre_submit=lambda: dispatch.refreshed_queue_gate(intent))
    evidence = json.loads(next(Path(temporary).glob('intent.json.queue-*.json')).read_text())
    check('BUG end-to-end mocked fresh queue reaches single fake submit', len(submits) == 1 and result['job_id'] == '999')
    check('BUG queue evidence records no error/conflict despite empty hostnames', evidence['unknown_error'] is None and evidence['conflicts'] == [])
    findings.append(dict(id='EMPTY_EXPANDED_NODE_SET', blocking=True, source='dispatch_once.py', line=49,
        input_queue_row=row, input_scontrol_job=job, synthetic_scontrol_hostnames_stdout='',
        actual_result=dict(conflicts=conflicts, fake_submit_calls=len(submits), queue_error=evidence['unknown_error']),
        expected='Reject empty expanded hostname set as unknown before creating intent/submitting; or preserve h100-9 risk.',
        no_real_scheduler_calls=True, recorded_tool_argv=tools_called))
check('explicit h100-9 hostnames conflicts', len(dispatch.queue_conflicts([row], lambda _: job, lambda _: ['h100-9'])) == 1)
check('truly unknown null nodes conflict', len(dispatch.queue_conflicts([row], lambda _: job.replace('ReqNodeList=h100-9', 'ReqNodeList=(null)'), lambda _: [])) == 1)
check('non-tree own GPU job deliberately excluded by current prefix scope', dispatch.queue_conflicts([row.replace('tree-pending', 'grpo-pending')], lambda _: job, lambda _: ['h100-9']) == [],
      'Scope disclosed; parent/root to determine whether own-job gate means all own GPU jobs or only Tree instances.')

payload = dict(status='BLOCKED_SOURCE_AND_DISPATCH_CPU_REVIEW', package_manifest_sha256=EXPECTED_MANIFEST,
    source_hashes=source_hashes, bundle_payload_sha256=MANIFEST['bundle_payload_sha256'], checks=checks, findings=findings,
    actual_gpu_probe_accepted=False, actual_optimizer_writes=0, remote_calls=0, real_scheduler_calls=0,
    real_gpu_calls=0, real_tensor_or_model_reads=0, frozen_package_modified=False,
    mock_scope='Metadata copied from exact frozen package; start exec intercepted; all scheduler APIs replaced with synthetic callables. Full package validation is parent scope.')
(HERE / 'INDEPENDENT_PROOF.json').write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
print(json.dumps(dict(check_count=len(checks), blocker_count=len(findings), status=payload['status']), sort_keys=True))
