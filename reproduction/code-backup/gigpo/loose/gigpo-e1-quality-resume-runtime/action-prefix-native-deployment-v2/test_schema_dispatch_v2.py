"""Only the 3297 actual-binding schema/CLI regressions; no GPU or target reads."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'work/gigpo-e1-quality-resume-runtime'
OUTPUT = ROOT / 'outputs/gigpo-e1-policy-safety'
BINDING = OUTPUT / 'ACTION-PREFIX-NATIVE-BINDING-2262942-v1.json'
METADATA = OUTPUT / 'ACTION-PREFIX-NATIVE-FAILURE3297-METADATA-v1.json'
BINDING_SHA = 'c5e184a458cc6ae10cb3c924c845564f1684e0347dfc9fb0391c2fd1b7bd7c76'
METADATA_SHA = 'a8ac827a468d8015264ac6768ca6443773ad588d4f72c04f0be58b0ccfe31cdd'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NATIVE = load('schema_native_v2', BASE / 'action-prefix-native-v2/action_prefix_native_control_v1.py')
OLD = load('schema_native_preserved_v1', BASE / 'action-prefix-native-v1/action_prefix_native_control_v1.py')
DEPLOY = load('schema_deploy_v2', BASE / 'action-prefix-native-deployment-v2/deployment_contract_v1.py')


class ActualBindingAndCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256(BINDING.read_bytes()).hexdigest() == BINDING_SHA
        assert hashlib.sha256(METADATA.read_bytes()).hexdigest() == METADATA_SHA
        cls.bound = json.loads(BINDING.read_text())
        cls.meta = json.loads(METADATA.read_text())
        assert cls.meta['binding']['sha256'] == BINDING_SHA
        assert cls.meta['binding_alias_summary']['files'] == dict(entries=42353, raw_duplicate_paths=[], resolved_duplicate_groups=1)
        assert cls.meta['binding_alias_summary']['evidence'] == dict(entries=421, raw_duplicate_paths=[], resolved_duplicate_groups=0)
        cls.alias = cls.meta['resolved_aliases'][0]['items']
        assert len(cls.alias) == 3
        assert {item['realpath'] for item in cls.alias} == {'/usr/bin/python3.10'}
        assert all(item['bytes'] == item['actual_bytes'] == 5937672 for item in cls.alias)
        assert len({(item['bytes'], item['sha256'], item['dev'], item['ino']) for item in cls.alias}) == 1
        cls.alias_map = {item['path']: item['realpath'] for item in cls.alias}
        # Only these three literal realpaths were preserved by the target probe.
        # All other recorded paths are unique placeholders in this CPU schema
        # replay, justified by the original target summary's one alias group.
        # They are NOT a claim that every other target realpath was observed here.
        cls.resolve = staticmethod(lambda path, strict=False: Path(cls.alias_map.get(str(path), str(path))))
        cls.recorded_sha = {item['path']: item['sha256'] for item in cls.bound['files'] + cls.bound['evidence']}
        files = {item['path']: item for item in cls.bound['files']}
        assert len(files) == len(cls.bound['files']) == 42353
        assert all({k: item[k] for k in ('path', 'bytes', 'sha256')} == files[item['path']] for item in cls.alias)

    def validate_actual(self, value, module=NATIVE):
        # This callback replays the original immutable SHA declarations; it
        # never reads a target model, environment or checkpoint. Target fullSHA
        # checks remain mandatory in native_run_safety_v1 and physical/once.
        with patch.object(Path, 'resolve', self.resolve):
            return module.validate_binding(value, value['helpers'], lambda path: self.recorded_sha[str(path)])

    def test_actual_c5_all_remaining_native_fields_and_equal_aliases_pass(self):
        self.assertTrue(self.validate_actual(copy.deepcopy(self.bound)))

    def test_actual_c5_exact_old_checker_reproduces_resolved_duplicate_failure(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate bound file'):
            self.validate_actual(copy.deepcopy(self.bound), OLD)

    def test_actual_raw_duplicate_inside_files_rejected(self):
        value = copy.deepcopy(self.bound)
        value['files'].append(copy.deepcopy(value['files'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate raw bound path in files'):
            self.validate_actual(value)

    def test_actual_raw_duplicate_inside_evidence_rejected(self):
        value = copy.deepcopy(self.bound)
        value['evidence'].append(copy.deepcopy(value['evidence'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate raw bound path in evidence'):
            self.validate_actual(value)

    def test_actual_cross_list_identical_path_bytes_sha_is_valid_reference(self):
        value = copy.deepcopy(self.bound)
        value['evidence'].append(copy.deepcopy(value['files'][0]))
        self.assertTrue(self.validate_actual(value))

    def test_actual_cross_list_different_sha_rejected(self):
        value = copy.deepcopy(self.bound)
        item = copy.deepcopy(value['files'][0]); item['sha256'] = '0' * 64
        value['evidence'].append(item)
        with self.assertRaisesRegex(ValueError, 'Conflicting cross-list bound identity'):
            self.validate_actual(value)

    def test_actual_cross_list_different_bytes_rejected(self):
        value = copy.deepcopy(self.bound)
        item = copy.deepcopy(value['files'][0]); item['bytes'] += 1
        value['evidence'].append(item)
        with self.assertRaisesRegex(ValueError, 'Conflicting cross-list bound identity'):
            self.validate_actual(value)

    def test_actual_alias_conflicting_sha_rejected(self):
        value = copy.deepcopy(self.bound)
        next(item for item in value['files'] if item['path'] == self.alias[1]['path'])['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Conflicting resolved alias identity'):
            self.validate_actual(value)

    def test_actual_alias_conflicting_bytes_rejected(self):
        value = copy.deepcopy(self.bound)
        next(item for item in value['files'] if item['path'] == self.alias[1]['path'])['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'Conflicting resolved alias identity'):
            self.validate_actual(value)

    def test_actual_alias_missing_bytes_rejected(self):
        value = copy.deepcopy(self.bound)
        del next(item for item in value['files'] if item['path'] == self.alias[1]['path'])['bytes']
        with self.assertRaisesRegex(ValueError, 'Incomplete bound identity'):
            self.validate_actual(value)

    def test_actual_budget_or_helper_still_not_bypassed(self):
        for key, wrong in [('expected_optimizer_steps_per_outer', 20), ('gpu_count', 4), ('native_actual_restore', True)]:
            value = copy.deepcopy(self.bound); value[key] = wrong
            with self.assertRaises(ValueError): self.validate_actual(value)
        value = copy.deepcopy(self.bound)
        value['files'] = [item for item in value['files'] if not item['path'].endswith('/native_run_safety_v1.py')]
        with self.assertRaisesRegex(ValueError, 'New helper omitted/changed'):
            self.validate_actual(value)

    def test_new_claim_paths_and_runtime_namespaces_are_distinct(self):
        for key in ('CANDIDATE', 'PHYSICAL', 'READY', 'BINDING', 'INTENT', 'RECEIPT'):
            self.assertTrue(str(getattr(DEPLOY, key)).endswith('-2262942-v2.json'))
        self.assertTrue(str(DEPLOY.HELPERS).endswith('action-prefix-native-2262942-v2'))
        self.assertTrue(str(DEPLOY.CONTROLS).endswith('action-prefix-native-2262942-control-v2'))

    def test_native_gpu_wrapper_and_resources_remain_original_bytes(self):
        old = BASE / 'action-prefix-native-v1'; new = BASE / 'action-prefix-native-v2'
        for path in old.iterdir():
            if path.is_file() and path.name not in ('action_prefix_native_control_v1.py', 'test_action_prefix_native_control_v1.py'):
                self.assertEqual(path.read_bytes(), (new / path.name).read_bytes(), path.name)

    def expression_case(self, directory, should_fail):
        path = BASE / directory / 'submit_once_v1.py'
        body = ast.parse(path.read_text()).body[-1].body[-1]
        expression = ast.get_source_segment(path.read_text(), body)
        with tempfile.TemporaryDirectory(prefix='native-v2-cli-cpu-') as temp:
            temp = Path(temp); calls = []
            result = dict(status='SUBMITTED_EXACTLY_ONCE', job='999999', synthetic=True)
            def fake_submit(value):
                with (temp / 'intent.json').open('x') as f: json.dump({'synthetic': True}, f)
                calls.append(value)
                with (temp / 'receipt.json').open('x') as f: json.dump(result, f)
                return result
            stream = io.StringIO(); env = dict(json=json, submit=fake_submit, sys=types.SimpleNamespace(argv=['fake', 'a' * 64]))
            with contextlib.redirect_stdout(stream):
                if should_fail:
                    with self.assertRaises(TypeError): exec(compile(expression, str(path), 'exec'), env)
                else: exec(compile(expression, str(path), 'exec'), env)
            self.assertEqual(calls, ['a' * 64])
            self.assertEqual(json.loads((temp / 'receipt.json').read_text()), result)
            if should_fail: self.assertEqual(stream.getvalue(), '')
            else: self.assertEqual(json.loads(stream.getvalue()), result)
            with self.assertRaises(FileExistsError): fake_submit('a' * 64)
            self.assertEqual(len(calls), 1)

    def test_actual_frozen_cli_expression_fails_only_after_fake_receipt(self):
        self.expression_case('action-prefix-native-deployment-v1', True)

    def test_correct_v2_cli_expression_prints_once_after_fake_receipt(self):
        self.expression_case('action-prefix-native-deployment-v2', False)

    def run_cli_stub(self, uncertain=False):
        """Execute the whole new __main__/submit body with no real command.

        Only READY/schema validation is a declared stub here. This is not a
        target READY test. Local exclusive files exercise the actual submit
        control's claim/binding/receipt and entry-point output ordering.
        """
        path = BASE / 'action-prefix-native-deployment-v2/submit_once_v1.py'
        with tempfile.TemporaryDirectory(prefix='native-v2-whole-cli-cpu-') as temp:
            temp = Path(temp); calls = []
            stub = types.ModuleType('deployment_contract_v1')
            stub.ROOT = temp; stub.SOURCE = DEPLOY.SOURCE
            stub.HELPERS = temp / 'synthetic-helpers'; stub.CONTROLS = path.parent
            for name in ('READY', 'BINDING', 'INTENT', 'RECEIPT'):
                setattr(stub, name, temp / (name.lower() + '.json'))
            def require(value, message):
                if not value: raise ValueError(message)
            stub.require = require
            stub.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            def write_once(p, value):
                with Path(p).open('x') as f: json.dump(value, f)
            stub.write_once = write_once
            def read_bound(record):
                require(stub.sha(record['path']) == record['sha256'], 'Synthetic READY changed')
                return json.loads(Path(record['path']).read_text())
            stub.read_bound = read_bound
            stub.validate_ready = lambda value: {'repo': str(temp / 'synthetic-repo')}
            record_path = temp / 'synthetic-proof.json'; record_path.write_text('{}\n')
            record = dict(path=str(record_path), bytes=record_path.stat().st_size, sha256=stub.sha(record_path))
            ready = dict(files=[record], evidence=[record], deployment_review=record,
                         candidate_record=record, physical_record=record, synthetic=True)
            stub.write_once(stub.READY, ready); expected = stub.sha(stub.READY)
            def fake_run(command, **kwargs):
                calls.append(command)
                if command[0] in ('squeue', 'nvidia-smi'):
                    return types.SimpleNamespace(returncode=0, stdout='', stderr='')
                require(command[0] == 'sbatch' and command[1] == '--parsable', 'Unexpected synthetic command')
                require(len(command) == 7 and command[3] == str(stub.BINDING)
                        and command[4] == stub.sha(stub.BINDING), 'CLI changed exact sbatch four arguments')
                if uncertain: raise TimeoutError('SYNTHETIC_NO_COMMAND_EXECUTED')
                return types.SimpleNamespace(returncode=0, stdout='999998\n', stderr='')
            stream = io.StringIO()
            with patch.dict(sys.modules, {'deployment_contract_v1': stub}), \
                 patch.object(sys, 'argv', [str(path), expected]), \
                 patch.object(subprocess, 'run', fake_run), \
                 patch.object(socket, 'gethostname', lambda: 'h100-8'), \
                 contextlib.redirect_stdout(stream):
                if uncertain:
                    with self.assertRaisesRegex(TimeoutError, 'SYNTHETIC_NO_COMMAND_EXECUTED'):
                        runpy.run_path(str(path), run_name='__main__')
                else:
                    runpy.run_path(str(path), run_name='__main__')
                self.assertEqual(len(calls), 3)
                self.assertTrue(stub.INTENT.exists() and stub.BINDING.exists() and stub.RECEIPT.exists())
                receipt = json.loads(stub.RECEIPT.read_text())
                self.assertEqual(receipt['status'], 'UNCERTAIN_NO_RESUBMISSION' if uncertain else 'SUBMITTED_EXACTLY_ONCE')
                if uncertain: self.assertEqual(stream.getvalue(), '')
                else:
                    self.assertEqual(json.loads(stream.getvalue()), receipt)
                    self.assertEqual(receipt['job'], '999998')
                with self.assertRaisesRegex(ValueError, 'Existing claim/uncertain result'):
                    runpy.run_path(str(path), run_name='__main__')
                self.assertEqual(len(calls), 3)

    def test_full_v2_cli_stub_durable_claim_receipt_print_and_no_second_dispatch(self):
        self.run_cli_stub()

    def test_full_v2_cli_stub_uncertain_dispatch_is_preserved_and_no_retry(self):
        self.run_cli_stub(uncertain=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
