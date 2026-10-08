"""Pure payload and local mkdir fixtures; never executes the publisher as a whole."""
import ast
import base64
import copy
import hashlib
import io
import json
import pathlib
import tempfile
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = (HERE / 'publisher.py').read_text()
PAYLOAD = json.loads((HERE / 'payload.original.json').read_bytes())
PREPARATION = json.loads((HERE / 'preparation.original.json').read_bytes())
TREE = ast.parse(SOURCE)


def validation_only(payload):
    # Exact original AST from payload=... up to def syncdir: host/source guards,
    # filesystem publication and result emission are not executed in this fixture.
    start = next(i for i, node in enumerate(TREE.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'payload' for t in node.targets))
    stop = next(i for i, node in enumerate(TREE.body) if isinstance(node, ast.FunctionDef) and node.name == 'syncdir')
    module = ast.Module(body=TREE.body[start:stop], type_ignores=[])
    context = {'pathlib': pathlib, 'base64': base64, 'hashlib': hashlib, 'json': json,
               'sys': types.SimpleNamespace(stdin=types.SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())))}
    exec(compile(module, str(HERE / 'publisher.py') + ':pure-payload-fixture', 'exec'), context)
    return context


class PublicationPreparation(unittest.TestCase):
    def test_only_six_explicit_literal_differences(self):
        original = pathlib.Path(PREPARATION['stable_input_records']['successful_base_publisher']['physical_path']).read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), PREPARATION['base_publisher_sha256'])
        expected = original
        for old, new in PREPARATION['literal_only_differences']:
            self.assertEqual(expected.count(old), 1)
            expected = expected.replace(old, new, 1)
        self.assertEqual(len(PREPARATION['literal_only_differences']), 6)
        self.assertEqual(SOURCE, expected)
        # The reviewed write/readback source is unchanged, not a new semantic common.
        self.assertEqual(SOURCE[SOURCE.index('def syncdir'):SOURCE.index("print(json.dumps")],
                         original[original.index('def syncdir'):original.index("print(json.dumps")].replace("'final200-eval-native-cpu-'", "'final200-prerequisites-'", 1))

    def test_actual_payload_all_40_plus_original_manifest(self):
        context = validation_only(PAYLOAD)
        self.assertEqual(len(context['files']), 40)
        self.assertEqual(context['m']['kind'], 'STRICT_FINAL200_PREREQUISITES_READ_ONLY_RUNTIME_BUNDLE')
        self.assertEqual(context['m']['source_commit'], 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6')
        self.assertEqual(context['m']['tools']['common.py'], 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581')
        self.assertEqual(context['m']['tools']['verify_actual200_prerequisites.py'], '7dff811b01b302f45c729ee12ae9ee22bb561b359f24fcd0d37352192db2d075')
        self.assertFalse(PREPARATION['publisher_executed'])
        self.assertIsNone(PREPARATION['actual_remote_deployment'])

    def test_wrong_manifest_digest_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['manifest_sha256'] = '0' * 64
        with self.assertRaises(AssertionError):
            validation_only(value)

    def test_modified_manifest_body_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['manifest_base64'] = base64.b64encode(base64.b64decode(value['manifest_base64']) + b' ').decode()
        with self.assertRaises(AssertionError):
            validation_only(value)

    def test_corrupt_semantic_common_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['files']['common.py'] = base64.b64encode(b'native-ee31-control-common-fixture').decode()
        with self.assertRaises(AssertionError):
            validation_only(value)

    def test_extra_member_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['files']['extra.py'] = ''
        with self.assertRaises(AssertionError):
            validation_only(value)

    def test_missing_member_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['files'].pop('common.py')
        with self.assertRaises(AssertionError):
            validation_only(value)

    def test_malformed_base64_rejected(self):
        value = copy.deepcopy(PAYLOAD)
        value['files']['common.py'] = '***invalid***'
        with self.assertRaises(ValueError):
            validation_only(value)

    def test_existing_directory_and_symlink_refuse_without_overwrite(self):
        node = next(n for n in TREE.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'out' for t in n.targets))
        mkdir = next(n for n in TREE.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'mkdir')
        code = compile(ast.Module(body=[node, mkdir], type_ignores=[]), str(HERE / 'publisher.py') + ':mkdir-only-fixture', 'exec')
        with tempfile.TemporaryDirectory() as name:
            parent = pathlib.Path(name)
            out = parent / 'final200-prerequisites-85f014e167c8'
            out.mkdir()
            original = out / 'original'; original.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                exec(code, {'parent': parent, 'sha': PREPARATION['bundle_manifest_sha256']})
            self.assertEqual(original.read_bytes(), b'keep')
            original.unlink(); out.rmdir()
            target = parent / 'existing'; target.mkdir()
            out.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                exec(code, {'parent': parent, 'sha': PREPARATION['bundle_manifest_sha256']})
            self.assertTrue(out.is_symlink())
            self.assertEqual(list(target.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
