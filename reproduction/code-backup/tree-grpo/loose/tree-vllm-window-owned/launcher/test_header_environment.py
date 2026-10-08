"""Own syntheticCPU environment proof; no shared headers, Slurm, GPU or PT IO."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import supervisor
import probe_package


class HeaderEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.include=self.root/'python-dev/usr/include';self.primary=self.include/'python3.10'
        self.headers=[self.primary/'Python.h',self.primary/'pyconfig.h',
                      self.include/'x86_64-linux-gnu/python3.10/pyconfig.h']
        for header in self.headers:
            header.parent.mkdir(parents=True,exist_ok=True);header.write_bytes(b'/* own synthetic fixture */\n')
        self.root_patch=patch.object(supervisor,'ROOT',self.root);self.root_patch.start()
    def tearDown(self):
        self.root_patch.stop();self.temp.cleanup()
    def test_exact_existing_legacy_include_environment(self):
        source=Path(__file__).resolve().parents[2]/'tree-safety110-runtime/supervisor.py'
        tree=ast.parse(source.read_text())
        values=[node.value for node in ast.walk(tree) if isinstance(node,ast.keyword) and node.arg=='C_INCLUDE_PATH']
        self.assertEqual(len(values),1)
        original=eval(compile(ast.Expression(values[0]),str(source),'eval'),{'ROOT':self.root})
        self.assertEqual(supervisor.compiler_header_environment(),{'C_INCLUDE_PATH':original})
    def test_missing_header_root_refused(self):
        shutil.rmtree(self.root/'python-dev')
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_missing_python_h_refused(self):
        self.headers[0].unlink()
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_missing_pyconfig_wrapper_refused(self):
        self.headers[1].unlink()
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_missing_multiarch_pyconfig_refused(self):
        self.headers[2].unlink()
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_missing_primary_include_directory_refused(self):
        shutil.rmtree(self.primary)
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_header_escape_refused(self):
        outside=self.root/'outside-header.h';outside.write_bytes(b'/* own outside fixture */')
        self.headers[2].unlink();self.headers[2].symlink_to(outside)
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_empty_header_refused(self):
        self.headers[0].write_bytes(b'')
        with self.assertRaises(ValueError):supervisor.compiler_header_environment()
    def test_unreadable_header_refused(self):
        with patch.object(Path,'open',side_effect=PermissionError('own synthetic read error')):
            with self.assertRaises(PermissionError):supervisor.compiler_header_environment()
    def test_actual_child_inherits_exact_new_value_not_unrelated_old_include(self):
        with patch.dict(os.environ,{'C_INCLUDE_PATH':'/unrelated/old/include'}):
            os.environ.update(supervisor.compiler_header_environment())
            child=subprocess.run([sys.executable,'-B','-c','import os; print(os.environ["C_INCLUDE_PATH"])'],
                env={'C_INCLUDE_PATH':os.environ['C_INCLUDE_PATH']},text=True,capture_output=True,check=True)
            self.assertEqual(child.stdout.strip(),str(self.primary)+':'+str(self.include))
            self.assertNotIn('/unrelated/old/include',child.stdout)


class MinimalDiffTests(unittest.TestCase):
    def test_original_header_and_owned_resource_functions_unchanged(self):
        root=Path(supervisor.__file__).resolve().parent
        parent=root.parent.parent/'tree-probability-probe-owned/supervisor-candidate-r3'
        def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        before,after=funcs(parent/'supervisor.py'),funcs(root/'supervisor.py')
        for name in ('compiler_header_environment','wait_probe','process','protected','healthy_guard','owned_adapter'):
            self.assertEqual(before[name],after[name],name)
        old_adapter=root.parent.parent/'tree-probability-probe-owned/resource_adapter_v2.py'
        before,after=funcs(old_adapter),funcs(root/'resource_adapter_window.py')
        for name in ('fdinfo_exclusive_owner','process','allocation','guard_record','check','installed_bindings'):
            self.assertEqual(before[name],after[name],name)
    def test_start_bytes_and_header_application_before_spawn_preserved(self):
        root=Path(supervisor.__file__).resolve().parent
        parent=root.parent.parent/'tree-probability-probe-owned/supervisor-candidate-r3'
        for name in ('start.py','start.sh'):
            self.assertEqual((root/name).read_bytes(),(parent/name).read_bytes(),name)
        text=(root/'supervisor.py').read_text()
        self.assertLess(text.index('os.environ.update(compiler_header_environment())'),text.index('probe=owned.spawn'))
        self.assertIn("compiler_environment=dict(C_INCLUDE_PATH=os.environ['C_INCLUDE_PATH'])",text)


class HeaderIntentTests(unittest.TestCase):
    def test_parent_header_intent_preserved_and_new_window_is_separate(self):
        root=Path(supervisor.__file__).resolve().parent
        parent=root.parent.parent/'tree-probability-probe-owned/supervisor-candidate-r3/probe_package.py'
        self.assertIn('PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json',parent.read_text())
        self.assertNotEqual(probe_package.DISPATCH_INTENT,str(probe_package.ROOT/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json'))
    def test_new_fixed_intent_only_old_and_alternative_refused(self):
        from test_candidate import MetadataTests
        template,manifest,receipt=MetadataTests().envelopes()
        expected=str(probe_package.ROOT/'instruct300/SINGLE_TP1_WINDOW_AFTER_3282_ONCE.json')
        self.assertEqual(probe_package.DISPATCH_INTENT,expected)
        self.assertTrue(probe_package.validate_release(manifest,'5'*64,template,receipt))
        for rejected in ('PROBABILITY_PROBE_ONCE-3268.json','arbitrary-retry.json'):
            receipt['dispatch_intent_path']=str(probe_package.ROOT/'instruct300'/rejected)
            with self.assertRaises(ValueError):probe_package.validate_release(manifest,'5'*64,template,receipt)


if __name__=='__main__':unittest.main(verbosity=2)
