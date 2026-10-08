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
    def test_only_header_function_environment_application_and_launch_record_in_source(self):
        current=Path(supervisor.__file__).resolve()
        previous=current.parent.parent/'supervisor-candidate-r2/supervisor.py'
        old=ast.parse(previous.read_text());new=ast.parse(current.read_text())
        added=[node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=='compiler_header_environment']
        self.assertEqual(len(added),1);new.body.remove(added[0])
        main=next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        expressions=[node for node in main.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
            and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Attribute)
            and isinstance(node.value.func.value.value,ast.Name) and node.value.func.value.value.id=='os'
            and node.value.func.value.attr=='environ' and node.value.func.attr=='update'
            and len(node.value.args)==1 and isinstance(node.value.args[0],ast.Call)
            and isinstance(node.value.args[0].func,ast.Name) and node.value.args[0].func.id=='compiler_header_environment']
        self.assertEqual(len(expressions),1)
        self.assertLess(main.body.index(expressions[0]),next(i for i,n in enumerate(main.body) if isinstance(n,ast.Try)))
        main.body.remove(expressions[0])
        receipts=[node for node in ast.walk(main) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id=='dict' and any(key.arg=='compiler_environment' for key in node.keywords)]
        self.assertEqual(len(receipts),1)
        receipt=receipts[0];field=next(key for key in receipt.keywords if key.arg=='compiler_environment')
        self.assertEqual(ast.dump(field.value,include_attributes=False),ast.dump(ast.parse(
            "dict(C_INCLUDE_PATH=os.environ['C_INCLUDE_PATH'])",mode='eval').body,include_attributes=False))
        receipt.keywords.remove(field)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
    def test_dispatch_start_manifest_and_existing_test_bytes_unchanged(self):
        root=Path(supervisor.__file__).resolve().parent;old=root.parent/'supervisor-candidate-r2'
        for name in ('dispatch_once.py','start.py','start.sh','freeze_package.py','test_candidate.py'):
            self.assertEqual((root/name).read_bytes(),(old/name).read_bytes(),name)


class HeaderIntentTests(unittest.TestCase):
    def test_only_explicit_new_intent_and_error_literal_changed(self):
        root=Path(supervisor.__file__).resolve().parent
        old=(root.parent/'supervisor-candidate-r2/probe_package.py').read_text()
        new=(root/'probe_package.py').read_text()
        normalized=new.replace('PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json','PROBABILITY_PROBE_ONCE-3268.json').replace(
            'Header repair must use its new fixed intent; old or arbitrary intent paths are refused',
            'This probe must keep its original fixed single-attempt intent path')
        self.assertEqual(normalized,old)
    def test_new_fixed_intent_only_old_and_alternative_refused(self):
        from test_candidate import MetadataTests
        template,manifest,receipt=MetadataTests().envelopes()
        expected=str(probe_package.ROOT/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json')
        self.assertEqual(probe_package.DISPATCH_INTENT,expected)
        self.assertTrue(probe_package.validate_release(manifest,'5'*64,template,receipt))
        for rejected in ('PROBABILITY_PROBE_ONCE-3268.json','arbitrary-retry.json'):
            receipt['dispatch_intent_path']=str(probe_package.ROOT/'instruct300'/rejected)
            with self.assertRaises(ValueError):probe_package.validate_release(manifest,'5'*64,template,receipt)


if __name__=='__main__':unittest.main(verbosity=2)
