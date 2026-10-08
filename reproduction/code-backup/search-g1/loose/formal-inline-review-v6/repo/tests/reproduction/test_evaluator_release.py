import ast
import contextlib
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from reproduction.a800 import runtime

class EvaluatorReleaseTests(unittest.TestCase):
    def suspend(self,allocated):
        source=Path(__file__).resolve().parents[2]/'probe_server.py';tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='suspend');fn.decorator_list=[]
        ns=dict(STATE=dict(version='r:0:v',model=object(),tok=object()),LOCK=threading.Lock(),evaluator_gpu_lock=contextlib.nullcontext,EvaluatorControl=object,
                gc=types.SimpleNamespace(collect=lambda:None),torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,memory_allocated=lambda:allocated)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
        return ns
    def test_actual_endpoint_clears_encoder_and_reports_release(self):
        ns=self.suspend(0);result=ns['suspend'](types.SimpleNamespace(version='r:0:v'))
        self.assertTrue(result['suspended']);self.assertIsNone(ns['STATE']['model']);self.assertIsNone(ns['STATE']['tok'])
    def test_wrong_version_or_retained_allocation_rejected(self):
        ns=self.suspend(0)
        with self.assertRaises(AssertionError):ns['suspend'](types.SimpleNamespace(version='other'))
        self.assertIsNotNone(ns['STATE']['model'])
        ns=self.suspend(256*1024*1024)
        with self.assertRaises(AssertionError):ns['suspend'](types.SimpleNamespace(version='r:0:v'))
    def test_refit_cannot_start_without_actual_release_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'artifacts/live').mkdir(parents=True)
            (base/'artifacts/live/manifest.json').write_text(json.dumps({'version':'r:0:v'}))
            response=types.SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(version='r:0:v',suspended=True,allocated_bytes_after_release=256*1024*1024))
            session=types.SimpleNamespace(post=lambda *a,**kw:response,trust_env=False)
            with patch.dict(os.environ,{'SEARCH_G1_BASE':d}),patch('requests.Session',return_value=session),patch.object(runtime.subprocess,'run') as run:
                with self.assertRaises(AssertionError):runtime.refresh_after_checkpoint(50,'/unused')
                run.assert_not_called()

if __name__=='__main__':unittest.main()
