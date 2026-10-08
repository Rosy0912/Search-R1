import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('capture-hardware-devicefix.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CaptureTests(unittest.TestCase):
    def make(self,root,name='runs/hardware/log.txt',payload=b'actual evidence',kind=None):
        archive=root/'raw.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            item=tarfile.TarInfo(name);item.size=len(payload)
            if kind:item.type=kind;item.linkname='/etc/passwd';item.size=0
            tf.addfile(item,None if kind else io.BytesIO(payload))
        inventory={'files':[{'archive_path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
        return archive,inventory
    def test_valid_inventory_exact_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,inv=self.make(root)
            m.safe_extract(archive,root/'extract',inv)
            self.assertEqual((root/'extract/runs/hardware/log.txt').read_bytes(),b'actual evidence')
    def test_traversal_and_absolute_refused(self):
        for name in ('../outside','/absolute'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);archive,inv=self.make(root,name)
                with self.assertRaises(ValueError):m.safe_extract(archive,root/'extract',inv)
                self.assertFalse((root/'extract').exists())
    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,inv=self.make(root,kind=tarfile.SYMTYPE)
            with self.assertRaises(ValueError):m.safe_extract(archive,root/'extract',inv)
    def test_uninventoried_file_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,inv=self.make(root);inv['files']=[]
            with self.assertRaises(ValueError):m.safe_extract(archive,root/'extract',inv)
    def test_changed_file_sha_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,inv=self.make(root);inv['files'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):m.safe_extract(archive,root/'extract',inv)
    def test_worker_script_compiles_fixed_identity(self):
        script=m.worker_code(123,'JobId=123 JobState=COMPLETED ExitCode=0:0\n')
        compile(script,'worker-capture','exec')
        self.assertIn(m.SOURCE,script)
        self.assertIn("RAW_CAPTURE_ONLY_REQUIRES_NONAUTHOR_INDEPENDENT_REVIEW",script)
        self.assertNotIn("PASS_INDEPENDENT",script)
    def test_existing_extraction_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,inv=self.make(root);(root/'extract').mkdir()
            with self.assertRaises(ValueError):m.safe_extract(archive,root/'extract',inv)

class SchedulerFallbackTests(unittest.TestCase):
    raw=b'JobId=123 JobName=bapo-instruct300-hardware JobState=COMPLETED NodeList=h100-5 ExitCode=0:0\n'
    def run_cases(self, responses):
        import subprocess
        calls=[]
        def runner(cmd):
            calls.append(cmd);code,out=responses.pop(0)
            return subprocess.CompletedProcess(cmd,code,stdout=out,stderr=b'query failed' if code else b'')
        return runner,calls
    def test_live_original_preferred_no_fallback(self):
        runner,calls=self.run_cases([(0,self.raw)])
        raw,origin,_=m.scheduler_terminal(123,runner)
        self.assertEqual(raw,self.raw);self.assertEqual(origin['origin'],'LIVE_SCONTROL');self.assertEqual(len(calls),1)
    def test_failed_live_uses_preserved_original_and_origin(self):
        runner,calls=self.run_cases([(1,b''),(0,self.raw)])
        raw,origin,_=m.scheduler_terminal(123,runner)
        self.assertEqual(raw,self.raw);self.assertEqual(origin['origin'],'HEAD_WATCHER_PRESERVED_ORIGINAL')
        self.assertIn('hardware-job-id.txt',calls[1]);self.assertIn('hardware-123-terminal.txt',calls[1])
    def test_failed_both_never_terminal(self):
        runner,_=self.run_cases([(1,b''),(1,b'')])
        with self.assertRaises(ValueError):m.scheduler_terminal(123,runner)
    def test_active_live_does_not_fallback(self):
        runner,calls=self.run_cases([(0,self.raw.replace(b'COMPLETED',b'RUNNING'))])
        with self.assertRaises(ValueError):m.scheduler_terminal(123,runner)
        self.assertEqual(len(calls),1)
    def test_fallback_wrong_identity_refused(self):
        for old,new in [(b'JobId=123',b'JobId=999'),(b'NodeList=h100-5',b'NodeList=h100-4'),
                        (b'JobName=bapo-instruct300-hardware',b'JobName=other')]:
            runner,_=self.run_cases([(1,b''),(0,self.raw.replace(old,new))])
            with self.assertRaises(ValueError):m.scheduler_terminal(123,runner)
    def test_fallback_active_or_malformed_refused(self):
        for raw in (b'',self.raw.replace(b'COMPLETED',b'PENDING'),self.raw+b' JobId=123'):
            runner,_=self.run_cases([(1,b''),(0,raw)])
            with self.assertRaises(ValueError):m.scheduler_terminal(123,runner)
    def test_failure_exit_is_real_terminal_not_pass(self):
        raw=self.raw.replace(b'COMPLETED',b'FAILED').replace(b'ExitCode=0:0',b'ExitCode=1:0')
        runner,_=self.run_cases([(1,b''),(0,raw)])
        got,origin,_=m.scheduler_terminal(123,runner)
        self.assertEqual(got,raw);self.assertNotIn('PASS',json.dumps(origin))

if __name__=='__main__':unittest.main()
