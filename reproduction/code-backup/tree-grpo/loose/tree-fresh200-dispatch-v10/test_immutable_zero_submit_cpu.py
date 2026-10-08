import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import submit_tree_fresh200_once as s


def require(condition,message):
    if not condition:raise ValueError(message)


class ImmutableZeroSubmitTests(unittest.TestCase):
    def fixture(self,tmp,mutation='change',out=b'777;wuhan-h100\n',err=b'',timeout=False):
        root=Path(tmp);(root/'instruct300').mkdir();(root/'logs').mkdir()
        target=root/'instruct300/candidates/test';target.mkdir(parents=True)
        launch=(Path(__file__).resolve().parents[1]/'tree-fresh200-mu-runtime-v10/launch.sbatch').read_bytes()
        script=target/'launch.sbatch';script.write_bytes(launch)
        common=SimpleNamespace(ROOT=root,STAGE='fresh_mu_calibration',COMMIT='af67976ec3b544eddaaa68c35411a46b9db7af68',
            WRAPPER_HASHES={'launch.sbatch':hashlib.sha256(launch).hexdigest()},
            require=require,protected_processes=lambda:None,queue_clear=lambda x:None)
        calls=[];submitted=[]
        def run(argv,**kw):
            calls.append(argv)
            if argv==['scontrol','--local','show','config']:return SimpleNamespace(returncode=0,stdout='ClusterName = wuhan-h100\n',stderr='')
            if argv==['scontrol','--local','show','federation']:return SimpleNamespace(returncode=0,stdout='',stderr='')
            if argv[0]=='squeue':
                if mutation=='change':script.write_bytes(launch.replace(b'03:00:00',b'999:00:00'))
                elif mutation=='delete':script.unlink()
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            if argv[0]=='sbatch':
                submitted.append(kw['input'])
                self.assertIs(type(kw['input']),bytes)
                self.assertEqual(kw['input'],launch)
                self.assertFalse(kw['text'])
                self.assertNotIn(str(script),argv)
                journal=json.loads(next((root/'instruct300').glob('SUBMISSION_ONCE-*.json')).read_text())
                self.assertEqual(journal['actual_submit_stdin_sha256'],hashlib.sha256(kw['input']).hexdigest())
                self.assertEqual(base64.b64decode(journal['actual_submit_stdin_base64']),kw['input'])
                self.assertEqual(journal['actual_submit_argv'],argv)
                self.assertEqual(journal['status'],'SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN')
                if timeout:raise subprocess.TimeoutExpired(argv,60,output=b'77\xff',stderr=b'partial')
                return SimpleNamespace(returncode=0,stdout=out,stderr=err)
            raise AssertionError('Unexpected actual command: '+repr(argv))
        return common,target,script,launch,run,calls,submitted

    def submit(self,c,target,run):
        with contextlib.redirect_stdout(io.StringIO()):return s.submit_once(c,target,'a'*64,'b'*64,{},run)

    def test_real_function_queue_mutation_submits_reviewed_bytes_and_refuses_second_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,target,script,launch,run,calls,submitted=self.fixture(tmp)
            result=self.submit(c,target,run)
            self.assertEqual(result['status'],'SUBMITTED')
            self.assertEqual(result['returned_cluster'],'wuhan-h100')
            script.write_bytes(launch)
            with self.assertRaises(FileExistsError):self.submit(c,target,run)
            self.assertEqual(submitted,[launch])

    def test_queue_deleted_source_path_still_submits_only_reviewed_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,target,script,launch,run,calls,submitted=self.fixture(tmp,mutation='delete')
            self.assertEqual(self.submit(c,target,run)['job_id'],777)
            self.assertFalse(script.exists());self.assertEqual(submitted,[launch])

    def test_timeout_keeps_original_input_and_raw_partial_outputs_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,target,script,launch,run,calls,submitted=self.fixture(tmp,timeout=True)
            with self.assertRaises(ValueError):self.submit(c,target,run)
            result=json.loads(next((c.ROOT/'instruct300').glob('SUBMISSION_ONCE-*.json')).read_text())
            self.assertEqual(result['status'],'SUBMISSION_UNKNOWN_DO_NOT_RETRY')
            self.assertEqual(base64.b64decode(result['exception_stdout']['data']),b'77\xff')
            self.assertEqual(base64.b64decode(result['actual_submit_stdin_base64']),launch)
            script.write_bytes(launch)
            with self.assertRaises(FileExistsError):self.submit(c,target,run)
            self.assertEqual(len(submitted),1)

    def test_invalid_or_foreign_native_outputs_preserve_unknown_raw_bytes(self):
        for output,err in [(b'\xff',b''),(b'777;foreign\n',b''),(b'777\n',b'warning')]:
            with tempfile.TemporaryDirectory() as tmp:
                c,target,script,launch,run,calls,submitted=self.fixture(tmp,out=output,err=err)
                with self.assertRaises(ValueError):self.submit(c,target,run)
                result=json.loads(next((c.ROOT/'instruct300').glob('SUBMISSION_ONCE-*.json')).read_text())
                self.assertEqual(result['status'],'SUBMISSION_UNKNOWN_DO_NOT_RETRY')
                self.assertEqual(base64.b64decode(result['stdout_raw']['data']),output)
                self.assertNotIn('job_id',result)

    def test_unreviewed_symlink_or_nul_launch_refuses_before_client_calls(self):
        for kind in ('changed','symlink','nul'):
            with tempfile.TemporaryDirectory() as tmp:
                c,target,script,launch,run,calls,submitted=self.fixture(tmp,mutation=None)
                if kind=='changed':script.write_bytes(b'changed')
                elif kind=='symlink':
                    other=target/'other';other.write_bytes(launch);script.unlink();script.symlink_to(other)
                else:
                    raw=launch+b'\0';script.write_bytes(raw);c.WRAPPER_HASHES['launch.sbatch']=hashlib.sha256(raw).hexdigest()
                with self.assertRaises((ValueError,OSError)):self.submit(c,target,run)
                self.assertEqual(calls,[]);self.assertEqual(submitted,[])
                self.assertEqual(list((c.ROOT/'instruct300').glob('SUBMISSION_ONCE-*.json')),[])


if __name__=='__main__':unittest.main()
