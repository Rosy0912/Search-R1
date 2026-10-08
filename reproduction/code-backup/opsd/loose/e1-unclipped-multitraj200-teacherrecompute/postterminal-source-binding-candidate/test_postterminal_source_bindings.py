"""Small isolated fixtures; never executes production main or claims stored3304 terminal."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
H=HERE.parent
WORK=H.parent


def source_module(path,name):
    m=types.ModuleType(name);m.__file__=str(path);m.__package__=''
    exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True,optimize=0),m.__dict__)
    return m


a=source_module(HERE/'collect_postterminal_source_bindings.py','fixture_adapter')
c=source_module(H/'final200-runtime-bundle-f11580b6a7bc/common.py','fixture_common')
logs=source_module(H/'final200-runtime-bundle-f11580b6a7bc/consume_final200_logs_terminal.py','fixture_logs')


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name).resolve()
        self.run=self.base/'runs/formal-3304';(self.run/'evidence').mkdir(parents=True)
        self.out=self.base/'output';self.out.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def record(self,path,raw):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return c.read_file(path)[1]
    def request(self,child=b'{"returncode":0}',wrapper=b'1\n',scheduler=None):
        if scheduler is None:scheduler=b'JobId=3304 UserId=root(0) JobState=FAILED ExitCode=1:0\n'
        return dict(version=1,kind='POSTTERMINAL_SOURCE_BINDING_READONLY_REQUEST',source_commit=a.SOURCE,config_sha256=a.CONFIG,stored_training_job_id='3304',accepted5_receipt_sha256=a.RECEIPT,
          ready=self.record(self.run/'approval-ready.json',b'{}'),terminal=dict(
            training_child_exit=self.record(self.run/'evidence/train-process-exit.json',child),wrapper_exit=self.record(self.run/'job-exit.txt',wrapper),scheduler_original=self.record(self.run/'evidence/scheduler-terminal.original',scheduler)))

    def test_bootstrap_same_verified_bytes(self):
        path=self.base/'common.py';path.write_bytes((H/'final200-runtime-bundle-f11580b6a7bc/common.py').read_bytes())
        with mock.patch.object(a,'COMMON_PATH',path):m=a.bootstrap_common()
        self.assertEqual(m.__file__,str(path));self.assertEqual(m.FORMAL,a.FORMAL_SHA)

    def test_bootstrap_changed_bytes_reject(self):
        path=self.base/'common.py';path.write_bytes(b'raise AssertionError("must not execute")')
        with mock.patch.object(a,'COMMON_PATH',path),self.assertRaisesRegex(ValueError,'exact source'):a.bootstrap_common()

    def test_bootstrap_symlink_reject(self):
        target=self.base/'target';target.write_bytes(b'x');path=self.base/'common.py';path.symlink_to(target)
        with mock.patch.object(a,'COMMON_PATH',path),self.assertRaisesRegex(ValueError,'physical'):a.bootstrap_common()

    def test_literal_nonzero_wrapper_preserved(self):
        request=self.request()
        with mock.patch.object(a,'RUN',self.run):layers,_=a.terminal_first(c,logs,request,self.out,time.time_ns())
        self.assertEqual(layers['training_child_exit']['returncode'],0)
        self.assertEqual(layers['wrapper_exit']['returncode'],1)
        self.assertEqual(layers['scheduler']['state'],'FAILED');self.assertEqual(layers['scheduler']['exit_code'],'1:0')
        self.assertEqual((self.out/'wrapper_exit.original').read_bytes(),b'1\n')

    def test_child_bool_reject(self):
        request=self.request(child=b'{"returncode":false}')
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'bool'):a.terminal_first(c,logs,request,self.out,time.time_ns())

    def test_nonterminal_scheduler_reject(self):
        request=self.request(scheduler=b'JobId=3304 UserId=root(0) JobState=RUNNING ExitCode=0:0\n')
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'not terminal'):a.terminal_first(c,logs,request,self.out,time.time_ns())

    def test_other_job_scheduler_reject(self):
        request=self.request(scheduler=b'JobId=33040 UserId=root(0) JobState=FAILED ExitCode=1:0\n')
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'job/UID'):a.terminal_first(c,logs,request,self.out,time.time_ns())

    def test_terminal_created_after_start_reject(self):
        started=time.time_ns();request=self.request()
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'after this inspection'):a.terminal_first(c,logs,request,self.out,started)

    def test_terminal_outside_run_reject(self):
        request=self.request();request['terminal']['scheduler_original']=self.record(self.base/'outside',b'JobId=3304 UserId=root(0) JobState=FAILED ExitCode=1:0')
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'outside run'):a.terminal_first(c,logs,request,self.out,time.time_ns())

    def test_request_identity_is_typed(self):
        request=self.request()
        with mock.patch.object(a,'RUN',self.run):
            a.validate_request(request,c)
            for key,value in [('version',True),('source_commit','c5bf4c8db33562a259fa227a940dc69543901b98'),('stored_training_job_id',3304),('config_sha256','0'*64)]:
                v=dict(request);v[key]=value
                with self.assertRaisesRegex(ValueError,'Request identity'):a.validate_request(v,c)

    def test_unknown_terminal_role_reject(self):
        request=self.request();request['terminal']['receipt']={}
        with mock.patch.object(a,'RUN',self.run),self.assertRaisesRegex(ValueError,'Three actual'):a.validate_request(request,c)

    def test_copy_original_change_reject(self):
        path=self.base/'raw.json';rec=self.record(path,b'{"original":1}');path.write_bytes(b'{"original":2}')
        with self.assertRaisesRegex(ValueError,'Pinned bytes'):a.copy_record(c,rec,self.out/'copied')
        self.assertFalse((self.out/'copied').exists())

    def test_unknown_once_reject_without_overwrite(self):
        path=self.base/'once';a.reserve(path);(path/'previous').write_bytes(b'unknown')
        with self.assertRaises(FileExistsError):a.reserve(path)
        self.assertEqual((path/'previous').read_bytes(),b'unknown')

    def test_actual_local_git_wait_and_streams(self):
        repo=WORK/'opsd-unclipped-multitraj200-teacherrecompute'
        with mock.patch.object(a,'REPO',repo):
            result=a.git_original(c,self.out,'test','head')
        self.assertIs(result['waited'],True);self.assertEqual(result['returncode'],0)
        self.assertEqual(set(result['originals']),{'argv','stdout','stderr','wait'})
        self.assertEqual(c.strict_json((self.out/'test-git-head.wait.original.json').read_bytes())['returncode'],0)
        self.assertEqual((self.out/'test-git-head.stdout.original').read_text().strip(),a.SOURCE)

    def test_actual_local_git_failure_wait_preserved(self):
        with mock.patch.object(a,'REPO',self.base),self.assertRaisesRegex(ValueError,'Git child'):a.git_original(c,self.out,'bad','head')
        value=c.strict_json((self.out/'bad-git-head.wait.original.json').read_bytes())
        self.assertIs(value['waited'],True);self.assertEqual(value['returncode'],128)
        self.assertGreater((self.out/'bad-git-head.stderr.original').stat().st_size,0)

    def test_failure_keeps_request_terminal_and_once(self):
        request=self.request();request_path=self.run/'evidence/request.json';request_path.write_bytes(c.canonical(request))
        request_record=c.read_file(request_path)[1];own=c.read_file(HERE/'collect_postterminal_source_bindings.py')[1]
        common=c.read_file(H/'final200-runtime-bundle-f11580b6a7bc/common.py')[1];out=self.base/'once'
        with mock.patch.object(a,'RUN',self.run),mock.patch.object(a,'LOGS_PATH',H/'final200-runtime-bundle-f11580b6a7bc/consume_final200_logs_terminal.py'),mock.patch.object(a,'inspect_sources',side_effect=ValueError('synthetic source mismatch')):
            with self.assertRaisesRegex(ValueError,'synthetic source'):a.collect(c,request,request_record,own,common,dict(scope='LOCAL_FIXTURE_NOT_NATIVE_EXECUTION'),out,time.time_ns())
        self.assertEqual((out/'request.original.json').read_bytes(),request_path.read_bytes())
        self.assertEqual((out/'wrapper_exit.original').read_bytes(),b'1\n')
        self.assertEqual(c.strict_json((out/'failure.original.json').read_bytes())['message'],'synthetic source mismatch')
        self.assertFalse((out/'source-bindings.original.json').exists())
        with self.assertRaises(FileExistsError):a.reserve(out)

    def formal_fixture(self):
        tools=self.base/'tools/formal-71d2501d4512';tools.mkdir(parents=True)
        frozen=H/'formal-71d2501d4512'
        for p in frozen.iterdir():
            if p.is_file():shutil.copyfile(p,tools/p.name)
        manifest=c.strict_json((tools/'manifest.json').read_bytes());repo=self.base/'repo-fcecbea5d939'
        for name,pin in manifest['source_files'].items():
            src=WORK/'opsd-unclipped-multitraj200-teacherrecompute'/name;raw=src.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),pin)
            path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        owned=source_module(tools/'owned_runtime.py','owned_runtime')
        with mock.patch.dict(sys.modules,{'owned_runtime':owned}):formal=source_module(tools/'formal_supervisor.py','fixture_frozen_formal')
        formal.ROOT=self.base  # Explicit tiny physical-path fixture, no production scope.
        role_files={}
        for role in sorted(formal.ROLES):
            path=self.run/'bindings'/f'{role}.json';rec=self.record(path,c.canonical(dict(role=role)))
            role_files[role]=dict(path=str(path),sha256=rec['sha256'])
        raw=c.canonical(dict(source_commit=a.SOURCE,run_root=str(self.run),files=role_files))
        binding=self.run/'bindings/manifest.json';binding.write_bytes(raw)
        ready=dict(binding_sha256=hashlib.sha256(raw).hexdigest())
        env={'E1_FORMAL_MANIFEST_SHA256':a.FORMAL_SHA,'E1_ENTRY_SCRIPT':str(tools/'formal_job.sh'),'SEARCH_R1_REPO':str(repo)}
        return formal,tools,repo,ready,env

    def test_exact_frozen_functions_tiny_physical_fixture(self):
        formal,tools,repo,ready,env=self.formal_fixture()
        with mock.patch.dict(os.environ,env),mock.patch.object(formal,'command',side_effect=[a.SOURCE+'\n','']):actual,manifest=formal.sources(tools)
        self.assertEqual(actual,repo);self.assertEqual(len(manifest['source_files']),19)
        self.assertEqual(set(formal.verify_binding(self.run,ready)['files']),formal.ROLES)
        self.assertEqual(formal.__file__,str(tools/'formal_supervisor.py'))

    def test_exact_frozen_function_source_changed_reject(self):
        formal,tools,repo,ready,env=self.formal_fixture();path=repo/'search_r1/baselines/e1_profile.py';path.write_bytes(path.read_bytes()+b'\n')
        with mock.patch.dict(os.environ,env),mock.patch.object(formal,'command',side_effect=[a.SOURCE+'\n','']),self.assertRaisesRegex(ValueError,'Frozen source file'):formal.sources(tools)

    def test_exact_frozen_function_bound_member_changed_reject(self):
        formal,tools,repo,ready,env=self.formal_fixture()
        path=self.run/'bindings/source.json';path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'Bound artifact'):formal.verify_binding(self.run,ready)

    def test_validation_environment_restores_caller_values(self):
        before={key:os.environ.get(key) for key in ('E1_FORMAL_MANIFEST_SHA256','E1_ENTRY_SCRIPT','SEARCH_R1_REPO')}
        with a.validation_environment() as values:self.assertEqual(values['E1_FORMAL_MANIFEST_SHA256'],a.FORMAL_SHA)
        self.assertEqual({key:os.environ.get(key) for key in before},before)


if __name__=='__main__':unittest.main(verbosity=2)
