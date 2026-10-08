"""Independent local mocks and real local children, never remote/Slurm/GPU."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parent / 'final-acceptance-candidate/checkpoint_repair_cpu.py'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class LocalLauncher:
    def __init__(self, root):
        self.root=root; self.read_hook=None; self.write_hook=None; self.allocations=0
    def need(self, condition, message):
        if not condition: raise ValueError(message)
    def canonical(self, value): return (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()
    def unique(self, value):
        def pairs(items):
            out={}
            for key,v in items:
                self.need(key not in out,'duplicate');out[key]=v
            return out
        return json.loads(value, object_pairs_hook=pairs)
    def pinned_bytes(self, path, expected):
        path=Path(path)
        if self.read_hook: self.read_hook(path, expected)
        self.physical(path)
        raw=path.read_bytes();self.need(hashlib.sha256(raw).hexdigest()==expected,'SHA differs')
        return raw
    def physical(self, path, directory=False):
        path=Path(path)
        self.need(path.is_absolute() and path.resolve()==path,'alias')
        self.need(path.is_dir() if directory else path.is_file(),'physical type')
        return path
    def write(self, path, value):
        if self.write_hook: self.write_hook(Path(path),value)
        self.physical(Path(path).parent,True)
        with Path(path).open('xb') as out:
            out.write(self.canonical(value));out.flush();os.fsync(out.fileno())
    def now(self): return 'FIXTURE_ONLY'
    def sha(self,path): return sha(path)
    def allocation(self): self.allocations+=1;return {'scope':'MOCK_H100_ALLOCATION_ONLY'}
    def process(self, pid, require_live=False):
        return {'pid':pid,'present':True,'start_ticks':'71002','uids':[0]*4,
            'cgroup_original':'1:freezer:/job_3261/step_9\n','state':'R'}
    def exit_status(self, code): return code if code>=0 else 128-code

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.r=types.ModuleType('repair_test');self.r.__file__=str(SOURCE)
        raw=SOURCE.read_bytes();self.source_sha=hashlib.sha256(raw).hexdigest()
        exec(compile(raw,str(SOURCE),'exec'),self.r.__dict__)
        self.r.OUTPUT=root/'checkpoint-25-cpu-inspection-repair-v2'
        self.r.MARKER=root/'checkpoint-cpu-25-repair-v2-launch.json'
        self.r.HERE=root/'tools';self.r.HERE.mkdir()
        self.r.INVENTORY=root/'original-inventory.json';self.r.INVENTORY.write_text('{}\n')
        self.r.INVENTORY_SHA=sha(self.r.INVENTORY)
        self.l=LocalLauncher(root);self.manifest='b'*64
        self.tools={'source_commit':'188f4e5e7b63cf479290d0888a4a075499886750',
            'tools':{'inspect_author200_checkpoint_cpu.py':'f'*64}}
        self.context={'job_id':'3261','step_id':'9','scope':'MOCK_CPU_CONTEXT'}
        self.pipeline=types.SimpleNamespace(execution_context=lambda _:self.context,
            NATIVE_HELPER=root/'native-helper.py',BASE=root/'base',MODEL_SHA='c'*64)
    def initialize(self):
        self.r.OUTPUT.mkdir()
        self.l.write(self.r.MARKER,self.r.request_value(self.manifest))
        return sha(self.r.MARKER)
    def report(self,pid=34567):
        return {'scope':'NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION','passed':True,
            'source_commit':self.tools['source_commit'],'job_id':'3261','checkpoint_step':25,
            'cuda_is_initialized':False,'inspector_source_sha256':'f'*64,
            'execution_identity':{'execution_job_id':'3261','execution_step_id':'9',
                'pid':pid,'node':'h100-10','uid':0,'uids':[0]*4,'start_ticks':71002,
                'cgroup':'1:freezer:/job_3261/step_9\n'}}
    def inspect_fixture(self, code=0, mutate=None, hook=None):
        request=self.initialize()
        report=self.report()
        if mutate: mutate(report)
        def run(launcher,role,command):
            self.captured_command=command
            self.l.write(self.r.OUTPUT/'native-process-original.json',
                {'pid':34567,'process':self.l.process(34567)})
            self.l.write(self.r.OUTPUT/'native-exit-original.json',{'pid':34567,'returncode':code})
            if code==0:self.l.write(self.r.OUTPUT/'native.json',report)
            return code
        self.l.read_hook=hook
        with mock.patch.object(self.r,'module',return_value=self.pipeline), \
                mock.patch.object(self.r,'wait_child',side_effect=run), \
                mock.patch.object(self.r,'bootstrap',return_value=(self.l,self.tools)):
            self.r.inspect(self.l,self.tools,self.manifest,request)
        return json.loads((self.r.OUTPUT/'inspection-completed-review-required.json').read_bytes())
    def test_constants_actual_failure_only25(self):
        self.assertEqual(self.r.ORIGINAL_REQUEST,'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f')
        self.assertEqual(self.r.FAILURES,{'native-exit-original.json':'acf00633f70bcf95c11e5635df9724235531121177740d81e73768e692a86e46',
            'srun-exit-original.json':'faec00c6c152b566416d818c2de3fe5b629e27ef4920875ef8f688026c799a37',
            'controller-exit-original.json':'1fbf8672f6873a68c693c6dfcf0ea1263030f5d6ccfdca16bf12e7766c91694a'})
        self.assertEqual(self.r.FAILURE_STDERR,'43277dcc82f583862054290d83e2c608e497ce4815c5d4c5586600652ccc20f8')
    def test_read_request_exact_schema_and_counters(self):
        pin=self.initialize();self.r.read_request(self.l,self.manifest,pin)
        for field,value in [('version',True),('version',1.0),('checkpoint_step',25.0),
                ('checkpoint_step',26),('training_job_id',3261),('inventory_sha256','e'*64),
                ('original_request_sha256','e'*64),('scope','futurepass'),('source_commit','x'),
                ('output_directory',str(self.r.OUTPUT.parent)),('tools_manifest_sha256','a'*64)]:
            item=self.r.request_value(self.manifest);item[field]=value
            self.r.MARKER.write_bytes(self.l.canonical(item))
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                self.r.read_request(self.l,self.manifest,sha(self.r.MARKER))
    def test_request_wrong_pin_and_duplicate_key_rejected(self):
        pin=self.initialize()
        with self.assertRaises(ValueError):self.r.read_request(self.l,self.manifest,'a'*64)
        raw=self.r.MARKER.read_bytes();self.r.MARKER.write_bytes(raw.replace(b'{',b'{"version":1,',1))
        with self.assertRaises(ValueError):self.r.read_request(self.l,self.manifest,sha(self.r.MARKER))
    def test_request_output_alias_rejected(self):
        pin=self.initialize();self.r.OUTPUT.rmdir();self.r.OUTPUT.symlink_to(self.r.HERE,target_is_directory=True)
        with self.assertRaises(ValueError):self.r.read_request(self.l,self.manifest,pin)
    def test_unknown_popen_result_keeps_marker_no_retry(self):
        with mock.patch.object(self.r.subprocess,'Popen',side_effect=OSError('fixture unknown')):
            with self.assertRaises(OSError):self.r.start(self.l,self.manifest)
        self.assertTrue(self.r.MARKER.exists());self.assertTrue(self.r.OUTPUT.exists())
        pin=sha(self.r.MARKER)
        with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
        self.assertEqual(pin,sha(self.r.MARKER))
    def test_allocation_rejected_before_marker(self):
        with mock.patch.object(self.l,'allocation',side_effect=ValueError('foreign')):
            with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
        self.assertFalse(self.r.OUTPUT.exists());self.assertFalse(self.r.MARKER.exists())
    def test_native_failure_never_complete(self):
        with self.assertRaises(ValueError):self.inspect_fixture(code=7)
        self.assertFalse((self.r.OUTPUT/'inspection-completed-review-required.json').exists())
    def test_valid_report_completion_still_review_required(self):
        value=self.inspect_fixture()
        self.assertIs(value['independent_acceptance_pending'],True)
        self.assertEqual(value['original_failure_exits_preserved'],self.r.FAILURES)
        self.assertEqual(value['native_report_sha256'],sha(self.r.OUTPUT/'native.json'))
        self.assertEqual(self.captured_command[self.captured_command.index('--step')+1],'25')
        self.assertEqual(self.captured_command[self.captured_command.index('--execution-step-id')+1],'9')
    def test_native_report_changed_between_pin_and_parse_rejected(self):
        def hook(path,expected):
            if path.name=='native.json':
                self.l.read_hook=None;path.write_text('{}\n')
        with self.assertRaises(ValueError):self.inspect_fixture(hook=hook)
        self.assertFalse((self.r.OUTPUT/'inspection-completed-review-required.json').exists())
    def test_child_waits_actual_stdout_stderr_success(self):
        self.r.OUTPUT.mkdir()
        command=[sys.executable,'-c','import sys,time; print("ACTUAL_FIXTURE");sys.stderr.write("FIXTURE_ERR\\n");time.sleep(.05)']
        self.assertEqual(self.r.wait_child(self.l,'native',command),0)
        self.assertEqual((self.r.OUTPUT/'native-stdout-original.txt').read_text(),'ACTUAL_FIXTURE\n')
        self.assertEqual((self.r.OUTPUT/'native-stderr-original.txt').read_text(),'FIXTURE_ERR\n')
        record=json.loads((self.r.OUTPUT/'native-exit-original.json').read_text())
        self.assertEqual(record['returncode'],0);self.assertEqual(record['waited_by_pid'],os.getpid())
    def test_child_waits_actual_nonzero_and_signal(self):
        self.r.OUTPUT.mkdir()
        for role,command,code in [('exit7',[sys.executable,'-c','import sys;sys.exit(7)'],7),
                ('signal',[sys.executable,'-c','import os,signal;os.kill(os.getpid(),signal.SIGTERM)'],-15)]:
            self.assertEqual(self.r.wait_child(self.l,role,command),code)
            self.assertEqual(json.loads((self.r.OUTPUT/(role+'-exit-original.json')).read_text())['returncode'],code)
    def test_metadata_error_finally_waits_actual_child(self):
        self.r.OUTPUT.mkdir();done=Path(self.tmp.name)/'fixture-child-completed'
        command=[sys.executable,'-c',f'import time,pathlib;time.sleep(.05);pathlib.Path({str(done)!r}).write_text("done")']
        def hook(path,value):
            if path.name=='native-process-original.json':raise ValueError('fixture metadata')
        self.l.write_hook=hook
        with self.assertRaises(ValueError):self.r.wait_child(self.l,'native',command)
        self.assertTrue(done.exists())
    def test_controller_cpu_argv_does_not_execute_srun(self):
        pin=self.initialize();captured=[]
        with mock.patch.object(self.r,'bootstrap',return_value=(self.l,self.tools)), \
                mock.patch.object(self.r,'wait_child',side_effect=lambda l,role,command:captured.append((role,command)) or 0), \
                mock.patch.object(sys,'argv',['fixture','controller','--tools-manifest-sha256',self.manifest,'--request-sha256',pin]):
            with self.assertRaises(SystemExit) as result:self.r.main()
        self.assertEqual(result.exception.code,0)
        role,command=captured[0];self.assertEqual(role,'srun');self.assertEqual(command[:8],['srun','--jobid=3261','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10'])
        for env in ('CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1'):
            self.assertIn(env,command)

def report_mutation_test(field,value):
    def test(self):
        def mutate(record):
            if field.startswith('execution.'):
                record['execution_identity'][field.split('.',1)[1]]=value
            else:record[field]=value
        with self.assertRaises(ValueError):self.inspect_fixture(mutate=mutate)
        self.assertFalse((self.r.OUTPUT/'inspection-completed-review-required.json').exists())
    return test

for field,value in [('scope','futurepass'),('passed',1),('checkpoint_step',25.0),('checkpoint_step',200),
        ('cuda_is_initialized',0),('job_id',3261),('inspector_source_sha256','a'*64),
        ('execution.execution_job_id','2999'),('execution.execution_step_id','8'),
        ('execution.pid',34568),('execution.pid',True),('execution.node','a800'),
        ('execution.uid',False),('execution.start_ticks',71003),('execution.start_ticks',True),
        ('execution.uids',[False]*4),('execution.cgroup','1:freezer:/job_2999/step_9\n')]:
    setattr(RepairTests,'test_report_'+field.replace('.','_')+'_'+str(value).replace('/','_').replace(' ','_'),report_mutation_test(field,value))

if __name__=='__main__':unittest.main(verbosity=2)
