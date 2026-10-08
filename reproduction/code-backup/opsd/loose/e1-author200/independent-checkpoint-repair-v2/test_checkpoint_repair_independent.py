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

SOURCE = Path(os.environ.get('REPAIR_SOURCE',str(Path(__file__).resolve().parent / 'final-acceptance-candidate/checkpoint_repair_cpu.py')))
EXPECTED_SOURCE = 'a740a5106b7b8c338da294d14cd6437a711ac79e74b50b28770ed9814b5a60e2'

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
        root=Path(self.tmp.name).resolve()
        self.r=types.ModuleType('repair_test');self.r.__file__=str(SOURCE)
        raw=SOURCE.read_bytes();self.source_sha=hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.source_sha,EXPECTED_SOURCE)
        exec(compile(raw,str(SOURCE),'exec'),self.r.__dict__)
        self.r.OUTPUT=root/'checkpoint-25-cpu-inspection-repair-v2'
        self.r.MARKER=root/'checkpoint-cpu-25-repair-v2-launch.json'
        self.r.HERE=root/'tools';self.r.HERE.mkdir()
        self.r.INVENTORY=root/'original-inventory.json';self.r.INVENTORY.write_text('{}\n')
        self.r.INVENTORY_SHA=sha(self.r.INVENTORY)
        self.l=LocalLauncher(root);self.manifest='b'*64
        self.tools={'source_commit':'188f4e5e7b63cf479290d0888a4a075499886750',
            'tools':{'inspect_author200_checkpoint_cpu.py':'f'*64}}
        self.context={'job_id':'3261','step_id':'9','scope':'MOCK_CPU_CONTEXT',
            'hostname':'h100-10','process':self.l.process(12345)}
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
    def bootstrap_fixture(self, mutate=None):
        launcher_source=Path(__file__).resolve().parent/'final-acceptance-8441e10fa456/checkpoint_overlap.py'
        raw=launcher_source.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),self.r.LAUNCHER)
        (self.r.HERE/'checkpoint_overlap.py').write_bytes(raw)
        failed=Path(self.tmp.name).resolve()/'failed-original';failed.mkdir()
        self.r.ORIGINAL_OUTPUT=failed;self.r.INVENTORY=failed/'inventory.json'
        self.r.INVENTORY.write_text('{}\n');self.r.INVENTORY_SHA=sha(self.r.INVENTORY)
        self.r.FAILURES={}
        for name in ('native-exit-original.json','srun-exit-original.json','controller-exit-original.json'):
            self.l.write(failed/name,{'returncode':1})
            self.r.FAILURES[name]=sha(failed/name)
        stderr=failed/'native-stderr-original.txt';stderr.write_text('ValueError: Sampler position/config differs: seed\n')
        self.r.FAILURE_STDERR=sha(stderr)
        repair=self.r.HERE/'checkpoint_repair_cpu.py';repair.write_bytes(SOURCE.read_bytes())
        tools={'source_commit':self.tools['source_commit'],'tools':{'checkpoint_repair_cpu.py':sha(repair)}}
        if mutate:mutate(failed,tools)
        launcher=self.l;real_type=types.ModuleType
        class LoaderModule(real_type):
            def __getattribute__(this,name):
                if name=='tools':return lambda _:tools
                if name in ('pinned_bytes','need','unique'):return getattr(launcher,name)
                return super().__getattribute__(name)
        old=types.SimpleNamespace(load_request=lambda *_:({'tools_manifest_sha256':self.r.OLD_MANIFEST,'checkpoint_step':25},failed))
        original_stat=Path.stat;original_fstat=os.fstat
        class Owned:
            def __init__(this,value):this.value=value
            def __getattr__(this,key):return 0 if key=='st_uid' else getattr(this.value,key)
        with mock.patch.object(self.r.types,'ModuleType',side_effect=lambda name:LoaderModule(name)), \
                mock.patch.object(self.r,'module',return_value=old), \
                mock.patch.object(Path,'stat',lambda p,*a,**k:Owned(original_stat(p,*a,**k))), \
                mock.patch.object(self.r.os,'fstat',side_effect=lambda fd:Owned(original_fstat(fd))), \
                mock.patch.object(self.r.os,'getuid',return_value=0):
            return self.r.bootstrap(self.manifest)
    def test_bootstrap_actual_failure_shape_positive_fixture(self):
        self.bootstrap_fixture()
    def test_bootstrap_previous_exit_boolean_rejected(self):
        def mutate(failed,tools):
            p=failed/'native-exit-original.json';p.write_bytes(self.l.canonical({'returncode':True}));self.r.FAILURES[p.name]=sha(p)
        with self.assertRaises(ValueError):self.bootstrap_fixture(mutate)
    def test_bootstrap_previous_exit0_rejected(self):
        def mutate(failed,tools):
            p=failed/'controller-exit-original.json';p.write_bytes(self.l.canonical({'returncode':0}));self.r.FAILURES[p.name]=sha(p)
        with self.assertRaises(ValueError):self.bootstrap_fixture(mutate)
    def test_bootstrap_prior_native_report_rejected(self):
        with self.assertRaises(ValueError):self.bootstrap_fixture(lambda failed,tools:(failed/'native.json').write_text('{}'))
    def test_bootstrap_prior_completion_rejected(self):
        with self.assertRaises(ValueError):self.bootstrap_fixture(lambda failed,tools:(failed/'inspection-completed-review-required.json').write_text('{}'))
    def test_bootstrap_different_failure_stderr_rejected(self):
        def mutate(failed,tools):
            p=failed/'native-stderr-original.txt';p.write_text('Different failure\n');self.r.FAILURE_STDERR=sha(p)
        with self.assertRaises(ValueError):self.bootstrap_fixture(mutate)
    def test_bootstrap_inventory_pin_changed_rejected(self):
        with self.assertRaises(ValueError):self.bootstrap_fixture(lambda failed,tools:self.r.INVENTORY.write_text('changed'))
    def test_bootstrap_self_file_differs_from_pinned_tools_rejected(self):
        with self.assertRaises(ValueError):self.bootstrap_fixture(lambda failed,tools:tools['tools'].update({'checkpoint_repair_cpu.py':'a'*64}))
    def test_module_verifies_same_raw_bytes_before_exec(self):
        path=Path(self.tmp.name).resolve()/'small_module.py';path.write_text('proof = 7\n')
        actual=self.r.module(path,sha(path),self.l.pinned_bytes);self.assertEqual(actual.proof,7)
        with self.assertRaises(ValueError):self.r.module(path,'a'*64,self.l.pinned_bytes)
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
    def test_existing_marker_output_and_broken_alias_rejected(self):
        cases=('marker','output','marker_alias','output_alias')
        for case in cases:
            target=self.r.MARKER if case.startswith('marker') else self.r.OUTPUT
            if case.endswith('alias'):target.symlink_to(self.r.HERE/'missing')
            elif case=='marker':target.write_text('original')
            else:target.mkdir()
            with self.subTest(case=case),mock.patch.object(self.r.subprocess,'Popen') as spawn:
                with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
                spawn.assert_not_called()
            if case=='output':target.rmdir()
            else:target.unlink()
    def test_concurrent_start_one_actual_local_detached_child(self):
        done=Path(self.tmp.name).resolve()/'only-local-child'
        command=[sys.executable,'-c',f'import time,pathlib;time.sleep(.05);pathlib.Path({str(done)!r}).write_text("one")']
        original_popen=subprocess.Popen;children=[];errors=[];barrier=threading.Barrier(2)
        def popen(*args,**kwargs):
            child=original_popen(*args,**kwargs);children.append(child);return child
        def run():
            barrier.wait()
            try:self.r.start(self.l,self.manifest)
            except (ValueError,FileExistsError) as error:errors.append(str(error))
        with mock.patch.object(self.r,'self_command',return_value=command), \
                mock.patch.object(self.r.subprocess,'Popen',side_effect=popen), \
                mock.patch('builtins.print'):
            threads=[threading.Thread(target=run) for _ in range(2)]
            for t in threads:t.start()
            for t in threads:t.join()
        self.assertEqual(len(children),1);self.assertEqual(len(errors),1)
        self.assertEqual(children[0].wait(),0);self.assertEqual(done.read_text(),'one')
        self.assertTrue((self.r.OUTPUT/'durable-launch-original.json').exists())
        with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
    def test_durable_metadata_unknown_retains_live_child_and_no_retry(self):
        done=Path(self.tmp.name).resolve()/'local-child-completed'
        command=[sys.executable,'-c',f'import time,pathlib;time.sleep(.05);pathlib.Path({str(done)!r}).write_text("done")']
        original_popen=subprocess.Popen;children=[]
        def popen(*args,**kwargs):
            child=original_popen(*args,**kwargs);children.append(child);return child
        def hook(path,value):
            if path.name=='durable-launch-original.json':raise ValueError('unknown metadata result')
        self.l.write_hook=hook
        with mock.patch.object(self.r,'self_command',return_value=command), \
                mock.patch.object(self.r.subprocess,'Popen',side_effect=popen):
            with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
        self.assertEqual(len(children),1);self.assertEqual(children[0].wait(),0)
        self.assertTrue(done.exists());self.assertTrue(self.r.MARKER.exists())
        with self.assertRaises(ValueError):self.r.start(self.l,self.manifest)
    def test_native_failure_never_complete(self):
        with self.assertRaises(ValueError):self.inspect_fixture(code=7)
        self.assertFalse((self.r.OUTPUT/'inspection-completed-review-required.json').exists())
    def test_final_tools_recheck_failure_never_complete(self):
        request=self.initialize()
        def run(launcher,role,command):
            self.l.write(self.r.OUTPUT/'native-process-original.json',{'pid':34567,'process':self.l.process(34567)})
            self.l.write(self.r.OUTPUT/'native.json',self.report())
            return 0
        with mock.patch.object(self.r,'module',return_value=self.pipeline), \
                mock.patch.object(self.r,'wait_child',side_effect=run), \
                mock.patch.object(self.r,'bootstrap',side_effect=ValueError('tool drift')):
            with self.assertRaises(ValueError):self.r.inspect(self.l,self.tools,self.manifest,request)
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
        self.r.OUTPUT.mkdir();done=Path(self.tmp.name).resolve()/'fixture-child-completed'
        command=[sys.executable,'-c',f'import time,pathlib;time.sleep(.05);pathlib.Path({str(done)!r}).write_text("done")']
        def hook(path,value):
            if path.name=='native-process-original.json':raise ValueError('fixture metadata')
        self.l.write_hook=hook
        with self.assertRaises(ValueError):self.r.wait_child(self.l,'native',command)
        self.assertTrue(done.exists())
        self.assertEqual(json.loads((self.r.OUTPUT/'native-exit-original.json').read_text())['returncode'],0)
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
