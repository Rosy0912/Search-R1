"""V4 changed interfaces only: injected controller/NVML, real tiny tokenizer CPU.
No actual scheduler/GPU/model/tokenizer assets or remote operations.
"""
import ast
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as N
import unittest
from unittest.mock import patch
import build_package_draft as builder
import slurm_identity as identity
import guard_entry
import submit_once
import check_behavior_mu_cpu as checker
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'tree-behavior-mu-utf8-v2'
OWN={'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':'h100-9','SLURM_CLUSTER_NAME':identity.CLUSTER}
CGROUP='0::/slurm/job_99/step_batch\n'
ALLOCATION='JobId=99 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=96 Requeue=0 Restarts=0 TimeLimit=3-00:00:00 MinMemoryNode=256G'

def tiny_checker_child(queue):
    """Fresh spawn: execute current checker registration on tiny fixture only."""
    try:
        import search_r1.baselines.tree_raw_tokens as raw
        from test_tree_behavior_mu_data_cpu import BehaviorMuDataTests
        assert not raw._VALIDATION_TOKENIZERS
        fixture=BehaviorMuDataTests();fixture.setUp()
        try:
            data=fixture.output();raw._VALIDATION_TOKENIZERS.clear()
            try:fixture.validate(data)
            except ValueError as e:missing=str(e)
            else:raise AssertionError('empty process registry accepted')
            path=Path(fixture.tmp.name).resolve()
            for name in checker.TOKENIZER_FILES:
                if not (path/name).exists():(path/name).write_bytes(b'{"fixture_only":true}')
            package={'official_model_path':str(path),'model_identity':{'files':{
                name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in checker.TOKENIZER_FILES}}}
            loader_calls=[];utils=ModuleType('verl.utils')
            def load(value,**kwargs):loader_calls.append((value,kwargs));return fixture.tok
            utils.hf_tokenizer=load
            with patch.dict(sys.modules,{'verl.utils':utils}),patch.object(checker,'OFFICIAL_MODEL_PATH',path),patch.object(checker.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                tokenizer,binding=checker.initialize_validation_tokenizer(package)
                receipt=fixture.validate(data)
                assert raw._validation_tokenizer(binding) is tokenizer
                assert loader_calls==[(str(path),{'local_files_only':True,'trust_remote_code':False})]
                fixture.tok.parts[2]='WRONG'
                try:fixture.validate(data)
                except ValueError as e:drift=str(e)
                else:raise AssertionError('live semantic backend drift accepted')
            queue.put(dict(ok=True,actual_spawn=True,missing_registry=missing,registered_semantics_accepted=True,backend_drift=drift,loader_calls=loader_calls,official_assets=False,model_weights=False))
        finally:fixture.tearDown()
    except BaseException as error:queue.put(dict(ok=False,error=repr(error)))

class V4Tests(unittest.TestCase):
    def test_runtime_wrong_missing_sibling_mixed_before_any_scheduler_or_gpu_query(self):
        for text in ('','bad','0::/\n','0::/slurm/job_98/step_batch\n','0::/slurm/job_99/step_batch\n1:cpu:/slurm/job_98/step_batch\n'):
            calls=[]
            runner=lambda *a,**k:(calls.append(a),N(returncode=0,stdout='',stderr=''))[1]
            with self.subTest(text=text),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value=text),patch.dict(os.environ,OWN,clear=True):
                with self.assertRaises(ValueError):identity.verify_runtime(run=runner)
                with self.assertRaises(ValueError):identity.guarded_gpu_query(lambda:calls.append('GPU'))()
                self.assertEqual(calls,[])
    def test_legal_own_runtime_exact_allocation_and_query_order(self):
        calls=[]
        def runner(argv,**kw):
            calls.append((argv,kw))
            out='ClusterName = wuhan-h100\n' if argv[-1]=='config' else '' if argv[-1]=='federation' else ALLOCATION
            return N(returncode=0,stdout=out,stderr='')
        with patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value=CGROUP),patch.dict(os.environ,OWN,clear=True):
            result=identity.verify_runtime(run=runner)
            self.assertEqual(result['cgroup_identity']['job_id'],'99');self.assertEqual(result['allocated_cpus'],96)
            self.assertEqual([x[0][-1] for x in calls],['config','federation','99'])
            self.assertEqual(identity.guarded_gpu_query(lambda:'unchanged')(),'unchanged')
    def test_current_host_job_and_node_refuse_before_any_query(self):
        calls=[]
        for env,host in ((OWN,'h100-8'),(dict(OWN,SLURM_JOB_ID=''),'h100-9'),(dict(OWN,SLURM_JOB_NODELIST='h100-8'),'h100-9'),(dict(OWN,SLURM_CLUSTER_NAME='other'),'h100-9')):
            with self.subTest(env=env,host=host),patch.object(identity.socket,'gethostname',return_value=host),patch.object(identity.Path,'read_text',return_value=CGROUP):
                with self.assertRaises(ValueError):identity.verify_runtime(env,run=lambda *a,**k:calls.append(a))
        self.assertEqual(calls,[])
    def test_cpu_management_scope_distinct_no_job_no_gpu_and_legal_queue(self):
        env={'CUDA_VISIBLE_DEVICES':''};calls=[]
        runner=lambda argv,**kw:(calls.append(argv),N(returncode=0,stdout='',stderr=''))[1]
        with patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value='0::/user.slice/session.scope\n'):
            result=identity.verify_management_scope(env);self.assertFalse(result['gpu_queries_allowed'])
            self.assertEqual(submit_once.queue_clear(runner,env),'');self.assertEqual(len(calls),1)
            for bad in ({},dict(env,CUDA_VISIBLE_DEVICES='0'),dict(env,SLURM_JOB_ID='99'),dict(env,SLURM_JOB_NODELIST='h100-9'),dict(env,SLURM_CLUSTER_NAME='other')):
                calls.clear()
                with self.subTest(bad=bad),self.assertRaises(ValueError):submit_once.queue_clear(runner,bad)
                self.assertEqual(calls,[])
        with patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value=CGROUP):
            with self.assertRaises(ValueError):identity.verify_management_scope(env)
    def test_submitter_wrong_management_cgroup_before_queries_or_package_io(self):
        calls=[]
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=True),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value=CGROUP):
            with self.assertRaises(ValueError):submit_once.submit('/not/read','x','/not/read','x','/not/read','x',run=lambda *a,**kw:calls.append(a))
        self.assertEqual(calls,[])
    def test_original_nvml_functions_guarded_before_each_query_unchanged_outputs(self):
        names=('nvmlInit','nvmlDeviceGetCount','nvmlDeviceGetHandleByIndex','nvmlDeviceGetUUID','nvmlDeviceGetMemoryInfo','nvmlDeviceGetUtilizationRates','nvmlDeviceGetComputeRunningProcesses','nvmlDeviceGetGraphicsRunningProcesses','nvmlDeviceGetProcessUtilization')
        calls=[];module=N(**{name:(lambda *a,_name=name,**kw:(calls.append(_name),_name)[1]) for name in names})
        originals=guard_entry.install_nvml_cgroup_guards(module)
        with patch.object(guard_entry,'verify_runtime_cgroup',side_effect=ValueError('wrong cgroup')):
            for name in names:
                with self.assertRaises(ValueError):getattr(module,name)()
            self.assertEqual(calls,[])
        with patch.object(guard_entry,'verify_runtime_cgroup',return_value={'job_id':'99'}) as verified:
            for name in names:self.assertEqual(getattr(module,name)(),name)
            self.assertEqual(verified.call_count,len(names));self.assertEqual(calls,list(names))
        self.assertEqual(set(originals),set(names))
    def test_launch_supervisor_guard_checker_order_actual_ast(self):
        text=(HERE/'launch.sbatch').read_text();self.assertLess(text.index('verify_runtime_cgroup()'),text.index('identity=verify_runtime()'))
        sup=ast.parse((HERE/'supervisor.py').read_text());main=next(n for n in sup.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        cgroup=min(n.lineno for n in calls if n.func.id=='verify_runtime_cgroup');runtime=min(n.lineno for n in calls if n.func.id=='verify_runtime')
        self.assertLess(cgroup,runtime)
        check=ast.parse((HERE/'check_behavior_mu_cpu.py').read_text());verify=next(n for n in check.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
        calls=[n for n in ast.walk(verify) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertLess(min(n.lineno for n in calls if n.func.id=='initialize_validation_tokenizer'),min(n.lineno for n in calls if n.func.id=='validate_behavior_mu_batch'))
    def test_new_cpu_checker_fresh_process_real_registration_and_semantic_drift(self):
        context=multiprocessing.get_context('spawn');queue=context.Queue();process=context.Process(target=tiny_checker_child,args=(queue,));process.start();process.join(20)
        self.assertFalse(process.is_alive());self.assertEqual(process.exitcode,0);result=queue.get(timeout=3)
        self.assertTrue(result['ok'],result);self.assertIn('not registered',result['missing_registry']);self.assertIn('drifted',result['backend_drift'])
    def test_tokenizer_fixed_manifest_path_and_sha_refuse_before_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()
            for name in checker.TOKENIZER_FILES:(path/name).write_bytes(b'{"tiny":true}')
            package={'official_model_path':str(path),'model_identity':{'files':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in checker.TOKENIZER_FILES}}}
            utils=ModuleType('verl.utils');calls=[];utils.hf_tokenizer=lambda *a,**kw:calls.append(a)
            with patch.dict(sys.modules,{'verl.utils':utils}),patch.object(checker.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                with self.assertRaises(ValueError):checker.initialize_validation_tokenizer(package)
                with patch.object(checker,'OFFICIAL_MODEL_PATH',path):
                    package['model_identity']['files']['tokenizer.json']='0'*64
                    with self.assertRaises(ValueError):checker.initialize_validation_tokenizer(package)
            self.assertEqual(calls,[])
    def test_actual_owner_metadata_adapter_consumed_without_rebranding(self):
        path=HERE/'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        value,_,_=builder.verify_source_interface({'source_interface_manifest':{'path':str(path),'sha256':digest},'source_interface_receipt_sha256':digest,'code_commit':'02b1ea2688ff3fb032ee6699003a7e09d651f166'},SOURCE)
        original=Path(value['metadata_adapter']['original_owner_receipt_path']);actual=json.loads(original.read_text())
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),value['metadata_adapter']['original_owner_receipt_sha256'])
        for key,old in actual.items():self.assertEqual(value[key],old,key)
        self.assertFalse(value['metadata_adapter']['actual_new_tests_executed_by_adapter']);self.assertEqual(len(value['additional_changed_runtime_files']),4)

if __name__=='__main__':unittest.main()
