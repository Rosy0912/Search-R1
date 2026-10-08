"""Small local-only controller fixtures. No production main or GPU invoked."""
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_supervisor_fixture',HERE/'paired_supervisor.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
spec=importlib.util.spec_from_file_location('paired_cpu_fixture',HERE/'paired_cpu_prepare.py')
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)


class TestControl(unittest.TestCase):
    def test_real_wait0_7_sigterm_and_metadata_failure_preserves_wait(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve()
            for label,code,rc in [('good','print("preserved")',0),('bad','raise SystemExit(7)',7),
                ('signal','import os,signal;os.kill(os.getpid(),signal.SIGTERM)',-signal.SIGTERM)]:
                if rc==0:s.wait_child([sys.executable,'-B','-c',code],root,label,timeout=3)
                else:
                    with self.assertRaises(ValueError):s.wait_child([sys.executable,'-B','-c',code],root,label,timeout=3)
                v=json.loads((root/(label+'-exit.json')).read_bytes());self.assertIs(v['waited'],True);self.assertEqual(v['returncode'],rc)
            def bad_metadata(*a):raise RuntimeError('metadata fixture')
            with self.assertRaisesRegex(RuntimeError,'metadata fixture'):
                s.wait_child([sys.executable,'-B','-c','raise SystemExit(6)'],root,'meta',timeout=3,job='fixture',step='0',metadata=bad_metadata)
            self.assertEqual(json.loads((root/'meta-exit.json').read_bytes())['returncode'],6)
            self.assertEqual((root/'good.stdout').read_bytes(),b'preserved\n')

    def test_unknown_wait_retained_no_retry(self):
        class Child:
            pid=888;returncode=None
            def wait(self,timeout=None):raise subprocess.TimeoutExpired('fixture',timeout)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();child=Child()
            with patch.object(s.subprocess,'Popen',return_value=child) as pop,patch.object(s.time,'monotonic',side_effect=[0,2]):
                with self.assertRaises(TimeoutError):s.wait_child(['fixture'],root,'unknown',timeout=1)
            self.assertEqual(pop.call_count,1)
            v=json.loads((root/'unknown-exit.json').read_bytes());self.assertIs(v['waited'],False);self.assertIsNone(v['returncode'])
            with self.assertRaises(FileExistsError):s.wait_child(['fixture'],root,'unknown',timeout=1)

    def test_same_fd_physical_pin_exclusive_and_exact_bootstrap(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();p=root/'tiny.py';p.write_bytes(b'print("same-bytes")\n');pin=hashlib.sha256(p.read_bytes()).hexdigest()
            _,rec=s.capture(p,pin);self.assertEqual(rec['bytes'],len(p.read_bytes()))
            result=subprocess.run([sys.executable,'-B','-c',s.bootstrap(p,pin,[])],capture_output=True)
            self.assertEqual(result.returncode,0);self.assertEqual(result.stdout,b'same-bytes\n')
            p.write_bytes(b'print("changed")\n')
            result=subprocess.run([sys.executable,'-B','-c',s.bootstrap(p,pin,[])],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,b'')
            link=root/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):s.capture(link)
            hard=root/'hard';os.link(p,hard)
            with self.assertRaises(ValueError):s.capture(p)
            marker=root/'marker';s.write_once(marker,{'once':True})
            before=marker.read_bytes()
            with self.assertRaises(FileExistsError):s.write_once(marker,{'once':False})
            self.assertEqual(marker.read_bytes(),before)

    def test_sampler_only_actual_gone_and_foreign_unreadable_rejected(self):
        app=dict(pid=17,uuid='GPU-a',used_mib=1);known={17:dict(pid=17,start_ticks=9,job_id='new',step_id='0')}
        def gone(*a):raise FileNotFoundError(errno.ENOENT,'gone')
        v=s.classify_app(app,known,lambda:[],lambda p:True,gone,'new','0')
        self.assertEqual(v['classification'],'EXITED_BETWEEN_OBSERVATIONS')
        for owners,apps,proc in [({},[],True),(known,[app],True),(known,[],False)]:
            with self.assertRaises(ValueError):s.classify_app(app,owners,lambda:apps,lambda p:proc,gone,'new','0')
        def denied(*a):raise PermissionError(errno.EACCES,'unreadable')
        with self.assertRaises(ValueError):s.classify_app(app,known,lambda:[],lambda p:True,denied,'new','0')
        def foreign(*a):raise ValueError('foreign job')
        with self.assertRaisesRegex(ValueError,'foreign job'):s.classify_app(app,known,lambda:[],lambda p:True,foreign,'new','0')
        def reused(*a):return dict(pid=17,start_ticks=10,job_id='new')
        with self.assertRaisesRegex(ValueError,'PID reused'):s.classify_app(app,known,lambda:[],lambda p:True,reused,'new','0')

    def test_submission_preview_no_submit_fixed_gpu_budget_clean_submit_env(self):
        refs=[dict(path=str(s.ROOT/'prepared/request.json'),sha256='a'*64),dict(path=str(s.ROOT/'tools/new/paired_supervisor.py'),sha256='b'*64)]
        with patch.dict(os.environ,{'SBATCH_GRES':'bad','SRUN_NODELIST':'foreign','SLURM_JOB_ID':'3304'}):
            v=s.submission_preview(*refs)
        self.assertIsNone(v['actual_job_id']);self.assertIsNone(v['actual_wait'])
        self.assertIn('--gres=gpu:h100:8',v['argv']);self.assertIn('--no-requeue',v['argv']);self.assertIn('SBATCH_GRES',v['argv'])
        self.assertIn('exec srun --nodes=1 --ntasks=1 --cpus-per-task=192 --gres=gpu:h100:8',v['batch_script'])
        self.assertNotIn('SLURM_STEP_ID=',v['batch_script'])
        self.assertNotIn('ray stop',v['batch_script']);self.assertNotIn('pkill',v['batch_script'])
        with self.assertRaises(ValueError):s.submission_preview(dict(path='/tmp/foreign',sha256='a'*64),refs[1])
        with self.assertRaises(ValueError):s.submission_preview(dict(path=str(s.ROOT/'../escape'),sha256='a'*64),refs[1])

    def test_literal_slurm_allocation_and_typed_no_wrong_job(self):
        raw='JobId=901 UserId=root(0) JobState=RUNNING Restarts=0 NodeList=h100-10 BatchHost=h100-10 NumNodes=1 NumCPUs=192 NumTasks=1 OverSubscribe=NO TresPerNode=gres:gpu:h100:8 TRES=cpu=192,mem=1T,node=1'
        self.assertEqual(s.slurm_allocation(raw,'901')['NumCPUs'],'192')
        for wrong in [raw.replace('h100:8','h100:7'),raw.replace('JobId=901','JobId=9010'),raw.replace('OverSubscribe=NO','OverSubscribe=YES'),raw+' JobId=901']:
            with self.assertRaises(ValueError):s.slurm_allocation(wrong,'901')

    def test_actual_frozen_c5bf_full_config_and_proposed_stage_no_review_cycle(self):
        repo=HERE.parents[1]/'opsd-final200-paired-eval-source'
        # This is local authored-config compatibility, not the native94 import/GPU proof.
        sys.path.insert(0,str(repo))
        contract,_=s.exact_module(HERE/'paired_eval_contract.py',s.CONTRACT_SHA,'paired_config_fixture_contract')
        q=dict(common_manifest=dict(path=str(s.ROOT/'runs/formal-3304/bindings/data_manifest.json'),sha256=contract.COMMON),
               layered_receipt=dict(path=str(s.ROOT/'future-layered.json'),sha256='c'*64),source_inventory=dict(path=str(s.ROOT/'inventory.json'),sha256=s.INVENTORY_SHA),native_import_closure=dict(path=str(s.ROOT/'closure.json'),sha256=s.CLOSURE_SHA))
        inputs=dict(path=str(s.ROOT/'paired-final200-runs/eval-901/ordered-inputs.json'),sha256='d'*64)
        run=s.ROOT/'paired-final200-runs/eval-901'
        with patch.object(s,'REPO',repo):
            configs={r:cpu.compose_role(s,q,run,r,inputs,contract) for r in s.ROLES}
        for r in s.ROLES:
            self.assertTrue(configs[r]['trainer']['val_only']);self.assertEqual(configs[r]['trainer']['total_training_steps'],0)
            self.assertEqual(configs[r]['actor_rollout_ref']['model']['path'],str(s.MODELS[r]))
            for n in contract.INACTIVE:self.assertIsNone(configs[r]['baselines']['opsd_e1'][n])
            stage=s.proposed_stage(q,contract,dict(job_id='901'),r,{'config.json':'f'*64},inputs,{'source.py':'e'*64})
            self.assertIsNone(stage['root_pre_gpu_review']);self.assertTrue(stage['stage_construction_scope'].startswith('PROPOSED'))
            self.assertEqual(stage['input_manifest_sha256'],inputs['sha256'])
        wrong=json.loads(json.dumps(configs['official_base']));wrong['actor_rollout_ref']['actor']['use_kl_loss']=True
        with self.assertRaises(ValueError):contract.validate_resolved_config(wrong)

    def test_role2_not_entered_if_role1_cleanup_unknown(self):
        # Lifecycle proof uses controlled composition, not Ray/GPU fixtures.
        owner=dict(job_id='901',step_id='0');tracker=s.RoleProcesses(owner,'token',Path('/tmp/fixture'),Path('/tmp/fixture'))
        tracker.cleanup_invoked=True
        with self.assertRaisesRegex(ValueError,'do not automatically repeat'):tracker.cleanup([],None,dict(pid=1))
        source=(HERE/'paired_supervisor.py').read_text()
        self.assertIn("outcomes['official_base']['cleanup']['status']",source)
        self.assertNotIn("['ray', 'stop']",source);self.assertNotIn('pkill',source)

    def test_head_gone_eval_alive_scoped_cleanup_waits_real_child(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name).resolve();head=subprocess.Popen([sys.executable,'-B','-c','pass']);head.wait()
            child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])
            tracker=s.RoleProcesses(dict(job_id='fixture',step_id='0'),'fixture',folder,folder)
            rec=dict(pid=child.pid,start_ticks=1,job_id='fixture',step_id='0')
            def scan(roots):tracker.known[child.pid]=rec
            fake_sampler=type('FixtureSampler',(),{'sample':lambda self:dict(gpu_processes=[])})()
            try:
                with patch.object(tracker,'scan',side_effect=scan),patch.object(s,'same_process',side_effect=lambda r:child.poll() is None),\
                    patch.object(s,'explicit_gone',side_effect=lambda pid:child.poll() is not None):
                    result=tracker.cleanup([head,child],fake_sampler,dict(pid=-1))
                self.assertEqual(child.returncode,-signal.SIGTERM)
                self.assertTrue(all(v['waited'] for v in result['direct_waits']))
                self.assertTrue((folder/'cleanup-waits.original.json').exists())
                self.assertIn('if direct and not tracker.cleanup_invoked:',(HERE/'paired_supervisor.py').read_text())
            finally:
                if child.poll() is None:child.kill();child.wait()

    def test_complete_models_real_training_state_directory_and_unknown_entries(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();base=root/'base';hf=root/'hf';base.mkdir();hf.mkdir();(hf/'training_state').mkdir()
            for folder in (base,hf):
                for n in ('config.json','tokenizer.json','model.safetensors'):(folder/n).write_bytes(b'fixture')
            (base/'VERIFIED-MODEL-MANIFEST.json').write_bytes(b'fixture-manifest')
            contract=type('FixtureContract',(),{'MODEL_MANIFEST':hashlib.sha256((base/'VERIFIED-MODEL-MANIFEST.json').read_bytes()).hexdigest()})()
            with patch.object(s,'MODELS',dict(official_base=base,actual_hf200=hf)):
                maps,_=cpu.complete_models(s,contract);self.assertNotIn('training_state',maps['actual_hf200'])
                (hf/'dangling').symlink_to(hf/'missing')
                with self.assertRaises((ValueError,FileNotFoundError)):cpu.complete_models(s,contract)
                (hf/'dangling').unlink();os.mkfifo(hf/'fifo')
                with self.assertRaises(ValueError):cpu.complete_models(s,contract)

    def test_service_observation_view_no_added_forward_and_gpu_devices_strict(self):
        spec=importlib.util.spec_from_file_location('paired_service_fixture',HERE/'paired_service_entry.py')
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        class Index:
            d=768;ntotal=9
            def __init__(self,device):self.device=device
            def getDevice(self):return self.device
        class Shards:
            def count(self):return 8
            def at(self,i):return Index(i)
        class Model:
            def parameters(self):return [type('Parameter',(),{'device':'cuda:0','dtype':'torch.float16'})()]
            def __call__(self,*a):raise AssertionError('No added forward allowed')
        cfg=type('Cfg',(),dict(faiss_gpu=True,encoder_device='cuda',retrieval_topk=3,index_path='/index',corpus_path='/corpus',retrieval_model_path='/e5',retrieval_use_fp16=True))()
        retriever=type('Retriever',(),{'index':Shards(),'encoder':type('Encoder',(),dict(model=Model(),device='cuda'))()})()
        faiss=type('Faiss',(),{'downcast_index':staticmethod(lambda value:value)})()
        report=service.loaded_record(retriever,cfg,dict(pid=1),faiss)
        self.assertEqual(report['gpu_index']['devices'],list(range(8)))
        cfg.faiss_gpu=False
        with self.assertRaises(ValueError):service.loaded_record(retriever,cfg,dict(pid=1),faiss)

    def test_service_popen_registration_precedes_fallible_ledger_true_local_wait(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();entry=root/'paired_service_entry.py';entry.write_bytes(b'fixture')
            assets=dict(status='FULL_TOOL_ASSET_CONTENT_VERIFIED',results=dict(e5=[dict(path=str(root/'e5/model'),sha256='a'*64)],
                wiki_gzip=dict(path=str(root/'wiki'),sha256='b'*64),index=dict(path=str(root/'index'),sha256='c'*64),corpus=dict(path=str(root/'corpus'),sha256='d'*64)))
            q=dict(tool_asset_manifest={},control_tools={'paired_service_entry.py':'e'*64,'paired_supervisor.py':'f'*64,'paired_cpu_prepare.py':'1'*64})
            lease=type('FixtureLease',(),{'fds':[],'environment':lambda self:{}})();registered=[]
            actual_popen=subprocess.Popen;actual_write=s.write_once
            def start_fixture(*a,**kw):return actual_popen([sys.executable,'-B','-c','raise SystemExit(3)'],**kw)
            def failing_write(path,value):
                if path.name=='service-popen.json':raise OSError('ledger fixture failure')
                return actual_write(path,value)
            def command_fixture(*a,**kw):(root/'service-port-before.stdout').write_bytes(b'')
            try:
                with patch.object(s,'reference',return_value=(assets,{})),patch.object(s,'capture',return_value=(None,{})),\
                     patch.object(s,'command',side_effect=command_fixture),patch.object(s,'__file__',str(root/'paired_supervisor.py')),\
                     patch.object(s.subprocess,'Popen',side_effect=start_fixture),patch.object(s,'write_once',side_effect=failing_write):
                    with self.assertRaisesRegex(OSError,'ledger fixture failure'):
                        s.service_start(root,q,{}, {},lease,None,lambda *handles:registered.append(handles))
                self.assertEqual(len(registered),1);proc,out,err=registered[0]
                self.assertEqual(proc.wait(timeout=3),3)
                self.assertTrue((root/'service-command.json').exists());self.assertFalse((root/'service-popen.json').exists())
            finally:
                if registered:
                    proc,out,err=registered[0]
                    if proc.poll() is None:proc.kill();proc.wait()
                    out.close();err.close()


if __name__=='__main__':unittest.main()
