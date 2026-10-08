"""Local syntheticCPU proof: no Slurm/GPU/services/real tensors/HF are touched."""
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import probe_package as package
import dispatch_once as dispatch
import freeze_package as freeze
import supervisor


def foreign_holder(paths,ready,finish):
    files=[]
    try:
        for path in paths:
            stream=open(path,'a+');files.append(stream);fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready.send(True);finish.recv()
    finally:
        for stream in files:stream.close()


def mock_receipt(stdout='991;h100',code=0):
    return SimpleNamespace(stdout=stdout,stderr='',returncode=code)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self): self.temp.cleanup()
    def template(self):
        return dict(output=package.OUTPUT_TOKEN,status='GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED',runner_commit='1'*40,
            selection_file=dict(path='/unused/metadata',sha256=package.SELECTION_SHA),
            reviewed_resource_adapter=dict(path=str(package.ROOT/'candidate/adapter.py'),sha256='2'*64),
            probe_source_files=[dict(path=str(package.ROOT/'candidate/repo/a.py'),sha256='3'*64)],
            installed_sources=[dict(path='/unused/site/a.py',sha256='4'*64)],installed_versions=dict(torch='2.4.0'))
    def envelopes(self):
        template=self.template();manifest=dict(schema_version=1,protocol=copy.deepcopy(package.PROTOCOL),
            bundle_payload_sha256=package.digest(package.bundle_payload(template)),selection_sha256=package.SELECTION_SHA,
            runner_commit=template['runner_commit'],reviewed_resource_adapter=template['reviewed_resource_adapter'],
            probe_source_files=template['probe_source_files'],installed_sources=template['installed_sources'],
            installed_versions=template['installed_versions'])
        receipt=dict(status=package.RELEASE_STATUS,protocol=copy.deepcopy(package.PROTOCOL),
            package_manifest_sha256='5'*64,bundle_payload_sha256=manifest['bundle_payload_sha256'],
            selection_sha256=package.SELECTION_SHA,runner_commit='1'*40,actual_optimizer_writes=0,
            adapter_sha256='2'*64,dispatch_intent_path=package.DISPATCH_INTENT)
        return template,manifest,receipt
    def test_canonical_unicode_format(self):
        self.assertEqual(package.canonical({'z':1,'a':'中文'}),'{"a":"中文","z":1}'.encode())
    def test_strict_json_duplicate_rejected(self):
        with self.assertRaises(ValueError):package.decode(b'{"a":1,"a":2}')
    def test_strict_json_nonfinite_and_overflow_rejected(self):
        for value in (b'NaN',b'Infinity',b'1e999'):
            with self.assertRaises(ValueError):package.decode(value)
    def test_same_fd_metadata_verified(self):
        path=self.root/'meta';payload=b'{"ok":true}';path.write_bytes(payload)
        self.assertEqual(package.read_bound(path,hashlib.sha256(payload).hexdigest()),{'ok':True})
    def test_atomic_evidence_never_clobbered(self):
        path=self.root/'evidence';package.atomic_json(path,{'a':1})
        with self.assertRaises(FileExistsError):package.atomic_json(path,{'a':2})
        self.assertEqual(json.loads(path.read_text()),{'a':1})
    def test_runtime_output_projection_exact(self):
        template=self.template();runtime=copy.deepcopy(template)
        runtime['output']=package.runtime_output('3279');runtime['release']={'path':'x','sha256':'y'}
        self.assertEqual(package.digest(package.bundle_payload(template)),package.digest(package.bundle_payload(runtime,'3279')))
    def test_wrong_output_and_extra_token_rejected(self):
        template=self.template();template['output']='/wrong'
        with self.assertRaises(ValueError):package.bundle_payload(template)
        template=self.template();template['secret']=package.OUTPUT_TOKEN
        with self.assertRaises(ValueError):package.bundle_payload(template)
    def test_single_release_sufficient_no_second_envelope(self):
        template,manifest,receipt=self.envelopes()
        self.assertTrue(package.validate_release(manifest,'5'*64,template,receipt))
    def test_full_source_entry_tampering_changes_content(self):
        template,manifest,receipt=self.envelopes();template['probe_source_files'][0]['sha256']='9'*64
        with self.assertRaises(ValueError):package.validate_release(manifest,'5'*64,template,receipt)
    def test_false_or_wrong_gpu_release_rejected(self):
        for change in ({'status':'SOURCE_CPU_ONLY'},{'package_manifest_sha256':'9'*64},{'actual_optimizer_writes':False},
                       {'runner_commit':'9'*40},{'adapter_sha256':'9'*64}):
            template,manifest,receipt=self.envelopes();receipt.update(change)
            with self.assertRaises(ValueError):package.validate_release(manifest,'5'*64,template,receipt)
    def test_actual_allocation_normalized_memory(self):
        text='JobId=9 JobState=RUNNING NodeList=h100-9 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=64 TimeLimit=00:30:00 Requeue=0 MinMemoryNode=256G'
        self.assertEqual(package.allocation(text,'9'),64)
        self.assertEqual(package.allocation(text.replace('256G','262144M'),'9'),64)
    def test_requeue_memory_cpu_and_deadline_refused(self):
        text='JobId=9 JobState=RUNNING NodeList=h100-9 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=64 TimeLimit=00:30:00 Requeue=0 MinMemoryNode=256G'
        for old,new in (('Requeue=0','Requeue=1'),('256G','200G'),('CPUs/Task=32','CPUs/Task=16'),('00:30:00','01:00:00')):
            with self.assertRaises(ValueError):package.allocation(text.replace(old,new),'9')
    def test_guard_fresh_identity_and_dual_thresholds(self):
        uuids=['GPU-%d'%i for i in range(8)]
        row=dict(job_id='9',epoch=100,gpus=[dict(uuid=u,foreign_memory_fraction=.15,foreign_sm_upper=15) for u in uuids])
        self.assertEqual(package.guard_record(row,'9',uuids,115),row)
        for change in ({'epoch':99},{'epoch':116},{'event':'STOP_OWN_JOB'}):
            item=copy.deepcopy(row);item.update(change)
            with self.assertRaises(ValueError):package.guard_record(item,'9',uuids,115)
        for key,value in [('foreign_memory_fraction',.15001),('foreign_sm_upper',15.01),('foreign_sm_upper',float('nan'))]:
            item=copy.deepcopy(row);item['gpus'][0][key]=value
            with self.assertRaises(ValueError):package.guard_record(item,'9',uuids,115)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'intent.json';self.calls=[]
    def tearDown(self):self.temp.cleanup()
    def submit(self,argv):self.calls.append(argv);return mock_receipt()
    def test_success_one_attempt_durable_original_intent(self):
        row=dispatch.submit_once(self.path,['fake-sbatch'],{'release':'S'},self.submit)
        self.assertEqual(row['job_id'],'991');self.assertEqual(len(self.calls),1)
        self.assertEqual(json.loads(self.path.read_text())['status'],'SUBMISSION_ATTEMPTED_OUTCOME_UNKNOWN')
    def test_repeat_after_success_refused(self):
        dispatch.submit_once(self.path,['fake-sbatch'],{},self.submit)
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['different'],{},self.submit)
        self.assertEqual(len(self.calls),1)
    def test_nonzero_is_unknown_and_never_retried(self):
        def failure(argv):self.calls.append(argv);return mock_receipt('991',1)
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['fake'],{},failure)
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['fake'],{},self.submit)
        self.assertEqual(len(self.calls),1)
    def test_timeout_unknown_intent_precedes_side_effect(self):
        def timeout(argv):
            self.assertTrue(self.path.is_file());self.calls.append(argv);raise subprocess.TimeoutExpired(argv,30)
        with self.assertRaises(subprocess.TimeoutExpired):dispatch.submit_once(self.path,['fake'],{},timeout)
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['fake'],{},self.submit)
        self.assertEqual(len(self.calls),1)
    def test_malformed_response_no_guess(self):
        for index,answer in enumerate(('991\n992','Submitted batch job 991','991;h100;other','0','')):
            path=self.path.with_name('intent%d'%index)
            with self.assertRaises(ValueError):dispatch.submit_once(path,['fake'],{},lambda argv:mock_receipt(answer))
            self.assertTrue(path.is_file())
    def test_concurrent_attempt_never_second_submit(self):
        entered=threading.Event();finish=threading.Event();errors=[]
        def hold(argv):self.calls.append(argv);entered.set();self.assertTrue(finish.wait(5));return mock_receipt()
        def execute():
            try:dispatch.submit_once(self.path,['fake'],{},hold)
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=execute);thread.start();self.assertTrue(entered.wait(5))
        try:
            with self.assertRaises(BlockingIOError):dispatch.submit_once(self.path,['other'],{},self.submit)
        finally:finish.set();thread.join(5)
        self.assertFalse(errors);self.assertEqual(len(self.calls),1)
    def test_failure_to_persist_intent_zero_submissions(self):
        with patch.object(dispatch,'atomic_json',side_effect=OSError('disk synthetic')):
            with self.assertRaises(OSError):dispatch.submit_once(self.path,['fake'],{},self.submit)
        self.assertFalse(self.calls)
    def test_uncertain_confirmation_write_still_blocks_repeat(self):
        original=dispatch.atomic_json
        def fail_confirmation(path,value):
            if Path(path)!=self.path:raise OSError('synthetic response write')
            return original(path,value)
        with patch.object(dispatch,'atomic_json',side_effect=fail_confirmation):
            with self.assertRaises(OSError):dispatch.submit_once(self.path,['fake'],{},self.submit)
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['fake'],{},self.submit)
        self.assertEqual(len(self.calls),1)
    def test_command_exact_allocation_no_wrap_injection(self):
        args=dispatch.command('/same node/a','M','/same node/b','R')
        for value in ('--gres=gpu:8','--exclusive','--cpus-per-task=32','--mem=256G','--time=00:30:00','--no-requeue'):
            self.assertIn(value,args)
        self.assertFalse(any(x.startswith('--wrap') for x in args));self.assertIn('/same node/a/start.sh',args)
    def test_active_tree_gpu_queue_refused_pending_and_running(self):
        for state in ('RUNNING','PENDING'):
            rows=['81|tree-safety300|'+state+'|h100-9|gpu:8']
            text='JobId=81 NodeList=h100-9 ReqNodeList=h100-9 ReqTRES=cpu=32,gres/gpu=8'
            found=dispatch.queue_conflicts(rows,lambda job:text,lambda nodes:[nodes])
            self.assertEqual(len(found),1)
    def test_cpu_tree_or_fixed_other_node_queue_not_blocking(self):
        rows=['81|tree-cpu-audit|RUNNING|h100-9|N/A','82|tree-other|RUNNING|h100-16|gpu:8']
        texts={'81':'JobId=81 NodeList=h100-9 ReqNodeList=h100-9 ReqTRES=cpu=4,mem=32G',
               '82':'JobId=82 NodeList=h100-16 ReqNodeList=h100-16 ReqTRES=gres/gpu=8'}
        self.assertEqual(dispatch.queue_conflicts(rows,lambda job:texts[job],lambda nodes:[nodes]),[])
    def test_unassigned_pending_gpu_tree_is_potential_h1009(self):
        text='JobId=83 NodeList=(null) ReqNodeList=(null) ReqTRES=gres/gpu=8'
        self.assertEqual(len(dispatch.queue_conflicts(['83|tree-pending|PENDING|(null)|gpu:8'],
            lambda job:text,lambda nodes:[nodes])),1)
    def test_readonly_queue_refusal_prevents_attempt_marker_and_submit(self):
        def refused():raise ValueError('existing queued GPU tree')
        with self.assertRaises(ValueError):dispatch.submit_once(self.path,['fake'],{},self.submit,pre_submit=refused)
        self.assertFalse(self.path.exists());self.assertFalse(self.calls)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.streams=[]
        self.uuids=['GPU-%d'%i for i in range(8)]
        for uid in self.uuids:
            stream=(self.root/(uid+'.lock')).open('a+');self.streams.append(stream)
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.records=[dict(uuid=uid,fd=stream.fileno()) for uid,stream in zip(self.uuids,self.streams)]
    def tearDown(self):
        for stream in self.streams:stream.close()
        self.temp.cleanup()
    def test_current_real_same_ofd_locks_all_eight(self):
        self.assertTrue(package.validate_owned_lease_fds(self.records,self.uuids,self.root))
    def test_all_locks_lost_refused(self):
        for stream in self.streams:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        with self.assertRaises(ValueError):package.validate_owned_lease_fds(self.records,self.uuids,self.root)
    def test_foreign_holder_not_mistaken_for_parent(self):
        for stream in self.streams:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        ctx=multiprocessing.get_context('fork');ready_parent,ready_child=ctx.Pipe(False);finish_child,finish_parent=ctx.Pipe(False)
        child=ctx.Process(target=foreign_holder,args=([str(self.root/(u+'.lock')) for u in self.uuids],ready_child,finish_child));child.start()
        try:
            self.assertTrue(ready_parent.poll(5));self.assertTrue(ready_parent.recv())
            with self.assertRaises(BlockingIOError):package.validate_owned_lease_fds(self.records,self.uuids,self.root)
        finally:
            finish_parent.send(True);child.join(5)
            if child.is_alive():child.terminate();child.join(5)
        self.assertEqual(child.exitcode,0)


class ParentLoopTests(unittest.TestCase):
    def run_loop(self,guard_exit=None,gate_error=None,probe_exit=0):
        values=iter([None,probe_exit]);probe=SimpleNamespace(poll=lambda:next(values),returncode=probe_exit)
        guard=SimpleNamespace(poll=lambda:guard_exit);adapter=SimpleNamespace(check=unittest.mock.Mock(side_effect=gate_error,return_value={'ok':True}))
        with tempfile.TemporaryDirectory() as temp,patch.object(supervisor,'validate_owned_lease_fds') as leases, \
             patch.object(supervisor.time,'sleep'),patch.object(supervisor.time,'monotonic',return_value=1):
            supervisor.wait_probe(probe,guard,[],[],adapter,SimpleNamespace(baseline={1}),None,
                lambda pid:{'pid':pid},lambda scope:{1,2},temp,0)
            rows=(Path(temp)/'process-resources.jsonl').read_text().splitlines()
            return leases.call_count,adapter.check.call_count,rows
    def test_actual_live_poll_records_one_check_and_own_process(self):
        leases,checks,rows=self.run_loop();self.assertEqual((leases,checks),(1,1))
        self.assertEqual(json.loads(rows[0])['processes'],[{'pid':2}])
    def test_actual_guard_exit_refused_before_next_forward(self):
        with self.assertRaises(ValueError):self.run_loop(guard_exit=1)
    def test_actual_live_adapter_failure_refused(self):
        with self.assertRaises(RuntimeError):self.run_loop(gate_error=RuntimeError('lost lease'))
    def test_actual_probe_nonzero_refused(self):
        with self.assertRaises(ValueError):self.run_loop(probe_exit=1)


class RankTests(unittest.TestCase):
    def test_full_actual_rank_receipts_and_unknown_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);identity={'test':'synthetic'}
            row=dict(world_size=8,status='SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE',
                source_and_input_bundle_identity=identity,optimizer_instantiated=False,scheduler_instantiated=False,
                scheduler_advanced=False,backward_called=False,training_acceptance=False,probability_consistency_acceptance=False,
                gpu_training_capacity_proven=False,optimizer_committed=0,full_state_unchanged_since_postwarm_baseline=True,
                original_module_modes_restored=True,original_forward_calls=2)
            for rank in range(8):
                path=root/('rank-%02d'%rank);path.mkdir();package.atomic_json(path/'ready.json',dict(row,rank=rank))
            self.assertEqual(len(package.rank_records(root,identity)),8)
            (root/'rank-07/ready.json').unlink()
            with self.assertRaises(FileNotFoundError):package.rank_records(root,identity)
    def test_true_optimizer_receipt_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rank-00';path.mkdir();package.atomic_json(path/'ready.json',dict(rank=0,world_size=8,
                status='SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE',
                source_and_input_bundle_identity={},optimizer_instantiated=True))
            with self.assertRaises(ValueError):package.rank_records(temp,{})


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        template=MetadataTests.template(self);package.atomic_json(self.root/'bundle-template.json',template)
        required=['start.sh','start.py','supervisor.py','probe_package.py','dispatch_once.py',
            'repo/search_r1/baselines/preflight.py',*[('repo/search_r1/baselines/probability_probe_'+name+'.py')
                for name in ('runner','loader','capture','io','contracts','monitor')]]
        for name in required:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# own synthetic CPU source\n')
        self.spec=dict(runner_commit='1'*40,torchrun_port=26382,external_files={},
            **{field:template[field] for field in ('reviewed_resource_adapter','probe_source_files','installed_sources','installed_versions')})
    def tearDown(self):self.temp.cleanup()
    def test_full_staged_source_manifest_hashes_every_file(self):
        manifest=freeze.build_manifest(self.root,self.spec)
        self.assertEqual(manifest['files']['start.py'],package.file_sha(self.root/'start.py'))
        self.assertEqual(manifest['bundle_payload_sha256'],package.digest(package.bundle_payload(MetadataTests.template(self))))
        extra=self.root/'repo/new_unlisted_helper.py';extra.write_text('# additional actual source\n')
        rebuilt=freeze.build_manifest(self.root,self.spec)
        self.assertIn('repo/new_unlisted_helper.py',rebuilt['files']);self.assertNotEqual(package.digest(manifest),package.digest(rebuilt))
    def test_no_tensor_assets_or_bytecode_staging(self):
        for name in ('raw.pt','model.safetensors','unbound.pyc'):
            path=self.root/name;path.write_bytes(b'own synthetic placeholder')
            try:
                with self.assertRaises(ValueError):freeze.build_manifest(self.root,self.spec)
            finally:path.unlink()


if __name__=='__main__':unittest.main(verbosity=2)
