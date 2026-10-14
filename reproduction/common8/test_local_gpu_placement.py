"""Small CPU-only tests for launch placement and physical/UUID admission."""
import copy
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from reproduction.common8 import native_allocation as allocation


def inventory(indices=range(8)):
    return [[str(index),'GPU-'+str(index),'81920','0','0'] for index in indices]


def pending_run():
    return dict(profile='common8',training_world_size=4,execution_backend='ssh_local',
        slurm_job_id=None,execution_id='a'*32,run_id='b'*32,source_commit='c'*40,
        execution_host=socket.gethostname(),completed_updates=0,
        gpu_placement_policy=allocation.LOCAL_PLACEMENT_POLICY,gpu_placement_bound=False,
        training_physical_indices=[0,1,2,3],auxiliary_physical_indices=[5,6],
        retriever_physical_indices=[5,6],probe_physical_index=6,excluded_physical_indices=[4,7])


class IdleSelectionTests(unittest.TestCase):
    def test_cuda_uuid_reordering_refuses_before_any_context_allocation(self):
        indices=[1,2,3,4,5,6];expected=['GPU-'+str(i) for i in indices]
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda:6,synchronize=Mock()),empty=Mock(),uint8=object())
        def run_child(argv,**kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],','.join(map(str,indices)))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':kwargs['env']['CUDA_VISIBLE_DEVICES']}), \
                 patch.dict(sys.modules,{'torch':fake}),patch.object(sys,'argv',['-c',argv[3],argv[4]]), \
                 patch.object(allocation,'cuda_uuids',return_value=expected[::-1]):
                exec(argv[2],{})
        with patch.object(allocation.subprocess,'check_output',side_effect=run_child):
            with self.assertRaisesRegex(AssertionError,'before context creation'):
                allocation._local_cuda_probe(indices,expected)
        fake.empty.assert_not_called();fake.cuda.synchronize.assert_not_called()

    def test_rotating_foreign_zero_uses_idle_four_and_preserves_seven(self):
        training,excluded=allocation.select_idle_local_placement(inventory(),[['GPU-0','100'],['GPU-7','200']])
        self.assertEqual(training,[1,2,3,4])
        self.assertEqual(excluded,[0,7])

    def test_idle_seven_is_eligible_and_ids_are_not_counts(self):
        training,excluded=allocation.select_idle_local_placement(inventory(),[['GPU-0','100'],['GPU-1','101']])
        self.assertEqual((training,excluded),([2,3,4,7],[0,1]))
        sparse=inventory([1,2,3,4,5,6,17,21])
        self.assertEqual(allocation.select_idle_local_placement(sparse,[]),([1,2,3,4],[17,21]))

    def test_busy_auxiliary_or_insufficient_idle_cards_refuse_selection(self):
        with self.assertRaisesRegex(AssertionError,'Authorized local GPUs are occupied'):
            allocation.select_idle_local_placement(inventory(),[['GPU-5','100']])
        with self.assertRaisesRegex(AssertionError,'Insufficient idle local training GPUs'):
            allocation.select_idle_local_placement(inventory(),[['GPU-'+str(i),str(100+i)] for i in (0,1,7)])
        rows=inventory();rows[4][3]='257'
        with self.assertRaisesRegex(AssertionError,'Insufficient idle local training GPUs'):
            allocation.select_idle_local_placement(rows,[['GPU-0','100'],['GPU-7','200']])


class BoundPlacementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name).resolve();(self.base/'logs').mkdir()
        self.path=self.base/'logs/run-manifest.json';self.path.write_text(json.dumps(pending_run()))
        self.rows=inventory();self.foreign=[['GPU-0','100'],['GPU-7','200']]
        for target,value in (
            ('reproduction.common8.topology.validate_base',self.base),
            ('reproduction.a800.owned_processes.identity',dict(pid=300,start_ticks=400,sid=300,uid=os.getuid())),
            ('reproduction.common8.native_allocation._query_local_gpu_state',(self.rows,self.foreign))):
            p=patch(target,return_value=value);p.start();self.addCleanup(p.stop)
        p=patch.dict(os.environ,{'SEARCH_G1_TRAINING_WORLD_SIZE':'4'});p.start();self.addCleanup(p.stop)

    def bind(self):
        return allocation.bind_idle_local_placement(self.base)

    def test_binding_persists_actual_manifest_and_immutable_idempotent_receipt(self):
        run=self.bind();receipt=self.base/allocation.LOCAL_BINDING_RECEIPT
        self.assertEqual(json.loads(self.path.read_text()),run)
        self.assertEqual(run['training_physical_indices'],[1,2,3,4])
        self.assertEqual(run['training_gpu_uuids'],['GPU-'+str(i) for i in (1,2,3,4)])
        self.assertEqual(run['excluded_physical_indices'],[0,7])
        raw=receipt.read_bytes();manifest=self.path.read_bytes()
        self.assertEqual(self.bind(),run)
        self.assertEqual(receipt.read_bytes(),raw);self.assertEqual(self.path.read_bytes(),manifest)
        saved=json.loads(raw)
        self.assertFalse(saved['cuda_initialized']);self.assertFalse(saved['launch_claim'])
        self.assertEqual(saved['observed_compute_applications'],self.foreign)

    def test_foreign_arrival_and_stale_binding_never_reselect(self):
        run=self.bind();before=self.path.read_bytes()
        with patch.object(allocation,'_query_local_gpu_state',return_value=(self.rows,self.foreign+[['GPU-1','201']])):
            with self.assertRaisesRegex(AssertionError,'Authorized local GPUs are occupied'):self.bind()
        with patch.object(allocation.time,'time',return_value=run['gpu_placement_bound_unix']+121):
            with self.assertRaisesRegex(AssertionError,'placement is stale'):self.bind()
        self.assertEqual(self.path.read_bytes(),before)

    def test_unbound_and_false_complement_are_rejected(self):
        with self.assertRaisesRegex(AssertionError,'unbound preparation placeholder'):
            allocation.local_placement(pending_run())
        run=self.bind();false=copy.deepcopy(run);false['excluded_physical_indices']=[4,7]
        with self.assertRaisesRegex(AssertionError,'actual assignment complement'):
            allocation.local_placement(false)
        false=copy.deepcopy(run);false['training_physical_indices']=[1,2,3,5]
        with self.assertRaises(AssertionError):allocation.local_placement(false)

    def test_idle_recheck_precedes_first_cuda_and_foreign_arrival_refuses_admission(self):
        run=self.bind()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2,3,4'}), \
             patch.object(allocation,'_query_local_gpu_state',return_value=(self.rows,self.foreign+[['GPU-4','204']])), \
             patch.object(allocation,'_local_cuda_probe') as cuda:
            with self.assertRaisesRegex(AssertionError,'Authorized local GPUs are occupied'):
                allocation.admit_local(self.base,run)
            cuda.assert_not_called()
        self.assertFalse((self.base/'logs/native-allocation-admission.json').exists())

    def test_trainer_recheck_allows_auxiliary_services_but_refuses_foreign_training_context(self):
        run=self.bind();rows=copy.deepcopy(self.rows)
        rows[5][3]='20000';rows[5][4]='60'
        services=self.foreign+[['GPU-5','301'],['GPU-6','302']]
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,services)):
            allocation.assert_local_training_idle(run)
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,services+[['GPU-4','204']])):
            with self.assertRaisesRegex(AssertionError,'Authorized local GPUs are occupied'):
                allocation.assert_local_training_idle(run)

    def test_each_service_rechecks_idle_cards_or_authenticated_owned_births(self):
        run=self.bind();rows=copy.deepcopy(self.rows)
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,self.foreign)):
            allocation.assert_local_component_gpus_available(run,[5,6],[])
        contexts=self.foreign+[['GPU-6','301']];rows[6][3]='20000';rows[6][4]='75'
        record=dict(pid=301,start_ticks=500,sid=301,uid=os.getuid())
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,contexts)), \
             patch('reproduction.a800.owned_processes.same',return_value=True) as same:
            allocation.assert_local_component_gpus_available(run,[6],[record])
            same.assert_called_once_with(record)
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,contexts)), \
             patch('reproduction.a800.owned_processes.same',return_value=False):
            with self.assertRaisesRegex(AssertionError,'unowned or changed PID birth'):
                allocation.assert_local_component_gpus_available(run,[6],[record])
        with patch.object(allocation,'_query_local_gpu_state',return_value=(rows,contexts)):
            with self.assertRaisesRegex(AssertionError,'unowned or changed PID birth'):
                allocation.assert_local_component_gpus_available(run,[5,6],[])

    def test_admission_retains_four_rank_uuid_order_and_full_actual_inventory(self):
        run=self.bind();assigned=[1,2,3,4,5,6]
        cuda=dict(actual_cuda_runtime_uuids=['GPU-'+str(i) for i in assigned],
            context_logical_ordinals=list(range(6)),producer_identity=dict(pid=301))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2,3,4'}), \
             patch.object(allocation,'_local_cuda_probe',return_value=cuda) as probe:
            proof=allocation.admit_local(self.base,run)
        probe.assert_called_once_with(assigned,['GPU-'+str(i) for i in assigned])
        self.assertEqual(proof['training_world_size'],4)
        self.assertEqual(proof['actual_allocated_indices'],assigned)
        self.assertEqual(proof['actual_cuda_runtime_uuids'],['GPU-'+str(i) for i in (1,2,3,4)])
        self.assertEqual(proof['excluded_physical_indices'],[0,7])
        self.assertEqual(len(proof['physical_gpu_inventory']),8)
        self.assertEqual(proof['unassigned_compute_applications'],self.foreign)
        from reproduction.common8.formal_inline_safety import native_birth
        self.assertEqual(native_birth(proof)['training_physical_indices'],[1,2,3,4])

    def test_cuda_reordering_and_postproof_foreign_arrival_do_not_admit(self):
        run=self.bind();cuda=dict(actual_cuda_runtime_uuids=['GPU-'+str(i) for i in (2,1,3,4,5,6)],
            context_logical_ordinals=list(range(6)),producer_identity=dict(pid=301))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2,3,4'}),patch.object(allocation,'_local_cuda_probe',return_value=cuda):
            with self.assertRaisesRegex(AssertionError,'CUDA UUID mapping differs'):
                allocation.admit_local(self.base,run)
        cuda['actual_cuda_runtime_uuids']=['GPU-'+str(i) for i in (1,2,3,4,5,6)]
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2,3,4'}), \
             patch.object(allocation,'_local_cuda_probe',return_value=cuda), \
             patch.object(allocation,'_query_local_gpu_state',side_effect=[(self.rows,self.foreign),(self.rows,self.foreign+[['GPU-1','201']])]):
            with self.assertRaisesRegex(AssertionError,'foreign arrival after CUDA mapping proof'):
                allocation.admit_local(self.base,run)
        self.assertFalse((self.base/'logs/native-allocation-admission.json').exists())


if __name__=='__main__':unittest.main()
