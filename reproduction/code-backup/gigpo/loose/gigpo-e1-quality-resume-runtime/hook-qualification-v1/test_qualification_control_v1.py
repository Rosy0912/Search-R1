"""New qualification control tests with declared CPU adapter/forward spies.

These execute complete local orchestration, not GPU or Ray/FSDP interfaces.
No production/native CPU tests are repeated and no fake fixture becomes READY.
"""
import copy
import unittest
from unittest.mock import patch
import qualification_protocol_v1 as protocol
from qualification_contract_v1 import admit,assert_counter_result,assert_AB
from existing_forward_recorder_v1 import record_existing_actor_forwards,validate_recorded_coverage


class CPUActorSpy:
    def __init__(self): self.calls=[];self.output=([1],[2]);self.update=None
    def _forward_micro_batch(self,micro_batch,temperature):
        self.calls.append(('existing_forward',micro_batch,temperature));return self.output
    def _policy_safety_guard(self,monitor,temperature,control,identity,phase,writes):
        self.calls.append(('existing_guard',phase,writes))
        for micro in monitor:self._forward_micro_batch(micro,temperature)
        return 'original-guard-return-object'


class CPUBackendSpy:
    def __init__(self):self.state={'rng':1,'model':2,'Adam':500,'version':0};self.mutate=None
    def snapshot_micro_tensors(self,micro):return copy.deepcopy(micro)
    def snapshot_probability_tensors(self,*values):
        if self.mutate is not None:self.state[self.mutate]+=1
        return copy.deepcopy(values)
    def capture_observer_read_state(self):return copy.deepcopy(self.state)
    def micro_digest(self,micro):return repr(sorted(micro.items()))


class ExistingForwardCPU(unittest.TestCase):
    def test_original_call_return_identity_and_finally_restores_methods(self):
        actor=CPUActorSpy();backend=CPUBackendSpy();original=actor._forward_micro_batch;guard=actor._policy_safety_guard
        with record_existing_actor_forwards(actor,backend) as rows:
            output=actor._forward_micro_batch({'tokens':[1,2],'groups':[0]},1)
            self.assertIs(output,actor.output)
        self.assertEqual(len(actor.calls),1);self.assertEqual(len(rows),1)
        self.assertEqual(actor._forward_micro_batch,original);self.assertEqual(actor._policy_safety_guard,guard)

    def test_no_rng_model_adam_or_version_mutation_can_be_hidden_by_read(self):
        for field in ('rng','model','Adam','version'):
            with self.subTest(field=field):
                actor=CPUActorSpy();backend=CPUBackendSpy();backend.mutate=field
                original=actor._forward_micro_batch
                with self.assertRaisesRegex(ValueError,'altered actual model'):
                    with record_existing_actor_forwards(actor,backend):actor._forward_micro_batch({'tokens':[1]},1)
                self.assertEqual(actor._forward_micro_batch,original)
                self.assertEqual(len(actor.calls),1)

    def test_existing160_and20_prepost_guard_forward_coverage(self):
        actor=CPUActorSpy();backend=CPUBackendSpy();micro={'tokens':[1,2]}
        with record_existing_actor_forwards(actor,backend) as rows:
            for mini in range(10):
                actor._policy_safety_guard([micro]*16,1,None,{'mini_index':mini},'before',mini)
                for _ in range(16):actor._forward_micro_batch(micro,1)
                actor._policy_safety_guard([micro]*16,1,None,{'mini_index':mini},'after',mini+1)
        self.assertEqual(validate_recorded_coverage(rows),dict(gradient_forward_count=160,monitor_forward_count=320,extra_forward_count=0))
        self.assertEqual(len([row for row in actor.calls if row[0]=='existing_forward']),480)
        for case in ('missing','wrong_writes','wrong_temperature'):
            invalid=copy.deepcopy(rows)
            if case=='missing':invalid.pop()
            elif case=='wrong_writes':invalid[0]['identity']['actual_optimizer_writes']=1
            else:invalid[0]['temperature']=.8
            with self.assertRaises(ValueError):validate_recorded_coverage(invalid)


class CPUAdapterSpy:
    def __init__(self):self.events=[];self.state={'worker_rng':1,'driver_rng':2,'cursor':12800,'Adam':500};self.after_counter_mutation=None;self.AB_mutation=None;self.pending_phase=None
    def initialize_same_ray_actor_and_reference_roles(self,binding):self.events.append('declaredCPU-init-notRay')
    def actual_worker_pid_start_roles(self):return [(rank,rank+100,'5','actor&ref_CPU_fixture') for rank in range(8)]
    def restore_complete25_and_driver_iterator(self,label):
        self.pending_phase=label;self.state={'worker_rng':1,'driver_rng':2,'cursor':12800,'Adam':500}
        self.events.append('restore'+label)
        if self.AB_mutation=='restore' and label=='B':self.state['driver_rng']=99
        return dict(actual_saved_worker_state_equal=True,actual_saved_driver_state_equal=True,Adam=500,scheduler=25,cursor=12800)
    def capture_complete_worker_and_driver_state(self):return copy.deepcopy(self.state)
    def build_real512x5_old_ref_advantage_and_masks(self,label,probability):
        self.events.append('full2560CPUfixture'+label);self.state['cursor']=13312
        return dict(actual_trajectories=2560,actual_questions=512,five_response_groups_verified=True,raw_files_sha_verified=True)
    def actual_matched_rejection_rpc_on_cloned_metadata(self,case,batch):
        self.events.append('counter'+case)
        if self.after_counter_mutation:self.state[self.after_counter_mutation]+=1
        return [dict(rank=rank,matched_collective_rejected=True,actual_optimizer_steps_delta=0,actual_gradient_forward_delta=0) for rank in range(8)]
    def save_original_counter_receipts(self,*args):self.events.append('counter-receipt-CPU-fixture')
    def original_ten_Adam_update_with_existing_forward_witness(self,label,batch):
        self.events.append('diagnostic-Adam10'+label);self.state['Adam']=510
        return dict(actual_Adam_per_rank=10,gradient_micro_per_rank=160,monitor_prepost_count_per_rank=20,fixed_global256_coverage_verified=True)
    def save_and_actually_reload_worker_driver_and_hf(self,label):
        self.events.append('actual-reload-interface-CPU-fixture'+label)
        return dict(actual_all8_worker_roundtrip=True,actual_driver_roundtrip=True,HF_semantic_reload=True)
    def phase_receipts_for_AB(self,label,before,batch,result,reloaded):
        receipt=dict(actual_Adam_per_rank=10,gradient_micro_per_rank=160,retained_goal_contribution=0)
        keys=('before_worker_state','before_driver_state','actual_questions_and_groups','generated_tensors','balanced_scored_batch','old_and_ref_probabilities','gradient_micro_inputs_and_current','fixed_global256_monitor_inputs_and_current','loss_and_gradient_records','after_worker_state','after_driver_state','saved_reloaded_worker_state','saved_reloaded_driver_state')
        for key in keys:receipt[key]={'explicit_CPU_fixture':1}
        if self.AB_mutation in keys and label=='B':receipt[self.AB_mutation]={'explicit_CPU_fixture':2}
        return receipt


class FullOrchestrationCPU(unittest.TestCase):
    def test_null_actual_proofs_reject_before_any_initialization(self):
        adapter=CPUAdapterSpy()
        with self.assertRaisesRegex(ValueError,'DRAFT'):
            protocol.run_qualification(adapter,{'status':'DRAFT','actual_source':None})
        self.assertEqual(adapter.events,[])

    def run_admitted_CPU_fixture(self,adapter):
        # Admission itself needs original native/probability files. This mock is
        # deliberately only the complete state machine's prerequisite boundary.
        with patch.object(protocol,'admit',return_value={'CPU_probability_fixture':True}):
            return protocol.run_qualification(adapter,{'CPU_admission_fixture_not_READY':True})

    def test_complete_AB_one_outer_each_and_four_counters(self):
        adapter=CPUAdapterSpy();result=self.run_admitted_CPU_fixture(adapter)
        self.assertEqual(result['actual_diagnostic_Adam_per_rank'],20)
        self.assertEqual(result['retained_goal_contribution'],0)
        self.assertEqual([e for e in adapter.events if e.startswith('diagnostic-Adam')],['diagnostic-Adam10A','diagnostic-Adam10B'])
        self.assertEqual(len([e for e in adapter.events if e.startswith('counter') and 'receipt' not in e]),4)
        self.assertLess(adapter.events.index('restoreB'),adapter.events.index('diagnostic-Adam10B'))

    def test_each_real_training_state_change_on_counter_refuses_before_BAdam(self):
        for key in ('worker_rng','driver_rng','cursor','Adam'):
            with self.subTest(key=key):
                adapter=CPUAdapterSpy();adapter.after_counter_mutation=key
                with self.assertRaisesRegex(ValueError,'changed real full state'):self.run_admitted_CPU_fixture(adapter)
                self.assertNotIn('diagnostic-Adam10B',adapter.events)

    def test_iterator_restore_mismatch_is_not_absorbed_in_probability_tolerance(self):
        adapter=CPUAdapterSpy();adapter.AB_mutation='restore'
        with self.assertRaisesRegex(ValueError,'iterator/RNG'):self.run_admitted_CPU_fixture(adapter)
        self.assertNotIn('full2560CPUfixtureB',adapter.events)

    def test_current_grad_monitor_model_Adam_reload_differences_are_not_ABaccepted(self):
        for field in ('generated_tensors','old_and_ref_probabilities','gradient_micro_inputs_and_current',
                      'fixed_global256_monitor_inputs_and_current','loss_and_gradient_records',
                      'after_worker_state','saved_reloaded_driver_state'):
            with self.subTest(field=field):
                adapter=CPUAdapterSpy();adapter.AB_mutation=field
                with self.assertRaisesRegex(ValueError,'A/B '+field):self.run_admitted_CPU_fixture(adapter)


if __name__=='__main__':unittest.main(verbosity=2)
