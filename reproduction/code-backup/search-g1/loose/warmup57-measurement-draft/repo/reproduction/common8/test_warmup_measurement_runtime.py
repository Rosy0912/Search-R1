"""Independent empty/40 Adam proof CPU fixtures, not native state claims."""
import copy,unittest
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.warmup_measurement_plan import ROLE
from reproduction.common8.warmup_measurement_runtime import verify_optimizer_states,scheduler_projection


def states():
    run=dict(run_id='CPU-only',measurement_parent_run_id='actual-parent-fixture')
    plan=dict(slurm_job_id='123');sha='a'*64
    allocation=dict(accepted=True,run_id=run['run_id'],slurm_job_id='123',actual_cuda_runtime_uuids=['GPU-CPU-'+str(i) for i in range(8)])
    initial={};final={};hashes={}
    for rank in range(8):
        hashes[rank]=str(rank)*64
        initial[rank]=dict(schema_version=1,logical_update=0,run_id=run['run_id'],execution_role=ROLE,
            plan_sha256=sha,parent_production_run_id=run['measurement_parent_run_id'],slurm_job_id='123',rank=rank,
            pid=100+rank,start_ticks=200+rank,cuda_uuid=allocation['actual_cuda_runtime_uuids'][rank],
            initial_optimizer_writes=0,initial_adam_state_entries=0,final_optimizer_writes=0,
            parameter_tensor_count=2,parameter_numel=4,adam_state_entries=0,adam_state_tensor_count=0,adam_state_numel=0,
            adam_steps=[],all_parameters_and_adam_finite=True,initial_optimizer_proof_sha256=None,learning_rates=[consumed_lr(1)])
        final[rank]=dict(initial[rank],logical_update=2,final_optimizer_writes=40,
            adam_state_entries=2,adam_state_tensor_count=6,adam_state_numel=10,adam_steps=[40.,40.],
            initial_optimizer_proof_sha256=hashes[rank],learning_rates=[consumed_lr(2)])
    return final,run,plan,sha,allocation,initial,hashes


class WarmupMeasurementRuntimeTests(unittest.TestCase):
    def test_independent_empty_initial_and_actual40_state_bind_birth_device_and_lr(self):
        args=states();result=verify_optimizer_states(*args)
        self.assertEqual(len(result),8);self.assertEqual([row['final_optimizer_writes'] for row in result],[40]*8)

    def test_stale_twenty_state_wrong_initial_hash_nonfinite_or_changed_worker_rejected(self):
        for kind in ('20writes','initialHash','initialNonempty','changedBirth','wrongLR','nonfinite'):
            args=states();final,_,_,_,_,initial,_=args
            if kind=='20writes':final[7]['final_optimizer_writes']=20;final[7]['adam_steps']=[20.,20.]
            elif kind=='initialHash':final[7]['initial_optimizer_proof_sha256']='f'*64
            elif kind=='initialNonempty':initial[7]['adam_state_entries']=1
            elif kind=='changedBirth':final[7]['start_ticks']+=1
            elif kind=='wrongLR':final[7]['learning_rates']=[1e-6]
            else:final[7]['all_parameters_and_adam_finite']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):verify_optimizer_states(*args)

    def test_missing_or_duplicate_actual_rank_device_rejected(self):
        args=states();args[0].pop(7)
        with self.assertRaises(ValueError):verify_optimizer_states(*args)
        args=states();args[0][7]['cuda_uuid']=args[0][0]['cuda_uuid'];args[5][7]['cuda_uuid']=args[5][0]['cuda_uuid']
        with self.assertRaises(ValueError):verify_optimizer_states(*args)

    def test_scheduler_projection_survives_only_cleanup_hash_changes(self):
        rows={'0':dict(receipt_path='/CPU-receipt',selected_rows=[dict(kind='COMPLETE',actual_optimizer_writes=20)],
            selected_rows_sha256='a'*64,observed_file_sha256='b'*64,
            fatal_console_check=dict(observed_sha256='c'*64),owned_process_ledger_sha256='d'*64)}
        stable=scheduler_projection(rows);later=copy.deepcopy(rows)
        later['0']['fatal_console_check']['observed_sha256']='e'*64;later['0']['owned_process_ledger_sha256']='f'*64
        self.assertEqual(scheduler_projection(later),stable)
        later['0']['selected_rows'][0]['actual_optimizer_writes']=19
        self.assertNotEqual(scheduler_projection(later),stable)


if __name__=='__main__':unittest.main()
