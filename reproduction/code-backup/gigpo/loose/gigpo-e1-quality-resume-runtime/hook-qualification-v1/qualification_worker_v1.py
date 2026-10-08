"""Unlaunched actual Ray/FSDP candidate for diagnostic external-hook A/B.

The genuine metadata factory supplies outer Worker and inner (config, role).
Role dispatch and the original actor update perform FSDP/offload/Adam/scheduler.
No CPU result from this module is an eight-GPU acceptance.
"""
import os
import copy
from pathlib import Path
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from formal_probability_worker_v1 import ProductionProbabilityWorker
from qualification_contract_v1 import require
from qualification_torch_backend_v1 import ExistingForwardTensorBackend, actual_adam_step
from existing_forward_recorder_v1 import record_existing_actor_forwards, validate_recorded_coverage
from search_r1.baselines.recovery import atomic_torch_save, atomic_json_save


class HookQualificationWorker(ProductionProbabilityWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def qualification_phase(self, label):
        require(self._is_actor and label in ('A','B') and self.world_size == 8,
                'Qualification is actual eight-rank actor A/B only')
        require(actual_adam_step(self) == 500 and self.actor_lr_scheduler.last_epoch == 25,
                'A/B must each start from actual complete25, not a prior diagnostic update')
        self._qualification_label = label
        self._qualification_gradient_calls = 0
        self._qualification_completed_Adam = 0
        self._qualification_counter = None
        self._qualification_counter_ack = None
        return dict(rank=self.rank, pid=os.getpid(), label=label, actual_Adam=500, scheduler=25)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def qualification_actual_complete_state(self):
        state = self._complete_pause_state()
        return dict(rank=self.rank, complete=state, actual_Adam=actual_adam_step(self),
                    gradient_calls=self._qualification_gradient_calls,
                    diagnostic_Adam=self._qualification_completed_Adam,
                    retained_goal_contribution=0)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def qualification_counter(self, case):
        from qualification_protocol_v1 import COUNTERS
        require(self._qualification_label == 'B' and (case is None or case in COUNTERS)
                and actual_adam_step(self) == 500, 'Counterexample after an actual Adam is forbidden')
        self._qualification_counter, self._qualification_counter_ack = case, None
        return dict(rank=self.rank, case=case, actual_Adam=actual_adam_step(self))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def qualification_counter_acknowledgement(self):
        require(self._qualification_counter_ack is not None, 'RPC exception alone is not a matched counter receipt')
        return self._qualification_counter_ack

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def qualification_update_actor(self, data):
        """B reaches both real admission collectives; A calls original source once.

        Rejected B counters execute the same entry and throw before any original
        update. The driver must compare physical complete state and Adam before
        and after; a thrown RPC alone is not evidence of a zero-Adam result.
        """
        label = self._qualification_label
        require(label in ('A','B') and len(data) == 320, 'Wrong actual diagnostic DP chunk')
        before = actual_adam_step(self)
        directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'qualification' / label
        case = self._qualification_counter
        original_data, original_arm = data, self._production_arm
        import formal_probability_worker_v1 as formal_module
        original_gate, original_journal = formal_module.collective_gate, formal_module.atomic_json_save
        collective_decision = None
        def observed_gate(rows):
            nonlocal collective_decision
            # This function is reached by the frozen B method only after both
            # matching all_gather_object operations. Preserve all real proposals.
            collective_decision = copy.deepcopy(rows)
            return original_gate(rows)
        def journal(value,path):
            if case == 'rank6_admission_journal_io_failure' and self.rank == 6 and Path(path).name == 'iter-26-rank-6.json':
                raise OSError('Qualification declared pre-Adam local journal failure')
            return original_journal(value,path)
        if case is not None:
            require(label == 'B' and before == 500, 'Counters cannot execute on baseline A or a progressed actor')
            data = copy.deepcopy(data)
            if self.rank == 6:
                if case == 'wrong_dispatched_trajectory_provenance':
                    data.non_tensor_batch['audit_trajectory_row'][0] = 'DECLARED_INVALID_PROVENANCE_COUNTER'
                elif case == 'stale_post_sampling_worker_summary':
                    self._production_arm = copy.deepcopy(original_arm)
                    self._production_arm['complete_pause_state']['identity']['model_shards_sha256'] = '0'*64
                elif case == 'rank6_false_probability_acceptance_sha':
                    data.meta_info['formal_probability_gate']['manifest_sha256'] = '0'*64
            formal_module.collective_gate,formal_module.atomic_json_save = observed_gate,journal
        try:
            with record_existing_actor_forwards(self.actor, ExistingForwardTensorBackend(self)) as records:
                if label == 'A':
                    output = ActorRolloutRefWorker.update_actor(self, data)
                else:
                    output = ProductionProbabilityWorker.update_actor(self, data)
        except Exception:
            # Preserve real progress in memory for the later matched driver
            # state RPC; never label a partial update as zero Adam.
            self._qualification_gradient_calls += len([row for row in records
                if row['identity']['kind'] == 'gradient_full2560'])
            self._qualification_completed_Adam = actual_adam_step(self) - 500
            if case is not None:
                before_rows = collective_decision
                require(before_rows is not None and len(before_rows) == 8
                        and {row['rank'] for row in before_rows} == set(range(8))
                        and any(not row['local_ok'] for row in before_rows),
                        'No actual matching admission proposals; counter is unaccepted')
                self._qualification_counter_ack = dict(rank=self.rank,case=case,
                    matched_collective_rejected=True,actual_optimizer_steps_delta=actual_adam_step(self)-before,
                    actual_gradient_forward_delta=len([row for row in records if row['identity']['kind']=='gradient_full2560']),
                    actual_final_collective_proposals=before_rows)
            raise
        finally:
            formal_module.collective_gate,formal_module.atomic_json_save = original_gate,original_journal
            self._production_arm = original_arm if case is not None else self._production_arm
            data = original_data
        self._qualification_gradient_calls += len([row for row in records
            if row['identity']['kind'] == 'gradient_full2560'])
        self._qualification_completed_Adam = actual_adam_step(self) - 500
        coverage = validate_recorded_coverage(records)
        require(actual_adam_step(self) - before == 10
                and self.actor_lr_scheduler.last_epoch == 26, 'Actual diagnostic Adam/scheduler budget differs')
        # No metadata I/O occurs between forward observation and the original
        # source's optimizer. Save only after update returned; failure here is
        # reported as post-update failure with actual Adam retained.
        receipt = atomic_torch_save(dict(label=label,rank=self.rank,records=records,
            coverage=coverage, retained_goal_contribution=0),directory / f'existing-forwards-rank-{self.rank}.pt')
        atomic_json_save(dict(label=label,rank=self.rank,actual_Adam_before=before,
            actual_Adam_after=actual_adam_step(self),scheduler=26,coverage=coverage,
            witness=receipt,retained_goal_contribution=0),directory / f'existing-forwards-rank-{self.rank}.json')
        return output
