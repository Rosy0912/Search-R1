"""Auditable A/B orchestration; actual adapter obligations never become synthetic PASS.

No launch entry is supplied. Owned Slurm wrapper and concrete full25 adapter
qualification remain explicit pending interfaces. This state machine checks
actual receipts and orders every original update and counterexample boundary.
"""
from qualification_contract_v1 import admit, assert_same_state, assert_counter_result, assert_AB, require

COUNTERS = ('wrong_dispatched_trajectory_provenance','stale_post_sampling_worker_summary',
            'rank6_admission_journal_io_failure','rank6_false_probability_acceptance_sha')


def run_qualification(adapter, binding):
    probability = admit(binding)  # Must refuse the null DRAFT before adapter.initialize.
    adapter.initialize_same_ray_actor_and_reference_roles(binding)
    stable_pids = adapter.actual_worker_pid_start_roles()
    phases = {}
    for label in ('A','B'):
        # The actual adapter must reconstruct the same DataLoader iterator from
        # the same canonical driver RNG, then execute original full25 restore.
        # Missing iterator/engine seeds are recorded and refused, not tolerated.
        restored = adapter.restore_complete25_and_driver_iterator(label)
        require(restored['actual_saved_worker_state_equal'] and restored['actual_saved_driver_state_equal']
                and restored['Adam'] == 500 and restored['scheduler'] == 25 and restored['cursor'] == 12800,
                'Incomplete actual full25 restoration')
        require(adapter.actual_worker_pid_start_roles() == stable_pids, 'A/B replaced actual actor/ref workers')
        before = adapter.capture_complete_worker_and_driver_state()
        if label == 'B': assert_same_state(phases['A']['before'], before, 'A/B full25 + iterator/RNG boundary')
        batch = adapter.build_real512x5_old_ref_advantage_and_masks(label, probability)
        require(batch['actual_trajectories'] == 2560 and batch['actual_questions'] == 512
                and batch['five_response_groups_verified'] and batch['raw_files_sha_verified'],
                'Real full rollout and original GiGPO state groups required')
        if label == 'B':
            for case in COUNTERS:
                paused = adapter.capture_complete_worker_and_driver_state()
                acknowledgement = adapter.actual_matched_rejection_rpc_on_cloned_metadata(case, batch)
                after = adapter.capture_complete_worker_and_driver_state()
                assert_counter_result(paused, after, acknowledgement)
                adapter.save_original_counter_receipts(case, paused, after, acknowledgement)
        result = adapter.original_ten_Adam_update_with_existing_forward_witness(label, batch)
        require(result['actual_Adam_per_rank'] == 10 and result['gradient_micro_per_rank'] == 160
                and result['monitor_prepost_count_per_rank'] == 20
                and result['fixed_global256_coverage_verified'], 'Actual update/monitor coverage differs')
        reloaded = adapter.save_and_actually_reload_worker_driver_and_hf(label)
        require(reloaded['actual_all8_worker_roundtrip'] and reloaded['actual_driver_roundtrip']
                and reloaded['HF_semantic_reload'], 'Saved bytes alone do not prove complete state/HF reload')
        phases[label] = dict(before=before, **adapter.phase_receipts_for_AB(label, before, batch, result, reloaded))
    return assert_AB(phases['A'], phases['B'])
