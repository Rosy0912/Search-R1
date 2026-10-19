"""Reject wrong checkpoint/partial state/science drift before any GPU action."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from . import entry
from reproduction.official_eval_v2.protocol import PROFILE, SCIENCE, digest


def fixture():
    files = {str(i): str(i) for i in range(16)}
    proof = dict(job_id=entry.TRAIN_JOB, source=entry.TRAIN_SOURCE,
        plan_sha256=entry.TRAIN_PLAN, completed_native_step=50,
        local_small_evidence_sha_verified=True, external_cpu_only=True,
        no_cuda_initialization=True, process_rng_unchanged=True, no_training_mutation=True,
        workers=[dict(rank=i, scheduler_last_epoch=50, optimizer_states=29,
            actual_adam_min=1000, actual_adam_max=1000, all_complete_state_finite=True,
            stored_cpu_numpy_python_and_cuda_rng_verified=True) for i in range(8)],
        physical_rank_files={str(i): {} for i in range(8)},
        sampler=dict(rows=169028, cursor=25600), hf_identity=dict(files=files, sha256='hf'))
    reference = dict(kind='base', root='/old/base-r1', job='old', source='/old/source',
        source_commit='old', source_files={n: n for n in entry.CORE}, model='/base',
        model_revision='base', model_files={}, model_identity='base', profile=PROFILE,
        science=SCIENCE, repeat=1, repeat_count=3, gpus=8, cpus=64, memory_gib=600,
        data_sha256='fixed-data', id_gold_identity='fixed-ids')
    plan = dict(reference, kind='cgrts50', root='/new/cgrts50-r1', job=entry.JOB, repeat_count=1,
        source='/new/source', source_commit='new', model='/train/checkpoints/actor/global_step_50',
        model_revision='139134-step50', model_files=files,
        model_identity=digest(dict(training_job=entry.TRAIN_JOB,
            training_source=entry.TRAIN_SOURCE, completed_step=50, hf_identity=proof['hf_identity'])),
        milestone_checkpoint=dict(step=50, training_job=entry.TRAIN_JOB, training_root='/train',
            reference_model_label_only=False, not_final200=True,
            accepted_scientific_repeat_count_for_milestone=1))
    plan['source_files'] = dict(reference['source_files'], **{entry.STARTUP_FILE: entry.STARTUP_SHA})
    return plan, reference, proof


class Contracts(unittest.TestCase):
    def test_valid_bound_milestone(self):
        entry.validate_contract(*fixture())

    def test_reject_science_data_core_or_hardware_drift(self):
        for key, value in [('science', {}), ('data_sha256', 'wrong'),
                           ('gpus', 16), ('source_files', {})]:
            p, r, a = fixture(); p[key] = value
            with self.subTest(key=key), self.assertRaises((AssertionError, KeyError)):
                entry.validate_contract(p, r, a)

    def test_reject_wrong_checkpoint_or_unfinished_proof(self):
        for key, value in [('job_id', '139101'), ('completed_native_step', 5),
                           ('local_small_evidence_sha_verified', False), ('workers', [])]:
            p, r, a = fixture(); a[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                entry.validate_contract(p, r, a)

    def test_reject_missing_rng_and_zero_or_divergent_adam(self):
        for key, value in [('stored_cpu_numpy_python_and_cuda_rng_verified', False),
                           ('actual_adam_min', 0), ('actual_adam_max', 999)]:
            p, r, a = fixture(); a['workers'][4][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                entry.validate_contract(p, r, a)

    def test_no_mutation_of_reference_or_frozen_driver(self):
        p, r, a = fixture(); before = copy.deepcopy((p, r, a))
        fn = entry.driver.run; verify = entry.driver.verify; worker = entry.driver.worker
        entry.validate_contract(p, r, a)
        self.assertEqual((p, r, a), before)
        self.assertIs(entry.driver.run, fn)
        self.assertIs(entry.driver.verify, verify)
        self.assertIs(entry.driver.worker, worker)

    def test_entry_uses_private_globals_and_same_run_bytecode(self):
        def private_probe(planfile):
            return verify, worker, planfile
        old_verify, old_worker = entry.driver.verify, entry.driver.worker
        with patch.object(entry.driver, 'run', private_probe):
            actual_verify, actual_worker, path = entry.run('fixture-only')
        self.assertIs(actual_verify, entry.verify)
        self.assertIs(actual_worker, entry.port_recovery.startup_worker)
        self.assertEqual(path, 'fixture-only')
        self.assertIs(entry.driver.verify, old_verify)
        self.assertIs(entry.driver.worker, old_worker)

    def test_stage_result_cannot_claim_final_or_three_repeats(self):
        for field, value in [('not_final200', False),
                             ('accepted_scientific_repeat_count_for_milestone', 3)]:
            p, r, a = fixture(); p['milestone_checkpoint'][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                entry.validate_contract(p, r, a)


if __name__ == '__main__':
    unittest.main()
