"""Recovery entry boundaries with tiny scalar files; no native GPU or launch.

Production source/data and full-model decoding are replaced for these fixtures.
The actual candidate, Head-pair, namespace and unattempted-entry checks run.
"""
import contextlib
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.a800_common import run_recovery_pipeline as pipeline
from reproduction.a800_common.predecessor_admission import read_owned_json
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_namespace import namespace_origin
from tests.reproduction.test_actor_probe_restore_guard import fixture, write


@contextlib.contextmanager
def entry_fixture(root, with_verifier=False):
    base, run, _ = fixture(root)
    # Only prelaunch records survive; fixture endpoint admissions are unrelated
    # to this entry test and do not stand for an actual new allocation.
    (base/'logs/native-allocation-admission.json').unlink()
    run['run_start_semantics'] = 'same_experiment_full_state_resume'
    repo = base/'repo'; repo.mkdir()
    run['source_sha256'] = {}
    if with_verifier:
        entry = repo/pipeline.CHAIN_ENTRY; entry.parent.mkdir(parents=True)
        entry.write_text('# synthetic source-bound entry fixture; no acceptance\n')
        run['source_sha256'][pipeline.CHAIN_ENTRY] = digest(entry)
    candidate = base/'logs/actor-probe-recovery-candidate.json'; write(candidate, run)
    prep_path = base/'logs/actor-probe-recovery-preparation.json'
    import json
    prep = json.loads(prep_path.read_text()); prep['candidate_manifest_sha256'] = digest(candidate)
    write(prep_path, prep)
    staged_path = base/'logs/actor-probe-recovery-heads-staged.json'
    staged = json.loads(staged_path.read_text()); staged['preparation_sha256'] = digest(prep_path)
    write(staged_path, staged)
    with contextlib.ExitStack() as stack:
        for name in ('validate_base','assert_formal_run'):
            stack.enter_context(patch.object(pipeline, name))
        stack.enter_context(patch.object(pipeline, '__file__', str(repo/'reproduction/a800_common/run_recovery_pipeline.py')))
        stack.enter_context(patch.object(pipeline, 'supported_execution_role', return_value='production'))
        stack.enter_context(patch.object(pipeline, 'bound_ray_tmp_dir', return_value='/tmp/synthetic-ray-root'))
        stack.enter_context(patch('reproduction.common8.actor_probe_head_admission.verify_inputs_and_source'))
        context = stack.enter_context(patch('search_g1.actor_probe_recovery.read_recovery_context'))
        allocation = stack.enter_context(patch('reproduction.common8.native_allocation.admit',
            side_effect=AssertionError('Native allocation must not be reached by fixture')))
        stack.enter_context(patch.dict(os.environ, SEARCH_G1_BASE=str(base),
            SEARCH_G1_TRAINING_WORLD_SIZE='4', CUDA_VISIBLE_DEVICES='0,1,2,3',
            RAY_TMPDIR='/tmp/synthetic-ray-root'))
        yield base, run, context, allocation


class RecoveryPipelineTests(unittest.TestCase):
    def test_absent_or_changed_final_verifier_rejects_before_parent_load_or_gpu_claim(self):
        for present in (False, True):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as root:
                with entry_fixture(root, present) as (base, run, context, allocation):
                    if present: (base/'repo'/pipeline.CHAIN_ENTRY).write_text('# changed source fixture\n')
                    with self.assertRaises((AssertionError, FileNotFoundError)):
                        pipeline.launch(base, run)
                    context.assert_not_called(); allocation.assert_not_called()
                    self.assertFalse((base/'logs/pipeline-launch-claim.json').exists())
                    self.assertFalse((base/'gpu-job.lock').exists())

    def test_candidate_and_pair_are_checked_before_native_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            with entry_fixture(root, True) as (base, run, context, allocation):
                self.assertEqual(pipeline.preflight(base, run), namespace_origin(run))
                context.assert_called_once_with(run, 0); allocation.assert_not_called()
                candidate = base/'logs/actor-probe-recovery-candidate.json'
                original = candidate.read_bytes(); candidate.write_text('{"fixture":"rebound candidate"}')
                with self.assertRaises(AssertionError): pipeline.preflight(base, run)
                candidate.write_bytes(original)
                second = base/'artifacts/boundary_probe.pkl'
                second.write_text('{"fixture":"different binary Head"}')
                with self.assertRaises(AssertionError): pipeline.preflight(base, run)
                allocation.assert_not_called()

    def test_existing_attempt_claim_or_broken_symlink_is_never_overwritten(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as root:
                with entry_fixture(root, True) as (base, run, context, allocation):
                    claim = base/'logs/pipeline-launch-claim.json'
                    if symlink: claim.symlink_to(base/'missing.json')
                    else: claim.write_text('{"fixture":"earlier attempt"}')
                    with self.assertRaises(AssertionError): pipeline.launch(base, run)
                    if symlink: self.assertTrue(claim.is_symlink())
                    else: self.assertEqual(claim.read_text(), '{"fixture":"earlier attempt"}')
                    context.assert_not_called(); allocation.assert_not_called()

    def test_saved_pair_version_and_archive_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, _ = fixture(root, 100)
            self.assertEqual(pipeline.verify_saved_heads(base, run)['actor_step'], 100)
            original = copy.deepcopy(run)
            run['initial_probe_version'] = 'different-pair'
            with self.assertRaises(AssertionError): pipeline.verify_saved_heads(base, run)
            run = original
            archive = base/'artifacts/restored-version-manifest.json'
            archive.write_text('{"fixture":"different archive"}')
            with self.assertRaises(AssertionError): pipeline.verify_saved_heads(base, run)

    def test_only_explicit_historical_ledger_can_use_larger_bound(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root).resolve()/'process-ledger.json'; path.write_text('{"fixture":true}')
            actual_stat = Path.stat
            from types import SimpleNamespace
            def oversized(target, *args, **kwargs):
                actual = actual_stat(target, *args, **kwargs)
                if target == path:
                    return SimpleNamespace(st_uid=actual.st_uid, st_mode=actual.st_mode,
                        st_size=33*1024**2)
                return actual
            with patch.object(Path, 'stat', oversized):
                with self.assertRaises(AssertionError): read_owned_json(path)
                self.assertEqual(read_owned_json(path, limit=256*1024**2), {'fixture':True})
                with self.assertRaises(AssertionError): read_owned_json(path, limit=32*1024**2)


if __name__ == '__main__': unittest.main()
