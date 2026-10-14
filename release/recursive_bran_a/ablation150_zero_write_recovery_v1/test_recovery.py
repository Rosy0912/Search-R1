"""Small protocol fixtures and real CPU sampler/RNG; no model or remote access."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import patch

from .recovery import (FAILURE, FIELDS, IDENTITY_ENV, MANIFEST_ENV, SHA_ENV, manifest,
    restore_partial_driver, validate_initialization, validate_zero_write_evidence)


def fixture(base='/condition'):
    identity = dict(job_id='3393', run_id='run', code_commit='source', config_sha256='config',
                    data_identity='data', model_identity='official', attempted_outer=1,
                    completed_outer=0, attempt_id=1, training_protocol='tree_fresh200_five_leaf_v1')
    environ = {env: identity[key] for key, env in IDENTITY_ENV.items()}
    receipt = dict(base=base, group='C1-root-only', run_id='run', source_commit='source',
                   config_sha256='config', data_identity='data')
    workers = [dict(rank=r, world_size=8, matched_all_ranks=True, safety_guard=False,
        committed_optimizer_calls=0, scheduler_step_before_actual=0, scheduler_step_after_actual=0,
        status='PARTIAL_WORKER_STATE_INCOMPLETE', path=None, sha256=None, bytes=0,
        error='ValueError: '+FAILURE, refusal='PolicySafetyRefusal: Global partial worker context preparation refusal')
        for r in range(8)]
    value = dict(protocol='tree_partial_state_v1', status='PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP',
        identity=identity, outer_completed=False, automatic_rollback=False,
        completed_outer_updates=0, attempted_outer=1,
        driver=dict(consumed_cursor=512, consumed_epoch=0),
        actor_inputs=dict(rows=2560, fields=list(FIELDS)),
        worker_outcome=dict(protocol='tree_partial_state_v1', status='PARTIAL_WORKER_STATE_INCOMPLETE',
            identity_consistent=False, identity={}, scheduler_advanced_actual=[False]*8, worker_states=workers))
    return value, receipt, environ


def initial(identity):
    return dict(workers=[dict(rank=r, world_size=8, optimizer_state_entries=0,
        optimizer_step_values=[], scheduler_step=0, optimizer_committed=0, scheduler_advanced=False,
        configured_model_path='/official', expected_model_path='/official',
        **{key:identity[key] for key in IDENTITY_ENV}) for r in range(8)])


class AdmissionTests(unittest.TestCase):
    def test_exact_eight_rank_preupdate_refusal_is_admitted(self):
        value, receipt, environ = fixture()
        self.assertIs(validate_zero_write_evidence(value, receipt, environ), value)

    def test_one_write_or_scheduler_step_or_unmatched_rank_is_refused(self):
        for key, damage in (('committed_optimizer_calls', 1), ('scheduler_step_after_actual', 1),
                            ('matched_all_ranks', False), ('safety_guard', True), ('bytes', 1)):
            with self.subTest(key=key):
                value, receipt, environ = fixture()
                value['worker_outcome']['worker_states'][3][key] = damage
                with self.assertRaises(ValueError):
                    validate_zero_write_evidence(value, receipt, environ)

    def test_changed_provenance_batch_or_later_boundary_is_refused(self):
        for damage in ('source', 'run', 'row', 'schema', 'outer', 'rank'):
            with self.subTest(damage=damage):
                value, receipt, environ = fixture()
                if damage == 'source': environ['SEARCH_R1_BASELINE_COMMIT'] = 'new-source'
                elif damage == 'run': environ['SEARCH_R1_BASELINE_RUN_ID'] = 'new-run'
                elif damage == 'row': value['actor_inputs']['rows'] = 1280
                elif damage == 'schema': value['actor_inputs']['fields'].remove('advantages')
                elif damage == 'outer': value['identity']['attempted_outer'] = 2
                else: value['worker_outcome']['worker_states'].pop()
                with self.assertRaises(ValueError):
                    validate_zero_write_evidence(value, receipt, environ)

    def test_initial_adam_must_really_be_empty(self):
        value, _, _ = fixture()
        provenance = initial(value['identity'])
        validate_initialization(provenance, value['identity'])
        provenance['workers'][1]['optimizer_state_entries'] = 1
        with self.assertRaises(ValueError): validate_initialization(provenance, value['identity'])

    def test_manifest_sha_and_members_are_bound_to_original_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            from release.recursive_bran_a.ablation150_v1 import PROTOCOL
            value, receipt, environ = fixture(tmp)
            config = Path(tmp)/'config.yaml'; config.write_text('static-small-fixture')
            receipt.update(protocol=PROTOCOL, config=str(config),
                           config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            value['identity']['config_sha256'] = receipt['config_sha256']
            environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = receipt['config_sha256']
            receipt_path = Path(tmp)/'receipt.json'; receipt_path.write_text(json.dumps(receipt))
            parent = Path(tmp)/'checkpoints'/'training_state'/'partial_attempt_000001_outer_000001'
            parent.mkdir(parents=True)
            for key, filename in (('driver','driver.pt'), ('actor_inputs','actor-inputs.pt')):
                member = parent/filename; member.write_bytes(b'tiny fixture')
                value[key]['path'] = str(member)
            source = parent/'manifest.json'; source.write_text(json.dumps(value))
            environ.update(RECURSIVE_ABLATION_RECEIPT=str(receipt_path))
            environ[MANIFEST_ENV] = str(source)
            environ[SHA_ENV] = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(manifest(environ)['identity'], value['identity'])
            source.write_text(source.read_text()+' ')
            with self.assertRaises(ValueError): manifest(environ)

    def test_restore_cursor_and_rng_then_next_batch_skips_consumed_questions(self):
        from search_r1.baselines.training_state import CheckpointBatchSampler, digest, rng_state
        value, _, _ = fixture()
        old_sampler = CheckpointBatchSampler(1024, 512, 42, 'data')
        first = next(iter(old_sampler))
        tree_rng = random.Random(42); tree_rng.random()
        stage = dict(stage='training', seen=0)
        saved = dict(version=3, snapshot_kind='tree_partial_state_v1', identity=value['identity'],
            attempted_outer=1, completed_outer_updates=0, outer_completed=False,
            consumed_sampler=old_sampler.state_dict(), rng=rng_state(),
            algorithm=dict(method='tree_grpo', stage=copy.deepcopy(stage), tree_rng=tree_rng.getstate()),
            fresh_initialization_provenance=initial(value['identity']), recovery_provenance=None,
            guard_continue_accounting=dict(training_attempts=1, completed_outer_updates=0,
                refused_attempts=0, guard_refused_attempts=0, partial_evidence=[],
                completed_optimizer_calls_per_rank={}, partial_optimizer_calls_per_rank={},
                total_optimizer_calls_per_rank={str(r):0 for r in range(8)}))
        value['driver']['logical_state_sha256'] = digest(saved)
        live_stage = dict(stage='initial')
        trainer = N(global_steps=0, actual_outer_updates=0,
            checkpoint_sampler=CheckpointBatchSampler(1024, 512, 999, 'data'),
            fresh_initialization_provenance=initial(value['identity']),
            baseline_controller=N(method='tree_grpo', tree_rng=random.Random(999),
                stage=N(state_dict=lambda:copy.deepcopy(live_stage),
                        load_state_dict=lambda state: (live_stage.clear(), live_stage.update(state)))))
        random.seed(999)
        restore_partial_driver(trainer, saved, value)
        self.assertEqual((trainer.global_steps, trainer.actual_outer_updates, trainer.training_attempts), (1,0,1))
        self.assertEqual((trainer.checkpoint_sampler.cursor, trainer.checkpoint_sampler.epoch), (512,0))
        self.assertEqual(digest(rng_state()), digest(saved['rng']))
        second = next(iter(trainer.checkpoint_sampler))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(second, saved['consumed_sampler']['order'][512:1024])

    def test_prepared_update_is_called_once_then_native_credit_is_applied(self):
        from .recovery import complete_prepared_outer
        trace = []
        trainer = N(global_steps=1, actual_outer_updates=0, use_critic=False,
                    normal_guard_rounds=0, zero_write_sampling_rounds=0)
        batch = N(meta_info={})
        output = N(meta_info={'metrics': {'actor/optimizer_steps': 2},
            'tree_fresh_actor_result': {'ranks': [{'total_optimizer_calls': 2}]}})
        def actor_update(actual):
            self.assertIs(actual, batch)
            self.assertEqual(actual.meta_info['tree_partial_context'], {'replay': True})
            trace.append('actor')
            return output
        trainer.actor_rollout_wg = N(update_actor=actor_update)
        trainer.baseline_controller = N(ranker=None, after_actor=lambda b,m:trace.append('after_actor'))
        partial = ModuleType('search_r1.baselines.tree_partial_state')
        partial.partial_context = lambda t: {'replay': True}
        partial.stop_on_partial_actor_result = lambda t,b,o: False
        partial.preserve_unmatched_actor_failure = lambda *args: self.fail('No unmatched failure expected')
        guard = ModuleType('search_r1.baselines.tree_guard_continue')
        guard.validate_completed_result = lambda t,o: trace.append('all_rank_validate')
        def record(t, o):
            trace.append('native_credit')
            t.completed_optimizer_calls_per_rank = {str(r):2 for r in range(8)}
        guard.record_completed_result = record
        fresh = ModuleType('search_r1.baselines.tree_fresh200')
        fresh.validate_completed_outer = lambda m,s: trace.append('outer_validate')
        native = ModuleType('verl.trainer.ppo.ray_trainer')
        native.reduce_metrics = lambda m: dict(m)
        with patch.dict(sys.modules, {m.__name__:m for m in (partial, guard, fresh, native)}):
            metrics = complete_prepared_outer(trainer, batch)
        self.assertEqual(trace, ['actor','all_rank_validate','after_actor','outer_validate','native_credit'])
        self.assertEqual((trainer.global_steps, trainer.actual_outer_updates), (1,1))
        self.assertEqual(metrics['training/reused_saved_prepared_actor_batch'], 1)
        self.assertEqual(metrics['training/replayed_round_timing_excludes_original_sampling'], 1)
        self.assertGreaterEqual(metrics['timing_s/update_actor'], 0)
        self.assertGreaterEqual(metrics['timing_s/step'], metrics['timing_s/update_actor'])


if __name__ == '__main__': unittest.main()
