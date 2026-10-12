"""RAM sampler/RNG and constructed scalar receipts; no native training claim."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from reproduction.common8.actor_probe_driver_restore import verify_worker_restores, restore_driver_and_actor
from reproduction.common8.actor_probe_checkpoint import driver_payload
from reproduction.common8.recoverable_attempt import AttemptProgress
from reproduction.common8.formal_contract import formal_mathematics
from reproduction.common8.recoverable_attempt import record_attempt
from reproduction.common8.update_evidence import METRICS, record
from search_g1.actor_probe_namespace import namespace_origin, validate_driver_origin, validate_worker_origin
from search_g1.actor_probe_recovery import binding_digest, rebound_probe_manifest, identity
from search_g1.actor_probe_contract import MODE
from search_g1.actor_probe_sampler import RecordedRandomSampler, epoch_batches, restored_iterator
from search_g1.actor_probe_state import rng_state, tree_digest
from tests.reproduction.test_actor_probe_recovery import documents


class ResumeLoopTests(unittest.TestCase):
    def test_saved_epoch_then_new_epoch_matches_uninterrupted_batch_numbers_and_rng(self):
        for consumed in (3, 9):
            with self.subTest(consumed=consumed):
                torch.manual_seed(103)
                data = list(range(37))
                loader = DataLoader(data, batch_size=4, drop_last=True, sampler=RecordedRandomSampler(data))
                original = epoch_batches(loader); epoch, batches = next(original)
                for _ in range(consumed): next(batches)
                sampler, rng = loader.sampler.state_dict(), rng_state()
                expected_tail = list(batches)
                next_epoch, expected_next = next(original); expected_next = list(expected_next)
                expected_rng = tree_digest(rng_state())
                other = DataLoader(data, batch_size=4, drop_last=True, sampler=RecordedRandomSampler(data))
                resume = dict(iterator=restored_iterator(other, sampler, rng), epoch=epoch,
                              next_batch_index=consumed)
                actual = epoch_batches(other, resume)
                resumed_epoch, actual_tail = next(actual); actual_tail = list(actual_tail)
                actual_epoch, actual_next = next(actual); actual_next = list(actual_next)
                self.assertEqual((resumed_epoch, actual_epoch), (epoch, next_epoch))
                for expected, observed in ((expected_tail, actual_tail), (expected_next, actual_next)):
                    self.assertEqual([n for n, _ in expected], [n for n, _ in observed])
                    self.assertTrue(all(torch.equal(a, b) for (_, a), (_, b) in zip(expected, observed)))
                self.assertEqual(tree_digest(rng_state()), expected_rng)

    def fixture(self, directory):
        parent, run, binding, _ = documents()
        run.update(completed_updates=0, acceptance_update=5)
        run['budget'].update(formal_mathematics(run), profile='common8', contract_kind='fresh200_warmup57',
            model_id='Qwen/Qwen2.5-3B-Instruct', model_revision=run['model_revision'],
            common_manifest_sha256=run['input_sha256']['manifest.json'])
        base = Path(directory).resolve(); (base/'logs').mkdir()
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        return base, run

    def attempt(self, run, ordinal=6, count=20):
        return dict(status='complete' if count else 'recoverable_guard', logical_update=6,
            completed_outer=6 if count else 5, complete_outer=bool(count), scheduler_advances=int(bool(count)),
            attempt_id=f'{ordinal:032x}', attempt_index=ordinal, all_ranks_agreed=True, real_rank_count=4,
            optimizer_writes_before=5, optimizer_writes_after=5+count, optimizer_writes_this_attempt=count,
            opportunities_processed=20, opportunity_statuses=['written']*count+['skipped_guard']*(20-count))

    def metrics(self):
        result = {key: 0. for key in METRICS}
        result.update({'actor/optimizer_steps_this_update':20., 'actor/update_opportunities_this_attempt':20.,
                       'train/reward_logical_update':6, 'train/reward_binary_phase':0.})
        return result

    def actor_binding(self, run, result):
        return dict(run_id=run['run_id'], execution_id=run['execution_id'], completed_outer=5,
            logical_update=6, optimizer_writes=5, head_version='saved-head',
            attempt_id=result['attempt_id'], attempt_index=result['attempt_index'],
            row_order_verified=True, feature_rows=5120)

    def test_real_writers_begin_at_saved_cursor_and_never_copy_parent_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            base, run = self.fixture(root)
            skipped, complete = self.attempt(run,6,0), self.attempt(run,7,20)
            with patch.dict(os.environ, SEARCH_G1_BASE=str(base)):
                record_attempt(skipped); record_attempt(complete)
                with patch('reproduction.common8.behavior_runtime.read_gate_receipt', return_value={
                        'proof_sha256':'7'*64, 'private_trace_sha256':'8'*64}):
                    update = record(6, self.metrics(), 'saved-head', attempt_result=complete,
                                    probe_actor_binding=self.actor_binding(run, complete))
            attempts = [json.loads(line) for line in (base/'logs/actual-attempts.jsonl').read_text().splitlines()]
            self.assertEqual([row['attempt_index'] for row in attempts], [6,7])
            self.assertEqual(json.loads((base/'logs/actual-updates.jsonl').read_text()), update)
            self.assertEqual((update['step'], update['optimizer_writes_after']), (6,25))
            self.assertTrue(all(row['run_id'] == run['run_id'] for row in attempts))

    def test_copied_parent_rows_or_missing_recovered_attempt_cannot_be_appended(self):
        for failure in ('parent_prefix', 'attempt_hole', 'saved_cursor', 'old_cursor', 'wrong_logical', 'lost_writes'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                base, run = self.fixture(root); result = self.attempt(run)
                with patch.dict(os.environ, SEARCH_G1_BASE=str(base)):
                    if failure != 'parent_prefix':
                        if failure=='attempt_hole':result['attempt_index']=7
                        elif failure=='saved_cursor':result['attempt_index']=5
                        elif failure=='old_cursor':result['attempt_index']=4
                        elif failure=='wrong_logical':result['logical_update']=5
                        else:result['optimizer_writes_before']=25
                        with self.assertRaises(AssertionError): record_attempt(result)
                        self.assertFalse((base/'logs/actual-attempts.jsonl').exists())
                    else:
                        path=base/'logs/actual-updates.jsonl'
                        raw=json.dumps(dict(step=1,run_id='a'*32))+'\n'; path.write_text(raw)
                        with self.assertRaises(AssertionError):
                            record(6,self.metrics(),'saved-head',attempt_result=result,
                                   probe_actor_binding=self.actor_binding(run,result))
                        self.assertEqual(path.read_text(), raw)

    def test_origin_must_match_actual_saved_driver_counters(self):
        _, run, binding, _ = documents()
        payload=dict(completed_outer=5, attempt_progress=dict(completed_outer=5,pending=None,
            cumulative_writes=5,attempt_index=5,aborted_attempts=0))
        self.assertEqual(validate_driver_origin(run,payload), namespace_origin(run))
        worker=dict(completed_outer=5,optimizer_writes=5,actor_accounting={'last_attempt_index':5},
                    outer_accounting={'last_attempt_index':5})
        self.assertEqual(validate_worker_origin(run,worker), namespace_origin(run))
        for key in ('actor_accounting','outer_accounting'):
            wrong=copy.deepcopy(worker);wrong[key]['last_attempt_index']+=1
            with self.assertRaises(AssertionError):validate_worker_origin(run,wrong)
        for key in ('cumulative_writes', 'attempt_index', 'aborted_attempts'):
            wrong=copy.deepcopy(payload); wrong['attempt_progress'][key]+=1
            with self.assertRaises(AssertionError): validate_driver_origin(run,wrong)
        wrong=copy.deepcopy(run);wrong['actor_probe_recovery']['attempt_index']+=1
        with self.assertRaises(AssertionError): namespace_origin(wrong)

    def test_missing_rank_or_wrong_restore_binding_is_rejected(self):
        _, run, binding, receipt=documents();receipt['canonical_model_receipt_sha256']='9'*64
        context=dict(binding=binding,receipt=receipt)
        workers=[dict(rank=rank,identity=identity(run),last_attempt_index=5,policy_rejections=0,
            completed_outer=5,optimizer_writes=5,canonical_model_loaded=True,
            canonical_model_receipt_sha256='9'*64,recovery_binding_sha256=binding_digest(binding),
            original_worker_state_sha256='a'*64,rebound_worker_state_sha256='b'*64,
            local_actor_parameters_sha256='c'*64) for rank in range(4)]
        self.assertEqual(verify_worker_restores(run,context,workers),workers)
        for failure in ('missing_rank', 'wrong_model', 'wrong_writes', 'malformed_hash','other_child','wrong_cursor'):
            wrong=copy.deepcopy(workers)
            if failure=='missing_rank':wrong.pop()
            elif failure=='wrong_model':wrong[2]['canonical_model_loaded']=False
            elif failure=='wrong_writes':wrong[2]['optimizer_writes']+=20
            elif failure=='malformed_hash':wrong[2]['rebound_worker_state_sha256']='z'*64
            elif failure=='other_child':wrong[2]['identity']['run_id']='8'*32
            else:wrong[2]['last_attempt_index']+=1
            with self.subTest(failure=failure),self.assertRaises(AssertionError):
                verify_worker_restores(run,context,wrong)

    def test_driver_entry_orders_actual_restores_before_data_and_rejects_bad_origin(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as root:
                base=Path(root).resolve(); (base/'logs').mkdir(); (base/'artifacts/live').mkdir(parents=True)
                parent, run, binding, receipt=documents();receipt['canonical_model_receipt_sha256']='9'*64
                paths={}
                for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
                    # Text scalar placeholders, not fitted model or tensor files.
                    path=base/'artifacts'/name;raw=('constructed scalar placeholder '+key).encode()
                    path.write_bytes(raw);paths[key]=path;receipt['probe_head_hashes'][name]=hashlib.sha256(raw).hexdigest()
                manifest=dict(run_id=parent['run_id'],version=receipt['probe_version'],round=0,actor_step=0,
                    representation_source=MODE,calibration_data_sha256='1'*64,
                    head_sha256=receipt['probe_head_hashes'],commit='parent-scalar',boundary='parent-scalar')
                live=rebound_probe_manifest(manifest,binding,run,parent,receipt,paths)
                (base/'artifacts/live/manifest.json').write_text(json.dumps(live))
                admission=dict(role='actor_probe_full_recovery_launch_admission',accepted=True,
                    identity={key:run[key] for key in ('run_id','execution_id','source_commit',
                        'execution_backend','training_world_size','profile','slurm_job_id')},
                    origin=namespace_origin(run),recovery_binding_sha256=binding_digest(binding))
                (base/'logs/actor-probe-recovery-launch-admission.json').write_text(json.dumps(admission))
                (base/'logs/training-launch-claim.json').write_text(json.dumps({'scalar_fixture':'claim boundary'}))
                torch.manual_seed(39);data=list(range(37))
                old_loader=DataLoader(data,batch_size=4,drop_last=True,sampler=RecordedRandomSampler(data))
                it=iter(old_loader)
                for _ in range(3):next(it)
                old=SimpleNamespace(global_steps=5,train_dataloader=old_loader,
                    reward_fn=SimpleNamespace(alpha=.3,ema_em=.4,ema_trust=.6,_init=True,_calls=5))
                progress=AttemptProgress(parent);progress.completed_outer=5;progress.cumulative_writes=5;progress.attempt_index=5
                saved=driver_payload(old,parent,progress,0,2,manifest)
                if malformed:saved['attempt_progress']['cumulative_writes']+=1
                receipt['driver']={'path':'RAM-only-patched-load'}
                context=dict(binding=binding,parent=parent,receipt=receipt,probe_manifest=manifest)
                workers=[dict(rank=rank,identity=identity(run),last_attempt_index=5,policy_rejections=0,
                    completed_outer=5,optimizer_writes=5,canonical_model_loaded=True,
                    canonical_model_receipt_sha256='9'*64,recovery_binding_sha256=binding_digest(binding),
                    original_worker_state_sha256='a'*64,rebound_worker_state_sha256='b'*64,
                    local_actor_parameters_sha256='c'*64) for rank in range(4)]
                actor=SimpleNamespace(restore_probe_training_checkpoint=unittest.mock.Mock(return_value=workers))
                reference=SimpleNamespace(probe_isolation_state=unittest.mock.Mock(return_value={'scalar_fixture':'fixed'}))
                loader=DataLoader(data,batch_size=4,drop_last=True,sampler=RecordedRandomSampler(data))
                fresh=SimpleNamespace(global_steps=0,train_dataloader=loader,reward_fn=SimpleNamespace(),
                    actor_rollout_wg=actor,ref_policy_wg=reference)
                attempts=AttemptProgress(run)
                with patch.dict(os.environ,SEARCH_G1_BASE=str(base)), \
                     patch('search_g1.actor_probe_recovery.read_recovery_context',return_value=context), \
                     patch('reproduction.common8.actor_probe_launch_admission.verify_current',return_value=admission), \
                     patch('torch.load',return_value=saved):
                    if malformed:
                        with self.assertRaises(AssertionError):restore_driver_and_actor(fresh,run,attempts)
                        actor.restore_probe_training_checkpoint.assert_not_called()
                        self.assertEqual((fresh.global_steps,attempts.cumulative_writes),(0,0))
                    else:
                        resume=restore_driver_and_actor(fresh,run,attempts)
                        self.assertEqual((fresh.global_steps,attempts.cumulative_writes,resume['next_batch_index']),(5,5,3))
                        actor.restore_probe_training_checkpoint.assert_called_once_with()
                        self.assertEqual(reference.probe_isolation_state.call_count,2)
                        proof=json.loads((base/'logs/actor-probe-recovery-restored.json').read_text())
                        self.assertTrue(proof['accepted'] and proof['reference_unchanged'] and proof['no_new_optimizer_updates'])


if __name__=='__main__': unittest.main()
