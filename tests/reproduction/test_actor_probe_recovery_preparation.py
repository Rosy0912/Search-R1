"""Scalar preparation fixtures cannot admit a native recovery or real model."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.a800_common.prepare_actor_probe_recovery import candidate_manifest, _verify_official_model, _publish_candidate, _assert_clean_candidate
from reproduction.common8.formal_contract import formal_mathematics
from reproduction.common8.precision_contract import BF16_PROFILE, SAMPLING_JOURNAL_POLICY
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import identity
from tests.reproduction.test_actor_probe_namespace_audit import fixture


def saved_documents(root, step=50):
    base, parent, _, _ = fixture(root, start=0, recovery=False)
    parent.update(precision_profile=BF16_PROFILE, behavior_probability_policy=SAMPLING_JOURNAL_POLICY)
    parent['budget'].update(formal_mathematics(parent))
    receipt = dict(role='actor_probe_full_state_checkpoint', identity=identity(parent),
        completed_outer=step, next_logical_update=step+1, actual_optimizer_writes=step*20-7,
        workers=[dict(rank=rank, completed_outer=step, optimizer_writes=step*20-7) for rank in range(4)],
        probe_version=parent['run_id']+f':{(step//10)*10 if step>5 else 0}:unit',
        probe_actor_step=(step//10)*10 if step>5 else 0)
    driver = dict(role='actor_probe_full_driver_state', identity=identity(parent),
        completed_outer=step, next_logical_update=step+1, epoch=0, consumed_batch_index=step+3,
        attempt_progress=dict(world_size=4, completed_outer=step, cumulative_writes=step*20-7,
            attempt_index=step+3, aborted_attempts=3, pending=None),
        probe_version=receipt['probe_version'], probe_actor_step=receipt['probe_actor_step'],
        reward_state=dict(alpha=.6, ema_em=.4, ema_trust=.5, _init=True, _calls=step+3))
    args = dict(base='/data/shaoshitong/Search-G1-a800-fresh200-new', source_commit='6'*40,
        source_sha256={'actual.py':'7'*64}, run_id='8'*32, execution_id='9'*32,
        started_unix=100., parent_base=str(base), parent_manifest_sha256='a'*64,
        receipt_sha256='b'*64, experiment_root_run_id=parent['run_id'])
    return parent, receipt, driver, args


class RecoveryPreparationTests(unittest.TestCase):
    def test_saved_state_origin_and_recipe_derive_independent_child_without_parent_mutation(self):
        for step in (5,50,100,150):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as root:
                parent, receipt, driver, args = saved_documents(root, step)
                parent.update(pipeline_identity={'pid':111}, probe_logical_ordinal=0, probe_visible_device='6')
                original = copy.deepcopy((parent, receipt, driver))
                child = candidate_manifest(parent, receipt, driver, **args)
                self.assertEqual((parent,receipt,driver), original)
                self.assertEqual(namespace_origin(child),dict(completed_outer=step, actual_optimizer_writes=step*20-7,
                    attempt_index=step+3, aborted_attempts=3))
                self.assertNotIn('pipeline_identity', child)
                self.assertNotIn('probe_logical_ordinal', child)
                self.assertIsNone(child['probe_visible_device'])
                self.assertEqual(child['initial_probe_version'],receipt['probe_version'])
                self.assertEqual(child['run_start_semantics'],'same_experiment_full_state_resume')
                self.assertEqual(child['initial_model'],child['reference_model'])
                self.assertEqual(child['budget'],parent['budget'])
                self.assertEqual(child['input_sha256'],parent['input_sha256'])
                self.assertEqual(child['precision_profile'],BF16_PROFILE)
                self.assertEqual(child['checkpoint_dir'],args['base']+'/checkpoints/search-g1-common8')
                self.assertEqual(child['ray_tmp_dir'],'/tmp/g-'+args['execution_id'])

    def test_other_root_reused_identity_lost_write_totals_and_pending_state_rejected(self):
        for problem in ('other_root', 'run', 'execution', 'lost_writes', 'pending', 'wrong_head', 'wrong_step'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as root:
                parent, receipt, driver, args = saved_documents(root)
                if problem == 'other_root': args['experiment_root_run_id']='0'*32
                elif problem == 'run': args['run_id']=parent['run_id']
                elif problem == 'execution': args['execution_id']=parent['execution_id']
                elif problem == 'lost_writes': driver['attempt_progress']['cumulative_writes']+=20
                elif problem == 'pending': driver['attempt_progress']['pending']={'attempt_id':'c'*32}
                elif problem == 'wrong_head': driver['probe_version']='other:50:unit'
                else: receipt['completed_outer']=51
                with self.assertRaises((AssertionError,ValueError)):
                    candidate_manifest(parent,receipt,driver,**args)

    def test_recovery_of_recovery_preserves_explicit_original_root(self):
        with tempfile.TemporaryDirectory() as root:
            parent, receipt, driver, args = saved_documents(root)
            parent['actor_probe_experiment_root_run_id']='1'*32
            parent['actor_probe_recovery']=dict(role='actor_probe_same_experiment_full_state_recovery',
                state_semantics='full_model_optimizer_scheduler_rng_sampler_and_heads',
                parent_run_id='2'*32, parent_execution_id='3'*32, completed_outer=5,
                actual_optimizer_writes=93, attempt_index=8, aborted_attempts=3)
            args['experiment_root_run_id']='1'*32
            child=candidate_manifest(parent,receipt,driver,**args)
            self.assertEqual(child['actor_probe_experiment_root_run_id'],'1'*32)
            self.assertEqual(child['actor_probe_recovery']['experiment_root_run_id'],'1'*32)

    def test_json_string_model_path_reaches_real_file_hash_verifier(self):
        # Nine tiny JSON/text files exercise the real file/hash verifier; these
        # are not tensor/model weights and cannot confer native model admission.
        with tempfile.TemporaryDirectory() as root:
            model = Path(root).resolve()/'model'; model.mkdir()
            files = {
                'model.safetensors.index.json': json.dumps({'weight_map':{'a':'weight-a.json','b':'weight-b.json'}}),
                'config.json': json.dumps({'model_type':'qwen2','hidden_size':2048}),
                'weight-a.json':'{"fixture":1}', 'weight-b.json':'{"fixture":2}',
                'tokenizer.json':'{}', 'tokenizer_config.json':'{}', 'vocab.json':'{}',
                'merges.txt':'fixture', 'generation_config.json':'{}',
            }
            records=[]; hashes={}
            for name, text in files.items():
                payload=text.encode(); (model/name).write_bytes(payload)
                sha=hashlib.sha256(payload).hexdigest(); hashes[name]=sha
                records.append(dict(file=name,bytes=len(payload),sha256=sha,
                    git_blob_sha1=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()))
            official=Path(root)/'official-fixture.json'
            official.write_text(json.dumps(dict(repo='Qwen/Qwen2.5-3B-Instruct',
                revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', files=records)))
            parent=dict(initial_model=str(model),initial_model_sha256=hashes)
            with patch('reproduction.common8.prepare_run.OFFICIAL_MANIFEST_SHA',
                       hashlib.sha256(official.read_bytes()).hexdigest()):
                self.assertEqual(_verify_official_model(parent,official),hashes)
                (model/'weight-a.json').write_text('{"fixture":3}')
                with self.assertRaises(AssertionError): _verify_official_model(parent,official)

    def test_completion_published_after_manifest_and_binds_actual_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve(); (base/'logs').mkdir()
            parent,receipt,driver,args=saved_documents(root)
            child=candidate_manifest(parent,receipt,driver,**args)
            evidence=dict(role='actor_probe_full_recovery_preparation',prepared=True,
                identity=identity(child),production_admission=False)
            from reproduction.common8.calibration_runtime import write_once as actual_write
            order=[]
            def traced_write(path, document):
                if Path(path).name=='actor-probe-recovery-preparation.json':
                    self.assertTrue((base/'logs/run-manifest.json').exists())
                    self.assertEqual(document['candidate_manifest_sha256'],
                        hashlib.sha256((base/'logs/run-manifest.json').read_bytes()).hexdigest())
                order.append(Path(path).name); actual_write(path,document)
            with patch('reproduction.common8.calibration_runtime.write_once',traced_write):
                proof=_publish_candidate(base,child,evidence)
            self.assertEqual(order,['run-manifest.json','actor-probe-recovery-preparation.json'])
            self.assertEqual(json.loads((base/'logs/actor-probe-recovery-preparation.json').read_text()),proof)
            self.assertNotIn('candidate_manifest_sha256',evidence)

    def test_manifest_write_failure_cannot_publish_prepared_marker(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve(); (base/'logs').mkdir()
            parent,receipt,driver,args=saved_documents(root)
            child=candidate_manifest(parent,receipt,driver,**args)
            evidence=dict(prepared=True,identity=identity(child),production_admission=False)
            with patch('reproduction.common8.calibration_runtime.write_once',side_effect=OSError('fixture I/O failure')) as writer:
                with self.assertRaises(OSError): _publish_candidate(base,child,evidence)
            self.assertEqual(writer.call_count,1)
            self.assertFalse((base/'logs/actor-probe-recovery-preparation.json').exists())

    def test_old_driver_ledgers_and_dangling_links_rejected_before_preparation(self):
        for name in ('actual-attempts.jsonl','actual-updates.jsonl','run-manifest.json','optimizer-rank-0.jsonl'):
            for symlink in (False,True):
                with self.subTest(name=name,symlink=symlink), tempfile.TemporaryDirectory() as root:
                    base=Path(root).resolve(); (base/'logs').mkdir()
                    _assert_clean_candidate(base)
                    path=base/'logs'/name
                    if symlink: path.symlink_to(base/'missing-target')
                    else: path.write_text('{"fixture":"old namespace"}\n')
                    with self.assertRaises(AssertionError): _assert_clean_candidate(base)
                    self.assertFalse((base/'logs/actor-probe-recovery-preparation.json').exists())


if __name__=='__main__':unittest.main()
