"""Synthetic CPU evidence fixtures; never production results or tensor audits."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
try:
    from . import reader
    from .common import bind_helper, checked_file, exclusive, read
except ImportError:
    import reader
    from common import bind_helper, checked_file, exclusive, read
import hashlib


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent


def ref(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def original(name, modules):
    root = CONTROL / name
    return bind_helper(root, ref(root / 'source-manifest.json'), modules)


NATIVE = original('opsd_signal_weight_v1', ('checkpoint', 'evaluation'))
COMPLETION = original('opsd_signal_weight_completion_v1', ('assemble',))['assemble']
ANALYSIS = original('opsd_signal_weight_analysis_v1', ('summary', 'replay_math'))


def data_rows():
    return [dict(question_id='synthetic-' + str(i), question='synthetic', answers=['synthetic'],
                 data_source='nq' if i < 256 else 'hotpotqa') for i in range(512)]


def predictions(data, request, config, hits):
    return [dict(validation_index=i, question_id=q['question_id'], dataset=q['data_source'],
                 row_semantic_sha256=canonical(q), primary=True, status='complete',
                 version=request['version'], completed_outer=request['completed_outer'],
                 evaluation_id=request['evaluation_id'], run_id=config['run_id'], rank=i % 8, slot=i // 8,
                 protocol_em=int(i in hits), protocol_valid_final_action=True,
                 failure_class=None if i in hits else 'valid_gold_mismatch', search_calls=1,
                 termination='answer', hook_control=dict(accepted=True, excluded_from_primary_denominator=True),
                 zero_effect_control=dict(accepted=True)) for i, q in enumerate(data)]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


class Bundle:
    """Actual file/metadata checks on small synthetic payloads, not model tensors."""
    def __init__(self, directory, variant='G_POS', number=15):
        self.root = Path(directory); self.variant = variant; self.number = number; self.index = number - 1
        self.output = self.root / 'group' / 'runs'; self.output.mkdir(parents=True)
        self.data = data_rows()
        validation = self.root / 'validation.jsonl'
        validation.write_text(''.join(json.dumps(x) + '\n' for x in self.data))
        initial = exclusive(self.root / 'initial.json', dict(adapter_tensor_sha256='1' * 64))
        cohort = exclusive(self.root / 'cohort.json', dict(synthetic=True))
        source = exclusive(self.root / 'source.json', dict(synthetic=True))
        runtime = exclusive(self.root / 'runtime.json', dict(synthetic=True))
        value = dict(run_id='a' * 32, variant=variant, study_id='synthetic', precision=dict(mode='bf16'),
                     common_initial_adapter_manifest=initial, common_cohort_manifest=cohort,
                     source_model_manifest=source, runtime_manifest=runtime, validation_identity=ref(validation),
                     owner=dict(uid=1001), scientific=dict(global_batch=32), paths=dict(output_root=str(self.output)))
        self.cf = exclusive(self.root / 'config.json', value); self.cfg = dict(value, _config_sha256=self.cf['sha256'])
        self.prior_updates = number - 1; self.updates = number
        dummy = exclusive(self.root / 'dummy-training.json', dict(synthetic=True))
        self.prior = self.checkpoint(self.index - 1, self.prior_updates, dummy, initial, dict(synthetic='prior'), self.prior_updates * 32)
        self.outer_root = self.output / 'outers' / ('outer-%05d' % self.index)
        self.eval_root = self.output / 'evaluations' / ('outer-%02d' % number)
        self.outer_root.mkdir(parents=True); self.eval_root.mkdir(parents=True)
        sampling = dict(synthetic='actual outer start', checkpoint=self.prior, optimizer_updates=self.prior_updates)
        outer = dict(schema='opsd-online-outer-manifest-v1', outer_index=self.index, run_id=self.cfg['run_id'],
                     config_sha256=self.cf['sha256'], variant=variant, contract=NATIVE['checkpoint'].science_contract(self.cfg),
                     runtime_identity=runtime['sha256'], source_model_identity=source['sha256'],
                     common_initial_adapter_manifest=initial, common_cohort_manifest=cohort,
                     previous_complete_checkpoint=self.prior, explicit_fresh=False, outer_old_adapter=self.prior,
                     cumulative_optimizer_updates_before=self.prior_updates, P=32, sampling_model_identity=sampling)
        self.outer = exclusive(self.outer_root / 'outer-manifest.json', outer)
        self.cp = self.checkpoint(self.index, self.updates, self.outer, self.prior, sampling, self.updates * 32)
        complete = dict(schema='opsd-online-outer-complete-v1', accepted=True, run_id=self.cfg['run_id'],
                        outer_index=self.index, P=32, K=32, optimizer_updates_delta=1,
                        optimizer_updates=self.updates, scheduler_advanced_updates=0,
                        checkpoint=self.cp, cumulative_pair_exposure=self.updates * 32)
        self.complete = exclusive(self.outer_root / 'training' / 'outer-complete.json', complete)
        trainer = self.phase_exit('training-running', 57000, self.outer_root / 'trainer-phase.json')
        self.trainer_exit = exclusive(self.outer_root / 'trainer-logs' / 'attempt-00000-exit.json', trainer)
        self.child, self.parent, self.summary, self.eval_request = self.evaluation()
        self.target = dict(variant=variant, completed_outer=number, outer_manifest=self.outer,
                           outer_complete=self.complete, raw_checkpoint=self.cp, trainer_exit=self.trainer_exit,
                           parent=self.parent, child=self.child, summary=self.summary, evaluation_request=self.eval_request)
        # Only traces are synthetic. Production calls original aggregate with True.
        original_aggregate = NATIVE['evaluation'].aggregate
        self.native = dict(checkpoint=NATIVE['checkpoint'], evaluation=types.SimpleNamespace(
            PROTOCOL=NATIVE['evaluation'].PROTOCOL,
            aggregate=lambda records, data, config, request, verify_traces:
                original_aggregate(records, data, config, request, verify_traces=False)))

    def checkpoint(self, index, updates, outer_ref, old_ref, sampling, exposure):
        root = self.output / 'checkpoints' / ('update-%08d-outer-%05d' % (updates, index))
        root.mkdir(parents=True)
        def payload(name, body):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body); return ref(path)
        ranks = [dict(rank=r, file=payload('rank-%d-state.pt' % r, ('synthetic-state-%d' % r).encode())) for r in range(4)]
        adapter_files = [payload('adapter/adapter_config.json', b'{"synthetic":true}\n'),
                         payload('adapter/adapter_model.safetensors', b'synthetic-payload-not-tensors')]
        value = dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=self.cfg['run_id'],
                     config_sha256=self.cf['sha256'], runtime_identity=self.cfg['runtime_manifest']['sha256'],
                     source_model_identity=self.cfg['source_model_manifest']['sha256'],
                     contract=NATIVE['checkpoint'].science_contract(self.cfg), common_initial_adapter=self.cfg['common_initial_adapter_manifest'],
                     outer_old_adapter=old_ref, all_rank_scaler_state=[dict(enabled=False)] * 4,
                     optimizer_updates=updates, cumulative_pair_exposure=exposure, complete_rank_count=4, equal_optimizer_steps=True,
                     ranks=ranks, optimizer_storage='shared_exact_four_rank_optimizer_v1',
                     shared_optimizer=payload('optimizer-shared.pt', b'synthetic-optimizer-not-tensors'),
                     all_rank_optimizer_tensor_sha256=['2' * 64] * 4,
                     adapter=dict(path=str(root / 'adapter'), files=adapter_files), all_rank_adapter_sha256=['3' * 64] * 4,
                     all_rng=['python', 'numpy', 'torch_cpu', 'cuda_all'], scheduler=dict(kind='none', warmup_updates=0),
                     cursor=dict(outer_index=index, next_pair_cursor=32, usable_pair_count=32),
                     training_input=dict(outer_manifest=outer_ref, pairs=outer_ref, order=outer_ref, encoding=outer_ref),
                     sampling_model_identity=sampling)
        return exclusive(root / 'manifest.json', value)

    def phase_exit(self, phase, pid, path):
        identity = dict(pid=pid, uid=1001, starttick=pid * 10, run_id=self.cfg['run_id'])
        evidence = [dict(identity=identity, actual_wait_returncode=0)]
        terminal = exclusive(path, dict(run_id=self.cfg['run_id'], config_sha256=self.cf['sha256'],
                                        phase=phase, child_exit_evidence=evidence))
        return dict(accepted=True, child_exit_evidence=evidence, actual_terminal=terminal,
                    gpu_release=dict(all_owned_release=dict(accepted=True)))

    def evaluation(self):
        execution = self.eval_root / 'execution'; execution.mkdir()
        cp, _ = read(self.cp)
        raw = dict(kind='raw_end_of_outer', manifest=self.cp, cursor=cp['cursor'], optimizer_updates=self.updates,
                   rank_count=4, no_best_or_cache=True)
        identity = dict(protocol=NATIVE['evaluation'].PROTOCOL, source_model_manifest=self.cfg['source_model_manifest'],
                        runtime=dict(manifest=self.cfg['runtime_manifest']), validation=self.cfg['validation_identity'],
                        common_initial=self.cfg['common_initial_adapter_manifest'], expected_adapter_tensor_sha256='3' * 64,
                        raw_checkpoint=raw)
        request = dict(evaluation_id='fixture-evaluation', input_identity=identity, completed_outer=self.number,
                       version=self.variant, config=self.cf)
        request_ref = exclusive(execution / 'request.json', request)
        hits = set(range(110)) | set(range(256, 385)); hits.remove(0); hits.update((400, 401))
        records = predictions(self.data, request, self.cfg, hits)
        result_path = execution / 'results.jsonl'; result_path.write_text(''.join(json.dumps(x) + '\n' for x in records))
        summary = NATIVE['evaluation'].aggregate(records, self.data, self.cfg, request, verify_traces=False)
        summary_ref = exclusive(execution / 'summary.json', summary)
        exits = []; proofs = []
        for rank in range(8):
            directory = execution / ('worker-%d' % rank); directory.mkdir()
            worker_results = directory / 'results.jsonl'
            rr = records[rank::8]; worker_results.write_text(''.join(json.dumps(x) + '\n' for x in rr))
            worker_result_ref = ref(worker_results)
            process = dict(pid=58000 + rank, starttick=580000 + rank, uid=1001, run_id=self.cfg['run_id'])
            exits.append(dict(identity=process, actual_wait_returncode=0))
            worker = dict(schema='opsd-signal-native512-worker-v1', accepted=True, status='COMPLETE',
                          evaluation_id=request['evaluation_id'], run_id=self.cfg['run_id'], config=self.cf,
                          request=request_ref, primary_rows=64, rank=rank, results=worker_result_ref,
                          actual_loaded_adapter=dict(tensor_sha256='3' * 64), process=process,
                          actual_loaded_forward=dict(accepted=True, adapter_name=self.variant, layers=252,
                                                     layer_checks=[{}] * 252, scaling=2., rank=16, alpha=32,
                                                     dropout=0., A_B_dtype='torch.float32'))
            proofs.extend((exclusive(directory / 'complete.json', worker), worker_result_ref))
        child = dict(schema='opsd-signal-native512-complete-v1', accepted=True, run_id=self.cfg['run_id'], config=self.cf,
                     variant=self.variant, study_id=self.cfg['study_id'], completed_outer=self.number, primary_slots=512,
                     actual_child_exit_and_release=True, no_training=True, checkpoint_metadata_unchanged=True,
                     child_exit_evidence=exits, gpu_release=dict(released=True), input_identity=identity, raw_checkpoint=raw,
                     results=ref(result_path), request=request_ref, summary=summary_ref, version=self.variant,
                     evaluation_id=request['evaluation_id'], worker_proofs=proofs)
        child_ref = exclusive(self.eval_root / 'child-complete.json', child)
        parent = dict(schema='opsd-signal-evaluation-parent-v1', accepted=True, run_id=self.cfg['run_id'],
                      config_sha256=self.cf['sha256'], completed_outer=self.number, raw_checkpoint=self.cp, child=child_ref,
                      isolated_child_no_training_rng_consumed=True, raw_state_unchanged=True, parent_torch_imported=False,
                      exit_proof=self.phase_exit('evaluation-running', 59000, self.eval_root / 'parent-phase.json'))
        parent_ref = exclusive(self.eval_root / 'parent-complete.json', parent)
        return child_ref, parent_ref, summary_ref, request_ref

    def rewrite(self, reference, operation):
        value, _ = read(reference); operation(value)
        Path(reference['path']).write_text(json.dumps(value) + '\n'); return ref(reference['path'])

    def verify(self):
        return reader.verify_target(self.target, self.cfg, self.cf, self.native, COMPLETION, self.data)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.bundle = Bundle(self.temp.name)

    def test_real_original_acceptance_interfaces(self):
        proof, summary, records = self.bundle.verify()
        self.assertEqual((proof['K'], proof['complete_cursor'], summary['overall']['hits'], len(records)), (32, 32, 240, 512))

    def test_outer25_uses_raw24(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Bundle(directory, variant='H_NEG', number=25)
            self.assertEqual(fixture.verify()[0]['endpoint']['completed_outer'], 25)

    def test_other_outer_or_Adam_counter_not_endpoint(self):
        self.bundle.target['completed_outer'] = 14
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_wrong_raw_checkpoint_rejected(self):
        self.bundle.target['raw_checkpoint'] = self.bundle.prior
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_sampling_cache_path_rejected_even_matching_bytes(self):
        copied = self.bundle.outer_root / 'merged-model-cache' / 'manifest.json'; copied.parent.mkdir()
        copied.write_bytes(Path(self.bundle.cp['path']).read_bytes())
        self.bundle.target['raw_checkpoint'] = ref(copied)
        self.bundle.target['outer_complete'] = self.bundle.rewrite(self.bundle.complete, lambda x:x.update(checkpoint=ref(copied)))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_uncommitted_tail_rejected(self):
        self.bundle.target['raw_checkpoint'] = self.bundle.rewrite(self.bundle.cp, lambda x:x['cursor'].update(next_pair_cursor=31))
        self.bundle.target['outer_complete'] = self.bundle.rewrite(self.bundle.complete, lambda x:x.update(checkpoint=self.bundle.target['raw_checkpoint']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_unknown_trainer_wait_rejected(self):
        self.bundle.target['trainer_exit'] = self.bundle.rewrite(self.bundle.trainer_exit, lambda x:x['child_exit_evidence'][0].update(actual_wait_returncode='UNKNOWN'))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_nonzero_parent_wait_rejected(self):
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x['exit_proof']['child_exit_evidence'][0].update(actual_wait_returncode=1))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_unknown_logical_worker_wait_rejected(self):
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, lambda x:x['child_exit_evidence'][5].update(actual_wait_returncode=None))
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_wrong_worker_slice_rejected(self):
        child, _ = read(self.bundle.child)
        wrong_slice = child['worker_proofs'][1]
        worker_ref = self.bundle.rewrite(child['worker_proofs'][2], lambda x:x.update(results=wrong_slice))
        def change(value):
            value['worker_proofs'][2] = worker_ref
            value['worker_proofs'][3] = wrong_slice
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, change)
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_incomplete512_rejected(self):
        child, _ = read(self.bundle.child); path = Path(child['results']['path'])
        path.write_bytes(b'\n'.join(path.read_bytes().splitlines()[:-1]) + b'\n')
        result_ref = ref(path)
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, lambda x:x.update(results=result_ref))
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_summary_pin_cannot_be_other_endpoint(self):
        self.bundle.target['summary'] = self.bundle.outer
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_actual_worker_adapter_digest_rejected(self):
        child, _ = read(self.bundle.child); wr = child['worker_proofs'][0]
        fresh = self.bundle.rewrite(wr, lambda x:x['actual_loaded_adapter'].update(tensor_sha256='4' * 64))
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, lambda x:x['worker_proofs'].__setitem__(0, fresh))
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_child_missing_release_cannot_be_success(self):
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, lambda x:x['gpu_release'].update(released=False))
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_worker_scaling_mismatch_rejected(self):
        child, _ = read(self.bundle.child)
        fresh = self.bundle.rewrite(child['worker_proofs'][0], lambda x:x['actual_loaded_forward'].update(scaling=1.))
        self.bundle.target['child'] = self.bundle.rewrite(self.bundle.child, lambda x:x['worker_proofs'].__setitem__(0, fresh))
        self.bundle.target['parent'] = self.bundle.rewrite(self.bundle.parent, lambda x:x.update(child=self.bundle.target['child']))
        with self.assertRaises(ValueError): self.bundle.verify()

    def test_zero_config_must_be_actual_G_POS(self):
        cfg = self.bundle.cfg; other = dict(self.bundle.cf, sha256='f' * 64)
        with self.assertRaises(ValueError):
            reader.verify_zero(dict(config=other), {'G_POS': cfg}, {'G_POS': self.bundle.cf}, self.bundle.native, COMPLETION, self.bundle.data)


class PairedTests(unittest.TestCase):
    def setUp(self):
        data = data_rows(); cfg = dict(run_id='a' * 32)
        base = set(range(110)) | set(range(256, 385)); end = base - {0} | {400, 401}
        self.left = predictions(data, dict(version='source100', completed_outer=0, evaluation_id='zero'), cfg, base)
        self.right = predictions(data, dict(version='G_POS', completed_outer=15, evaluation_id='end'), cfg, end)

    def test_original_paired_arithmetic_all512_and_datasets(self):
        pristine = copy.deepcopy((self.left, self.right))
        value, slots = reader.pair_counts(self.left, self.right, ANALYSIS)
        self.assertEqual((value['gained'], value['lost'], value['preserved_correct'], value['still_wrong']), (2, 1, 238, 271))
        self.assertEqual((value['left_hits'], value['right_hits'], len(slots)), (239, 240, 512))
        self.assertEqual((value['datasets']['nq']['gained'], value['datasets']['nq']['lost']), (0, 1))
        self.assertEqual((value['datasets']['hotpotqa']['gained'], value['datasets']['hotpotqa']['lost']), (2, 0))
        self.assertEqual((self.left, self.right), pristine)
        self.assertEqual(value['probability_metric_changes'], 'NA_no_probability_replay_performed')
        self.assertTrue(all(set(row) == {'validation_index', 'dataset', 'zero_em', 'endpoint_em', 'outcome'} for row in slots))

    def test_missing512_not_silently_paired(self):
        with self.assertRaises(ValueError): reader.pair_counts(self.left[:-1], self.right[:-1], ANALYSIS)

    def test_cross_question_or_dataset_mismatch_rejected(self):
        self.right[0]['question_id'] = 'wrong-fixture'
        with self.assertRaises(ValueError): reader.pair_counts(self.left, self.right, ANALYSIS)

    def test_partial_metric_or_unexpected_replay_not_fabricated(self):
        self.right[0]['replay'] = dict(metrics={})
        with self.assertRaises(ValueError): reader.pair_counts(self.left, self.right, ANALYSIS)


class PublicationTests(unittest.TestCase):
    def test_publication_must_bind_exact_explicit_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'one.py').write_text('x=1\n')
            manifest = exclusive(root / 'manifest.json', dict(files={'one.py': ref(root / 'one.py')}))
            pub = exclusive(root / 'publication.json', dict(accepted=True, branch='search-e1', source_commit='a' * 40,
                                                         remote_readback_sha='a' * 40, source_manifest_sha256=manifest['sha256']))
            spec = dict(helper_root=str(root), source_manifest=manifest, publication=pub)
            self.assertEqual(reader.published(spec)['source_manifest']['sha256'], manifest['sha256'])
            (root / 'one.py').write_text('x=2\n')
            with self.assertRaises(ValueError): reader.published(spec)

    def test_publication_unknown_or_not_matching_readback_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'one.py').write_text('x=1\n')
            manifest = exclusive(root / 'manifest.json', dict(files={'one.py': ref(root / 'one.py')}))
            pub = exclusive(root / 'publication.json', dict(accepted=True, branch='search-e1', source_commit='a' * 40,
                                                         remote_readback_sha='b' * 40, source_manifest_sha256=manifest['sha256']))
            with self.assertRaises(ValueError):
                reader.published(dict(helper_root=str(root), source_manifest=manifest, publication=pub))


if __name__ == '__main__':
    unittest.main()
