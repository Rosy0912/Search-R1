"""Small actual CPU fixtures; no model, optimizer creation, HTTP or CUDA calls."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('explicit_cp_cpu_reader', HERE / 'read_checkpoint_cpu.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SCIENCE = dict(learning_rate=5e-6, component_clip=10,
               optimizer=dict(kind='AdamW', betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01))


def proof(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **m.stat_from(path.stat()))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return proof(path)


def native_mask(tokenizer, text, spans):
    # Exact fixture expression from SHA-bound frozen response_encoding.
    import re
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    information = [(x.start(), x.end()) for x in re.finditer(r'<information>.*?(?:</information>|$)', text, re.S)]
    return offsets['input_ids'], [int(b > a and not any(a < y and b > x for x, y in information)
        and (spans is None or any(x <= a and b <= y for x, y in spans))) for a, b in offsets['offset_mapping']]


def tail_fixture(actual=3):
    text = 'x<information>hidden</information>y'
    pair_row = dict(question_id='fixture-q', student=text, reference='fixture-reference',
                    reference_reward=1, student_reward=0, student_policy_spans=[[0, 1], [len(text)-1, len(text)]])
    row_sha = m.canonical_sha(pair_row)
    sampling = dict(source_model_identity='b'*64, checkpoint=None, optimizer_updates=0, adapter_tensor_sha256='ZERO_EFFECT_FRESH_LORA')
    config = dict(run_id='fixture-run', source_model_manifest=dict(sha256='b'*64), scientific=copy.deepcopy(SCIENCE),
                  logging=dict(topk_max_tokens_per_outer=8192), _outer_sampling_identity=sampling)
    update = dict(outer_index=0, optimizer_updates=7, global_actual_pairs=actual, pre_update_adapter_sha256='c'*64)
    header = dict(schema='opsd-online-training-token-header-v1', run_id='fixture-run', outer_index=0, optimizer_update=7,
                  rank=0, pair_index=0, encoded_sha256='d'*64, student_pre_update_adapter_sha256='c'*64,
                  frozen_teacher_identity='b'*64, sampling_model_identity=sampling, component_clip=10,
                  topk_cap=8192, loss_scope=m.SCOPE, full_vocab_persisted=False,
                  identity=dict(question_id=pair_row['question_id'], pair_row_sha256=row_sha,
                      reference_reward=1, student_reward=0,
                      student_response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      reference_response_sha256=hashlib.sha256(pair_row['reference'].encode()).hexdigest()),
                  token_ids=[10, 11, 12], policy_mask=[1, 0, 1],
                  token_offsets=[[0, 1], [1, len(text)-1], [len(text)-1, len(text)]], prefix_token_lengths=[5, 8])
    records = []
    for pos in (0, 2):
        records.append(dict(token_index=pos, token_id=header['token_ids'][pos], token_offsets=header['token_offsets'][pos],
            policy_mask=True, loss_scope=m.SCOPE, teacher_forced_fixed_student_prefix=True, actual_generation_scores=False,
            pre_clip_full_vocabulary_kl=2.0, post_clip_full_vocabulary_loss=1.0,
            clipped_vocabulary_count=1, vocabulary_size=4, clipped_vocabulary_fraction=0.25,
            clipped_teacher_probability_mass=0.3, trajectory_contribution=0.5, global_batch_contribution=0.5/actual))
    pair = dict(pair_index=0, valid_tokens=2, raw_loss=2.0, post_loss=1.0, clipped_vocabulary_count=2)
    return header, records, pair, 0, update, config, ['d'*64], {row_sha: pair_row}, native_mask


class FixedReads(unittest.TestCase):
    def test_fixed_handle_actual_torch_cpu_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'; torch.save(dict(x=torch.tensor([1., 2.])), path)
            reader = m.Reader(); state, p = m.fixed_handle_state(proof(path), torch, reader)
            self.assertEqual(state['x'].device.type, 'cpu'); self.assertTrue(torch.equal(state['x'], torch.tensor([1., 2.])))
            self.assertEqual(p['sha256'], proof(path)['sha256']); self.assertEqual(reader.finish(), 1)

    def test_wrong_hash_refused_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'; torch.save(dict(x=torch.ones(1)), path)
            ref = dict(proof(path), sha256='0'*64)
            with patch.object(torch, 'load', side_effect=AssertionError('must not load wrong SHA')):
                with self.assertRaises(ValueError): m.fixed_handle_state(ref, torch, m.Reader())

    def test_replaced_inode_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'; torch.save(dict(x=torch.ones(1)), path); ref = proof(path)
            replacement = Path(tmp) / 'replacement.pt'; torch.save(dict(x=torch.ones(1)), replacement); replacement.replace(path)
            with self.assertRaises(ValueError): m.fixed_handle_state(ref, torch, m.Reader())

    def test_end_metadata_mutation_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'file'; path.write_bytes(b'one'); r = m.Reader(); r.read(proof(path))
            path.write_bytes(b'changed-longer')
            with self.assertRaises(ValueError): r.finish()

    def test_path_escape_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'owned'; root.mkdir(); outside = Path(tmp) / 'outside'; outside.write_bytes(b'a')
            with self.assertRaises(ValueError): m.path_under(outside, root)
            link = root / 'alias'; link.symlink_to(outside)
            with self.assertRaises(ValueError): m.path_under(link, root)

    def test_actual_frozen305_bare_manifest_mask_and_no_weight_read(self):
        actual_core = HERE.parents[4] / 'search-e1-analysis-output/opsd-preparation3-xformers-backup-20261008/snapshot/runtime/opsd-preparation3-xformers/search_e1/core.py'
        # Portable caller can select the same accepted core source explicitly.
        override = os.environ.get('OPSD_ONLINE_TEST_RUNTIME')
        if override: actual_core = Path(override) / 'search_e1/core.py'
        self.assertTrue(actual_core.is_file(), 'Actual accepted frozen core needed for this fixture')
        raw = actual_core.read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(), '0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); model = base / 'model'; model.mkdir(); runtime = base / 'runtime'; (runtime / 'search_e1').mkdir(parents=True)
            (runtime / 'search_e1/core.py').write_bytes(raw); files = []
            for index in range(12):
                p = model / ('file-%d' % index); p.write_bytes(b'fixture'); files.append(proof(p))
            source = write_json(base / 'source.json', dict(files=files))
            runtime_ref = write_json(base / 'runtime.json', {'search_e1/core.py': hashlib.sha256(raw).hexdigest()})
            config = dict(paths=dict(model=str(model), runtime=str(runtime)), source_model_manifest=source, runtime_manifest=runtime_ref)
            reader = m.Reader(); original_read = reader.read
            def guard(ref):
                self.assertFalse(Path(ref['path']).is_relative_to(model), 'This reader must not rehash source weight bytes')
                return original_read(ref)
            with patch.object(reader, 'read', side_effect=guard):
                mask, identity = m.bind_source_and_mask(config, reader)
            data = list(tail_fixture()); data[-1] = mask
            self.assertTrue(m.validate_tail(*data)['actual_policy_mask_reconstructed_from_frozen_char_offsets'])
            self.assertEqual(len(identity['expected_source_files']), 12)

    def test_science_module_shadow_refused_and_exact_source_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = {}
            texts = dict(objective='VALUE = 17\n', checkpoint='VALUE = 18\n',
                         trainer='from objective import VALUE\ndef validate_science(config): return True\n')
            for name, text in texts.items():
                p = root / (name + '.py'); p.write_text(text); files[p.name] = dict(sha256=proof(p)['sha256'], bytes=p.stat().st_size)
            manifest = write_json(root / 'source-manifest.json', dict(files=files))
            config = dict(paths=dict(helper_root=str(root)), helper_source_manifest=manifest)
            saved = {name: sys.modules.get(name) for name in texts}
            try:
                import types
                shadow = types.ModuleType('objective'); shadow.__file__ = str(root.parent / 'wrong-objective.py')
                sys.modules['objective'] = shadow
                with self.assertRaisesRegex(ValueError, 'shadowed'): m.bind_dependencies(config, m.Reader())
                shadow.__file__ = str(root / 'objective.py'); shadow.VALUE = -1
                modules, _ = m.bind_dependencies(config, m.Reader())
                self.assertEqual(modules['trainer'].VALUE, 17)  # Cached same-path object replaced by verified bytes.
                self.assertEqual(Path(modules['trainer'].__file__).resolve(), (root / 'trainer.py').resolve())
            finally:
                for name, value in saved.items():
                    if value is None: sys.modules.pop(name, None)
                    else: sys.modules[name] = value


class OptimizerAndAdapter(unittest.TestCase):
    def optimizer(self):
        return dict(state={0: dict(step=torch.tensor(7.), exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2))},
                    param_groups=[dict(params=[0], lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=.01)])

    def test_actual_moments_and_group_config(self):
        self.assertEqual(m.inspect_adam(self.optimizer(), 7, 1, SCIENCE, torch)['finite_optimizer_tensors'], 3)

    def test_wrong_step(self):
        with self.assertRaises(ValueError): m.inspect_adam(self.optimizer(), 8, 1, SCIENCE, torch)

    def test_missing_moments_and_shape(self):
        for field in ('missing', 'shape', 'nonfinite', 'negative', 'group'):
            opt = self.optimizer()
            if field == 'missing': del opt['state'][0]['exp_avg']
            if field == 'shape': opt['state'][0]['exp_avg_sq'] = torch.ones(3)
            if field == 'nonfinite': opt['state'][0]['exp_avg'][0] = float('nan')
            if field == 'negative': opt['state'][0]['exp_avg_sq'][0] = -1.
            if field == 'group': opt['param_groups'][0]['lr'] = 1e-6
            with self.subTest(field=field), self.assertRaises(ValueError): m.inspect_adam(opt, 7, 1, SCIENCE, torch)

    def test_cross_rank_actual_moment_difference(self):
        first = self.optimizer(); second = copy.deepcopy(first)
        m.recursive_equal(first, second, torch); second['state'][0]['exp_avg'][0] = .01
        with self.assertRaises(ValueError): m.recursive_equal(first, second, torch)

    def test_adapter_change_and_shape_refusal(self):
        old = {'lora_A': torch.ones(2)}; new = {'lora_A': torch.tensor([1., 1.1])}
        self.assertEqual(m.adapter_delta(new, old, torch)['changed'], 1)
        with self.assertRaises(ValueError): m.adapter_delta(old, old, torch)
        with self.assertRaises(ValueError): m.adapter_delta({'lora_A': torch.ones(3)}, old, torch)


class TailRecords(unittest.TestCase):
    def test_actual_global_tail_one_three_thirty_two(self):
        for n in (1, 3, 32):
            with self.subTest(n=n): self.assertEqual(m.validate_tail(*tail_fixture(n))['native_tail_denominator'], n)

    def test_bad_mask_values_and_information_inclusion(self):
        for mask in ([1, .0, 1], [1, 2, 1], [1, 1, 1]):
            args = list(tail_fixture()); args[0]['policy_mask'] = mask
            with self.subTest(mask=mask), self.assertRaises(ValueError): m.validate_tail(*args)

    def test_count_positions_duplicate_wrong_denominator(self):
        for failure in ('count', 'duplicate', 'denominator', 'offset', 'nonfinite', 'clipfraction', 'summary', 'encoded'):
            args = list(tail_fixture())
            if failure == 'count': args[2]['valid_tokens'] = 3
            if failure == 'duplicate': args[1][1]['token_index'] = 0
            if failure == 'denominator': args[1][0]['global_batch_contribution'] = .5 / 4
            if failure == 'offset': args[1][0]['token_offsets'] = [0, 2]
            if failure == 'nonfinite': args[1][0]['post_clip_full_vocabulary_loss'] = float('nan')
            if failure == 'clipfraction': args[1][0]['clipped_vocabulary_fraction'] = .5
            if failure == 'summary': args[2]['raw_loss'] = 3.
            if failure == 'encoded': args[0]['encoded_sha256'] = '0'*64
            with self.subTest(failure=failure), self.assertRaises(ValueError): m.validate_tail(*args)

    def test_disabled_clip_and_native_header_scope(self):
        args = list(tail_fixture()); args[5]['scientific']['component_clip'] = None; args[0]['component_clip'] = None
        with self.assertRaises(ValueError): m.validate_tail(*args)
        args = list(tail_fixture()); args[1][0]['actual_generation_scores'] = True
        with self.assertRaises(ValueError): m.validate_tail(*args)


class NativeSummary(unittest.TestCase):
    def build(self, tmp, K=33, cursor=33):
        outer_dir = Path(tmp) / 'outer'; training = outer_dir / 'training'; training.mkdir(parents=True)
        previous = dict(cumulative_pair_exposure=10, all_rank_adapter_sha256=['a'*64]*8)
        cp = dict(cursor=dict(next_pair_cursor=cursor, usable_pair_count=K, outer_index=1), optimizer_updates=5+math.ceil(cursor/32), cumulative_pair_exposure=10+cursor, outer_update_receipts=[], all_rank_adapter_sha256=['c'*64]*8)
        outer = dict(outer_dir=str(outer_dir), cumulative_optimizer_updates_before=5)
        config = dict(run_id='fixture-run', scientific=SCIENCE)
        order = list(reversed(range(K))); offset = 0; chain = 'a'*64
        for number in range(1, math.ceil(cursor/32)+1):
            actual = min(32, cursor-offset); batch = order[offset:offset+actual]; metrics = []
            for rank in range(8):
                pairs = []
                for index in batch[rank::8]:
                    token_path = training / ('u%d-r%d-p%d.jsonl' % (number, rank, index)); token_path.write_bytes(b'{}\n')
                    pairs.append(dict(pair_index=index, token_log=proof(token_path), valid_tokens=2, raw_loss=2., post_loss=1., clipped_vocabulary_count=2))
                metrics.append(dict(rank=rank, grad_norm=1., pairs=pairs))
            after = 'c'*64 if offset+actual == cursor else 'b'*64
            value = dict(schema='opsd-online-real-update-v1', accepted=True, run_id='fixture-run', outer_index=1,
                         optimizer_updates=5+number, global_actual_pairs=actual, global_valid_tokens=actual*2,
                         cumulative_pair_exposure=10+offset+actual, loss_reduction='trajectory_mean', micro_batch=1,
                         learning_rate=5e-6, loss_scope=m.SCOPE, reference_student_prefix_shared=True, frozen_teacher=True,
                         pre_update_adapter_sha256=chain, post_update_adapter_sha256=after, all_rank_actual_token_metrics=metrics)
            cp['outer_update_receipts'].append(write_json(training / ('update%d.json'%number), value)); chain = after; offset += actual
        return cp, outer, order, m.Reader(), config, previous

    def test_actual_33_tail_and_partial32(self):
        for cursor in (32, 33):
            with tempfile.TemporaryDirectory() as tmp:
                args = self.build(tmp, cursor=cursor); summaries, last, paths = m.validate_updates(*args)
                self.assertEqual(last['global_actual_pairs'], 1 if cursor == 33 else 32)
                self.assertEqual(len(paths), cursor); self.assertEqual(len(summaries), math.ceil(cursor/32))

    def test_rank_partition_and_native_count_refusal(self):
        for failure in ('partition', 'globalcount', 'validcount', 'updates', 'exposure', 'digest', 'gradnorm'):
            with tempfile.TemporaryDirectory() as tmp:
                args = list(self.build(tmp)); cp = args[0]; ref = cp['outer_update_receipts'][-1]; value = json.loads(Path(ref['path']).read_text())
                if failure == 'partition': value['all_rank_actual_token_metrics'][0]['pairs'][0]['pair_index'] = 2
                if failure == 'globalcount': value['global_actual_pairs'] = 4
                if failure == 'validcount': value['global_valid_tokens'] = 4
                if failure == 'updates': value['optimizer_updates'] += 1
                if failure == 'exposure': value['cumulative_pair_exposure'] += 1
                if failure == 'digest': value['post_update_adapter_sha256'] = 'd'*64
                if failure == 'gradnorm': value['all_rank_actual_token_metrics'][0]['grad_norm'] = float('nan')
                cp['outer_update_receipts'][-1] = write_json(Path(ref['path']), value)
                with self.subTest(failure=failure), self.assertRaises(ValueError): m.validate_updates(*args)

    def test_k0_explicit_unsupported_and_bad_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = list(self.build(tmp, K=0, cursor=0))
            with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_K0'): m.validate_updates(*args)
        with tempfile.TemporaryDirectory() as tmp:
            args = list(self.build(tmp, K=33, cursor=31))
            with self.assertRaisesRegex(ValueError, 'Partial cursor'): m.validate_updates(*args)


if __name__ == '__main__':
    unittest.main()
