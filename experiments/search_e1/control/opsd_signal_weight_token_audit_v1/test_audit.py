"""Pure stdlib synthetic committed-forward fixtures; no tensors or raw assets."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit


def ref(path, digest='d' * 64):
    return dict(path=path, sha256=digest)


def source_config():
    root = Path(os.environ.get('OPSD_SIGNAL_TOKEN_TEST_ORIGINAL', HERE.parent / 'opsd_signal_weight_v1')).resolve()
    return dict(schema='opsd-signal-weight-config-v1', run_id='a' * 32, variant='G_POS', precision={'mode': 'bf16'},
                paths={'helper_root': str(root)}, helper_source_manifest=ref(str(root / 'source-manifest.json'), audit.ORIGINAL_MANIFEST_SHA),
                scientific={'component_clip': 10, 'learning_rate': 1e-6}, source_model_manifest=ref('/fixture/source100.json'))


def fixture(functions, *, fallback=False, missing_answer=False):
    cfg = source_config(); selected = 3; encoded = 'e' * 64; old = ref('/fixture/initial.json')
    pair = dict(question_id='synthetic-q', question='synthetic?', answers=['synthetic'],
                reference='synthetic-B', student='xyz', reference_reward=1, student_reward=0, pair_weight=1.,
                mining=dict(student_sibling=selected, reference_sibling=0,
                    sibling_row_sha256=[str(slot) * 64 for slot in range(5)], rewards_by_sibling=[1, 0, 0, 0, 0]))
    header = dict(schema='opsd-online-training-token-header-v1', run_id=cfg['run_id'], outer_index=0,
                  optimizer_update=1, rank=0, pair_index=0, retry=0, encoded_sha256=encoded,
                  contract=functions['science_contract'](cfg), frozen_teacher_identity=cfg['source_model_manifest']['sha256'],
                  component_clip=10, loss_scope='actual_training_forward_before_optimizer',
                  student_pre_update_adapter_sha256='b' * 64, outer_old_adapter=old,
                  sampling_model_identity={'source_model_identity': cfg['source_model_manifest']['sha256']},
                  mining=pair['mining'], token_ids=[4, 5, 6], policy_mask=[1, 0, 1],
                  token_offsets=[[0, 1], [1, 2], [2, 3]], token_regions=['think_text', 'other', 'final_answer'],
                  identity=dict(question_id=pair['question_id'], pair_row_sha256=audit.canonical(pair),
                      student_response_sha256=hashlib.sha256(pair['student'].encode()).hexdigest(),
                      reference_response_sha256=hashlib.sha256(pair['reference'].encode()).hexdigest(),
                      student_reward=0, reference_reward=1))
    update = dict(run_id=cfg['run_id'], outer_index=0, optimizer_updates=1, retries=0,
                  global_actual_pairs=7, pre_update_adapter_sha256='b' * 64, contract=header['contract'],
                  outer_old_adapter=old, loss_scope=header['loss_scope'])
    batch = dict(preupdate_adapter_sha256='b' * 64, cursor=0, order_sha256='f' * 64,
                 checkpoint=ref('/fixture/cp0.json'))
    rows = []
    for index in (0, 2):
        rows.append(dict(token_index=index, token_id=header['token_ids'][index], token_offsets=header['token_offsets'][index],
            region=header['token_regions'][index], policy_mask=True, valid_trajectory_tokens=2, global_actual_pairs=7,
            target_kind='teacher', direction='student_to_fixed_target_reverse', loss_scope=header['loss_scope'],
            teacher_forced_fixed_student_prefix=True, actual_generation_scores=False, outer_old_identity={'manifest': old},
            pre_clip_full_vocabulary_kl=.2, post_clip_full_vocabulary_loss=.2, student_to_teacher_raw_kl=.2,
            student_to_outer_old_raw_kl=.3, student_to_mixture_raw_kl=.2,
            student_observed_token_logp=-.7, teacher_observed_token_logp=-.5,
            student_entropy=.6, teacher_entropy=.4, trajectory_contribution=.1,
            clipped_vocabulary_count=0, vocabulary_size=32, clipped_vocabulary_fraction=0.,
            clipped_student_probability_mass=0., clipped_teacher_probability_mass=0., clipped_mixture_probability_mass=0.))
    siblings = []
    for slot in range(5):
        item = dict(sibling=slot, original_row_sha256=pair['mining']['sibling_row_sha256'][slot],
                    encoded_sha256=encoded if slot == selected else str(slot) * 64)
        if fallback:
            item.update(answer_gap=None, student_entropy=None, skipped_reason='sibling_encoding_failure')
        else:
            gap = -.2 if slot == selected else slot * .1
            item.update(answer_gap=gap, student_entropy=.6 if slot == selected else .2 + slot * .1,
                        answer_valid_tokens=1, answer_token_indices=[2], valid_policy_token_indices=[0, 2],
                        policy_student_logp=-.7, policy_teacher_logp=-.5,
                        raw_teacher_reverse_kl=.2, post_clip_teacher_loss=.2, gap_missing_reason=None)
            if missing_answer and slot == selected:
                item.update(answer_gap=None, answer_valid_tokens=0, answer_token_indices=[],
                            gap_missing_reason='no_valid_final_answer_payload_tokens')
        siblings.append(item)
    normalization = functions['weights']([item['answer_gap'] for item in siblings], 1,
                    neutral_reason='sibling_encoding_failure' if fallback else None)
    score = dict(schema='opsd-current-preupdate-five-sibling-scores-v1', accepted=True, variant='G_POS', signal='answer_gap',
                 scoring_scope='no_grad_current_preupdate_hf_replay_not_generation_scores', original_training_objective=False,
                 selected_D_only_training=True, question_id=pair['question_id'], selected_pair_sha256=audit.canonical(pair),
                 reference_response_sha256=header['identity']['reference_response_sha256'], selected_sibling=selected,
                 siblings=siblings, normalization=normalization, selected_score=siblings[selected],
                 selected_pair_weight=normalization['weights'][selected], neutral_details=[{'synthetic': 'context_limit'}] if fallback else [],
                 state_restore={'state_restored': True, 'rng_consumed': False, 'rng_before_sha256': 'c' * 64,
                                'restored_rng_sha256': 'c' * 64},
                 preupdate_identity=dict(run_id=cfg['run_id'], variant='G_POS', outer_index=0, optimizer_update=1,
                    rank=0, pair_index=0, retry=0, adapter_tensor_sha256='b' * 64, source_model_identity=cfg['source_model_manifest']['sha256'],
                    encoded_sha256=encoded, cursor=0, order_sha256=batch['order_sha256'], checkpoint=batch['checkpoint']))
    score['sha256'] = audit.canonical(score)
    weight = score['selected_pair_weight']; header['pair_weight'] = weight
    for row in rows:
        row['pair_weight'] = weight; row['global_batch_contribution'] = weight * .1 / 7
    comparison = functions['compare_selected_training_metrics'](score, rows)
    metrics = dict(pair_index=0, valid_tokens=2, pair_weight=weight, raw_loss=.2, post_loss=.2,
                   weighted_post_loss=weight * .2, student_to_teacher_raw_kl=.2,
                   student_to_outer_old_raw_kl=.3, student_to_mixture_raw_kl=.2,
                   target_kind='teacher', direction='reverse', clipped_vocabulary_count=0,
                   score_training_match=comparison,
                   signal_scalar=dict(question_id=pair['question_id'], sibling=selected, strict_reward=0, success_fraction=.2, weight=weight))
    return header, rows, score, metrics, update, batch, cfg, encoded, pair, functions


class TokenAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions, cls.source = audit.original_functions(source_config(), audit.ORIGINAL_MANIFEST_SHA)

    def test_tail_actual_seven_and_selected_weight_reconstructs_native_mean(self):
        values = fixture(self.functions); result = audit.check_pair(*values)
        self.assertEqual(result['valid_tokens'], 2); self.assertTrue(result['scorer_checked'])
        self.assertNotEqual(result['weight'], 1.); self.assertEqual(result['scorer_fallback_na'], False)
        self.assertLess(max(result['max_absolute_errors'].values()), 1e-12)
        self.assertNotIn('torch', sys.modules)

    def test_wrong_global_denominator_and_weight_sum_are_rejected(self):
        for wrong in (32, 7 * fixture(self.functions)[0]['pair_weight']):
            with self.subTest(wrong=wrong):
                values = fixture(self.functions)
                for row in values[1]: row['global_batch_contribution'] = row['pair_weight'] * .1 / wrong
                with self.assertRaisesRegex(ValueError, 'actual_pair_global_contribution'): audit.check_pair(*values)
        values = fixture(self.functions); values[1][0]['global_actual_pairs'] = 32
        with self.assertRaisesRegex(ValueError, 'global denominator'): audit.check_pair(*values)

    def test_changed_signal_and_changed_selected_weight_refused(self):
        values = fixture(self.functions); values[2]['siblings'][0]['answer_gap'] += .01
        values[2]['sha256'] = audit.canonical({key: value for key, value in values[2].items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'normalization differ'): audit.check_pair(*values)
        values = fixture(self.functions); values[0]['pair_weight'] = 1.
        with self.assertRaisesRegex(ValueError, 'weight changed'): audit.check_pair(*values)

    def test_mask_duplicate_id_and_foreign_preupdate_model_refused(self):
        values = fixture(self.functions); values[0]['policy_mask'][1] = 1
        with self.assertRaisesRegex(ValueError, 'effective mask'): audit.check_pair(*values)
        values = fixture(self.functions); values[1][1]['token_index'] = 0
        with self.assertRaisesRegex(ValueError, 'effective mask'): audit.check_pair(*values)
        values = fixture(self.functions); values[1][0]['token_id'] = 31
        with self.assertRaisesRegex(ValueError, 'token identity'): audit.check_pair(*values)
        values = fixture(self.functions); values[2]['preupdate_identity']['adapter_tensor_sha256'] = '0' * 64
        values[2]['sha256'] = audit.canonical({key: value for key, value in values[2].items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'preupdate model'): audit.check_pair(*values)

    def test_na_encoding_fallback_is_explicit_unchecked_and_all_five_neutral(self):
        values = fixture(self.functions, fallback=True); result = audit.check_pair(*values)
        self.assertFalse(result['scorer_checked']); self.assertTrue(result['scorer_fallback_na']); self.assertEqual(result['weight'], 1.)
        values[3]['score_training_match'] = dict(accepted=True, checked=True, differences={})
        with self.assertRaisesRegex(ValueError, 'comparison differs'): audit.check_pair(*values)

    def test_missing_answer_only_neutralizes_that_sibling_with_explicit_reason(self):
        values = fixture(self.functions, missing_answer=True); result = audit.check_pair(*values)
        self.assertTrue(result['gap_missing']); self.assertFalse(result['scorer_fallback_na']); self.assertEqual(result['weight'], 1.)
        values[2]['siblings'][3]['gap_missing_reason'] = None
        with self.assertRaisesRegex(ValueError, 'zero-answer-token reason'): audit.check_pair(*values)

    def test_entropy_weight_remains_dynamic_when_answer_gap_unavailable(self):
        values = fixture(self.functions, missing_answer=True)
        header, rows, score, metrics, update, batch, cfg, encoded, pair, functions = values
        cfg['variant'] = score['variant'] = score['preupdate_identity']['variant'] = 'H_POS'
        score['signal'] = 'student_entropy'
        score['normalization'] = self.functions['weights']([row['student_entropy'] for row in score['siblings']], 1)
        weight = score['selected_pair_weight'] = score['normalization']['weights'][score['selected_sibling']]
        score['sha256'] = audit.canonical({key: value for key, value in score.items() if key != 'sha256'})
        header['contract'] = update['contract'] = self.functions['science_contract'](cfg)
        header['pair_weight'] = metrics['pair_weight'] = metrics['signal_scalar']['weight'] = weight
        metrics['weighted_post_loss'] = weight * metrics['post_loss']
        for row in rows:
            row['pair_weight'] = weight; row['global_batch_contribution'] = weight * row['trajectory_contribution'] / update['global_actual_pairs']
        result = audit.check_pair(*values)
        self.assertTrue(result['gap_missing']); self.assertTrue(result['scorer_checked'])
        self.assertFalse(result['scorer_fallback_na']); self.assertNotEqual(result['weight'], 1.)

    def test_answer_indices_reject_duplicate_unsorted_and_nonpolicy_position(self):
        for invalid in ([2, 2], [2, 0], [1]):
            with self.subTest(invalid=invalid):
                values = fixture(self.functions)
                values[2]['siblings'][0]['answer_token_indices'] = invalid
                values[2]['siblings'][0]['answer_valid_tokens'] = len(invalid)
                with self.assertRaisesRegex(ValueError, 'sorted unique'): audit.check_pair(*values)

    def test_native_forward_not_generation_log_and_nonfinite_scalar_rejected(self):
        values = fixture(self.functions); values[1][0]['actual_generation_scores'] = True
        with self.assertRaisesRegex(ValueError, 'scope'): audit.check_pair(*values)
        values = fixture(self.functions); values[1][0]['student_entropy'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'): audit.check_pair(*values)

    def test_committed_native_reference_order_and_failed_attempt_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'committed.json'
            value = dict(schema='opsd-online-real-update-v1', accepted=True, run_id='a' * 32,
                outer_index=0, optimizer_updates=1, global_actual_pairs=1, global_valid_tokens=2,
                all_rank_actual_token_metrics=[dict(rank=rank, pairs=[]) for rank in range(4)])
            token = Path(directory) / 'token.jsonl'; token.write_text('{}\n')
            signal = Path(directory) / 'signal.json'; signal.write_text('{}')
            value['all_rank_actual_token_metrics'][0]['pairs'] = [dict(pair_index=0, valid_tokens=2,
                token_log=ref(str(token), hashlib.sha256(token.read_bytes()).hexdigest()),
                signal_score=ref(str(signal), hashlib.sha256(signal.read_bytes()).hexdigest()), score_training_match={'accepted': True})]
            path.write_text(json.dumps(value)); committed = ref(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
            (Path(directory) / 'failed-attempt.json').write_text('{"accepted":false}')
            manifest = dict(cursor={'next_pair_cursor': 1, 'outer_index': 0}, optimizer_updates=1, run_id='a' * 32)
            count = self.functions['verify_committed_receipts']([committed], manifest, 0, audit.checked_json, audit.checked_file, [0])
            self.assertEqual(count, 2)
            with self.assertRaisesRegex(ValueError, 'shuffled pair order'):
                self.functions['verify_committed_receipts']([committed], manifest, 0, audit.checked_json, audit.checked_file, [1])

    def test_complete_pinned_tail_checkpoint_audit_is_compact_and_committed_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value, lines=False):
                path = root / name
                path.write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))
                return ref(str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
            cfg = source_config()
            cfg['source_model_manifest'] = write('source.json', {'synthetic': 'source identity only'})
            cfg['runtime_manifest'] = write('runtime.json', {'synthetic': 'runtime identity only'})
            cfg['common_initial_adapter_manifest'] = write('initial.json', {'synthetic': 'no tensor fixture'})
            config_ref = write('config.json', cfg); actual_cfg = copy.deepcopy(cfg)
            before = 'b' * 64; after = 'c' * 64
            cp0 = write('cp0.json', dict(run_id=cfg['run_id'], config_sha256=config_ref['sha256'],
                    optimizer_updates=0, all_rank_adapter_sha256=[before] * 4))
            all_metrics = [dict(rank=rank, pairs=[]) for rank in range(4)]
            all_scores = [dict(rank=rank, scores=[]) for rank in range(4)]
            pairs = []; encodings = []; buffers = []
            for index in range(7):
                header, rows, score, metrics, update, batch, _, encoded, pair, functions = fixture(self.functions)
                encoded = hashlib.sha256(('synthetic encoding ' + str(index)).encode()).hexdigest()
                pair['question_id'] = 'synthetic-private-q-' + str(index)
                header.update(pair_index=index, rank=index % 4, encoded_sha256=encoded,
                    frozen_teacher_identity=cfg['source_model_manifest']['sha256'],
                    outer_old_adapter=cfg['common_initial_adapter_manifest'],
                    sampling_model_identity={'source_model_identity': cfg['source_model_manifest']['sha256']})
                header['identity'].update(question_id=pair['question_id'], pair_row_sha256=audit.canonical(pair))
                score.update(question_id=pair['question_id'], selected_pair_sha256=audit.canonical(pair))
                score['selected_score']['encoded_sha256'] = encoded
                score['preupdate_identity'].update(pair_index=index, rank=index % 4, encoded_sha256=encoded,
                    source_model_identity=cfg['source_model_manifest']['sha256'], checkpoint=cp0)
                for row in rows: row['outer_old_identity']['manifest'] = cfg['common_initial_adapter_manifest']
                metrics.update(pair_index=index); metrics['signal_scalar']['question_id'] = pair['question_id']
                pairs.append(pair); encodings.append(encoded); buffers.append((header, rows, score, metrics))
            pair_ref = write('pairs.jsonl', pairs, lines=True)
            encoding_ref = write('encoding.json', dict(sha256=audit.canonical(encodings), rows=encodings, K=7, P=7))
            order_ref = write('order.json', dict(pair_indices=list(range(7)), outer_index=0, encoded_sha256=audit.canonical(encodings)))
            for index, (header, rows, score, metrics) in enumerate(buffers):
                score['preupdate_identity']['order_sha256'] = order_ref['sha256']
                score['sha256'] = audit.canonical({key: value for key, value in score.items() if key != 'sha256'})
                score_ref = write('score-' + str(index) + '.json', score)
                header['signal_score'] = score_ref; metrics['signal_score'] = score_ref
                all_scores[index % 4]['scores'].append(dict(pair_index=index, score=score_ref))
                all_metrics[index % 4]['pairs'].append(metrics)
            batch_ref = write('batch.json', dict(schema='opsd-signal-score-batch-v1', accepted=True,
                run_id=cfg['run_id'], outer_index=0, optimizer_update=1, cursor=0,
                global_actual_pairs=7, order_sha256=order_ref['sha256'], preupdate_adapter_sha256=before,
                checkpoint=cp0, all_rank_scores=all_scores))
            for index, (header, rows, score, metrics) in enumerate(buffers):
                header['score_batch'] = batch_ref
                metrics['token_log'] = write('tokens-' + str(index) + '.jsonl', [header] + rows, lines=True)
            update_ref = write('update1.json', dict(schema='opsd-online-real-update-v1', accepted=True,
                run_id=cfg['run_id'], outer_index=0, optimizer_updates=1, retries=0, global_actual_pairs=7,
                global_valid_tokens=14, pre_update_adapter_sha256=before, post_update_adapter_sha256=after,
                contract=functions['science_contract'](cfg), outer_old_adapter=cfg['common_initial_adapter_manifest'],
                loss_scope='actual_training_forward_before_optimizer', learning_rate=1e-6,
                loss_reduction='trajectory_mean', micro_batch=1, reference_student_prefix_shared=True,
                frozen_teacher=True, score_batch=batch_ref, all_rank_actual_token_metrics=all_metrics))
            sampling = {'source_model_identity': cfg['source_model_manifest']['sha256']}
            outer_ref = write('outer.json', dict(run_id=cfg['run_id'], config_sha256=config_ref['sha256'],
                outer_index=0, sampling_model_identity=sampling, outer_old_adapter=cfg['common_initial_adapter_manifest'],
                cumulative_optimizer_updates_before=0))
            cp_ref = write('cp1.json', dict(schema='opsd-online-checkpoint-v1', accepted=True,
                run_id=cfg['run_id'], config_sha256=config_ref['sha256'], optimizer_updates=1,
                source_model_identity=cfg['source_model_manifest']['sha256'], runtime_identity=cfg['runtime_manifest']['sha256'],
                contract=functions['science_contract'](cfg), cumulative_pair_exposure=7,
                cursor=dict(outer_index=0, next_pair_cursor=7, usable_pair_count=7,
                            order_sha256=order_ref['sha256'], encoded_sha256=audit.canonical(encodings)),
                training_input=dict(outer_manifest=outer_ref, pairs=pair_ref, order=order_ref, encoding=encoding_ref),
                sampling_model_identity=sampling, outer_old_adapter=cfg['common_initial_adapter_manifest'],
                outer_update_receipts=[update_ref], all_rank_adapter_sha256=[after] * 4))
            write('failed-unreferenced.json', {'accepted': False})
            result = audit.audit(config_ref, cp_ref, 1, 0, audit.ORIGINAL_MANIFEST_SHA)
            self.assertTrue(result['accepted']); self.assertEqual(result['actual_pair_exposures'], 7)
            self.assertEqual(result['valid_token_rows'], 14); self.assertEqual(result['committed_updates_read'], 1)
            self.assertEqual(result['updates'][0]['scorer_checked'], 7)
            self.assertNotIn('synthetic-private-q', json.dumps(result)); self.assertNotIn('token_ids', json.dumps(result))
            self.assertNotIn('torch', sys.modules)
            token_ref = all_metrics[0]['pairs'][0]['token_log']
            saved_rows = [json.loads(line) for line in Path(token_ref['path']).read_text().splitlines()]
            saved_rows[0]['sampling_model_identity']['optimizer_updates'] = 999
            changed_token = write('tokens-0.jsonl', saved_rows, lines=True)
            update_body = json.loads(Path(update_ref['path']).read_text())
            update_body['all_rank_actual_token_metrics'][0]['pairs'][0]['token_log'] = changed_token
            changed_update = write('update1.json', update_body)
            cp_body = json.loads(Path(cp_ref['path']).read_text()); cp_body['outer_update_receipts'] = [changed_update]
            changed_cp = write('cp1.json', cp_body)
            with self.assertRaisesRegex(ValueError, 'Full recorded sampling model identity'):
                audit.audit(config_ref, changed_cp, 1, 0, audit.ORIGINAL_MANIFEST_SHA)


if __name__ == '__main__':
    unittest.main()
