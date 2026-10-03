from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import pandas as pd
from omegaconf import OmegaConf
from search_r1.baselines.bapo import answer_f1, ordinary_answer_f1, boundary_scores, resample_rows, BoundaryStage
from search_r1.baselines.holdout import split_frames, prepare_common_holdout, question_key
from search_r1.baselines.controller import Controller


class BapoFixes(unittest.TestCase):
    def test_author_reward_boundary_counterexamples(self):
        for prediction, target in [('yes indeed', 'yes'), ('no answer', 'no'), ('the', 'a')]:
            self.assertEqual(answer_f1(prediction, [target]), 0)
        self.assertAlmostEqual(ordinary_answer_f1('yes indeed', ['yes']), 2 / 3)
        self.assertEqual(answer_f1('yes', ['yes']), 1)
        stage = BoundaryStage(early=False)
        answers = ['yes indeed'] * 4 + ['I do not know']
        self.assertEqual(boundary_scores(answers, [['yes']] * 5, ['q'] * 5, stage)[-1], .5)
        self.assertEqual(resample_rows(['yes indeed'] * 5, [['yes']] * 5, ['q'] * 5, stage), list(range(5)))

    def test_question_group_split_deterministic_disjoint_and_uid_guard(self):
        train = pd.DataFrame([dict(prompt=[dict(content=f'q{i}')], extra_info={'index': i}) for i in range(100)])
        # Repeated questions stay on the same side, and cannot leak into stage validation.
        train = pd.concat([train, train.iloc[:2]], ignore_index=True)
        final = train.iloc[:5]
        a, h = split_frames(train, final)
        b, j = split_frames(train.sample(frac=1, random_state=7), final)
        keys = lambda f: set(f.prompt.map(question_key))
        self.assertEqual(keys(h), keys(j))
        self.assertEqual(len(keys(h)), 2)
        self.assertFalse(keys(a) & keys(h))
        self.assertFalse(keys(final) & keys(h))
        train.at[9, 'extra_info'] = {'index': 8}
        with self.assertRaisesRegex(ValueError, 'index'):
            split_frames(train, final)

    def test_persisted_split_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame([dict(prompt=[dict(content=f'q{i}')], extra_info={'index': i}) for i in range(100)]).to_parquet(root / 'input.parquet')
            pd.DataFrame([dict(prompt=[dict(content='final')])]).to_parquet(root / 'final.parquet')
            def config():
                return OmegaConf.create(dict(data=dict(train_files=str(root/'input.parquet'), val_files=str(root/'final.parquet'), prompt_key='prompt'),
                    baselines=dict(holdout=dict(enabled=True,fraction=.02,seed=42,output_dir=str(root/'split')),bapo=dict(stage_validation_files=None))))
            c = config(); m = prepare_common_holdout(c)
            self.assertEqual((m['train_rows'], m['holdout_rows']), (98, 2))
            d = config(); self.assertEqual(m, prepare_common_holdout(d))
            self.assertEqual(c.data.train_files, d.data.train_files)
            Path(c.data.train_files).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'modified'):
                prepare_common_holdout(config())

    def test_actor_batch_validation_before_worker_initialization(self):
        from test_models_and_driver import config
        from search_r1.baselines.controller import validate_config
        c = config('gigpo')
        c.actor_rollout_ref.actor.ppo_micro_batch_size = 1
        with self.assertRaisesRegex(ValueError, 'DP'):
            validate_config(c)
        c = config('opsd')
        c.actor_rollout_ref.actor.use_kl_loss = False
        c.actor_rollout_ref.actor.entropy_coeff = 0
        c.algorithm.kl_ctrl.kl_coef = 0
        c.baselines.distillation.teacher_path = '/teacher'
        c.baselines.distillation.privileged_answers = True
        c.actor_rollout_ref.actor.ppo_micro_batch_size = 8
        validate_config(c)
        c.actor_rollout_ref.actor.ppo_micro_batch_size = 16
        validate_config(c)
        c.actor_rollout_ref.actor.ppo_micro_batch_size = 24
        with self.assertRaisesRegex(ValueError, 'divisible by microbatch'):
            validate_config(c)

    def test_independent_phase_frequency_and_restore_on_error(self):
        c = Controller.__new__(Controller)
        c.method = 'bapo'; c.config = SimpleNamespace(bapo=SimpleNamespace(stage_validation_freq=1))
        c.stage = BoundaryStage(); c.stage_loader = 'holdout'; c.stage_dataset = 'holdout'
        t = SimpleNamespace(val_reward_fn=None, tokenizer=None, val_dataloader='final', val_dataset='final')
        c.trainer = t
        def validate():
            self.assertEqual(t.val_dataloader, 'holdout')
            self.assertIsNotNone(t.val_reward_fn)
            return {'val/test_score/nq': .5, 'val/evaluated_count/nq': 2}
        t._validate = validate
        for step in range(1, 7): c.stage_validation(step)
        self.assertFalse(c.stage.early); self.assertEqual(c.stage.stalled, 5)
        self.assertIsNone(t.val_reward_fn); self.assertEqual(t.val_dataloader, 'final')
        t._validate = lambda: (_ for _ in ()).throw(RuntimeError('test'))
        with self.assertRaises(RuntimeError): c.stage_validation(7)
        self.assertIsNone(t.val_reward_fn); self.assertEqual(t.val_dataset, 'final')

if __name__ == '__main__': unittest.main()
