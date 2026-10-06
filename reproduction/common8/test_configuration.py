"""Compose the real Hydra config; reject stale bindings and wrong fresh reference."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import configuration as c


class ConfigurationTests(unittest.TestCase):
    def fixture(self, root):
        # macOS temp roots may spell the same directory /var or /private/var;
        # match production's canonical namespace before constructing bindings.
        base = (Path(root) / 'fixture with spaces').resolve()
        (base / 'logs').mkdir(parents=True)
        model = str(base / 'official model')
        run = dict(profile='common8', run_id='cpu-fixture-only', completed_updates=0, initial_model=model,
                   contract_kind='fresh200_warmup57',target_updates=200,acceptance_update=5,
                   reference_model=model, model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
                   budget=json.loads((c.REPO / 'reproduction/common8/budget.json').read_text()),
                   train_log=str(base / 'logs/training.log'),
                   checkpoint_dir=str(base / 'checkpoints/search-g1-common8'),
                   source_sha256={n: c.sha(c.REPO / n) for n in
                       ('reproduction/common8/recipe.json', 'reproduction/common8/budget.json',
                        'verl/trainer/config/ppo_trainer.yaml')})
        (base / 'logs/run-manifest.json').write_text(json.dumps(run))
        (base / 'logs/training.log').write_text('constructed CPU fixture console; no training run\n')
        return base, run

    def test_real_hydra_composes_budget_paths_and_interpolations(self):
        with tempfile.TemporaryDirectory() as root:
            base, run = self.fixture(root)
            _, config, args = c.resolved(base)
            self.assertEqual(len(args), len(json.loads(c.RECIPE.read_text())))
            self.assertEqual(config.actor_rollout_ref.actor.compute_dtype, "float32")
            self.assertFalse(config.actor_rollout_ref.ref.allow_tf32)
            self.assertEqual(config.actor_rollout_ref.rollout.dtype, "float32")
            self.assertEqual(config.data.train_batch_size * config.actor_rollout_ref.rollout.n_agent, 2560)
            self.assertEqual(2560 // config.actor_rollout_ref.actor.ppo_mini_batch_size, 20)
            self.assertEqual(config.actor_rollout_ref.rollout.prompt_length, 4096)
            self.assertEqual(config.actor_rollout_ref.rollout.response_length, 500)
            self.assertEqual(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz, False)
            self.assertEqual(config.actor_rollout_ref.ref.ulysses_sequence_parallel_size, 1)
            self.assertEqual(config.actor_rollout_ref.model.path, run['initial_model'])
            self.assertEqual(config.data.val_batch_size, 64)
            self.assertEqual(config.trainer.acceptance_step, 5)
            self.assertEqual(config.trainer.logger, ['console'])
            self.assertEqual(config.trainer.total_training_steps,201)
            self.assertEqual(config.actor_rollout_ref.actor.optim.lr_warmup_steps,57)
            self.assertEqual(config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio,0.)
            self.assertEqual(config.actor_rollout_ref.actor.entropy_coeff,0.)
            self.assertTrue(config.actor_rollout_ref.actor.use_kl_loss)
            self.assertEqual(config.actor_rollout_ref.actor.kl_loss_coef,0.)
            self.assertEqual(config.actor_rollout_ref.actor.kl_loss_type,'low_var_kl')

    def test_stale_recipe_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base, run = self.fixture(root)
            run['source_sha256']['reproduction/common8/recipe.json'] = 'wrong'
            (base / 'logs/run-manifest.json').write_text(json.dumps(run))
            with self.assertRaises(AssertionError): c.arguments(base)

    def test_different_reference_and_nonfresh_run_rejected(self):
        for key, value in (('reference_model', 'old checkpoint'), ('completed_updates', 1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                base, run = self.fixture(root); run[key] = value
                (base / 'logs/run-manifest.json').write_text(json.dumps(run))
                with self.assertRaises(AssertionError): c.arguments(base)

    @unittest.skipUnless(Path('/proc').exists(), 'Actual parent birth check requires Linux')
    def test_claimed_launcher_records_exact_exec_arguments_without_starting_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            base, run = self.fixture(root)
            parent = os.getppid()
            stat = Path(f'/proc/{parent}/stat').read_text().rsplit(')', 1)[1].split()
            claim = dict(run_id=run['run_id'], slurm_job_id='cpu-fixture-only',
                         shell_pid=parent, shell_start_ticks=int(stat[19]))
            (base / 'logs/training-launch-claim.json').write_text(json.dumps(claim))
            _, args = c.arguments(base)
            with patch.dict(os.environ, SLURM_JOB_ID='cpu-fixture-only'), patch.object(c.os, 'execv') as execute:
                c.launch(base)
                execute.assert_called_once_with(sys.executable,
                    [sys.executable, '-m', 'verl.trainer.main_ppo_eco', *args])
                proof = json.loads((base / 'logs/resolved-training-config.json').read_text())
                self.assertEqual(proof['actual_argv_overrides'], args)
                self.assertEqual(proof['resolved_config']['actor_rollout_ref']['actor']['optim']['lr'], 1e-6)
                with self.assertRaises(FileExistsError): c.launch(base)

    def test_wrong_claim_cannot_start_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            base, _ = self.fixture(root)
            (base / 'logs/training-launch-claim.json').write_text(json.dumps(dict(run_id='other-run')))
            with patch.object(c.os, 'execv') as execute:
                with self.assertRaises(AssertionError): c.launch(base)
                execute.assert_not_called()


if __name__ == '__main__': unittest.main()
