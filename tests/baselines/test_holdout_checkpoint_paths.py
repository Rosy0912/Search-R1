"""Actual trainer-save layout must pass holdout entry without weakening gates.

The save dispatch and GPU/evaluation services are mocked, but the trainer path,
holdout orchestration, candidate checks and on-disk manifest checks are real.
The fixture files exercise the manifest contract, not production HF inference.
"""
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from search_r1.baselines.main import main
from search_r1.baselines.select_teacher import CANDIDATES, model_manifest
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

ROOT = Path(__file__).resolve().parents[2]
SPLIT = {'holdout_rows': 10, 'content_identity': {'sha256': 'test-split'}}
RESULT = {'val/qa_count/all': 10, 'val/f1/all': .5}


def saved_candidate(tmp_path, step):
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='ppo_baselines', overrides=['baselines.method=grpo', '+trainer.val_only=true'])
    OmegaConf.resolve(cfg)
    cfg.trainer.default_local_dir = str(tmp_path / 'run')
    cfg.trainer.default_hdfs_dir = None
    cfg.trainer.val_only = True
    cfg.trainer.val_before_train = True
    cfg.baselines.holdout_eval_only = True
    cfg.baselines.holdout_record_path = str(tmp_path / 'holdout-record.json')
    captured = []

    def save_dispatch(local_path, remote_path):
        captured.append(Path(local_path))
        assert remote_path is None
        directory = Path(local_path)
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
        (directory / 'tokenizer_config.json').write_text('{}')
        (directory / 'model.safetensors').write_bytes(b'manifest-fixture-weights')

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = cfg
    trainer.global_steps = step
    trainer.use_critic = False
    trainer.baseline_controller = None
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=save_dispatch)
    trainer._save_checkpoint()
    directory, = captured
    assert directory == tmp_path / 'run' / 'actor' / f'global_step_{step}'
    cfg.actor_rollout_ref.model.path = str(directory)
    return cfg, directory


def evaluate(cfg, scoring=None):
    remote = Mock(return_value=dict(RESULT)) if scoring is None else Mock(side_effect=scoring)
    task = SimpleNamespace(remote=remote)
    with ExitStack() as stack:
        stack.enter_context(patch('search_r1.baselines.preflight.validate_gpu_lease'))
        stack.enter_context(patch('search_r1.baselines.holdout.prepare_common_holdout', return_value=SPLIT))
        stack.enter_context(patch('search_r1.baselines.preflight.validate_assets_and_retriever'))
        stack.enter_context(patch('verl.trainer.main_ppo.main_task', task))
        stack.enter_context(patch('ray.is_initialized', return_value=True))
        stack.enter_context(patch('ray.get', side_effect=lambda result: result))
        # validate_config, path checks and model_manifest are intentionally real.
        main.__wrapped__(cfg)
    return remote


@pytest.mark.parametrize('step', CANDIDATES)
def test_real_save_directory_passes_holdout_entry_and_records_hash(tmp_path, step):
    cfg, directory = saved_candidate(tmp_path, step)
    expected = model_manifest(directory)['sha256']
    remote = evaluate(cfg)
    remote.assert_called_once()
    assert cfg.data.final_val_files is None
    record = json.loads(Path(cfg.baselines.holdout_record_path).read_text())
    assert record['outer_update'] == step
    assert record['hf_directory'] == str(directory.resolve())
    assert record['checkpoint_sha256'] == expected
    assert record['evaluated_count'] == SPLIT['holdout_rows']
    assert record['split_content_identity'] == 'test-split'


def test_real_save_unlisted_step_is_still_rejected(tmp_path):
    cfg, _ = saved_candidate(tmp_path, 6)
    with pytest.raises(ValueError, match='specified saved actor update'):
        evaluate(cfg)
    assert not Path(cfg.baselines.holdout_record_path).exists()


def test_real_save_missing_weight_manifest_is_still_rejected(tmp_path):
    cfg, directory = saved_candidate(tmp_path, 5)
    (directory / 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='Missing HF weights'):
        evaluate(cfg)
    assert not Path(cfg.baselines.holdout_record_path).exists()


def test_real_save_mutation_during_evaluation_is_still_rejected(tmp_path):
    cfg, directory = saved_candidate(tmp_path, 5)

    def score(_cfg):
        (directory / 'model.safetensors').write_bytes(b'changed-during-evaluation')
        return dict(RESULT)

    with pytest.raises(ValueError, match='Checkpoint changed during evaluation'):
        evaluate(cfg, score)
    assert not Path(cfg.baselines.holdout_record_path).exists()
