"""One Hydra recipe for actual common8 training and actual model admission."""
import hashlib
import json
import os
import time
from pathlib import Path

from reproduction.common8.batch_contract import check_global

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / 'reproduction/common8/recipe.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(base):
    base = Path(base).resolve()
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    budget = json.loads((REPO / 'reproduction/common8/budget.json').read_text())
    assert run['profile'] == 'common8' and run['budget'] == budget
    assert run['model_revision'] == budget['model_revision']
    assert run['initial_model'] == run['reference_model']
    assert run['completed_updates'] == 0, 'Admission is for a fresh run only'
    for name in ('reproduction/common8/recipe.json', 'verl/trainer/config/ppo_trainer.yaml'):
        assert sha(REPO / name) == run['source_sha256'][name], 'Recipe changed after run binding'
    values = json.loads(RECIPE.read_text())
    overrides = []
    for value in values:
        key, raw = value.split('=', 1)
        assert key not in [x.split('=', 1)[0] for x in overrides], 'Duplicate recipe override'
        if '@BASE@' in raw or '@MODEL@' in raw:
            raw = raw.replace('@BASE@', str(base)).replace('@MODEL@', run['initial_model'])
            # Hydra receives these as individual argv strings, never shell code.
            raw = json.dumps(raw)
        overrides.append(key + '=' + raw)
    return run, overrides


def resolved(base):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    run, overrides = arguments(base)
    with initialize_config_dir(config_dir=str(REPO / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=overrides)
    OmegaConf.resolve(config)
    check_global(config.actor_rollout_ref, config.trainer.n_gpus_per_node,
                 config.actor_rollout_ref.actor.ulysses_sequence_parallel_size)
    assert config.data.train_batch_size == 512
    assert config.actor_rollout_ref.actor.grad_clip == 1.0
    for role in ('actor', 'ref'):
        part = config.actor_rollout_ref[role]
        assert part.compute_dtype == 'float32' and part.allow_tf32 is False
        assert part.fsdp_config.model_dtype == 'fp32'
    mixed = config.actor_rollout_ref.actor.fsdp_config.mixed_precision
    assert all(mixed[k] == 'fp32' for k in ('param_dtype', 'reduce_dtype', 'buffer_dtype'))
    assert config.actor_rollout_ref.rollout.dtype == 'float32'
    assert config.trainer.total_training_steps == 301
    assert config.actor_rollout_ref.model.path == run['initial_model']
    assert config.trainer.default_local_dir == run['checkpoint_dir']
    assert config.data.train_files == str(Path(base).resolve() / 'data/train.parquet')
    assert config.data.val_files == str(Path(base).resolve() / 'data/stage.parquet')
    return run, config, overrides


def launch(base):
    import sys
    from omegaconf import OmegaConf
    base = Path(base).resolve()
    run, config, overrides = resolved(base)
    claim = json.loads((base / 'logs/training-launch-claim.json').read_text())
    assert claim['run_id'] == run['run_id']
    assert claim['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    assert claim['shell_pid'] == os.getppid(), 'Only the claimed shell may launch'
    stat = Path(f'/proc/{claim["shell_pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[19]) == claim['shell_start_ticks']
    document = OmegaConf.to_container(config, resolve=True)
    canonical = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    proof = dict(run_id=run['run_id'], slurm_job_id=claim['slurm_job_id'],
                 checked_unix=time.time(), recipe_sha256=sha(RECIPE),
                 resolved_config_sha256=hashlib.sha256(canonical).hexdigest(),
                 resolved_config=document, actual_argv_overrides=overrides,
                 semantics='resolved launch configuration; not actual optimizer or model evidence')
    with (base / 'logs/resolved-training-config.json').open('x') as f:
        json.dump(proof, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.execv(sys.executable, [sys.executable, '-m', 'verl.trainer.main_ppo_eco', *overrides])


if __name__ == '__main__':
    launch(os.environ['SEARCH_G1_BASE'])
