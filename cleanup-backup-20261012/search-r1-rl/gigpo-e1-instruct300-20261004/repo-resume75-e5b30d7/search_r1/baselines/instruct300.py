"""Fresh Instruct GiGPO: exact update limit, complete snapshots and live gate."""
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from .recovery import (RecoverableSampler, atomic_json_save, atomic_torch_save,
                       file_sha256, publish_checkpoint_manifest, rng_state)

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'


def enabled(config):
    return config.trainer.get('instruct300', {}).get('enabled', False)


def prepare_sampler(config, size):
    a = config.actor_rollout_ref.actor
    r = config.actor_rollout_ref.rollout
    from .instruct_resume import enabled as restore_enabled, start_step
    restored = restore_enabled(config)
    origin = start_step(config)
    if (config.trainer.resume.enabled or config.baselines.method != 'gigpo'
            or config.baselines.loss_reduction != 'sequence_mean'
            or config.baselines.holdout.input_mode != 'prebuilt_manifest'
            or config.baselines.holdout.manifest_sha256 != DATA_SHA
            or size != 169028 or config.data.train_batch_size != 512
            or a.ppo_mini_batch_size != 128 or a.ppo_micro_batch_size != 16
            or a.ppo_epochs != 1 or a.optim.lr != 1e-6 or a.optim.lr_warmup_steps != 57
            or a.entropy_coeff != 0 or not a.use_kl_loss or a.kl_loss_coef != .001
            or a.kl_loss_type != 'low_var_kl' or a.clip_ratio != .2 or a.grad_clip != 1
            or r.log_prob_micro_batch_size != 32
            or config.actor_rollout_ref.ref.log_prob_micro_batch_size != 32
            or r.n_agent != 5 or r.n != 1 or r.tensor_model_parallel_size != 1
            or r.temperature != 1 or r.top_p != 1 or r.top_k != -1
            or config.max_turns != 4 or config.trainer.total_training_steps != 300
            or config.trainer.save_freq != 25 or config.trainer.acceptance_step != origin + 5
            or config.trainer.test_freq != 50 or config.trainer.val_before_train != (not restored)
            or config.data.val_data_num is not None or config.data.train_data_num is not None
            or config.actor_rollout_ref.ref.model_path != config.actor_rollout_ref.model.path):
        raise ValueError('Fresh Instruct300 configuration differs from approved protocol')
    lengths = [config.data.max_prompt_length, config.data.max_start_length,
               config.data.max_response_length, config.data.max_obs_length]
    if lengths != [4096, 2048, 500, 500]:
        raise ValueError('Fresh Instruct300 length contract differs')
    model = Path(config.actor_rollout_ref.model.path)
    sidecar = model / 'VERIFIED-MODEL-MANIFEST.json'
    if file_sha256(sidecar) != '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44':
        raise ValueError('Pinned official model inventory changed')
    identity = json.loads(sidecar.read_text())
    if len(identity['files']) != 11 or len({r['file'] for r in identity['files']}) != 11:
        raise ValueError('Incomplete official model inventory')
    if identity['repo'] != 'Qwen/Qwen2.5-3B-Instruct' or identity['revision'] != REVISION:
        raise ValueError('Expected pinned official Instruct actor and reference')
    for record in identity['files']:
        p = model / record['file']
        if p.resolve().parent != model.resolve() or record['revision'] != REVISION:
            raise ValueError('Official model inventory path/revision changed')
        if p.stat().st_size != record['bytes'] or file_sha256(p) != record['sha256']:
            raise ValueError('Official model artifact changed: ' + str(p))
    return RecoverableSampler(size, config.data.shuffle_train_dataloader, config.baselines.seed)


def equal_state(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left == right


def recheck_inputs(trainer):
    from .holdout import assert_prebuilt_unchanged
    assert_prebuilt_unchanged(trainer.config, trainer.instruct_data_identity)


def check_manifest(manifest):
    identity = manifest['identity']
    if file_sha256(manifest['path']) != manifest['sha256'] or json.loads(Path(manifest['path']).read_text()) != identity:
        raise RuntimeError('Checkpoint manifest identity changed')
    files = identity['files']
    if len(files) != len({r['path'] for r in files}):
        raise RuntimeError('Duplicate checkpoint receipt')
    ranks = [r['pause_identity']['rank'] for r in files if 'pause_identity' in r]
    if len(ranks) != 8 or set(ranks) != set(range(8)):
        raise RuntimeError('Incomplete eight-rank state coverage')
    for r in files:
        path = Path(r['path'])
        if path.stat().st_size != r['bytes'] or file_sha256(path) != r['sha256']:
            raise RuntimeError('Checkpoint receipt changed: ' + str(path))
    checkpoint = Path(manifest['path']).parent.parent
    actual = {str(p) for p in checkpoint.iterdir() if p.is_file()}
    hf = {r['path'] for r in files if Path(r['path']).parent == checkpoint}
    if actual != hf or not {str(checkpoint / n) for n in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json')} <= hf:
        raise RuntimeError('HF checkpoint file coverage changed')
    if sum(r['path'].endswith('/driver.pt') for r in files) != 1 or sum('/algorithm/' in r['path'] for r in files) != 1:
        raise RuntimeError('Missing complete driver/algorithm receipts')

def save_complete(trainer, checkpoint, worker_receipts):
    step = trainer.global_steps
    algorithm = Path(trainer.config.trainer.default_local_dir) / 'algorithm' / f'global_step_{step}' / 'state.json'
    algorithm_bytes = algorithm.read_bytes()
    driver = dict(schema=1, logical_step=step, actual_new_updates=trainer.actual_outer_updates,
                  source=trainer.config.actor_rollout_ref.model.path,
                  source_revision=REVISION, sampler=trainer.recovery_sampler.state_dict(),
                  rng=rng_state(), kl_controller=dict(vars(trainer.kl_ctrl)),
                  algorithm_state=json.loads(algorithm_bytes),
                  config=OmegaConf.to_container(trainer.config, resolve=True))
    receipt = atomic_torch_save(driver, Path(checkpoint) / 'training_state/driver.pt')
    extra = dict(path=str(algorithm), bytes=len(algorithm_bytes), sha256=file_sha256(algorithm))
    trainer.instruct_manifest = publish_checkpoint_manifest(
        checkpoint, worker_receipts, receipt, driver['source'], step, trainer.actual_outer_updates,
        recovery_mode=('full_state_instruct75' if trainer.config.trainer.instruct300.get('restore_checkpoint')
                       else 'fresh_official_instruct'), extra_receipts=[extra])


def roundtrip_and_wait(trainer, final=False):
    """Reload the actual saved rank states into the same workers before the gate."""
    manifest = trainer.instruct_manifest
    from .instruct_resume import start_step
    origin = start_step(trainer.config)
    gate = 300 if final else origin + 5
    expected_actual = 300 - origin if final else 5
    if trainer.global_steps != gate or trainer.actual_outer_updates != expected_actual:
        raise RuntimeError('Live acceptance must occur at exactly five real updates')
    recheck_inputs(trainer)
    check_manifest(manifest)
    expected = [r['pause_identity'] for r in manifest['identity']['files'] if 'pause_identity' in r]
    shas = {str(r['pause_identity']['rank']): r['sha256'] for r in manifest['identity']['files']
            if 'pause_identity' in r}
    if trainer.actor_rollout_wg.checkpoint_pause_identity() != expected:
        raise RuntimeError('Worker state changed before live state roundtrip')
    checkpoint = str(Path(manifest['path']).parent.parent)
    loaded = trainer.actor_rollout_wg.load_training_state(checkpoint, shas, gate, gate * 20)
    if len(loaded) != 8 or {v['rank'] for v in loaded} != set(range(8)) or any(v['logical_step'] != gate for v in loaded):
        raise RuntimeError('Incomplete eight-rank checkpoint roundtrip')
    if trainer.actor_rollout_wg.checkpoint_pause_identity() != expected:
        raise RuntimeError('Model/Adam/scheduler/RNG/PID changed after actual roundtrip')
    # Driver is preserved in place; validate its saved ordering/cursor without fetching row6.
    import torch
    from .recovery import restore_rng
    driver_receipt = next(r for r in manifest['identity']['files'] if r['path'].endswith('/driver.pt'))
    if file_sha256(driver_receipt['path']) != driver_receipt['sha256']:
        raise RuntimeError('Driver state SHA changed')
    state = torch.load(driver_receipt['path'], map_location='cpu', weights_only=False)
    if (state['schema'] != 1 or state['logical_step'] != gate or state['actual_new_updates'] != expected_actual
            or state['source_revision'] != REVISION
            or state['source'] != trainer.config.actor_rollout_ref.model.path
            or state['config'] != OmegaConf.to_container(trainer.config, resolve=True)):
        raise RuntimeError('Driver configuration/source/update identity changed')
    algorithm_receipt = next(r for r in manifest['identity']['files'] if '/algorithm/' in r['path'])
    if state['algorithm_state'] != json.loads(Path(algorithm_receipt['path']).read_text()):
        raise RuntimeError('Driver GiGPO state differs from complete algorithm receipt')
    if not equal_state(state['rng'], rng_state()):
        raise RuntimeError('Driver RNG changed before roundtrip')
    before = trainer.recovery_sampler.state_dict()
    trainer.recovery_sampler.load_state_dict(state['sampler'])
    after = trainer.recovery_sampler.state_dict()
    if (before['order'] != after['order'] or before['cursor'] != after['cursor']
            or before['cursor'] != gate * 512 or not torch.equal(before['generator'], after['generator'])):
        raise RuntimeError('Live data order/cursor changed in checkpoint roundtrip')
    if dict(vars(trainer.kl_ctrl)) != state['kl_controller']:
        raise RuntimeError('KL controller state changed')
    restore_rng(state['rng'])
    if not equal_state(state['rng'], rng_state()):
        raise RuntimeError('Driver RNG roundtrip failed')
    expected_driver_rng = rng_state()
    approval_path = Path(trainer.config.trainer.instruct300.approval_path)
    if final:
        atomic_json_save(dict(manifest=manifest, actual_state_roundtrip=loaded,
            checkpoint_pause_identity=expected, driver_cursor=after['cursor'],
            logical_step=300, actual_new_updates=expected_actual, retained_original_updates=origin),
            approval_path.with_name('final300-state-roundtrip.json'))
        return
    atomic_json_save(dict(manifest=manifest, actual_state_roundtrip=loaded,
                          checkpoint_pause_identity=expected, driver_cursor=after['cursor']),
                     approval_path.with_name('resume80-ready.json' if gate == 80 else 'instruct5-ready.json'))
    deadline = time.monotonic() + trainer.config.trainer.instruct300.wait_seconds
    expected_binding = dict(job_id=os.environ['SLURM_JOB_ID'], source=manifest['identity']['source'],
                            logical_step=gate, actual_new_updates=5, manifest_sha256=manifest['sha256'])
    if gate == 80:
        expected_binding['restore_manifest_sha256'] = trainer.instruct_restore_manifest['sha256']
    while time.monotonic() < deadline:
        if approval_path.exists():
            approval = json.loads(approval_path.read_text())
            if any(approval.get(k) != v for k, v in expected_binding.items()):
                raise RuntimeError('Independent approval belongs to another live experiment')
            if file_sha256(approval['review_path']) != approval['review_sha256']:
                raise RuntimeError('Independent acceptance bytes changed')
            review = json.loads(Path(approval['review_path']).read_text())
            if review.get('status') != 'PASS' or any(review.get(k) != v for k, v in expected_binding.items()):
                raise RuntimeError('Independent reviewer did not accept these five updates')
            if trainer.actor_rollout_wg.checkpoint_pause_identity() != expected:
                raise RuntimeError('Live worker continuity changed during independent review')
            if not equal_state(expected_driver_rng, rng_state()):
                raise RuntimeError('Driver RNG changed during acceptance pause')
            recheck_inputs(trainer)
            check_manifest(manifest)
            return
        time.sleep(1)
    raise TimeoutError('Step5 waiting for independent review; sixth fetch never authorized')


def validate_stage64(trainer):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    previous_dataset, previous_loader = trainer.val_dataset, trainer.val_dataloader
    try:
        trainer.val_dataset = RLHFDataset(
            trainer.config.baselines.bapo.stage_validation_files, trainer.tokenizer,
            prompt_key=trainer.config.data.prompt_key,
            max_prompt_length=trainer.config.data.max_prompt_length,
            return_raw_chat=trainer.config.data.get('return_raw_chat', False), truncation='error')
        if len(trainer.val_dataset) != 64:
            raise RuntimeError('Expected exact common64 stage dataset')
        trainer.val_dataloader = DataLoader(trainer.val_dataset, batch_size=32, shuffle=False,
                                           drop_last=False, collate_fn=collate_fn)
        metrics = trainer._validate()
        if metrics.get('val/qa_count/all') != 64:
            raise RuntimeError('Actual step5 stage evaluation count differs')
        record_evaluation(trainer, metrics, 'stage64', trainer.config.baselines.bapo.stage_validation_files)
        return metrics
    finally:
        trainer.val_dataset, trainer.val_dataloader = previous_dataset, previous_loader


def record_evaluation(trainer, metrics, phase, files):
    """Persist exact phase/data/job identities without depending on Ray forwarding."""
    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not root:
        raise RuntimeError('Instruct evaluation requires the actual audit directory')
    paths = [files] if isinstance(files, str) else list(files)
    expected_count = {'initial512': 512, 'periodic512': 512,
                      'stage64': 64, 'final51713': 51713}[phase]
    if metrics.get('val/qa_count/all') != expected_count:
        raise RuntimeError('Evaluation question count differs from its phase')
    record = dict(phase=phase, logical_step=trainer.global_steps,
        actual_new_updates=getattr(trainer, 'actual_outer_updates', 0), pid=os.getpid(),
        job_id=os.environ['SLURM_JOB_ID'], commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'], data_identity=DATA_SHA,
        dataset_files=[dict(path=str(p), sha256=file_sha256(p)) for p in paths], metrics=metrics)
    with (Path(root) / 'evaluations.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
