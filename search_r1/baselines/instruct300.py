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
    from .quality_resume import enabled as quality_enabled
    from .fresh200 import enabled as fresh200_enabled, validate_protocol, stop_step
    fresh200 = fresh200_enabled(config)
    validate_protocol(config)
    quality = quality_enabled(config)
    restored = restore_enabled(config)
    origin = start_step(config)
    if (config.trainer.resume.enabled or config.baselines.method != 'gigpo'
            or config.baselines.loss_reduction != 'sequence_mean'
            or config.baselines.holdout.input_mode != 'prebuilt_manifest'
            or config.baselines.holdout.manifest_sha256 != DATA_SHA
            or size != 169028 or config.data.train_batch_size != 512
            or a.ppo_mini_batch_size != (256 if quality else 128) or a.ppo_micro_batch_size != 16
            or a.ppo_epochs != 1 or a.optim.lr != 1e-6 or a.optim.lr_warmup_steps != 57
            or a.entropy_coeff != 0 or not a.use_kl_loss or a.kl_loss_coef != (0 if fresh200 else .001)
            or a.kl_loss_type != 'low_var_kl' or a.clip_ratio != .2 or a.grad_clip != 1
            or r.log_prob_micro_batch_size != 32
            or config.actor_rollout_ref.ref.log_prob_micro_batch_size != 32
            or r.n_agent != 5 or r.n != 1 or r.tensor_model_parallel_size != 1
            or r.temperature != 1 or r.top_p != 1 or r.top_k != -1
            or config.max_turns != 4 or config.trainer.total_training_steps != stop_step(config)
            or config.trainer.save_freq != (50 if fresh200 else 25) or config.trainer.acceptance_step != origin + 5
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


def _semantic_config(config):
    # The loader replaces the source directory and CPU execution settings. All
    # saved architecture, version, dtype, and token IDs remain bound.
    result = config.to_dict()
    result.pop('_name_or_path', None)
    for key in ('_attn_implementation', '_attn_implementation_internal', '_attn_implementation_autoset'):
        result.pop(key, None)
    return result


def _tokenizer_identity(tokenizer):
    import hashlib

    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)

    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return dict(tokenizer_class=type(tokenizer).__name__, vocabulary_size=len(tokenizer),
        vocabulary_sha256=hashlib.sha256(vocabulary.encode()).hexdigest(),
        special_tokens=plain(tokenizer.special_tokens_map_extended),
        special_token_ids=list(tokenizer.all_special_ids),
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, chat_template=tokenizer.chat_template)


def hf_export_identity(state_dict, config, tokenizer):
    """Bind every exported parameter/persistent buffer to the real HF loader."""
    import torch
    from .recovery import tensor_digest
    tensors = {}
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError('HF export has a nonfinite/non-tensor state: ' + name)
        tensors[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                            sha256=tensor_digest([(name, tensor)]))
    if not tensors:
        raise ValueError('Empty HF model export')
    return dict(tensors=tensors, config=_semantic_config(config), tokenizer=_tokenizer_identity(tokenizer))


def _hf_file_inventory(checkpoint):
    return [dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path))
            for path in sorted(Path(checkpoint).iterdir()) if path.is_file()]


def _hf_checkpoint_probe(checkpoint, expected):
    """Runs only in an independent CPU process, never in the live driver."""
    import hashlib
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('HF roundtrip probe requires an isolated CPU process')
    torch.set_num_threads(2)
    files_before = _hf_file_inventory(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if _semantic_config(config) != expected['config']:
        raise RuntimeError('Actual HF config loader differs from the saved export')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if _tokenizer_identity(tokenizer) != expected['tokenizer']:
        raise RuntimeError('Actual HF tokenizer loader differs from the live saved tokenizer')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, device_map='cpu',
        low_cpu_mem_usage=True, attn_implementation='eager')
    actual = hf_export_identity(model.state_dict(), model.config, tokenizer)
    if actual != expected:
        raise RuntimeError('Actual FP32 HF loader tensor/config/tokenizer identity differs from export')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise RuntimeError('HF semantic loader placed a parameter outside CPU')
    if _hf_file_inventory(checkpoint) != files_before:
        raise RuntimeError('HF checkpoint files changed during actual semantic loader roundtrip')
    serialized = json.dumps(expected, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return dict(schema=1, status='ACTUAL_CPU_HF_LOADER_ROUNDTRIP_PASS',
        checkpoint=str(Path(checkpoint).resolve()), probe_pid=os.getpid(),
        logical_step=int(Path(checkpoint).name.removeprefix('global_step_')),
        job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
        source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
        config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
        loader='AutoModelForCausalLM.from_pretrained', config_loader='AutoConfig.from_pretrained',
        tokenizer_loader='AutoTokenizer.from_pretrained', device='cpu', dtype='torch.float32',
        local_files_only=True, trust_remote_code=False, isolated_process=True,
        tensor_scope='all_exported_state_dict_keys_including_persistent_buffers',
        hf_files=files_before,
        tensors=actual['tensors'], config=actual['config'], tokenizer=actual['tokenizer'],
        export_identity_sha256=hashlib.sha256(serialized.encode()).hexdigest())


def reload_hf_checkpoint_cpu(checkpoint, expected):
    """Actual offline HF reload with bounded time and isolated RNG/CUDA state."""
    import subprocess
    import sys
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONHASHSEED='0')
    script = ('import json,sys; '
              'from search_r1.baselines.instruct300 import _hf_checkpoint_probe; '
              'request=json.load(sys.stdin); '
              'result=_hf_checkpoint_probe(request["checkpoint"],request["expected"]); '
              'print("HF_ROUNDTRIP_RECEIPT="+json.dumps(result,sort_keys=True,allow_nan=False))')
    completed = subprocess.run([sys.executable, '-c', script],
        input=json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()), expected=expected), allow_nan=False),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=300, check=False)
    if completed.returncode:
        raise RuntimeError('CPU HF loader failed: ' + completed.stderr[-2048:])
    receipts = [line.removeprefix('HF_ROUNDTRIP_RECEIPT=') for line in completed.stdout.splitlines()
                if line.startswith('HF_ROUNDTRIP_RECEIPT=')]
    if len(receipts) != 1:
        raise RuntimeError('CPU HF loader did not return exactly one actual receipt')
    identity = json.loads(receipts[0])
    if (identity.get('status') != 'ACTUAL_CPU_HF_LOADER_ROUNDTRIP_PASS'
            or identity.get('tensors') != expected['tensors']
            or identity.get('config') != expected['config']
            or identity.get('tokenizer') != expected['tokenizer']):
        raise RuntimeError('CPU HF loader receipt differs from the actual export')
    if identity.get('hf_files') != _hf_file_inventory(checkpoint):
        raise RuntimeError('Actual HF loader receipt bytes changed before receipt publication')
    folder = Path(checkpoint) / 'training_state'
    folder.mkdir(parents=True, exist_ok=True)
    receipt = atomic_json_save(identity, folder / 'hf-loader-roundtrip.json')
    return dict(**receipt, bytes=Path(receipt['path']).stat().st_size, identity=identity)


def _live_gigpo_state(trainer, step):
    """GiGPO has rollout-local groups; its only mutable saved controller is stage."""
    controller = trainer.baseline_controller
    if controller.method != 'gigpo' or controller.ranker is not None or controller.teacher is not None:
        raise RuntimeError('GiGPO roundtrip found unsupported live algorithm state')
    state = dict(method=controller.method, global_step=step, bapo=controller.stage.state_dict(),
        base_commit=controller.config.base_commit, algorithm_source_commit=controller.config.algorithm_source_commit,
        teacher_path=controller.config.distillation.teacher_path, resume_supported=True,
        gigpo_parameters=OmegaConf.to_container(controller.config.gigpo, resolve=True),
        loss_reduction=controller.config.loss_reduction,
        groups='stateless_per_rollout_with_step_ids_and_advantage_broadcast')
    if trainer.config.get('actor_token_identity') == 'sampled_action_prefix_v1':
        state.update(actor_token_identity='sampled_action_prefix_v1', reward_token_identity='legacy_canonical',
                     action_probability_context='original_sampler_prefix_each_action')
    return state


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
    probes = [record for record in files if Path(record['path']).name == 'hf-loader-roundtrip.json']
    if probes:
        saved_hf = sorted([dict(path=str(Path(record['path']).resolve()), bytes=record['bytes'], sha256=record['sha256'])
                           for record in files if Path(record['path']).parent == checkpoint], key=lambda value: value['path'])
        if (len(probes) != 1 or probes[0]['identity']['hf_files'] != saved_hf
                or json.loads(Path(probes[0]['path']).read_text()) != probes[0]['identity']):
            raise RuntimeError('HF loader receipt is not bound to the exact manifest HF bytes')
    if sum(r['path'].endswith('/driver.pt') for r in files) != 1 or sum('/algorithm/' in r['path'] for r in files) != 1:
        raise RuntimeError('Missing complete driver/algorithm receipts')

def save_complete(trainer, checkpoint, worker_receipts):
    from .instruct_resume import start_step
    step = trainer.global_steps
    algorithm = Path(trainer.config.trainer.default_local_dir) / 'algorithm' / f'global_step_{step}' / 'state.json'
    algorithm_bytes = algorithm.read_bytes()
    algorithm_state = json.loads(algorithm_bytes)
    if algorithm_state != _live_gigpo_state(trainer, step):
        raise RuntimeError('Saved GiGPO state differs from the actual live controller before complete save')
    driver = dict(schema=1, logical_step=step, actual_new_updates=trainer.actual_outer_updates,
                  driver_pid=os.getpid(),
                  source=trainer.config.actor_rollout_ref.model.path,
                  source_revision=REVISION, sampler=trainer.recovery_sampler.state_dict(),
                  rng=rng_state(), kl_controller=dict(vars(trainer.kl_ctrl)),
                  algorithm_state=algorithm_state,
                  config=OmegaConf.to_container(trainer.config, resolve=True))
    from .quality_resume import enabled as quality_enabled
    if quality_enabled(trainer.config):
        driver['quality_health'] = trainer.quality_health
        driver['validation_trace_serial'] = trainer.validation_trace_serial
    from .fresh200 import enabled as fresh200_enabled, driver_flow_state
    if fresh200_enabled(trainer.config):
        driver['fresh200_driver_flow'] = driver_flow_state(trainer)
    receipt = atomic_torch_save(driver, Path(checkpoint) / 'training_state/driver.pt')
    extra = dict(path=str(algorithm), bytes=len(algorithm_bytes), sha256=file_sha256(algorithm))
    extras = [extra]
    if fresh200_enabled(trainer.config):
        hf_records = [value['hf_loader_roundtrip'] for value in worker_receipts
                      if value is not None and 'hf_loader_roundtrip' in value]
        if len(hf_records) != 1:
            raise RuntimeError('Fresh200 requires one actual rank0 CPU HF loader receipt')
        hf = hf_records[0]
        if (Path(hf['path']).resolve() != (Path(checkpoint) / 'training_state/hf-loader-roundtrip.json').resolve()
                or hf['identity']['status'] != 'ACTUAL_CPU_HF_LOADER_ROUNDTRIP_PASS'
                or hf['identity']['logical_step'] != step
                or hf['identity']['checkpoint'] != str(Path(checkpoint).resolve())
                or hf['identity']['hf_files'] != _hf_file_inventory(checkpoint)
                or json.loads(Path(hf['path']).read_text()) != hf['identity']):
            raise RuntimeError('Actual HF loader receipt is not bound to this saved checkpoint')
        extras.append(hf)
    trainer.instruct_manifest = publish_checkpoint_manifest(
        checkpoint, worker_receipts, receipt, driver['source'], step, trainer.actual_outer_updates,
        recovery_mode=(f'full_state_instruct{start_step(trainer.config)}' if trainer.config.trainer.instruct300.get('restore_checkpoint')
                       else 'fresh_official_instruct'), extra_receipts=extras)


def roundtrip_and_wait(trainer, final=False):
    """Reload the actual saved rank states into the same workers before the gate."""
    manifest = trainer.instruct_manifest
    from .instruct_resume import start_step
    origin = start_step(trainer.config)
    from .fresh200 import stop_step
    target = stop_step(trainer.config)
    gate = target if final else origin + 5
    expected_actual = target - origin if final else 5
    if trainer.global_steps != gate or trainer.actual_outer_updates != expected_actual:
        raise RuntimeError('Live acceptance must occur at exactly five real updates')
    recheck_inputs(trainer)
    check_manifest(manifest)
    hf_roundtrip = None
    from .fresh200 import enabled as fresh200_enabled
    if fresh200_enabled(trainer.config):
        checkpoint_path = Path(manifest['path']).parent.parent
        hf_records = [record for record in manifest['identity']['files']
                      if Path(record['path']).resolve() == (checkpoint_path / 'training_state/hf-loader-roundtrip.json').resolve()]
        if len(hf_records) != 1:
            raise RuntimeError('Acceptance requires the manifest-bound actual HF loader roundtrip')
        hf_roundtrip = hf_records[0]
        identity = json.loads(Path(hf_roundtrip['path']).read_text())
        if (identity != hf_roundtrip['identity'] or identity['logical_step'] != gate
                or identity['checkpoint'] != str(checkpoint_path.resolve())
                or identity['status'] != 'ACTUAL_CPU_HF_LOADER_ROUNDTRIP_PASS'
                or identity['device'] != 'cpu' or identity['dtype'] != 'torch.float32'
                or identity['isolated_process'] is not True or not identity['tensors']
                or identity['job_id'] != os.environ.get('SLURM_JOB_ID')
                or identity['source_commit'] != os.environ.get('SEARCH_R1_BASELINE_COMMIT')
                or identity['config_sha256'] != os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256')):
            raise RuntimeError('Actual HF loader evidence belongs to another checkpoint/run')
    expected = [r['pause_identity'] for r in manifest['identity']['files'] if 'pause_identity' in r]
    shas = {str(r['pause_identity']['rank']): r['sha256'] for r in manifest['identity']['files']
            if 'pause_identity' in r}
    if trainer.actor_rollout_wg.checkpoint_pause_identity() != expected:
        raise RuntimeError('Worker state changed before live state roundtrip')
    runtime_before = trainer.actor_rollout_wg.checkpoint_runtime_metadata()
    if len(runtime_before) != 8 or {value['rank'] for value in runtime_before} != set(range(8)):
        raise RuntimeError('Incomplete live eight-rank runtime metadata before reload')
    checkpoint = str(Path(manifest['path']).parent.parent)
    from .quality_resume import optimizer_steps
    loaded = trainer.actor_rollout_wg.load_training_state(checkpoint, shas, gate, optimizer_steps(trainer.config, gate))
    if (len(loaded) != 8 or {v['rank'] for v in loaded} != set(range(8))
            or any(v['logical_step'] != gate or v.get('actual_rank_pt_reload') is not True
                   or v.get('actual_complete_runtime_readback') is not True for v in loaded)):
        raise RuntimeError('Incomplete eight-rank checkpoint roundtrip')
    if trainer.actor_rollout_wg.checkpoint_pause_identity() != expected:
        raise RuntimeError('Model/Adam/scheduler/RNG/PID changed after actual roundtrip')
    runtime_after = trainer.actor_rollout_wg.checkpoint_runtime_metadata()
    if not equal_state(runtime_before, runtime_after):
        raise RuntimeError('Complete worker RNG/scheduler/module modes/placement changed after actual reload')
    # Actually reload the live driver at this completed outer, without fetching row6.
    import torch
    from .recovery import restore_rng
    driver_receipt = next(r for r in manifest['identity']['files'] if r['path'].endswith('/driver.pt'))
    if file_sha256(driver_receipt['path']) != driver_receipt['sha256']:
        raise RuntimeError('Driver state SHA changed')
    state = torch.load(driver_receipt['path'], map_location='cpu', weights_only=False)
    if (state['schema'] != 1 or state['logical_step'] != gate or state['actual_new_updates'] != expected_actual
            or state['source_revision'] != REVISION
            or state['source'] != trainer.config.actor_rollout_ref.model.path
            or state.get('driver_pid') != os.getpid()
            or state['config'] != OmegaConf.to_container(trainer.config, resolve=True)):
        raise RuntimeError('Driver configuration/source/update identity changed')
    from .quality_resume import enabled as quality_enabled
    if quality_enabled(trainer.config):
        if state.get('quality_health') != trainer.quality_health or state.get('validation_trace_serial') != trainer.validation_trace_serial:
            raise RuntimeError('Quality monitoring or validation trace progress changed before roundtrip')
    algorithm_receipt = next(r for r in manifest['identity']['files'] if '/algorithm/' in r['path'])
    if state['algorithm_state'] != json.loads(Path(algorithm_receipt['path']).read_text()):
        raise RuntimeError('Driver GiGPO state differs from complete algorithm receipt')
    algorithm_before = _live_gigpo_state(trainer, gate)
    if algorithm_before != state['algorithm_state']:
        raise RuntimeError('Saved GiGPO state differs from the actual live algorithm before reload')
    if not equal_state(state['rng'], rng_state()):
        raise RuntimeError('Driver RNG changed before roundtrip')
    flow_before, flow_roundtrip = None, None
    if fresh200_enabled(trainer.config):
        from .fresh200 import driver_flow_state, restore_driver_flow
        flow_before = driver_flow_state(trainer)
        if flow_before != state.get('fresh200_driver_flow'):
            raise RuntimeError('Saved driver iterator/validation progress differs from the actual live flow')
        flow_roundtrip = restore_driver_flow(trainer, state['fresh200_driver_flow'])
        if driver_flow_state(trainer) != flow_before:
            raise RuntimeError('Actual driver iterator/validation progress readback differs after reload')
    before = trainer.recovery_sampler.state_dict()
    trainer.recovery_sampler.load_state_dict(state['sampler'])
    after = trainer.recovery_sampler.state_dict()
    if (not equal_state(before, after) or not equal_state(after, state['sampler'])
            or before['cursor'] != gate * 512):
        raise RuntimeError('Live data order/cursor changed in checkpoint roundtrip')
    kl_before = dict(vars(trainer.kl_ctrl))
    if not equal_state(kl_before, state['kl_controller']):
        raise RuntimeError('KL controller state changed')
    vars(trainer.kl_ctrl).update(state['kl_controller'])
    if not equal_state(dict(vars(trainer.kl_ctrl)), kl_before):
        raise RuntimeError('Actual KL controller reload differs from the live pre-reload state')
    trainer.baseline_controller.stage.load_state_dict(state['algorithm_state']['bapo'])
    algorithm_after = _live_gigpo_state(trainer, gate)
    if algorithm_after != algorithm_before or algorithm_after != state['algorithm_state']:
        raise RuntimeError('Actual live GiGPO controller stage reload differs from saved/live state')
    restore_rng(state['rng'])
    if not equal_state(state['rng'], rng_state()):
        raise RuntimeError('Driver RNG roundtrip failed')
    expected_driver_rng = rng_state()
    driver_roundtrip = dict(actual_sampler_reload=True, actual_kl_controller_reload=True,
        actual_controller_stage_reload=True, actual_rng_reload=True, live_algorithm=algorithm_after,
        grouping_scope='stateless_per_rollout; no persistent group cache or live rollout at completed outer',
        driver_pid=os.getpid(), driver_cursor=after['cursor'], actual_driver_flow_roundtrip=flow_roundtrip)
    worker_runtime_roundtrip = dict(actual_all8_saved_state_readback=True,
        complete_live_before_after_equal=True,
        scope=['python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng', 'sharding_rng',
               'full_scheduler_state', 'module_training_modes', 'parameter_buffer_local_shard_placement',
               'optimizer_tensor_placement'])
    approval_path = Path(trainer.config.trainer.instruct300.approval_path)
    if final:
        atomic_json_save(dict(manifest=manifest, actual_state_roundtrip=loaded,
            hf_loader_roundtrip=hf_roundtrip, actual_driver_state_roundtrip=driver_roundtrip,
            actual_worker_runtime_roundtrip=worker_runtime_roundtrip,
            checkpoint_pause_identity=expected, driver_cursor=after['cursor'],
            logical_step=target, actual_new_updates=expected_actual, retained_original_updates=origin),
            approval_path.with_name(f'final{target}-state-roundtrip.json'))
        return
    atomic_json_save(dict(manifest=manifest, actual_state_roundtrip=loaded,
                          hf_loader_roundtrip=hf_roundtrip, actual_driver_state_roundtrip=driver_roundtrip,
                          actual_worker_runtime_roundtrip=worker_runtime_roundtrip,
                          checkpoint_pause_identity=expected, driver_cursor=after['cursor']),
                     approval_path.with_name(f'resume{gate}-ready.json' if origin else 'instruct5-ready.json'))
    deadline = time.monotonic() + trainer.config.trainer.instruct300.wait_seconds
    expected_binding = dict(job_id=os.environ['SLURM_JOB_ID'], source=manifest['identity']['source'],
                            logical_step=gate, actual_new_updates=5, manifest_sha256=manifest['sha256'])
    if origin:
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
            if not equal_state(runtime_after, trainer.actor_rollout_wg.checkpoint_runtime_metadata()):
                raise RuntimeError('Complete worker runtime metadata changed during independent review')
            if not equal_state(expected_driver_rng, rng_state()):
                raise RuntimeError('Driver RNG changed during acceptance pause')
            if (not equal_state(trainer.recovery_sampler.state_dict(), after)
                    or not equal_state(dict(vars(trainer.kl_ctrl)), kl_before)
                    or _live_gigpo_state(trainer, gate) != algorithm_after):
                raise RuntimeError('Live driver sampler/KL/GiGPO state changed during acceptance pause')
            if flow_before is not None and driver_flow_state(trainer) != flow_before:
                raise RuntimeError('Live iterator/validation trace progress changed during acceptance pause')
            recheck_inputs(trainer)
            check_manifest(manifest)
            return
        time.sleep(1)
    raise TimeoutError('Step5 waiting for independent review; sixth fetch never authorized')


def validate_first5(trainer):
    from .fresh200 import enabled as fresh200_enabled
    if fresh200_enabled(trainer.config):
        metrics = trainer._validate()
        if metrics.get('val/qa_count/all') != 512:
            raise RuntimeError('Fresh200 first5 requires the actual common512 debug set')
        record_evaluation(trainer, metrics, 'first5_512', trainer.config.data.val_files)
        return metrics
    return validate_stage64(trainer)


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
    expected_count = {'initial512': 512, 'first5_512': 512, 'periodic512': 512, 'restored512': 512, 'quality512': 512,
                      'stage64': 64, 'final51713': 51713}[phase]
    if metrics.get('val/qa_count/all') != expected_count:
        raise RuntimeError('Evaluation question count differs from its phase')
    record = dict(phase=phase, logical_step=trainer.global_steps,
        actual_new_updates=getattr(trainer, 'actual_outer_updates', 0), pid=os.getpid(),
        job_id=os.environ['SLURM_JOB_ID'], commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'], data_identity=DATA_SHA,
        dataset_files=[dict(path=str(p), sha256=file_sha256(p)) for p in paths], metrics=metrics)
    if getattr(trainer, 'config', None) is not None and getattr(trainer.config.trainer, 'trajectory_logging', False):
        record['validation_trace_serial'] = trainer.validation_trace_serial
    with (Path(root) / 'evaluations.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
