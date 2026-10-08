"""External production backend contract; metadata reads only, not probability acceptance.

Observer and validator descend from frozen calibrationv9. Only the explicit
production identity arguments replace its calibration-native env binding.
No logits, tensors, forwards, model/Adam/RNG/data writes occur here.
"""
import os
from formal_probability_v1 import require

EXPECTED_BACKEND='FLASH_ATTN'
CALIBRATION_RUNTIME_SHA='e1fb643f082eac44d3e91c6151c1956f635ff50f4810caa57a93c97243f67143'


def require_flash_environment():
    require(os.environ.get('VLLM_ATTENTION_BACKEND')==EXPECTED_BACKEND,
            'Explicit production FLASH_ATTN environment missing/different')


def observe_backend_identity(engine, rank, source_commit, binding_sha256):
    """Read selected runner and every attention implementation; no tensor computation."""
    runner = engine.llm_engine.model_executor.worker.model_runner
    backend = runner.attn_backend
    rows = []
    for name, module in runner.model.named_modules():
        if type(module).__module__ == 'vllm.attention.layer' and type(module).__name__ == 'Attention':
            implementation = module.impl
            rows.append(dict(name=name,
                implementation_class=type(implementation).__module__ + '.' + type(implementation).__name__,
                head_size=implementation.head_size, num_heads=implementation.num_heads,
                num_kv_heads=implementation.num_kv_heads,
                sliding_window=list(implementation.sliding_window)
                    if isinstance(implementation.sliding_window, (tuple, list)) else implementation.sliding_window))
    return dict(scope='ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY', rank=rank, pid=os.getpid(),
        source_commit=source_commit,
        binding_sha256=binding_sha256,
        requested_backend=os.environ.get('VLLM_ATTENTION_BACKEND'),
        runner_backend_class=backend.__module__ + '.' + backend.__name__,
        runner_backend_name=backend.get_name(), model_dtype=str(runner.model_config.dtype),
        cache_dtype=runner.cache_config.cache_dtype, block_size=runner.cache_config.block_size,
        model_sliding_window=runner.model_config.get_sliding_window(), attention_layers=rows,
        observation_forward_calls=0, observation_tensor_value_reads=0,
        actual_vllm_runner_positions_observed=False, probability_accepted=False)

def validate_actual_flash_backend(rows, binding, binding_sha256):
    """An environment request or missing fallback log is insufficient for admission."""
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
            and len({row['pid'] for row in rows}) == 8, 'Missing actual eight-worker backend identity')
    for row in rows:
        require(row['scope'] == 'ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY'
                and row['source_commit'] == binding['code_commit']
                and row['binding_sha256'] == binding_sha256,
                'Backend evidence identity differs')
        require(row['requested_backend'] == 'FLASH_ATTN'
                and row['runner_backend_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionBackend'
                and row['runner_backend_name'] == 'flash-attn', 'Actual vLLM backend fell back or differs')
        require(row['model_dtype'] == 'torch.bfloat16' and row['cache_dtype'] == 'auto'
                and row['block_size'] == 16 and row['model_sliding_window'] is None,
                'Actual Flash dtype/cache/block/sliding configuration differs')
        layers = row['attention_layers']
        require(len(layers) == 36 and len({layer['name'] for layer in layers}) == 36,
                'Missing actual full-model attention implementation metadata')
        for layer in layers:
            require(layer['implementation_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionImpl'
                    and layer['head_size'] == 128 and layer['num_heads'] == 16
                    and layer['num_kv_heads'] == 2 and layer['sliding_window'] == [-1, -1],
                    'Actual attention implementation differs or silently falls back')
        require(row['observation_forward_calls'] == 0 and row['observation_tensor_value_reads'] == 0
                and row['actual_vllm_runner_positions_observed'] is False
                and row['probability_accepted'] is False, 'Backend metadata scope overstated')
    return dict(status='PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY',
                expected_backend='FLASH_ATTN', rank_count=8, layers_per_rank=36,
                actual_observations=rows, probability_accepted=False)


def observe_before_original_fit(trainer,binding,run,binding_sha256):
    """After init_workers, before original fit can restore25, validate or update."""
    from pathlib import Path
    from search_r1.baselines.recovery import atomic_json_save
    from formal_probability_proofs_v1 import file_record
    require_flash_environment()
    path=Path(run)/'actual-vllm-backend-identity.json'
    require(not path.exists() and not path.is_symlink(),'Existing startup backend material cannot be overwritten')
    rows=trainer.actor_rollout_wg.production_backend_identity()
    atomic_json_save(dict(status='OBSERVED_UNREVIEWED',actual_observations=rows,probability_accepted=False),path)
    checked=validate_actual_flash_backend(rows,binding,binding_sha256)
    atomic_json_save(checked,path)
    trainer.production_backend_record=file_record(path)
    return checked


def audit_backend_startup(run,binding,workers,binding_sha256):
    """Read original startup metadata and compare actual retained worker PIDs."""
    from pathlib import Path
    from formal_probability_proofs_v1 import file_record,read_json_record
    record=file_record(Path(run)/'actual-vllm-backend-identity.json')
    raw=read_json_record(record,run)
    checked=validate_actual_flash_backend(raw['actual_observations'],binding,binding_sha256)
    require(raw==checked and len(workers)==8 and set(workers)==set(range(8)),
            'Startup backend physical identity or full retained workers incomplete')
    require(all(row['pid']==workers[row['rank']]['pid'] for row in checked['actual_observations']),
            'Actual Flash backend belongs to different rollout workers')
    return dict(status='PHYSICAL_METADATA_RECONSTRUCTION_NOT_INDEPENDENT_ACCEPTANCE',
        actual_backend_record=record,expected_backend='FLASH_ATTN',same_eight_worker_pids=True,
        sampler_runner_positions_observed=False,probability_accepted=False)
