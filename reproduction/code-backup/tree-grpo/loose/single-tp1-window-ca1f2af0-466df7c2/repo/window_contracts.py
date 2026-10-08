"""Metadata only contracts for one original TP1 generation window; no release."""
import hashlib
import json
from pathlib import Path

from search_r1.baselines.probability_probe_contracts import (
    ASSET_ROOT, require, require_sha, validate_fixed_config, _finite_json,
)
from search_r1.baselines.probability_probe_io import bound_descriptor

ORIGINAL_SELECTION_SHA = 'bca59a4508a0627a04b3f1f64e632b69488208650fa375aac52ab7d16029f995'
ORIGINAL_READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
ORIGINAL_CONFIG_SHA = 'e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b'
PLANNED = 'ORIGINAL_CALL5_METADATA_PLAN_FROZEN_PENDING_TENSOR_READ'
RELEASED = 'GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED'
PROFILE = 'original_tp1_vllm_single_window'
INSTALLED_PATHS = (
    '__init__.py', 'attention/backends/flash_attn.py', 'attention/layer.py',
    'entrypoints/llm.py', 'model_executor/layers/layernorm.py',
    'model_executor/layers/linear.py', 'model_executor/layers/logits_processor.py',
    'model_executor/layers/rotary_embedding.py', 'model_executor/layers/sampler.py',
    'model_executor/layers/vocab_parallel_embedding.py',
    'model_executor/model_loader/loader.py', 'model_executor/model_loader/utils.py',
    'model_executor/model_loader/weight_utils.py', 'model_executor/models/llama.py',
    'model_executor/models/qwen2.py', 'model_executor/sampling_metadata.py',
    'outputs.py', 'worker/model_runner.py', 'model_executor/layers/activation.py',
    'model_executor/models/utils.py', 'model_executor/parameter.py',
    'model_executor/utils.py', 'attention/selector.py', 'sequence.py', 'config.py',
    '_custom_ops.py', 'engine/llm_engine.py', 'utils.py', 'sampling_params.py',
)


def bound_json(path, sha):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field: '+key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    with bound_descriptor(path, require_sha(sha)) as stream:
        value = json.loads(stream.read(), object_pairs_hook=pairs, parse_constant=bad)
    _finite_json(value)
    return value


def anchors():
    result = [dict(id='rank%02d-first'%rank, rank=rank, input_row=0, response_offset=0)
              for rank in range(8)]
    result.append(dict(id='rank04-row160-offset17', rank=4, input_row=160, response_offset=17))
    return result


def validate_inputs(value, root=ASSET_ROOT):
    require(type(value) is dict and value.get('status') == PLANNED and value.get('frozen') is True,
            'Source-frozen complete call5 metadata plan required')
    require(value.get('schema_version') == 1 and value.get('generation_call') == 5,
            'Only original generation_call5 is in this diagnostic')
    require(value.get('original_selection_sha256') == ORIGINAL_SELECTION_SHA and
            value.get('original_ready_sha256') == ORIGINAL_READY_SHA and
            value.get('original_config_sha256') == ORIGINAL_CONFIG_SHA,
            'Original ready/config/selection identity changed')
    require(value.get('asset_root') == str(root) and value.get('anchors') == anchors(),
            'Fixed same-node input root/anchor scope changed')
    require(value.get('actual_tensor_rows_verified') is False and value.get('CPU_job_executed_by_this_plan') is False,
            'Metadata plan cannot claim actual tensor preparation')
    def bound_entry(entry):
        require(type(entry) is dict and set(entry) == {'path','sha256'}, 'Bound asset metadata missing')
        path = Path(entry['path'])
        require(path.is_absolute() and '..' not in path.parts and Path(root) in path.parents,
                'Input asset escaped same-node root')
        require_sha(entry['sha256'])
    for label in ('source_position_report','hf_config_file','original_ready'):
        bound_entry(value.get(label))
    require(value['original_ready']['sha256'] == ORIGINAL_READY_SHA, 'Original ready entry changed')
    rows = value.get('ranks')
    require(type(rows) is list and len(rows) == 8 and [r.get('rank') for r in rows] == list(range(8)),
            'Exactly eight canonical ranks required')
    counts = []
    for row in rows:
        n, width = row.get('local_rows'), row.get('prompt_width')
        require(type(n) is int and n == 320 and type(width) is int and 0 < width <= 4096,
                'Exactly320 original prompt rows and source-bound width required')
        require(row.get('original_physical_rows') == list(range(n)),
                'Original prompt rows must be complete, contiguous, unreordered')
        require(row.get('expected_actor_micro4_calls') == 80,
                'Actual original local4 forward-count mismatch')
        counts.append((n+3)//4)
        for label in ('source_snapshot','source_record'): bound_entry(row.get(label))
        require(row.get('actual_tensor_rows_verified') is False and row.get('source_response_width') == 500,
                'Source metadata cannot claim actual load or change response width')
        identities = row.get('source_full_input_identity')
        require(type(identities) is dict and set(identities) == {'input_ids','attention_mask','position_ids'},
                'Complete source input identity missing')
        for identity in identities.values():
            require(set(identity) == {'dtype','shape','shape_and_bytes_sha256'} and
                    identity['dtype'] == 'torch.int64' and identity['shape'] == [320,width+500],
                    'Source input shape/dtype differs from original prompt/suffix layout')
            require_sha(identity['shape_and_bytes_sha256'])
    require(len(set(counts)) == 1, 'Actor FSDP forwards must be matched across ranks')
    require(rows[4]['local_rows'] > 160, 'Fixed saved worst-case anchor row is missing')
    require(value.get('maximum_new_emitted_tokens') == 1280000 and value.get('global_requests') == 2560,
            'New-generation upper token budget changed')
    require(value.get('actual_model_forwards') == 0 and value.get('actual_optimizer_writes') == 0,
            'CPU preparation must contain no model/optimizer work')
    return value


def validate_window_config(config):
    return validate_fixed_config(config, additional_expected={
        'actor_rollout_ref.rollout.enforce_eager': True,
        'actor_rollout_ref.rollout.free_cache_engine': True,
        'actor_rollout_ref.rollout.load_format': 'dummy_dtensor',
        'actor_rollout_ref.rollout.prompt_length': 4096,
        'actor_rollout_ref.rollout.response_length': 500,
    })


def bind_installed_sources(entries):
    require(type(entries) is dict and set(entries) == set(INSTALLED_PATHS),
            'Exactly the explicitly required installed source set is required')
    result = {}
    for relative, entry in entries.items():
        require(type(entry) is dict and set(entry) == {'path', 'sha256'}, 'Installed source identity incomplete')
        with bound_descriptor(entry['path'], entry['sha256']) as stream:
            stream.read()  # Same descriptor binding; no installed-module import.
        result[relative] = entry
    return result


def json_payload_sha(value):
    _finite_json(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()
