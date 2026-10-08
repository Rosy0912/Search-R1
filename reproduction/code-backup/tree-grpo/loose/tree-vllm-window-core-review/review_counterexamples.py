"""Independent tiny CPU fixtures. Never imports vLLM or loads PT/HF/model assets."""
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import torch

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CORE = BASE / 'work/tree-vllm-window-core/vllm_window_weights.py'
INSTALLED = BASE / 'work/tree-vllm-installed-source-20261006'
checks = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def expect_refusal(name, callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        checks.append(name)
    else:
        raise AssertionError('Unexpected acceptance: ' + name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

check('exact candidate helper bytes', sha(CORE) ==
      '46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350')
check('exact owner test bytes', sha(CORE.with_name('test_vllm_window_weights.py')) ==
      'd9f2c4b0acb8a81e6057a4b40ab4595019700367490a65f2e043055ca139b43f')
source_map = json.loads((INSTALLED / 'SOURCE_MAP.json').read_bytes())
check('complete18 captured installed source metadata', len(source_map) == 18 and all(
      (INSTALLED / key).stat().st_size == v['bytes'] and sha(INSTALLED / key) == v['sha256']
      for key, v in source_map.items()))
spec = importlib.util.spec_from_file_location('reviewed_observer_only', CORE)
M = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = M
spec.loader.exec_module(M)
linear_ast = ast.parse((INSTALLED / 'model_executor/layers/linear.py').read_bytes())
qclass = next(n for n in linear_ast.body if isinstance(n, ast.ClassDef) and n.name == 'QKVParallelLinear')
methods = [copy.deepcopy(n) for n in qclass.body if isinstance(n, ast.FunctionDef) and
           n.name in ('_get_shard_offset_mapping', '_get_shard_size_mapping')]
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             'isolated_actual_installed_qkv_metadata_functions', 'exec'), namespace)
fake = SimpleNamespace(num_heads=4, num_kv_heads=1, head_size=1)
check('actual source TP1 q/k/v offsets', [namespace['_get_shard_offset_mapping'](fake, k)
      for k in ('q', 'k', 'v', 'total')] == [0, 4, 5, 6])
check('actual source TP1 q/k/v row sizes', [namespace['_get_shard_size_mapping'](fake, k)
      for k in ('q', 'k', 'v')] == [4, 1, 1])
loader = BASE / 'work/tree-safety110-a632ba9-v2/source/repo/verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py'
loader_ast = ast.parse(loader.read_bytes())
fn = next(n for n in loader_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'qwen2_dtensor_weight_loader')
mapping = next(n for n in fn.body if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'stacked_params_mapping' for t in n.targets))
check('original DTensor q/k/v and gate/up loader plan', ast.literal_eval(mapping.value) == [
    ('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'),
    ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)])

# Expected data and target packing are hand constructed from the source layout,
# not from the helper mapping under test.
config = dict(layers=36, tensor_parallel_size=1, expected_dtype=torch.bfloat16,
              tie_word_embeddings=True, vocab_size=7, hidden_size=4,
              intermediate_size=3, q_rows=4, kv_rows=1, expected_device='cpu')
source = {}

def value(shape, tag):
    n = math_prod(shape)
    return (torch.arange(n, dtype=torch.float32).reshape(shape) / 256 + tag).requires_grad_(True)

def math_prod(shape):
    result = 1
    for x in shape:
        result *= x
    return result

source['model.embed_tokens.weight'] = value((7, 4), .03125)
source['lm_head.weight'] = source['model.embed_tokens.weight'].detach().clone().requires_grad_(True)
source['model.norm.weight'] = value((4,), .0625)
for i in range(36):
    prefix = f'model.layers.{i}.'
    for j, (suffix, shape) in enumerate([
        ('input_layernorm.weight', (4,)), ('post_attention_layernorm.weight', (4,)),
        ('self_attn.q_proj.weight', (4, 4)), ('self_attn.k_proj.weight', (1, 4)),
        ('self_attn.v_proj.weight', (1, 4)), ('self_attn.q_proj.bias', (4,)),
        ('self_attn.k_proj.bias', (1,)), ('self_attn.v_proj.bias', (1,)),
        ('self_attn.o_proj.weight', (4, 4)), ('mlp.gate_proj.weight', (3, 4)),
        ('mlp.up_proj.weight', (3, 4)), ('mlp.down_proj.weight', (4, 3))]):
        source[prefix + suffix] = value(shape, i / 64 + (j + 1) / 16)
actual = {k: v.detach().bfloat16().clone().requires_grad_(True) for k, v in source.items()
          if not any(fragment in k for fragment in ('q_proj.', 'k_proj.', 'v_proj.', 'gate_proj.', 'up_proj.'))}
for i in range(36):
    p = f'model.layers.{i}.'
    for suffix in ('weight', 'bias'):
        actual[p + 'self_attn.qkv_proj.' + suffix] = torch.cat([
            source[p + f'self_attn.{part}_proj.' + suffix].detach() for part in ('q', 'k', 'v')], 0).bfloat16().requires_grad_(True)
    actual[p + 'mlp.gate_up_proj.weight'] = torch.cat([
        source[p + 'mlp.gate_proj.weight'].detach(), source[p + 'mlp.up_proj.weight'].detach()], 0).bfloat16().requires_grad_(True)
actual['lm_head.weight'] = actual['model.embed_tokens.weight']
before_s = {k: v.detach().clone() for k, v in source.items()}
before_a = {k: v.detach().clone() for k, v in actual.items()}
rng = torch.random.get_rng_state().clone()
result = M.compare_tp1_weights(source, actual, **config)
check('independent full36 layer fixture435 to255 packed keys', result['serialized_source_keys'] == 435 and
      result['serialized_target_keys'] == 255 and result['all_parameter_values_match'] is True and
      len(result['records']) == 255 and sum(len(x['fragments']) for x in result['records']) == 435)
check('live alias and no acceptance flags', result['same_window_tied_alias'] is True and
      result['actual_parameter_device'] == 'cpu' and result['actual_device_is_cuda'] is False and all(
          result[k] is False for k in ('buffers_compared', 'live_window_provenance_proven_by_this_function',
                                      'historical_3268_identity_proven', 'probability_accepted', 'training_accepted')))
check('all source+target values unchanged', all(torch.equal(before_s[k], v) for k, v in source.items()) and
      all(torch.equal(before_a[k], v) for k, v in actual.items()))
check('RNG unchanged and no backward gradients', torch.equal(rng, torch.random.get_rng_state()) and
      all(x.grad is None for x in source.values()) and all(x.grad is None for x in actual.values()))
check('all target source coverage exact and fragment order',
      {x.target for x in M.qwen2_tp1_mapping(36)} == set(actual) and
      {k for x in M.qwen2_tp1_mapping(36) for k in x.sources} == set(source))
for suffix in ('weight', 'bias'):
    key = 'model.layers.35.self_attn.qkv_proj.' + suffix
    saved = actual[key]
    for order in itertools.permutations(('q', 'k', 'v')):
        if order == ('q', 'k', 'v'):
            continue
        actual[key] = torch.cat([source['model.layers.35.self_attn.' + x + '_proj.' + suffix].detach()
                                 for x in order], 0).bfloat16()
        bad = M.compare_tp1_weights(source, actual, **config)
        check('all wrong final-layer qkv permutations detected ' + suffix + str(order),
              bad['all_parameter_values_match'] is False and bad['different_elements'] > 0 and
              bad['max_abs_delta'] > 0 and bad['probability_accepted'] is False)
    actual[key] = saved
key = 'model.layers.35.mlp.gate_up_proj.weight'
saved = actual[key]
actual[key] = torch.cat([source['model.layers.35.mlp.up_proj.weight'].detach(),
                         source['model.layers.35.mlp.gate_proj.weight'].detach()], 0).bfloat16()
check('final layer gate/up reversal is a mismatch observation',
      M.compare_tp1_weights(source, actual, **config)['all_parameter_values_match'] is False)
actual[key] = saved
key = 'model.layers.35.mlp.down_proj.weight'
saved = actual[key]
actual[key] = actual[key].transpose(0, 1)
expect_refusal('final layer transpose cannot broadcast', lambda: M.compare_tp1_weights(source, actual, **config))
actual[key] = saved
saved = actual['lm_head.weight']
actual['lm_head.weight'] = saved.detach().clone()
expect_refusal('equal value head clone is not same live alias', lambda: M.compare_tp1_weights(source, actual, **config))
actual['lm_head.weight'] = saved
actual['unknown.new_key'] = torch.ones(1)
expect_refusal('extra actual key cannot be ignored', lambda: M.compare_tp1_weights(source, actual, **config))
del actual['unknown.new_key']
saved = source[key]
source[key] = saved.detach().clone()
source[key][0, 0] = math_nan = float('nan')
expect_refusal('late layer nonfinite source refused', lambda: M.compare_tp1_weights(source, actual, **config))
source[key] = saved
saved = actual[key]
actual[key] = saved.detach().clone()
actual[key][0, 0] = float('inf')
expect_refusal('late layer nonfinite target refused', lambda: M.compare_tp1_weights(source, actual, **config))
actual[key] = saved
expect_refusal('offloaded CPU cannot claim actual CUDA window', lambda: M.compare_tp1_weights(
    source, actual, **{**config, 'expected_device': 'cuda:0'}))
check('helper calls did not initialize CUDA', torch.cuda.is_initialized() is False)
proof = {'status': 'UNWIRED_TP1_QWEN2_OBSERVER_SOURCE_AND_TINY_CPU_COMPARISON_REVIEWED_ONLY',
    'helper_sha256': sha(CORE), 'test_sha256': sha(CORE.with_name('test_vllm_window_weights.py')),
    'installed_source_map_sha256': sha(INSTALLED / 'SOURCE_MAP.json'),
    'original_dtensor_loader_sha256': sha(loader), 'source_files_verified': 18,
    'checks_passed': len(checks), 'checks': checks, 'fixture_source_keys': 435,
    'fixture_packed_target_keys': 255, 'logical_layers': 36,
    'local_torch_version': torch.__version__, 'cuda_initialized': torch.cuda.is_initialized(),
    'actual_vllm_model_constructed': False, 'actual_same_window_runtime_proven': False,
    'model_optimizer_or_scheduler_instances': 0, 'actual_model_forwards': 0, 'remote_or_slurm_calls': 0,
    'real_pt_hf_assets_loaded': False, 'probability_accepted': False, 'training_accepted': False,
    'buffers_compared': False, 'frozen_source_modified': False,
    'limits': ['Synthetic CPU36-layer dimensions, not production-sized model or actual sampling window',
        'Function assumes caller provides stable materialized source/live attributes and does not prove provenance',
        'Future runner must bind config/quantization/TP1/device/method sources and defer errors across collective boundaries',
        'Per-parameter expected dtype cast is full fragment; chunked digest/deltas do not mean total allocation is only chunk size'],
}
(HERE / 'INDEPENDENT_COUNTEREXAMPLE_PROOF.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
print(json.dumps({'checks_passed': len(checks), 'proof_sha256': sha(HERE / 'INDEPENDENT_COUNTEREXAMPLE_PROOF.json')}))
