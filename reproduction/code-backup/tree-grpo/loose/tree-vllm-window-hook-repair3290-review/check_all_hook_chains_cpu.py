"""Synthetic source/annotation/decorator audit; never import native libraries."""
import ast
import copy
import contextlib
import functools
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import typing
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO = ROOT / 'work/single-tp1-window-e29b27d1-466df7c2/repo'
INSTALLED = ROOT / 'work/tree-vllm-installed-source-20261006'
BASE = ROOT / 'work/tree-vllm-installed-hook-base-source-20261006'
UTILS = ROOT / 'work/tree-vllm-installed-wrapper-source-20261006/vllm/utils.py'
CONTEXT = ROOT / 'work/tree-vllm-installed-wrapper-source-20261006/torch/utils/_contextlib.py'
CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(path):
    if path not in CACHE:
        CACHE[path] = path.read_bytes()
    return ast.parse(CACHE[path])


def function(path, name, klass=None):
    nodes = tree(path).body
    if klass:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == klass).body
    return [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name][-1]


def execute(path, node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[node.name]


def require(value, message):
    if not value:
        raise ValueError(message)


def entry(path):
    return dict(path=str(path), sha256=sha(path))


def fields(path, klass):
    cls = next(n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == klass)
    result = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            result.setdefault(n.target.id, []).append(n.lineno)
        if isinstance(n, ast.FunctionDef):
            if any(isinstance(d, ast.Name) and d.id == 'property' for d in n.decorator_list):
                result.setdefault(n.name, []).append(n.lineno)
            for sub in ast.walk(n):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == 'self' and isinstance(sub.ctx, ast.Store):
                    result.setdefault(sub.attr, []).append(sub.lineno)
    return result


def main():
    proof_path = OUT / 'ALL_10_HOOK_CHAIN_CPU_PROOF.json'
    assert not proof_path.exists()
    base_path = BASE / 'worker/model_runner_base.py'
    base_map = json.loads((BASE / 'SOURCE_MAP.json').read_bytes())
    assert sha(base_path) == base_map['worker/model_runner_base.py']['sha256'] == '49fe0eedd1667889daa63d5aa572d00f5eefbb794ea4368497f5243b02a54173'
    un = dict(F=typing.TypeVar('F'), T=typing.TypeVar('T'), Callable=typing.Callable, Any=typing.Any, Union=typing.Union, Optional=typing.Optional, partial=functools.partial, wraps=functools.wraps, warnings=warnings)
    execute(UTILS, function(UTILS, 'identity'), un)
    deprecated = execute(UTILS, function(UTILS, 'deprecate_kwargs'), un)
    dump = execute(base_path, function(base_path, 'dump_input_when_exception'), dict(Optional=typing.Optional, List=typing.List, wraps=functools.wraps))
    context = execute(CONTEXT, function(CONTEXT, 'context_decorator'), dict(functools=functools, inspect=inspect))
    bind_path = REPO / 'window_runtime.py'
    bind = execute(bind_path, function(bind_path, 'bind_method'), dict(Path=Path, inspect=inspect, require=require, file_sha=sha))
    old_allowed = [entry(UTILS), entry(CONTEXT)]
    allowed = old_allowed + [entry(base_path)]
    facade = REPO / 'verl/third_party/vllm/vllm_v_0_6_3/llm.py'
    engine = ROOT / 'work/tree-vllm-installed-engine-source-20261006/engine/llm_engine.py'
    qwen = INSTALLED / 'model_executor/models/qwen2.py'
    logits = INSTALLED / 'model_executor/layers/logits_processor.py'
    sampler = INSTALLED / 'model_executor/layers/sampler.py'
    runner = INSTALLED / 'worker/model_runner.py'
    methods = [
        ('engine.add_request', engine, 'LLMEngine', 'add_request', 'deprecate'),
        ('runner.execute_model', runner, 'ModelRunner', 'execute_model', 'dump'),
        ('facade._post_process_outputs', facade, 'LLM', '_post_process_outputs', None),
        ('facade.sync_model_weights', facade, 'LLM', 'sync_model_weights', None),
        ('model.forward', qwen, 'Qwen2ForCausalLM', 'forward', None),
        ('model.compute_logits', qwen, 'Qwen2ForCausalLM', 'compute_logits', None),
        ('model.sample', qwen, 'Qwen2ForCausalLM', 'sample', None),
        ('logits_processor.forward', logits, 'LogitsProcessor', 'forward', None),
        ('logits_processor._get_logits', logits, 'LogitsProcessor', '_get_logits', None),
        ('sampler.forward', sampler, 'Sampler', 'forward', None),
    ]
    records = []
    execute_method = None
    for label, path, klass, name, decoration in methods:
        actual = function(path, name, klass)
        prototype = copy.deepcopy(actual)
        prototype.decorator_list = []
        prototype.returns = None
        for arg in prototype.args.posonlyargs + prototype.args.args + prototype.args.kwonlyargs:
            arg.annotation = None
        if prototype.args.vararg:
            prototype.args.vararg.annotation = None
        if prototype.args.kwarg:
            prototype.args.kwarg.annotation = None
        prototype.body = [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='_calls', ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Constant(label)], keywords=[])), ast.Return(value=ast.Name(id='_marker', ctx=ast.Load()))]
        namespace = dict(_calls=[], _marker=object())
        original = execute(path, prototype, namespace)
        if decoration == 'deprecate':
            wrapped = deprecated('inputs', additional_message="Please use the 'prompt' parameter instead.")(original)
        elif decoration == 'dump':
            wrapped = context(lambda: contextlib.nullcontext(), dump(exclude_args=[0], exclude_kwargs=['self'])(original))
        else:
            assert not actual.decorator_list
            wrapped = original
        method = MethodType(wrapped, SimpleNamespace())
        chain = bind(method, entry(path), allowed)
        signature = inspect.signature(method)
        assert list(signature.parameters) == [a.arg for a in actual.args.args[1:] + actual.args.kwonlyargs]
        arguments = {k: object() for k, v in signature.parameters.items() if v.default is inspect.Parameter.empty}
        bound = signature.bind(**arguments)
        bound.apply_defaults()
        assert method(**arguments) is namespace['_marker'] and namespace['_calls'] == [label]
        records.append(dict(method=label, source_path=str(path.relative_to(ROOT)), source_sha256=sha(path), definition_line=actual.lineno, source_signature=ast.unparse(actual.args), source_decorators=[ast.unparse(d) for d in actual.decorator_list], synthetic_bound_signature=str(signature), verified_file_chain=[str(Path(item['path']).relative_to(ROOT)) for item in chain], wrapper_chain_binding_passed=True, exact_signature_parameter_names_preserved=True, synthetic_original_invoked_once_and_return_identity_preserved=True))
        if label == 'runner.execute_model':
            execute_method = method
    counterexamples = []
    for label, permitted, expected in [
        ('old_whitelist_missing_dump_source_refused', old_allowed, 'Actual native method source path differs'),
        ('new_dump_source_wrong_sha_refused', old_allowed + [dict(entry(base_path), sha256='0'*64)], 'SHA changed'),
    ]:
        try:
            bind(execute_method, entry(runner), permitted)
        except ValueError as error:
            assert expected in str(error)
            counterexamples.append(dict(test=label, passed=True, error=str(error)))
        else:
            raise AssertionError('Expected refusal')
    sequence = ROOT / 'work/tree-vllm-installed-source-extra-20261006/sequence.py'
    sampling = INSTALLED / 'model_executor/sampling_metadata.py'
    outputs = INSTALLED / 'outputs.py'
    attribute_specs = [
        (runner, 'ModelInputForGPU', ['input_tokens','input_positions','request_ids_to_seq_ids','query_lens']),
        (runner, 'ModelInputForGPUWithSamplingMetadata', ['sampling_metadata']),
        (sampling, 'SamplingMetadata', ['selected_token_indices','seq_groups','skip_sampler_cpu_output']),
        (sampling, 'SequenceGroupToSample', ['seq_ids','sample_indices','prompt_logprob_indices','seq_data','sampling_params','is_prompt']),
        (sequence, 'SequenceData', ['prompt_token_ids','output_token_ids']),
        (sampler, 'SamplerOutput', ['outputs']),
        (sequence, 'CompletionSequenceGroupOutput', ['samples']),
        (sequence, 'SequenceOutput', ['parent_seq_id','output_token','logprobs']),
        (outputs, 'RequestOutput', ['request_id','prompt_token_ids','prompt_logprobs','finished','outputs']),
        (outputs, 'CompletionOutput', ['index','token_ids','logprobs','finish_reason','stop_reason','cumulative_logprob']),
        (sequence, 'Logprob', ['logprob','rank','decoded_token']),
    ]
    attributes = []
    for path, klass, needed in attribute_specs:
        actual_fields = fields(path, klass)
        for name in needed:
            assert name in actual_fields, (klass, name)
            attributes.append(dict(klass=klass, field=name, definition_lines=sorted(set(actual_fields[name])), source_path=str(path.relative_to(ROOT)), source_sha256=sha(path)))
    for path, data in CACHE.items():
        assert path.read_bytes() == data
    proof = dict(status='ALL_10_SOURCE_HOOK_CHAINS_AND_SIGNATURES_CPU_RECONSTRUCTED_NO_GPU_RELEASE', audited_failed_package='single-tp1-window-e29b27d1-466df7c2', method_inventory=records, negative_counterexamples=counterexamples, method_count=10, observer_attribute_inventory=attributes, observer_attribute_count=len(attributes), missing_wrapper_file=base_map['worker/model_runner_base.py'], required_execute_chain=['torch/utils/_contextlib.py','vllm/worker/model_runner_base.py','vllm/worker/model_runner.py'], additional_method_decorator_files_found=[], other_reviewed_signature_or_attribute_API_mismatches_found=[], source_files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(data).hexdigest() for path,data in CACHE.items()}, scope='Exact saved decorator AST bodies and bind_method body. Native method bodies replaced by sentinel-return CPU prototypes preserving actual signatures. Saved context_decorator uses null context, not Torch inference execution. Code filenames/paths are local source captures, not actual remote loaded modules. No native method body, model, engine or kernel execution.', vllm_imported=False, torch_imported=False, remote_access=False, actual_assets_read=False, GPU_executed=False, production_edited=False, probability_accepted=False, training_accepted=False, dispatch_release=False)
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    print('10 method chains PASS; 2 refusal counterexamples PASS; observer attributes:', len(attributes))
    print('proof_sha256:', sha(proof_path))


if __name__ == '__main__':
    main()
