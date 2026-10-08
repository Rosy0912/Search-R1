"""Original-call construction and passive wrappers; no job launch or training."""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import sys
import types

import torch

from search_r1.baselines.probability_probe_capture import file_sha, require, tensor_digest
from search_r1.baselines.probability_probe_io import bound_descriptor


def load_bound_module(entry, name):
    """Execute verified source bytes from the same FD, never reopen its name."""
    with bound_descriptor(entry['path'], entry['sha256']) as stream:
        source = stream.read()
    module = types.ModuleType(name); module.__file__ = entry['path']
    require(name not in sys.modules, 'Bound diagnostic module name already imported')
    sys.modules[name] = module
    try:
        exec(compile(source, entry['path'], 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def extract_original_rollout(source_path, source_sha):
    with bound_descriptor(source_path, source_sha) as stream:
        source = stream.read()
    tree = ast.parse(source)
    methods = [method for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'ActorRolloutRefWorker'
               for method in cls.body if isinstance(method, ast.FunctionDef) and method.name == '_build_rollout']
    require(len(methods) == 1, 'Original rollout method missing/ambiguous')
    method = copy.deepcopy(methods[0]); require(not method.decorator_list, 'Original rollout gained a decorator')
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                    ('AdamW', 'step', 'backward', 'init', 'update_actor') for n in ast.walk(method)),
            'Original rollout constructor gained optimizer/Ray-init call')
    module = ast.Module(body=[method], type_ignores=[]); ast.fix_missing_locations(module)
    namespace = dict(torch=torch)
    from verl.utils.debug import log_gpu_memory_usage
    namespace['log_gpu_memory_usage'] = log_gpu_memory_usage
    exec(compile(module, source_path, 'exec'), namespace)
    return namespace['_build_rollout'], dict(original_source_sha256=source_sha,
        original_rollout_method_line=method.lineno,
        original_rollout_ast_sha256=hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        extra_engine_kwargs_added=False)


def model_metadata(model):
    return dict(modes={name:bool(child.training) for name,child in model.named_modules()},
        buffers={name:tensor_digest(value) for name,value in model.named_buffers(remove_duplicate=False)},
        dropout={name:float(child.p) for name,child in model.named_modules() if isinstance(child, torch.nn.Dropout)},
        parameters={name:tensor_digest(value) for name,value in model.named_parameters(remove_duplicate=False)})


def class_source(obj):
    cls = type(obj)
    return dict(module=cls.__module__, name=cls.__qualname__, source_path=inspect.getsourcefile(cls))


def bind_method(method, expected_entry, decorator_entries=()):
    # Decorated originals use __wrapped__; bind every wrapper code file too.
    entries = [expected_entry]+list(decorator_entries)
    allowed = {str(Path(e['path']).resolve()):e for e in entries}
    seen = []; current = method
    while current is not None:
        function = getattr(current, '__func__', current)
        path = inspect.getsourcefile(function)
        require(path is not None and str(Path(path).resolve()) in allowed,
                'Actual native method source path differs from bound source')
        entry = allowed[str(Path(path).resolve())]
        require(file_sha(path) == entry['sha256'], 'Actual native method source SHA changed')
        seen.append(dict(path=path, sha256=entry['sha256'], name=function.__qualname__))
        current = getattr(function, '__wrapped__', None)
    require(seen and Path(seen[-1]['path']).resolve() == Path(expected_entry['path']).resolve(),
            'Final unwrapped native implementation differs from bound source')
    return seen


def plain_object(value):
    """Exact primitive metadata, never repr() an unknown runtime object."""
    import enum
    if value is None or type(value) in (bool, int, float, str): return value
    if isinstance(value, enum.Enum): return dict(enum_class=type(value).__module__+'.'+type(value).__qualname__, value=plain_object(value.value))
    if isinstance(value, torch.dtype): return str(value)
    if isinstance(value, torch.device): return str(value)
    if isinstance(value, (list,tuple)): return [plain_object(v) for v in value]
    if isinstance(value, (set,frozenset)): return sorted(plain_object(v) for v in value)
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), 'Runtime metadata dictionary key must be string')
        return {k:plain_object(v) for k,v in value.items()}
    raise ValueError('Unsupported runtime metadata value: '+type(value).__module__+'.'+type(value).__qualname__)


def sampling_params_metadata(params):
    # msgspec.Struct (vLLM0.6.3) carries declared fields rather than __dict__.
    fields = getattr(type(params), '__struct_fields__', None)
    require(isinstance(fields, tuple) and fields, 'Actual SamplingParams declared fields unavailable')
    return {name:plain_object(getattr(params, name)) for name in fields}


def sampling_params_transition(before, after, sources):
    """Record every field; allow only native final-output delivery selection.

    Installed LLM._validate_and_add_requests mutates the caller's output_kind
    before request cloning. No distribution/stop/length field is exempted.
    The actual snapshots must be published even when this check refuses.
    """
    pins = {'entrypoints/llm.py': '1393441b1b5cf4750b2e3a72ff30272a7e0db2220fa23ffa459f135265686582',
            'sampling_params.py': 'f379042fe14ca7adb2a74eeb1b746de4583481c2a10416f7fe1b05aba81952bc'}
    require(all(sources[name]['sha256'] == sha for name, sha in pins.items()),
            'Native output-mode transition requires exact installed source pins')
    require(type(before) is dict and type(after) is dict,
            'Actual complete SamplingParams snapshots required')
    enum_class = 'vllm.sampling_params.RequestOutputKind'
    cumulative = dict(enum_class=enum_class, value=0)
    final_only = dict(enum_class=enum_class, value=2)
    reasons = []
    if set(before) != set(after): reasons.append('declared_sampling_fields_changed')
    if before.get('output_kind') not in (cumulative, final_only):
        reasons.append('unexpected_initial_output_kind')
    if after.get('output_kind') != final_only:
        reasons.append('native_final_only_output_kind_missing')
    changed = sorted(name for name in set(before) | set(after)
                     if name not in before or name not in after or before[name] != after[name])
    if any(name != 'output_kind' for name in changed):
        reasons.append('sampling_field_other_than_output_kind_changed')
    return dict(before=before, after=after, changed_fields=changed,
                allowed=not reasons, reasons=reasons,
                allowed_native_transition='CUMULATIVE_or_FINAL_ONLY_to_FINAL_ONLY_only',
                source_pins=pins, observer_changed_sampling_parameters=False,
                scope='Actual complete field snapshots; output delivery only, no distribution correction')


class NativeWindowWrappers:
    """Observe actual registration/input/RequestOutput and sync, originals once.

    Native failures propagate. Observer failures defer until all ranks have
    finished the native window; wrappers never return replacement values.
    """
    def __init__(self, llm, observer, weights, expected_sources):
        self.llm, self.observer, self.weights = llm, observer, weights
        self.engine = llm.llm_engine
        self.runner = self.engine.model_executor.worker.model_runner
        self.sources = expected_sources
        self.saved = []; self.errors = []; self.requests = 0; self.counts = dict(sync=0, execute=0, output=0)
        self.native_exception = None

    def defer(self, label, operation):
        try: return operation()
        except Exception as exc: self.errors.append(label+': '+type(exc).__name__+': '+str(exc))

    def patch(self, obj, name, wrapper):
        original = getattr(obj, name)
        local = name in vars(obj); self.saved.append((obj, name, local, vars(obj).get(name)))
        setattr(obj, name, wrapper(original))

    def __enter__(self):
        # This stage runs on all ranks before any native construction/window.
        self.method_receipts = {
            'register':bind_method(self.engine.add_request, self.sources['engine_add_request'], self.sources['decorators']),
            'execute':bind_method(self.runner.execute_model, self.sources['model_runner'], self.sources['decorators']),
            'output':bind_method(self.llm._post_process_outputs, self.sources['facade']),
            'sync':bind_method(self.llm.sync_model_weights, self.sources['facade']),
        }
        def register(original):
            signature = inspect.signature(original)
            def wrapped(*args, **kwargs):
                row = self.requests; self.requests += 1
                def capture():
                    bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                    prompt = bound.arguments['prompt']
                    require(bound.arguments.get('inputs') is None and isinstance(prompt, dict) and
                            'prompt_token_ids' in prompt and set(prompt) <= {'prompt_token_ids', 'prompt'},
                            'Original token-only prompt registration required')
                    self.observer.register_request(bound.arguments['request_id'], prompt['prompt_token_ids'], row)
                self.defer('actual request registration', capture)
                return original(*args, **kwargs)
            return wrapped
        def execute(original):
            signature = inspect.signature(original)
            def wrapped(*args, **kwargs):
                self.counts['execute'] += 1
                self.defer('actual ModelInput', lambda:self.observer.observe_model_input(signature.bind(*args, **kwargs).arguments['model_input']))
                return original(*args, **kwargs)
            return wrapped
        def output(original):
            signature = inspect.signature(original)
            def wrapped(*args, **kwargs):
                self.counts['output'] += 1
                self.defer('actual RequestOutput before pad', lambda:self.observer.observe_generated(
                    signature.bind(*args, **kwargs).arguments['request_outputs']))
                return original(*args, **kwargs)
            return wrapped
        def sync(original):
            signature = inspect.signature(original)
            def wrapped(*args, **kwargs):
                self.counts['sync'] += 1
                result = original(*args, **kwargs)
                def capture():
                    bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                    arguments = bound.arguments
                    require(arguments['load_format'] == 'dtensor', 'Original sharded DTensor sync required')
                    self.weights.after_sync(arguments['actor_weights'])
                self.defer('sync completed live weight proof', capture)
                return result
            return wrapped
        for obj, name, wrapper in [(self.engine,'add_request',register),(self.runner,'execute_model',execute),
                                  (self.llm,'_post_process_outputs',output),(self.llm,'sync_model_weights',sync)]:
            self.patch(obj, name, wrapper)
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None: self.native_exception = kind.__name__+': '+str(value)
        for obj,name,local,previous in reversed(self.saved):
            if local:setattr(obj,name,previous)
            else:delattr(obj,name)
        self.saved.clear()
        return False

    def receipt(self):
        require(not self.saved and not self.errors and self.native_exception is None,
                'Native passive-wrapper evidence failed: '+repr(self.errors)+'; native_exception='+repr(self.native_exception))
        require(self.counts['sync'] == self.counts['output'] == 1 and self.counts['execute'] > 0,
                'Expected exactly one native generation window and output observation')
        return dict(counts=self.counts, registered_requests=self.requests, source_methods=self.method_receipts,
                    instance_hooks_restored=True, distribution_changed=False)


class LiveWindowWeights:
    """Source-bound independent core, live GPU references before offload only."""
    def __init__(self, core, model, config, device, coordinator):
        self.core,self.model,self.config,self.device = core,model,config,torch.device(device)
        self.coordinator = coordinator
        self.pre = None; self.post = None; self.pre_metadata = None

    def after_sync(self, weights):
        def validate():
            require(self.pre is None, 'A second synchronization window is forbidden')
            require(isinstance(weights, dict), 'Original state_dict mapping required')
            plan = self.core.qwen2_tp1_mapping(int(self.config.num_hidden_layers))
            names = sorted({key for item in plan for key in item.sources})
            require(set(weights) == set(names), 'State_dict logical keys must exactly cover HF435')
            require(all(isinstance(weights[name],torch.Tensor) and callable(getattr(weights[name],'full_tensor',None)) and
                        weights[name].dtype == torch.float32 for name in names), 'All logical sources must be FP32 DTensors')
            return names
        names = self.coordinator.local_stage('same-window all-rank pre-gather source contract',validate)
        self.coordinator.signature([len(names),int(self.config.num_hidden_layers),int(self.config.hidden_size),
                                    int(self.config.vocab_size),int(self.config.num_attention_heads),int(self.config.num_key_value_heads)])
        # Identical sorted all-rank DTensor collectives; release full copies after
        # the pre comparison. No actual weights are exported or retained to Mac.
        source = {}
        for name in names:
            source[name] = self.coordinator.local_stage('original DTensor gather '+name,
                lambda name=name:weights[name].full_tensor().detach().cpu())
        actual = dict(self.model.named_parameters(remove_duplicate=False))
        cfg = self.config; head_dim = int(cfg.hidden_size)//int(cfg.num_attention_heads)
        self.pre = self.coordinator.local_stage('complete live TP1 source cast comparison',lambda:self.core.compare_tp1_weights(source, actual,
            layers=int(cfg.num_hidden_layers), tensor_parallel_size=1, expected_dtype=torch.bfloat16,
            tie_word_embeddings=bool(cfg.tie_word_embeddings), vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size), intermediate_size=int(cfg.intermediate_size),
            q_rows=int(cfg.num_attention_heads)*head_dim, kv_rows=int(cfg.num_key_value_heads)*head_dim,
            expected_device=self.device))
        # Finite value mismatches are the measurement, not an observer failure.
        # Only structure/device/finite/provenance errors refuse this diagnostic.
        self.pre_metadata = model_metadata(self.model)
        require(all(not mode for mode in self.pre_metadata['modes'].values()) and
                all(p == 0 for p in self.pre_metadata['dropout'].values()), 'Actual native inference model mode/dropout changed')
        del source, actual

    def before_offload(self):
        # A rejected second boundary cannot leave an older successful receipt.
        self.post = None
        require(self.pre is not None and self.pre_metadata is not None, 'No completed live post-sync weight proof')
        current = model_metadata(self.model)
        require(current == self.pre_metadata, 'Actual live sampler weights/buffers/modes changed within window')
        actual = dict(self.model.named_parameters(remove_duplicate=False))
        left,right = actual['model.embed_tokens.weight'],actual['lm_head.weight']
        require(left.device == right.device == self.device and bool(torch._C._is_alias_of(left,right)) and
                left.shape == right.shape and left.stride() == right.stride() and left.storage_offset() == right.storage_offset(),
                'Same-window tied embedding/head alias changed before offload')
        self.post = dict(status='LIVE_PRE_OFFLOAD_BYTES_AND_ALIAS_UNCHANGED', metadata=current,
                        same_window_tie=True, target_bytes_unchanged_since_live_postsync_snapshot=True,
                        source_cast_all_values_match=self.pre['all_parameter_values_match'],
                        CPU_offload_snapshot_used=False, historical_identity_proven=False)

    def receipt(self):
        require(self.pre is not None and self.post is not None, 'Both live-window weight boundaries required')
        return dict(status='LIVE_SYNC_AND_PRE_OFFLOAD_WEIGHTS_OBSERVED_NO_PROBABILITY_ACCEPTANCE',
                    pre=self.pre, post=self.post, buffers_compared=True, historical_3268_identity_proven=False,
                    probability_accepted=False, training_accepted=False)
