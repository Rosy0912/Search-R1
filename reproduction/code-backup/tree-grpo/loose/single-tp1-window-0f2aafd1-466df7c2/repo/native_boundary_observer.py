"""Source-only passive boundary observer; no runner, services or asset loading.

The owner must bind actual native methods and configuration before installation.
Hooks observe one original call, return None, and defer observation refusals until
the owner's matched all-rank post-native stage. No residual is reconstructed.
"""
import copy
import hashlib
import json

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_anchors():
    """Exact original window_contracts.anchors coordinates, not new selection."""
    result = [dict(id='rank%02d-first' % rank, rank=rank, input_row=0,
                   response_offset=0) for rank in range(8)]
    result.append(dict(id='rank04-row160-offset17', rank=4, input_row=160,
                       response_offset=17))
    return result


def boundary_names(config):
    n = config['num_hidden_layers']
    require(type(n) is int and n == 36, 'Bound original configuration must contain36 layers')
    names = ['embedding']
    for layer in range(n):
        names.extend(['layers.%02d.input_layernorm' % layer,
                      'layers.%02d.post_attention_layernorm' % layer])
    return names + ['final_norm']


def build_boundary_modules(model_core, bound_hf_config):
    """Use retained real Qwen .model module refs, before FSDP registry changes."""
    names = boundary_names(bound_hf_config)
    require(len(model_core.layers) == bound_hf_config['num_hidden_layers'],
            'Actual layer module count differs from bound HF configuration')
    modules = {'embedding': model_core.embed_tokens}
    for layer, module in enumerate(model_core.layers):
        modules['layers.%02d.input_layernorm' % layer] = module.input_layernorm
        modules['layers.%02d.post_attention_layernorm' % layer] = module.post_attention_layernorm
    modules['final_norm'] = model_core.norm
    require(list(modules) == names and len({id(m) for m in modules.values()}) == len(names),
            'Missing, reordered or aliased boundary modules')
    require(all(isinstance(m, torch.nn.Module) for m in modules.values()),
            'Every real boundary must be a torch Module')
    return modules


def _list(tensor):
    return tensor.detach().cpu().tolist()


def _identity(tensor):
    require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()),
            'Nonfinite or missing captured boundary vector')
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode() + b'\0' + str(tuple(value.shape)).encode() + b'\0')
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                full_shape_dtype_bytes_sha256=digest.hexdigest())


class NativeBoundaryObserver:
    """Observe actual normalized output rows on fixed original causal anchors.

    vLLM: observe_vllm_step(window_observer._step) immediately before the original
    execute_model. actor: prepare_actor_call with the already accepted original
    response_prediction_map, then run exactly the original immutable native call.
    clear_step follows each call. check_errors must be invoked only from an
    owner's all-rank matched stage after native calls, never inside a hook.
    """
    def __init__(self, backend, rank, anchors, modules, bound_hf_config,
                 actual_config, provenance, *, root_module=None):
        require(backend in ('actor', 'vllm') and type(rank) is int and 0 <= rank < 8,
                'Original actor/vllm backend and eight-rank identity required')
        require(json.dumps(anchors, sort_keys=True, separators=(',', ':'), allow_nan=False) ==
                json.dumps(fixed_anchors(), sort_keys=True, separators=(',', ':')),
                'Original nine anchor plan changed')
        require(type(bound_hf_config) is dict,
                'Caller-verified bound HF configuration fields required')
        actual_fields = {field: (actual_config.get(field) if type(actual_config) is dict
                         else getattr(actual_config, field, None))
                         for field in ('num_hidden_layers', 'hidden_size')}
        for field in ('num_hidden_layers', 'hidden_size'):
            require(type(bound_hf_config.get(field)) is int and bound_hf_config[field] > 0
                    and type(actual_fields.get(field)) is int
                    and actual_fields[field] == bound_hf_config[field],
                    'Actual configuration differs from bound HF field: ' + field)
        self.names = boundary_names(bound_hf_config)
        require(list(modules) == self.names and len(self.names) == 74
                and len({id(m) for m in modules.values()}) == 74,
                'Exactly74 unique original boundary modules required')
        require(all(isinstance(m, torch.nn.Module) for m in modules.values()), 'Actual module references required')
        require(type(provenance) is dict and provenance, 'Caller-bound source/configuration provenance required')
        # Metadata is retained, not converted into a claim of native binary use.
        json.dumps(provenance, allow_nan=False)
        require(backend != 'actor' or isinstance(root_module, torch.nn.Module),
                'Actor requires retained real top-level HF module for actual input verification')
        self.backend, self.rank = backend, rank
        self.modules, self.root = dict(modules), root_module
        self.config = {k: bound_hf_config[k] for k in ('num_hidden_layers', 'hidden_size')}
        self.actual_config = dict(actual_fields)
        self.provenance = copy.deepcopy(provenance)
        self.anchors = [dict(a) for a in anchors if a['rank'] == rank]
        self.errors, self.handles = [], []
        self.vectors, self.points, self.outcomes, self.records = {}, {}, {}, {}
        self.step = None
        self.active = False
        self.restored = False
        self.finalized = False
        self.native_failed = False
        self.hook_calls = {name: 0 for name in self.names}
        self._finite_checked = set()

    def _defer(self, label, operation):
        try:
            return operation()
        except Exception as error:
            message = label + ': ' + type(error).__name__ + ': ' + str(error)
            if message not in self.errors:
                self.errors.append(message)
            return None

    def __enter__(self):
        require(not self.active and not self.handles and not self.finalized,
                'Nested or reused boundary observation is unsupported')
        self.active = True
        try:
            if self.backend == 'actor':
                self.handles.append(self.root.register_forward_pre_hook(
                    self._actor_root_pre_hook, with_kwargs=True))
            for name, module in self.modules.items():
                def hook(current, args, output, name=name):
                    self.hook_calls[name] += 1
                    self._defer('boundary ' + name, lambda: self._capture(name, output))
                    return None  # The exact original result object remains native.
                self.handles.append(module.register_forward_hook(hook))
        except BaseException:
            self._remove_hooks()
            self.active = False
            raise
        return self

    def _remove_hooks(self):
        for handle in reversed(self.handles):
            self._defer('remove boundary hook', handle.remove)
        self.handles = []

    def __exit__(self, exc_type, exc_value, traceback):
        self.native_failed |= exc_type is not None
        self._remove_hooks()
        self.active = False
        self.restored = not self.handles
        self.step = None
        return False  # Original native exceptions propagate unchanged.

    def clear_step(self):
        self.step = None

    def observe_vllm_step(self, window_step):
        self.step = None
        def prepare():
            require(self.backend == 'vllm' and self.active, 'vLLM observer is not installed')
            require(type(window_step) is dict, 'Original WindowObserver ModelInput mapping missing')
            selected = window_step['selected']
            selected = _list(selected) if isinstance(selected, torch.Tensor) else list(selected)
            rows = window_step['rows']
            require(selected and sorted(rows) == list(range(len(selected))),
                    'Only original n1, no prompt-logprobs, complete selected sampling rows supported')
            require(all(type(i) is int and i >= 0 for i in selected)
                    and len(set(selected)) == len(selected), 'Ambiguous selected hidden row indices')
            points = []
            for native_row, row in rows.items():
                require(row['hidden_state_index'] == selected[native_row], 'Sampler row is not its actual hidden row')
                for anchor in self.anchors:
                    if (anchor['input_row'], anchor['response_offset']) != (row['input_row'], row['response_offset']):
                        continue
                    tokens, positions = row['context_token_ids'], row['context_positions']
                    require(tokens and len(tokens) == len(positions)
                            and tokens[-1] == row['input_token'] and positions[-1] == row['input_position'],
                            'Original complete causal context/input predecessor differs')
                    points.append(dict(anchor_id=anchor['id'], index=row['hidden_state_index'],
                        context_token_ids=list(tokens), context_positions=list(positions),
                        request_id=row['request_id'], input_row=row['input_row'],
                        response_offset=row['response_offset'], execution_index=row['execution_index']))
            # The accepted WindowObserver proved all query segments, cursor==len
            # tokens and every sampling row selects its segment end-1. With no
            # prompt-logprobs/graph padding, max(selected)+1 is the native T.
            self.step = dict(kind='vllm', expected_rows=max(selected)+1,
                             points=points, actual_inputs_verified=True)
        self._defer('prepare native vllm boundary geometry', prepare)

    def prepare_actor_call(self, micro_batch, mapping, input_start, call_index):
        self.step = None
        def prepare():
            require(self.backend == 'actor' and self.active, 'Actor observer is not installed')
            require(type(call_index) is int and type(input_start) is int and 0 <= call_index < 80
                    and input_start == call_index*4, 'Original adjacent local4 call order changed')
            ids, attention, positions, responses = (micro_batch[k] for k in
                ('input_ids', 'attention_mask', 'position_ids', 'responses'))
            require(ids.ndim == 2 and ids.shape[0] == 4 and positions.shape == attention.shape == ids.shape
                    and responses.ndim == 2 and responses.shape[0] == 4,
                    'Original four-row micro input geometry changed')
            width, response_width = ids.shape[1], responses.shape[1]
            require(0 < response_width < width and torch.equal(ids[:, -response_width:], responses)
                    and torch.equal(mapping['effective_mask'], attention[:, -response_width:].bool()),
                    'Mapping is not raw response attention / exact suffix')
            valid = attention.bool()
            rmpad = mapping['remove_padding']
            require(type(rmpad) is bool, 'Original packed/dense branch missing')
            expected_ids = ids[valid].unsqueeze(0) if rmpad else ids
            expected_positions = positions[valid].unsqueeze(0) if rmpad else positions
            require(mapping['expected_logit_rows'] == expected_ids.numel(), 'Original flattened output row count changed')
            points = []
            for anchor in self.anchors:
                local = anchor['input_row'] - input_start
                offset = anchor['response_offset']
                if not 0 <= local < 4:
                    continue
                hits = torch.nonzero((mapping['rows'] == local)
                    & (mapping['response_positions'] == offset), as_tuple=False).flatten()
                if not len(hits):
                    continue  # Only actual native early-finish outcome may explain it later.
                require(len(hits) == 1, 'Ambiguous original actor anchor predictor')
                hit = int(hits[0]); physical = width-response_width-1+offset
                require(bool(valid[local, physical]) and bool(valid[local, physical+1]),
                        'Actor anchor predecessor/target is padding')
                expected_index = (int(valid.flatten().long().cumsum(0)[local*width+physical])-1
                                  if rmpad else local*width+physical)
                require(int(mapping['logit_indices'][hit]) == expected_index
                        and int(mapping['labels'][hit]) == int(responses[local, offset]),
                        'Original physical predecessor/rolled label map differs')
                prefix_end = width-response_width+offset
                prefix_valid = valid[local, :prefix_end]
                points.append(dict(anchor_id=anchor['id'], index=expected_index,
                    input_row=anchor['input_row'], response_offset=offset,
                    context_token_ids=_list(ids[local, :prefix_end][prefix_valid]),
                    context_positions=_list(positions[local, :prefix_end][prefix_valid]),
                    target_token_id=int(responses[local, offset]), call_index=call_index))
            self.step = dict(kind='actor', expected_rows=expected_ids.numel(), points=points,
                actual_inputs_verified=False, expected_ids=expected_ids,
                expected_positions=expected_positions, expected_attention=None if rmpad else attention,
                expected_boundary_shape=tuple(expected_ids.shape)+(self.config['hidden_size'],))
        self._defer('prepare original actor boundary geometry', prepare)

    def _actor_root_pre_hook(self, module, args, kwargs):
        def verify():
            require(self.step is not None and self.step['kind'] == 'actor', 'Actor original-call mapping missing')
            if not self.step['points']:
                self.step['actual_inputs_verified'] = True
                return
            require(not args and kwargs.get('use_cache') is False,
                    'Original keyword-only actor forward / use_cache=False changed')
            for key, expected in [('input_ids', self.step['expected_ids']),
                                  ('position_ids', self.step['expected_positions'])]:
                actual = kwargs.get(key)
                require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                        and actual.dtype == expected.dtype and torch.equal(actual, expected),
                        'Actual actor input differs from original micro packing: '+key)
            attention = kwargs.get('attention_mask')
            expected = self.step['expected_attention']
            require(attention is None if expected is None else
                    isinstance(attention, torch.Tensor) and torch.equal(attention, expected),
                    'Actual original actor attention branch changed')
            self.step['actual_inputs_verified'] = True
        self._defer('actual HF root inputs', verify)
        return None

    def _capture(self, boundary, output):
        if self.step is None or not self.step['points']:
            return
        require(self.step['actual_inputs_verified'], 'Actual original input packing was not verified')
        if self.backend == 'vllm' and boundary not in ('embedding', 'layers.00.input_layernorm'):
            require(type(output) is tuple and len(output) == 2
                    and all(isinstance(x, torch.Tensor) for x in output),
                    'Original fused RMSNorm must return actual normalized tensor/residual tuple')
            value = output[0]  # Do not add hidden_states+residual or capture block output.
        else:
            require(isinstance(output, torch.Tensor), 'Original embedding/HF/first-vLLM norm output must be Tensor')
            value = output
        if self.backend == 'actor':
            require(value.ndim == 3 and tuple(value.shape) == self.step['expected_boundary_shape'],
                    'Actual actor boundary must be [B,T,bound hidden]')
            require(value.shape[0]*value.shape[1] == self.step['expected_rows'], 'Actual actor token geometry changed')
        else:
            require(value.ndim == 2 and value.shape == (self.step['expected_rows'], self.config['hidden_size']),
                    'Actual native boundary must be [T,bound hidden]')
        require(value.dtype.is_floating_point, 'Native normalization output must have floating dtype')
        for point in self.step['points']:
            anchor_id = point['anchor_id']
            require(boundary not in self.vectors.get(anchor_id, {}), 'Repeated native anchor boundary occurrence')
            index = point['index']
            if self.backend == 'actor':
                row, token = divmod(index, value.shape[1]); vector = value[row, token]
            else:
                vector = value[index]
            # Real vLLM fused normalization mutates residual storage later.
            # A detached view is insufficient: clone now on the original stream.
            snapshot = vector.detach().clone()
            require(snapshot.shape == (self.config['hidden_size'],), 'Actual boundary vector shape changed')
            self.vectors.setdefault(anchor_id, {})[boundary] = snapshot
            saved = self.points.setdefault(anchor_id, copy.deepcopy(point))
            require(saved == point, 'Native anchor context/coordinate changed across boundaries')
        require(sum(len(v) for v in self.vectors.values()) <= 74*len(self.anchors),
                'Fixed native boundary vector budget exceeded')

    def check_errors(self):
        """Call only after matched original native completion; no collectives here."""
        for anchor_id, values in self.vectors.items():
            for boundary, value in values.items():
                key = (anchor_id, boundary)
                if key in self._finite_checked:
                    continue
                def verify(value=value, key=key):
                    require(bool(torch.isfinite(value).all()), 'Actual captured boundary vector nonfinite')
                    self._finite_checked.add(key)
                self._defer('finite '+anchor_id+'/'+boundary, verify)
        require(not self.errors, '; '.join(self.errors))

    def finalize(self, window_anchor_outcomes):
        """Use original WindowObserver explicit sampled target / early stop ledger."""
        def finish():
            require(not self.active and self.restored and not self.native_failed and not self.finalized,
                    'Original window/calls and hook restoration must complete successfully')
            require(type(window_anchor_outcomes) is list, 'Original final window anchors missing')
            outcomes = {r['id']: r for r in window_anchor_outcomes}
            require(len(outcomes) == len(window_anchor_outcomes)
                    and set(outcomes) == {a['id'] for a in self.anchors}, 'Original local final anchor set changed')
            records = {}
            for anchor in self.anchors:
                aid = anchor['id']; outcome = outcomes[aid]
                require((outcome['input_row'], outcome['response_offset']) ==
                        (anchor['input_row'], anchor['response_offset']), 'Final anchor coordinate changed')
                if outcome['status'] == 'absent_native_early_finish':
                    require(anchor['response_offset'] >= outcome['true_response_length']
                            and outcome['finish_reason'] == 'stop'
                            and aid not in self.vectors and aid not in self.points,
                            'Missing boundary is not a real native early finish')
                    records[aid] = dict(anchor, status='absent_native_early_finish',
                        request_id=outcome['request_id'], true_response_length=outcome['true_response_length'],
                        finish_reason=outcome['finish_reason'], captured_boundary_count=0)
                    continue
                require(outcome['status'] == 'observed' and aid in self.points
                        and list(self.vectors.get(aid, {})) == self.names,
                        'Present anchor does not have all74 original boundaries')
                point = self.points[aid]
                require(point['context_token_ids'] == outcome['context_token_ids']
                        and point['context_positions'] == outcome['context_positions'],
                        'Actual captured context or positions differ from native sampler')
                require(self.backend != 'vllm' or point['request_id'] == outcome['request_id'],
                        'Captured native request changed')
                require(self.backend != 'actor' or point['target_token_id'] == outcome['target_token_id'],
                        'Captured actor target differs from actual sampled token')
                identities = {name: _identity(value) for name, value in self.vectors[aid].items()}
                context = dict(token_ids=point['context_token_ids'], positions=point['context_positions'],
                               target_token_id=outcome['target_token_id'])
                context_sha = hashlib.sha256(json.dumps(context, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()
                records[aid] = dict(anchor, status='ACTUAL_NATIVE_NORMALIZATION_BOUNDARIES_OBSERVED',
                    request_id=outcome['request_id'], target_token_id=outcome['target_token_id'],
                    context_length=len(context['token_ids']), context_identity_sha256=context_sha,
                    captured_boundary_count=74, boundary_tensor_identities=identities,
                    index=point['index'], index_scope='Actual predicting input token, not sampled target position',
                    embedding_and_normalization_outputs_only=True, residual_reconstructed=False)
            self.records = records
            self.outcomes = copy.deepcopy(outcomes)
            self.check_errors()
            self.finalized = True
        self._defer('final native boundary coverage and pairing', finish)
        return self.receipt()

    def receipt(self):
        complete = self.finalized and self.restored and not self.active and not self.errors and not self.native_failed
        return dict(status='PASSIVE_NATIVE_BOUNDARIES_OBSERVED_NO_ACCEPTANCE' if complete else
                    'PASSIVE_NATIVE_BOUNDARIES_REFUSED', observer_evidence_complete=complete,
            backend=self.backend, rank=self.rank, bound_model_fields=dict(self.config),
            actual_model_fields=dict(self.actual_config), provenance=copy.deepcopy(self.provenance),
            actual_loaded_binary_identity_proven=False,
            source_validation_scope='Owner-supplied bound method/config metadata, not a binary dispatch proof',
            records=copy.deepcopy(self.records), errors=list(self.errors), hooks_restored=self.restored,
            native_exception_seen=self.native_failed, hook_calls=dict(self.hook_calls),
            captured_vector_count=sum(len(v) for v in self.vectors.values()),
            maximum_vectors_per_rank=74*len(self.anchors), maximum_vectors_global_per_backend=74*9,
            observer_replaced_native_outputs=False, observer_reconstructed_residual=False,
            new_model_forwards=0, probability_accepted=False, training_accepted=False)

    def tensor_evidence(self):
        require(self.finalized and not self.errors, 'Cannot publish refused or incomplete native boundary tensors')
        return {aid: dict(values) for aid, values in self.vectors.items()}


def compare_boundary_observers(actor, vllm):
    """Observe finite differences; no equality tolerance or probability PASS."""
    require(actor.backend == 'actor' and vllm.backend == 'vllm' and actor.rank == vllm.rank,
            'Actual same-rank actor/vllm observers required')
    require(actor.receipt()['observer_evidence_complete'] and vllm.receipt()['observer_evidence_complete'],
            'Both restored native observations must be complete')
    require(actor.names == vllm.names and set(actor.records) == set(vllm.records), 'Boundary plans differ')
    comparisons = []
    for aid in actor.records:
        a, b = actor.records[aid], vllm.records[aid]
        require(a['status'] == b['status'], 'Backend presence/early-finish outcomes differ')
        if a['status'] == 'absent_native_early_finish':
            comparisons.append(dict(anchor_id=aid, status=a['status']))
            continue
        require(a['request_id'] == b['request_id'] and a['context_identity_sha256'] == b['context_identity_sha256']
                and a['target_token_id'] == b['target_token_id'], 'Actual paired causal identity differs')
        observations = []
        for name in actor.names:
            av, bv = actor.vectors[aid][name], vllm.vectors[aid][name]
            require(av.shape == bv.shape and av.dtype == bv.dtype, 'Paired actual boundary shape/dtype differs')
            delta = av.detach().double().cpu()-bv.detach().double().cpu()
            require(bool(torch.isfinite(delta).all()), 'Nonfinite boundary comparison')
            observations.append(dict(boundary=name, count=delta.numel(),
                max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                rms=float(delta.square().mean().sqrt()),
                recorded_identity_equal=a['boundary_tensor_identities'][name]==b['boundary_tensor_identities'][name]))
        first = next((r['boundary'] for r in observations if r['max_abs'] != 0), None)
        comparisons.append(dict(anchor_id=aid, status='SAME_CONTEXT_NATIVE_BOUNDARY_DIFFERENCES_OBSERVED',
            context_identity_sha256=a['context_identity_sha256'], target_token_id=a['target_token_id'],
            first_observed_different_boundary=first, boundaries=observations,
            unique_kernel_cause_proven=False, probability_accepted=False))
    return dict(rank=actor.rank, anchors=comparisons, probability_accepted=False, training_accepted=False,
        scope='First observed semantic boundary only; intervening attention, residual and normalization are not isolated kernel causes')
