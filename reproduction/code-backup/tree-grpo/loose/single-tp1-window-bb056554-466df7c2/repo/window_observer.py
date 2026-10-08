"""Passive, instance-local observers for one original vLLM generation window.

No vLLM imports, global patches, tensor replacement, or distribution changes.
The caller records actual request registration and actual native ModelInput.
"""
import copy
import math


def _ids(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def _logprob_dict(value):
    if value is None:
        return None
    # JSON objects require string keys; use the exact decimal native token ID.
    return {str(int(token)): {"logprob": float(item.logprob),
                         "rank": getattr(item, "rank", None),
                         "decoded_token": getattr(item, "decoded_token", None)}
            for token, item in value.items()}


class WindowObserver:
    """Anchors are {id, rank, input_row, response_offset}; at most nine.

    Call observe_prompts() before generation, register_request() with the actual
    engine.add_request arguments, and observe_model_input() before execute_model.
    Call observe_generated() on raw final RequestOutput objects before facade pad.
    Observer failures accumulate; original native method failures still propagate.
    """
    def __init__(self, runtime_model, installed_sha_map, anchors, rank):
        if len(anchors) > 9:
            raise ValueError("At most nine anchors are permitted")
        self.model = runtime_model
        self.installed_sha_map = copy.deepcopy(installed_sha_map)
        self.rank = int(rank)
        self.anchors = []
        all_ids = set()
        for anchor in anchors:
            if set(anchor) != {"id", "rank", "input_row", "response_offset"}:
                raise ValueError("Unknown anchor schema")
            item = dict(anchor)
            if not isinstance(item["id"], str) or not item["id"] or item["id"] in all_ids:
                raise ValueError("Anchor IDs must be unique nonempty strings")
            all_ids.add(item["id"])
            for field in ("rank", "input_row", "response_offset"):
                if type(item[field]) is not int or item[field] < 0:
                    raise ValueError("Anchor coordinates must be nonnegative integers")
            if item["rank"] == self.rank:
                self.anchors.append(item)
        self.errors = []
        self.expected_prompts = None
        self.requests = {}
        self.input_rows = {}
        self.anchor_records = {}
        self._tensors = {}
        self._ledger = {}
        self._context_input_ledger = {}
        self.generated = []
        self._saved = []
        self._step = None
        self._restored = False
        self._generated_seen = False
        self.counts = {"model_input": 0, "head": 0, "sampler": 0,
                       "request_registration": 0, "generated_observation": 0}

    def _error(self, label, exc):
        message = f"{label}: {type(exc).__name__}: {exc}"
        if message not in self.errors:
            self.errors.append(message)

    def _observe(self, label, callback):
        try:
            return callback()
        except Exception as exc:
            self._error(label, exc)
            return None

    def observe_prompts(self, prompt_token_ids):
        def capture():
            if self.expected_prompts is not None:
                raise ValueError("Prompt registration must happen once")
            values = [_ids(prompt) for prompt in prompt_token_ids]
            if not values or any(not prompt for prompt in values):
                raise ValueError("Expected prompts must be nonempty token lists")
            if any(a["input_row"] >= len(values) for a in self.anchors):
                raise ValueError("Anchor input row exceeds registered prompts")
            self.expected_prompts = values
        self._observe("observe_prompts", capture)

    def register_request(self, request_id, prompt_token_ids, input_row):
        def capture():
            self.counts["request_registration"] += 1
            if self.expected_prompts is None:
                raise ValueError("observe_prompts must precede actual requests")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("Actual request ID must be a nonempty string")
            if type(input_row) is not int or not 0 <= input_row < len(self.expected_prompts):
                raise ValueError("Actual request input row is outside registered prompts")
            if request_id in self.requests or input_row in self.input_rows:
                raise ValueError("Request ID and input row must each be unique")
            prompt = _ids(prompt_token_ids)
            if prompt != self.expected_prompts[input_row]:
                raise ValueError("Actual registered prompt differs from observed input row")
            self.requests[request_id] = {"input_row": input_row, "prompt_token_ids": prompt}
            self.input_rows[input_row] = request_id
        self._observe("register_request", capture)

    def _bind(self, obj, name, wrapper_factory):
        original = getattr(obj, name)
        local = name in vars(obj)
        self._saved.append((obj, name, local, vars(obj).get(name)))
        setattr(obj, name, wrapper_factory(original))

    def __enter__(self):
        def install():
            def head_wrapper(original):
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.counts["head"] += 1
                    self._observe("head_output", lambda: self._capture_logits(result, "head_logits"))
                    return result
                return wrapped
            def sampler_wrapper(original):
                def wrapped(*args, **kwargs):
                    logits = kwargs.get("logits", args[0] if args else None)
                    metadata = kwargs.get("sampling_metadata", args[1] if len(args) > 1 else None)
                    self.counts["sampler"] += 1
                    self._observe("sampler_input", lambda: self._capture_sampler_input(logits, metadata))
                    result = original(*args, **kwargs)
                    self._observe("sampler_output", lambda: self._capture_sampler_output(result))
                    return result
                return wrapped
            self._bind(self.model.logits_processor, "_get_logits", head_wrapper)
            self._bind(self.model.sampler, "forward", sampler_wrapper)
        self._observe("install_hooks", install)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for obj, name, local, value in reversed(self._saved):
            def restore(obj=obj, name=name, local=local, value=value):
                if local:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
            self._observe("restore_hooks", restore)
        self._saved.clear()
        self._restored = True
        return False

    def observe_model_input(self, model_input):
        self._step = None
        def capture():
            self.counts["model_input"] += 1
            metadata = model_input.sampling_metadata
            tokens, positions = _ids(model_input.input_tokens), _ids(model_input.input_positions)
            selected = _ids(metadata.selected_token_indices)
            request_map = model_input.request_ids_to_seq_ids
            if not isinstance(request_map, dict) or not request_map:
                raise ValueError("Native ModelInput lacks request-to-sequence mapping")
            reverse = {}
            for request_id, seq_ids in request_map.items():
                if request_id not in self.requests:
                    raise ValueError("ModelInput request was not actually registered")
                for seq_id in seq_ids:
                    seq_id = int(seq_id)
                    if seq_id in reverse:
                        raise ValueError("Native sequence maps to multiple requests")
                    reverse[seq_id] = request_id
            query_lens = [int(q) for q in model_input.query_lens]
            if len(tokens) != len(positions):
                raise ValueError("Native tokens and positions differ in length")
            rows, snapshots, cursor, ordinal = {}, [], 0, 0
            for group_index, group in enumerate(metadata.seq_groups):
                seq_ids = [int(s) for s in group.seq_ids]
                sample_rows = [int(s) for s in group.sample_indices]
                prompt_rows = [int(s) for s in group.prompt_logprob_indices]
                if set(sample_rows) & set(prompt_rows):
                    raise ValueError("Prompt and sample rows overlap")
                if sample_rows and len(sample_rows) != len(seq_ids):
                    raise ValueError("Only original single-step one-row-per-parent sampling is supported")
                settings = {name: getattr(group.sampling_params, name, None) for name in
                            ("n", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                             "frequency_penalty", "repetition_penalty", "min_tokens", "max_tokens",
                             "logprobs", "prompt_logprobs", "seed")}
                if settings["n"] != 1 or getattr(group.sampling_params, "logits_processors", None):
                    raise ValueError("Observer supports original n=1 without custom logits processors")
                if getattr(metadata, "skip_sampler_cpu_output", False):
                    raise ValueError("Deferred sampler CPU output is unsupported")
                snapshots.append({"seq_ids": seq_ids, "sample_rows": sample_rows,
                                  "prompt_rows": prompt_rows})
                for parent_index, seq_id in enumerate(seq_ids):
                    request_id = reverse[seq_id]
                    request = self.requests[request_id]
                    data = group.seq_data[seq_id]
                    prompt, prior = _ids(data.prompt_token_ids), _ids(data.output_token_ids)
                    if prompt != request["prompt_token_ids"]:
                        raise ValueError("SequenceData prompt differs from registered actual request")
                    context = prompt + prior
                    query_len = query_lens[ordinal]
                    ordinal += 1
                    if query_len <= 0 or query_len > len(context):
                        raise ValueError("Unknown native query length")
                    end = cursor + query_len
                    if tokens[cursor:end] != context[-query_len:] or positions[cursor:end] != list(range(len(context)-query_len, len(context))):
                        raise ValueError("Native token/position segment differs from SequenceData context")
                    history = self._context_input_ledger.setdefault(seq_id, {"request_id": request_id, "inputs": {}})
                    if history["request_id"] != request_id:
                        raise ValueError("Observed native sequence changed its actual request binding")
                    observed = history["inputs"]
                    for context_index, native_index in enumerate(range(cursor, end), start=len(context)-query_len):
                        actual = (tokens[native_index], positions[native_index])
                        if context_index in observed and observed[context_index] != actual:
                            raise ValueError("Repeated native context index changed token or position")
                        observed[context_index] = actual
                    if sample_rows:
                        if not group.is_prompt and query_len != 1:
                            raise ValueError("Multi-token decode mapping is unsupported")
                        row = sample_rows[parent_index]
                        hidden = selected[row]
                        if hidden != end-1 or row in rows:
                            raise ValueError("Sample row does not select the causal last input token")
                        if any(index not in observed for index in range(len(context))):
                            raise ValueError("Unobserved cached prefix prevents actual full-context position evidence")
                        if any(observed[index][0] != token or observed[index][1] != index for index, token in enumerate(context)):
                            raise ValueError("Accumulated actual native inputs differ from current complete context")
                        context_positions = [observed[index][1] for index in range(len(context))]
                        rows[row] = {"request_id": request_id, "input_row": request["input_row"],
                                     "seq_id": seq_id, "group_index": group_index,
                                     "response_offset": len(prior), "native_logits_row": row,
                                     "hidden_state_index": hidden, "input_token": tokens[hidden],
                                     "input_position": positions[hidden], "context_token_ids": context,
                                     "context_positions": context_positions,
                                     "prompt_token_ids": prompt, "prior_response_token_ids": prior,
                                     "sampling_params": settings,
                                     "execution_index": self.counts["model_input"]}
                    cursor = end
            if ordinal != len(query_lens) or cursor != len(tokens):
                raise ValueError("Unmapped input query segments or graph padding")
            if sorted(r for snap in snapshots for r in snap["sample_rows"]+snap["prompt_rows"]) != list(range(len(selected))):
                raise ValueError("Selected logits rows are not covered exactly")
            self._step = {"metadata": metadata, "selected": selected, "groups": snapshots, "rows": rows}
        self._observe("observe_model_input", capture)

    def _capture_logits(self, logits, key):
        if self._step is None or logits is None or len(logits.shape) != 2:
            raise ValueError("No stable ModelInput or rank-local logits tensor")
        if logits.shape[0] != len(self._step["selected"]):
            raise ValueError("Logits rows differ from native selected indices")
        for row, record in self._step["rows"].items():
            for anchor in self.anchors:
                if (anchor["input_row"], anchor["response_offset"]) != (record["input_row"], record["response_offset"]):
                    continue
                anchor_id = anchor["id"]
                if key in self._tensors.get(anchor_id, {}):
                    raise ValueError("Anchor was observed more than once")
                if anchor_id not in self.anchor_records:
                    self.anchor_records[anchor_id] = dict(anchor, **copy.deepcopy(record), status="observed")
                clone = logits[row].detach().clone()
                self._tensors.setdefault(anchor_id, {})[key] = clone
                self.anchor_records[anchor_id][key+"_dtype"] = str(clone.dtype)
                self.anchor_records[anchor_id][key+"_shape"] = list(clone.shape)

    def _capture_sampler_input(self, logits, metadata):
        if self._step is None or metadata is not self._step["metadata"]:
            raise ValueError("Sampler metadata differs from observed native ModelInput")
        if _ids(metadata.selected_token_indices) != self._step["selected"]:
            raise ValueError("Cached selected indices changed before sampler")
        for group, saved in zip(metadata.seq_groups, self._step["groups"]):
            if _ids(group.seq_ids) != saved["seq_ids"] or _ids(group.sample_indices) != saved["sample_rows"] or _ids(group.prompt_logprob_indices) != saved["prompt_rows"]:
                raise ValueError("Cached group row mapping changed before sampler")
        for row, record in self._step["rows"].items():
            group = metadata.seq_groups[record["group_index"]]
            data = group.seq_data[record["seq_id"]]
            if _ids(data.prompt_token_ids)+_ids(data.output_token_ids) != record["context_token_ids"]:
                raise ValueError("Sequence context changed between model execution and sampler")
        self._capture_logits(logits, "sampler_input_logits")

    def _capture_sampler_output(self, result):
        if self._step is None or len(result.outputs) != len(self._step["groups"]):
            raise ValueError("Native sampler output group mapping is unavailable")
        for group_index, output in enumerate(result.outputs):
            saved = self._step["groups"][group_index]
            if len(output.samples) != len(saved["sample_rows"]):
                raise ValueError("Native sampler branches or sample count differs")
            for sample in output.samples:
                parent = saved["seq_ids"].index(int(sample.parent_seq_id))
                row = saved["sample_rows"][0]+parent
                if row != saved["sample_rows"][parent]:
                    raise ValueError("Native get_logprobs parent-to-row mapping is not contiguous")
                record = self._step["rows"][row]
                token = int(sample.output_token)
                probability = float(sample.logprobs[token].logprob)
                if not math.isfinite(probability):
                    raise ValueError("Native sampled target logprob is nonfinite")
                ledger_key = (record["request_id"], record["response_offset"])
                if ledger_key in self._ledger:
                    raise ValueError("Native sample occurrence repeated")
                self._ledger[ledger_key] = {"target_token_id": token, "sampled_logprob": probability}
                for anchor in self.anchors:
                    if (anchor["input_row"], anchor["response_offset"]) == (record["input_row"], record["response_offset"]):
                        anchor_record = self.anchor_records[anchor["id"]]
                        anchor_record.update(target_token_id=token, sampled_logprob=probability,
                                             sampled_logprobs=_logprob_dict(sample.logprobs))

    def observe_generated(self, request_outputs):
        def capture():
            self.counts["generated_observation"] += 1
            if self._generated_seen:
                raise ValueError("Raw RequestOutput list must be observed once")
            self._generated_seen = True
            seen = set()
            for request in request_outputs:
                request_id = request.request_id
                registered = self.requests[request_id]
                if request_id in seen:
                    raise ValueError("Repeated final RequestOutput request")
                seen.add(request_id)
                prompt = _ids(request.prompt_token_ids)
                if prompt != registered["prompt_token_ids"] or not request.finished or len(request.outputs) != 1:
                    raise ValueError("Raw RequestOutput identity, finality, or n=1 differs")
                completion = request.outputs[0]
                token_ids = _ids(completion.token_ids)
                if completion.logprobs is None or len(completion.logprobs) != len(token_ids):
                    raise ValueError("Unpadded completion tokens/logprobs differ in true length")
                probabilities = [_logprob_dict(lp) for lp in completion.logprobs]
                for offset, token in enumerate(token_ids):
                    ledger = self._ledger[(request_id, offset)]
                    actual = probabilities[offset][str(token)]["logprob"]
                    if ledger != {"target_token_id": token, "sampled_logprob": actual}:
                        raise ValueError("Raw completion differs from actual sampler occurrence")
                if len([key for key in self._ledger if key[0] == request_id]) != len(token_ids):
                    raise ValueError("Sampler occurrences exceed final unpadded completion")
                self.generated.append({"request_id": request_id, "input_row": registered["input_row"],
                                       "prompt_token_ids": prompt,
                                       "prompt_logprobs": None if request.prompt_logprobs is None else [_logprob_dict(lp) for lp in request.prompt_logprobs],
                                       "finished": bool(request.finished), "completion_index": int(completion.index),
                                       "token_ids": token_ids, "logprobs": probabilities,
                                       "true_response_length": len(token_ids),
                                       "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                                       "cumulative_logprob": completion.cumulative_logprob})
            if seen != set(self.requests) or len(self.requests) != len(self.expected_prompts or []):
                raise ValueError("Registered request rows and final native outputs are incomplete")
        self._observe("observe_generated", capture)

    def receipt(self):
        if self.expected_prompts is None or not self.generated or len(self.requests) != len(self.expected_prompts):
            self._error("receipt", ValueError("Prompt/request/final-output coverage is incomplete"))
        if not self.counts["model_input"] or self.counts["model_input"] != self.counts["head"] or self.counts["head"] != self.counts["sampler"]:
            self._error("receipt", ValueError("Native execution/head/sampler observation counts differ"))
        records = copy.deepcopy(self.anchor_records)
        by_row = {record["input_row"]: record for record in self.generated}
        for anchor in self.anchors:
            anchor_id = anchor["id"]
            if anchor_id in records:
                if set(self._tensors.get(anchor_id, {})) != {"head_logits", "sampler_input_logits"} or "target_token_id" not in records[anchor_id]:
                    self._error("receipt", ValueError(f"Incomplete anchor {anchor_id}"))
                continue
            final = by_row.get(anchor["input_row"])
            if final and anchor["response_offset"] >= final["true_response_length"] and final["finish_reason"] == "stop":
                records[anchor_id] = dict(anchor, status="absent_native_early_finish",
                                         request_id=final["request_id"], true_response_length=final["true_response_length"],
                                         finish_reason=final["finish_reason"], stop_reason=final["stop_reason"])
            else:
                self._error("receipt", ValueError(f"Unexplained missing anchor {anchor_id}"))
                records[anchor_id] = dict(anchor, status="missing")
        complete = self._restored and self._generated_seen and not self.errors
        return {"status": "PASSIVE_WINDOW_OBSERVATION_COMPLETE" if complete else "PASSIVE_WINDOW_OBSERVATION_REFUSED",
                "observer_evidence_complete": complete, "rank": self.rank,
                "installed_sha_map": copy.deepcopy(self.installed_sha_map), "counts": dict(self.counts),
                "hooks_restored": self._restored, "errors": list(self.errors),
                "requests": copy.deepcopy(self.requests), "anchors": list(records.values()),
                "generated": copy.deepcopy(self.generated), "training_accepted": False,
                "probability_accepted": False, "observer_replaced_native_outputs": False,
                "observer_changed_sampling_parameters": False}

    def tensor_evidence(self):
        """Detached private clones; caller may save these only on target runtime."""
        return {anchor_id: dict(values) for anchor_id, values in self._tensors.items()}
