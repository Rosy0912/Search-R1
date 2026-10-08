"""Scoped, transparent recording around the frozen SearchAgent.run.

No generation options are added, no output is trimmed, no HTTP call is retried,
and all original return objects are returned unchanged. All captures are CPU
copies, made after the original operation. This module imports no torch.
"""
from contextlib import contextmanager
import hashlib
import json
import threading
import time

_LOCK = threading.Lock()


def snapshot(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return snapshot(value.to_dict())
    # StoppingCriteria objects must never be altered or stringified with an
    # address as a science identity. Record class plus their actual stop state.
    attributes = {}
    for name in ("start", "max_length", "eos_token_id"):
        if hasattr(value, name):
            attributes[name] = snapshot(getattr(value, name))
    return dict(python_type=type(value).__module__ + "." + type(value).__qualname__, attributes=attributes)


class TokenizerProxy:
    def __init__(self, tokenizer, events):
        object.__setattr__(self, "_original", tokenizer)
        object.__setattr__(self, "_events", events)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        self._events.append(dict(kind="tokenizer_call", unix=time.time(), args=snapshot(args),
                                 kwargs=snapshot(kwargs), input_ids=snapshot(result.input_ids)))
        return result


def _restore_attribute(obj, name, existed, original):
    if existed:
        setattr(obj, name, original)
    else:
        delattr(obj, name)


@contextmanager
def capture_native(agent, requests_module):
    """Yield trace; call agent.run in this scope. Always restore on failure."""
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Native capture cannot overlap another scoped hook")
    events = []
    trace = dict(schema="transparent-native-search-capture-v1", started_unix=time.time(),
                 generation_options_modified=False, return_objects_modified=False,
                 http_retries_added=0, processed_scores_recorded=False, events=events)
    model = agent.model
    tokenizer = agent.tokenizer
    own_generate = "generate" in vars(model)
    old_attribute = vars(model).get("generate")
    original_generate = model.generate
    original_post = requests_module.post
    processor_owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    processor_present = hasattr(processor_owner, "_get_logits_processor")
    own_processor = "_get_logits_processor" in vars(processor_owner)
    old_processor_attribute = vars(processor_owner).get("_get_logits_processor")
    original_processor = getattr(processor_owner, "_get_logits_processor", None)

    def generate(*args, **kwargs):
        event = dict(kind="model_generate", started_unix=time.time(), args=snapshot(args), kwargs=snapshot(kwargs))
        events.append(event)
        output = original_generate(*args, **kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        event.update(ended_unix=time.time(), output_type=type(output).__module__ + "." + type(output).__qualname__,
                     output_ids=snapshot(sequences))
        return output

    def post(*args, **kwargs):
        event = dict(kind="retriever_post", started_unix=time.time(), args=snapshot(args), kwargs=snapshot(kwargs))
        events.append(event)
        response = original_post(*args, **kwargs)
        body = response.content
        event.update(ended_unix=time.time(), status_code=response.status_code,
                     response_bytes=len(body), response_sha256=hashlib.sha256(body).hexdigest())
        # json.loads existing response bytes avoids an additional HTTP request,
        # preserves Response identity, and preserves native raise_for_status.
        try:
            event["response_json"] = json.loads(body)
        except (ValueError, UnicodeError):
            event["response_text"] = body.decode(errors="replace")
        return response

    def processors(*args, **kwargs):
        result = original_processor(*args, **kwargs)
        events.append(dict(kind="actual_hf_logits_processors", unix=time.time(), args=snapshot(args), kwargs=snapshot(kwargs),
                           processor_list_type=type(result).__module__ + "." + type(result).__qualname__,
                           processors=[dict(python_type=type(item).__module__ + "." + type(item).__qualname__,
                                            attributes=snapshot(vars(item))) for item in result],
                           returned_original_list_object=True, per_token_processed_scores_recorded=False))
        return result

    patched_generate = patched_tokenizer = patched_post = patched_processor = False
    try:
        model.generate = generate
        patched_generate = True
        agent.tokenizer = TokenizerProxy(tokenizer, events)
        patched_tokenizer = True
        requests_module.post = post
        patched_post = True
        if processor_present:
            processor_owner._get_logits_processor = processors
            patched_processor = True
        yield trace
    except BaseException as error:
        trace["exception"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if patched_post:
            requests_module.post = original_post
        if patched_processor:
            _restore_attribute(processor_owner, "_get_logits_processor", own_processor, old_processor_attribute)
        if patched_tokenizer:
            agent.tokenizer = tokenizer
        if patched_generate:
            _restore_attribute(model, "generate", own_generate, old_attribute)
        trace["ended_unix"] = time.time()
        trace["hooks_restored"] = True
        _LOCK.release()
