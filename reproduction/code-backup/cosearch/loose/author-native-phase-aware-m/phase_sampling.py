"""Transparent correction of original phase sampling propagation and undefined metrics.

This changes the effective training sampling distribution. It is not a pure-default
author reproduction. No reward, group filtering, batch tensor or optimizer change.
"""
import functools
import hashlib
import inspect
import numbers
import textwrap
from pathlib import Path

LOOP_SHA = 'fc63577b7b965f41e6a0de19ff7f57387c9c83edd1f4d275a45194271d7aeabd'
MANAGER_SHA = '8564d2e8536a2d78fd268a260c8f5a8ef68b0f60c25483764d7e7e2fa2ad03c8'
OLD = '''create_kwargs["reranker_sampling_params"] = self.tools[tool_name].config.get(
                "reranker_sampling_params", {"temperature": 0.0, "top_p": 1.0}
            )'''
NEW = '''create_kwargs.setdefault("reranker_sampling_params", self.tools[tool_name].config.get(
                "reranker_sampling_params", {"temperature": 0.0, "top_p": 1.0}
            ))'''


def corrected_sampling_method(original):
    module = inspect.getmodule(original)
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == LOOP_SHA
    source = inspect.getsource(original)
    assert source.count(OLD) == 1, 'Only the reviewed original sampling override may change'
    changed = source.replace(OLD, NEW)
    namespace = dict(original.__globals__)
    exec(compile(textwrap.dedent(changed), str(Path(__file__).resolve()), 'exec'), namespace)
    corrected = namespace[original.__name__]
    functools.update_wrapper(corrected, original)
    return corrected, dict(original_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          corrected_method_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                          correction='honor_original_worker_phase_sampling_in_request_kwargs',
                          effective_sampling_distribution_changed=True)


def install_sampling(record):
    from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop
    if getattr(CoSearchAgentLoop, '_native_phase_sampling_corrected', False):
        return
    method, metadata = corrected_sampling_method(CoSearchAgentLoop._handle_processing_tools_state)
    CoSearchAgentLoop._handle_processing_tools_state = method
    CoSearchAgentLoop._native_phase_sampling_corrected = True
    record('phase_sampling_correction_installed', metadata)


def undefined_group_means(metrics):
    """Identify exactly the original aggregator's unsupported group-size means."""
    nonempty = [m for m in metrics if m]
    undefined = {}
    for key in {key for m in nonempty for key in m}:
        if 'avg_group_size' in key:
            support_key = key.replace('avg_group_size', 'num_groups')
            weights = [m.get(support_key, 1.0) for m in nonempty]
            if sum(weights) == 0:
                undefined[key] = dict(values=[m.get(key, 0.0) for m in nonempty],
                                      weights=weights, support_count=0,
                                      status='undefined_no_group_support')
    return undefined


def aggregate_supported(original, instance, metrics, record):
    undefined = undefined_group_means(metrics)
    # Keep exact original metrics intact. Copies affect metadata only.
    forwarded = [{k: v for k, v in m.items() if k not in undefined} for m in metrics] if undefined else metrics
    result = original(instance, forwarded)
    record('original_ranker_metric_support', json_metadata(dict(original_metrics=metrics,
           qualified_groups=sum(m.get('post_filter/num_groups', 0) for m in metrics),
           undefined_means=undefined, numeric_undefined_values_omitted=True)))
    return result


def json_metadata(value):
    """Convert evidence copies only; preserve original NumPy metric values/data."""
    if isinstance(value, dict):
        return {str(k): json_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_metadata(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError('Unsupported original metric evidence type: '+type(value).__name__)


def install_aggregation(manager, record):
    cls = type(manager)
    module = inspect.getmodule(cls._aggregate_reranker_metrics)
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == MANAGER_SHA
    original = manager._aggregate_reranker_metrics
    support = []

    def aggregate(metrics):
        support[:] = [sum(m.get('post_filter/num_groups', 0) for m in metrics)]
        return aggregate_supported(lambda _, value: original(value), manager, metrics, record)

    manager._aggregate_reranker_metrics = aggregate
    return original, support


def require_training_support(support):
    assert len(support) == 1, 'Original complete ranker metrics must be observed before PPO'
    assert support[0] > 0, 'No original qualified ranker groups: reject before either PPO, including fallback'
