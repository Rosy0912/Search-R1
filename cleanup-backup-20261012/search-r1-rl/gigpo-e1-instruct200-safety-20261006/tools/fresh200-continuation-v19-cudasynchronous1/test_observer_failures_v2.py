"""UNEXECUTED CPU fault injections. No model, GPU, rollout or acceptance proof."""
from types import SimpleNamespace
import pytest
import torch
import raw_hf_observation_v1 as hf
import decisive_normalizer_v1 as normalizer
from observer_safety_v2 import observer_failures


class Module:
    def __init__(self, fault=None):
        self.fault = fault
        self._forward_pre_hooks = {}
        self._forward_pre_hooks_with_kwargs = {}
    def register_forward_pre_hook(self, hook, with_kwargs):
        if self.fault == 'install':
            raise RuntimeError('injected hook install')
        self._forward_pre_hooks[9] = hook
        self._forward_pre_hooks_with_kwargs[9] = True
        if self.fault == 'partial_install':
            raise RuntimeError('injected partial hook install')
        def remove():
            if self.fault == 'remove':
                raise RuntimeError('injected handle remove')
            self._forward_pre_hooks.pop(9, None)
            self._forward_pre_hooks_with_kwargs.pop(9, None)
        return SimpleNamespace(remove=remove)


class Actor:
    use_remove_padding = False
    ulysses_sequence_parallel_size = 1
    def __init__(self, fault=None):
        self.actor_module = Module(fault)
        self.original_calls = 0
    def _forward_flat_micro_batch(self, micro, temperature):
        self.original_calls += 1
        for hook in list(self.actor_module._forward_pre_hooks.values()):
            hook(self.actor_module, (), dict(input_ids=torch.tensor([[1]])))
        return torch.tensor([0.]), torch.tensor([-.5])


@pytest.mark.parametrize('fault', ['install', 'partial_install', 'remove', 'copy', 'alias_lookup'])
def test_hf_observer_fault_never_skips_original(fault, monkeypatch):
    actor = Actor(fault)
    if fault == 'copy':
        monkeypatch.setattr(hf, 'clone_cpu', lambda value: (_ for _ in ()).throw(RuntimeError('injected clone')))
    if fault == 'alias_lookup':
        # This fake original has no logprobs alias; fallback must still call it.
        assert 'logprobs_from_logits' not in Actor._forward_flat_micro_batch.__globals__
    with hf.capture_existing_flat_calls(actor) as calls:
        result = actor._forward_flat_micro_batch(dict(input_ids=torch.tensor([[1]])), 1.)
    assert actor.original_calls == 1 and torch.equal(result[1], torch.tensor([-.5]))
    assert observer_failures(calls)
    assert not actor.actor_module._forward_pre_hooks
    assert not actor.actor_module._forward_pre_hooks_with_kwargs
    assert '_forward_flat_micro_batch' not in vars(actor)


def test_sampler_setup_failure_still_allows_original_call():
    record = dict(observer_errors=[])
    count = []
    with normalizer.observe_vllm_sampler(SimpleNamespace(model=object()), None, record, [6]):
        count.append('original')
    assert count == ['original'] and record['observer_errors']


class Sampler:
    _do_penalties = _do_top_p_top_k = _do_min_p = False
    def __init__(self, repeats=1):
        self.repeats, self.calls = repeats, 0
    def forward(self, logits, metadata):
        self.calls += 1
        for _ in range(self.repeats):
            torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        return SimpleNamespace(outputs=[])


@pytest.mark.parametrize('repeats', [1, 2])
def test_sampler_normalizer_calls_are_counted_and_original_is_once(repeats, monkeypatch):
    sampler = Sampler(repeats)
    model = SimpleNamespace(sampler=sampler,
        compute_logits=lambda value: value,
        logits_processor=SimpleNamespace(scale=1, soft_cap=None, vocab_size=3))
    runner = SimpleNamespace(model=model)
    record = dict(observer_errors=[], groups=[])
    original_torch = torch
    # Metadata/copy failure must not abort the original sampler or RNG sequence.
    if repeats == 1:
        monkeypatch.setattr(normalizer, 'selected_rows_fingerprint',
            lambda *args: (_ for _ in ()).throw(RuntimeError('injected pre logits fingerprint')))
    with normalizer.observe_vllm_sampler(runner, None, record, [6]):
        value = sampler.forward(original_torch.tensor([[1., 2., 3.]]), SimpleNamespace(seq_groups=[]))
    assert value.outputs == [] and sampler.calls == 1
    assert record['actual_sampler_calls'][0]['observer_errors']
    assert torch is original_torch and 'forward' not in vars(sampler)


def test_unknown_original_exception_propagates_without_retry():
    actor = Actor()
    original_error = RuntimeError('unknown original collective failure')
    def fail(micro, temperature):
        actor.original_calls += 1
        raise original_error
    actor._forward_flat_micro_batch = fail
    with pytest.raises(RuntimeError) as raised:
        with hf.capture_existing_flat_calls(actor):
            actor._forward_flat_micro_batch({}, 1.)
    assert raised.value is original_error and actor.original_calls == 1


def test_runner_binding_failure_calls_original_execute_once():
    from raw_runner_observation_v1 import observe_actual_runner
    calls = []
    class Runner:
        model = Module()
        def prepare_model_input(self, *args, **kwargs):
            return SimpleNamespace()
        def execute_model(self, *args, **kwargs):
            calls.append('original_execute')
            return 'unchanged original output'
    runner = Runner()
    engine = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=SimpleNamespace(
        worker=SimpleNamespace(model_runner=runner))), _post_process_outputs=lambda requests: requests)
    with observe_actual_runner(engine) as observation:
        result = runner.execute_model('binding has no named model_input')
    assert result == 'unchanged original output' and calls == ['original_execute']
    assert observation['calls'][0]['observer_errors']
    assert 'execute_model' not in vars(runner)
