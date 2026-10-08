"""Actual native PPO implementation vs T observation, CPU only, zero RL.

Run in pinned native env with repo/verl and frozen T tools on PYTHONPATH.
CPU plumbing only: Gloo world1, no unpadding/SP, tiny FP32 HF weights and
native BF16 autocast. No replacement of forward/loss/backward/clip/Adam.
"""
import ast
import copy
import dataclasses
import hashlib
import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
import verl.utils.torch_functional as native_functional
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.workers.actor import dp_actor
from verl.workers.config import FSDPActorConfig
import launch_observed as observed
from bounded_sampling import input_digest


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_state():
    return copy.deepcopy((torch.get_rng_state(), random.getstate(), np.random.get_state()))


def set_rng(value):
    torch.set_rng_state(value[0]); random.setstate(value[1]); np.random.set_state(value[2])


def make_data(actor, kind):
    ids = torch.tensor([[0, 5, 6, 7, 8, 9, 2], [4, 5, 6, 7, 8, 2, 0],
                        [0, 0, 6, 7, 2, 0, 0], [4, 5, 6, 7, 9, 10, 2]])
    mask = (ids != 0).long()
    responses = ids[:, -3:].clone()
    response_mask = mask[:, -3:].clone()
    position = (mask.cumsum(-1) - 1).clamp_min(0)
    uids = np.array(['g'] * 4 if kind != 'small_group' else ['a', 'a', 'b', 'b'], dtype=object)
    scores = [0., 0., 0., 0.] if kind == 'constant' else [0., 1., .2, .9]
    rewards = torch.zeros(4, 3)
    for i, score in enumerate(scores):
        rewards[i, int(response_mask[i].sum()) - 1] = score
    advantages, _ = compute_grpo_outcome_advantage(rewards, response_mask, uids)
    inputs = dict(input_ids=ids, attention_mask=mask, position_ids=position, responses=responses)
    with torch.no_grad():
        _, logp = actor._forward_micro_batch(inputs, temperature=.9)
    data = DataProto.from_dict(tensors=dict(**inputs, response_mask=response_mask,
        old_log_probs=logp.detach().clone() + .03, ref_log_prob=logp.detach().clone() - .1,
        advantages=advantages, token_level_rewards=rewards, rm_scores=rewards.clone()),
        non_tensors={'uid': uids}, meta_info={'temperature': .9})
    return data


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    with tempfile.TemporaryDirectory(prefix='native-T-cpu-') as tmp:
        torch.distributed.init_process_group('gloo', init_method='file://' + str(Path(tmp) / 'gloo'), rank=0, world_size=1)
        old_device = dp_actor.get_device_id
        old_flash_ce = native_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE
        old_npu_ce = native_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE
        # Select the ORIGINAL native CPU logprob implementation; CUDA CE cannot run on CPU.
        native_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False
        native_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE = False
        dp_actor.get_device_id = lambda: 'cpu'  # only device plumbing, both paths
        events = []
        old_write = observed.write
        observed.write = lambda kind, record: events.append((kind, copy.deepcopy(record)))
        os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'] = str(Path(tmp) / 'evidence')
        results = []
        try:
            for kind in ['constant', 'small_group', 'qualified']:
                for mini in [4, 2]:  # on_policy and stored-old-logprob branches
                    torch.manual_seed(31)
                    config = FSDPActorConfig(ppo_mini_batch_size=mini, ppo_micro_batch_size_per_gpu=1,
                        use_dynamic_bsz=False, use_remove_padding=False, ulysses_sequence_parallel_size=1,
                        use_torch_compile=False, use_kl_loss=True, kl_loss_coef=.001, grad_clip=.02)
                    config = OmegaConf.create(dataclasses.asdict(config))
                    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
                        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, attention_dropout=0., pad_token_id=0, eos_token_id=2,
                        _attn_implementation='sdpa'))
                    engines = []
                    for _ in range(2):
                        m = copy.deepcopy(model)
                        opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=.01)
                        actor = dp_actor.DataParallelPPOActor(copy.deepcopy(config), m, opt)
                        scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=1)
                        calls = []
                        original_step = actor._optimizer_step
                        def traced_step(original=original_step, module=m, records=calls):
                            before = [None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]
                            result = original()
                            after = [None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]
                            records.append(dict(before=before, after=after, norm=result.detach().clone()))
                            return result
                        actor._optimizer_step = traced_step
                        engines.append((actor, scheduler, calls))
                    data = make_data(engines[0][0], kind)
                    initial_rng = rng_state()
                    snapshots = []
                    for index, (actor, scheduler, calls) in enumerate(engines):
                        set_rng(initial_rng)
                        original_data = copy.deepcopy(data)
                        original_digest = input_digest(original_data)
                        class Group:
                            count = 0
                            def update_actor(self, batch):
                                self.count += 1
                                metrics = actor.update_policy(batch)
                                lr = scheduler.get_last_lr()[0]
                                scheduler.step()  # exact native worker order, once per RPC
                                return dict(metrics=metrics, lr=lr)
                        group = Group()
                        forwarded = observed.ObservedPolicyGroup(group, 6, 'reranker') if index else group
                        outputs = [forwarded.update_actor(original_data), forwarded.update_actor(original_data)]
                        assert group.count == 2 and input_digest(original_data) == original_digest
                        snapshots.append(dict(outputs=outputs, gradients=calls,
                            params=copy.deepcopy(actor.actor_module.state_dict()),
                            adam=copy.deepcopy(actor.actor_optimizer.state_dict()),
                            scheduler=copy.deepcopy(scheduler.state_dict()), rng=rng_state()))
                    assert equal(snapshots[0], snapshots[1]), (kind, mini, 'Native PPO observation parity failure')
                    assert len(engines[0][2]) == 2 * (4 // mini)
                    assert len(snapshots[0]['adam']['state']) > 0
                    results.append(dict(kind=kind, mini=mini, original_update_calls=2,
                        adam_steps=len(engines[0][2]), all_gradient_clip_parameter_adam_scheduler_rng_data_exact=True))
            # Execute exact submitted nested generate with current bounded TWO attempts.
            tree = ast.parse(Path(observed.__file__).read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'generate')
            import observe_worker
            previous_preserve = observe_worker.preserve_raw_payload
            try:
                import phase_sampling
                scenarios = [([1], False, None), ([0, 1], False, None),
                    ([0, 0], False, None), ([0], True, None)]
                scenarios += [([0, 0], False, failure) for failure in
                    ['generation', 'context', 'evidence', 'malformed', 'missing_support', 'negative_support']]
                generation_cases = []
                for supports, validation, failure in scenarios:
                    support = []; calls = []; raw = []; records = []; original_outputs = []
                    batch = copy.deepcopy(data); batch.meta_info.update(global_steps=6, validate=validation)
                    before = input_digest(batch)
                    def original(value):
                        assert input_digest(value) == before
                        calls.append(value); torch.rand(1)
                        if failure != 'missing_support':
                            support.append(-1 if failure == 'negative_support' else supports[len(calls)-1])
                        if failure == 'generation': raise RuntimeError('original failure')
                        result = value if validation else (value, value)
                        if failure == 'malformed': result = object()
                        original_outputs.append(result)
                        return result
                    def preserve(kind, value):
                        if failure == 'evidence': raise OSError('raw evidence failure')
                        raw.append(value)
                    def reject(*args):
                        if failure == 'context': raise RuntimeError('context integrity failure')
                    observe_worker.preserve_raw_payload = preserve
                    namespace = dict(write=lambda k, v: records.append((k, v)), original_generate=original,
                        qualified_support=support, phase_sampling=phase_sampling, reject_context_before_ppo=reject)
                    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-T-generate', 'exec'), namespace)
                    state = torch.get_rng_state().clone(); failed = False
                    try:
                        returned = namespace['generate'](batch)
                    except (AssertionError, RuntimeError, OSError):
                        failed = True
                    end = torch.get_rng_state().clone(); torch.set_rng_state(state)
                    for _ in calls: torch.rand(1)
                    assert torch.equal(end, torch.get_rng_state()) and input_digest(batch) == before
                    expected = 1 if validation or failure or supports[0] > 0 else 2
                    assert len(calls) == expected and failed == bool(failure)
                    assert sum(k == 'generation_start' for k, v in records) == 1
                    assert sum(k == 'generation_end' for k, v in records) == (0 if failure else 1)
                    if not failure:
                        assert returned is original_outputs[-1] and len(raw) == expected
                        if not validation and supports[-1] == 0:
                            terminal = [v for k, v in records if k == 'rollout_attempt_return'][-1]
                            assert terminal['discarded_before_both_PPO'] is False
                            assert terminal['qualified_groups'] == 0
                            assert any(k == 'zero_support_original_fallback_forwarded' and
                                v['qualified_ranker_learning'] is False for k, v in records)
                    generation_cases.append(dict(supports=supports, validation=validation,
                        failure=failure, calls=len(calls), terminal_fallback_original_identity=True,
                        rejected=failed, RNG_natural=True, original_input_unchanged=True))
            finally:
                observe_worker.preserve_raw_payload = previous_preserve
            output = dict(passed=True, CUDA_initialized=torch.cuda.is_initialized(), RL_updates=0,
                CPU_synthetic_optimizer_calls_only=True, actual_native_update_policy=True,
                actual_submitted_wrapper=True, cases=results, generation_cases=generation_cases,
                current_settings_preserved=True, bounded_two_attempts=True,
                adaptations=['CPU Gloo world1', 'device_id CPU', 'tiny FP32 Qwen/native BF16 autocast',
                             'no remove_padding/SP1', 'original native CPU logprob fallback (CUDA/NPU CE disabled in test only)',
                             'native scheduler once after update_policy'],
                source_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in [__file__, observed.__file__, dp_actor.__file__]})
            assert not output['CUDA_initialized']
            Path(__file__).with_name('cpu-original-fallback-gate.json').write_text(json.dumps(output, indent=2) + '\n')
            print(json.dumps(output), flush=True)
        finally:
            observed.write = old_write
            dp_actor.get_device_id = old_device
            native_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = old_flash_ce
            native_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE = old_npu_ce
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
