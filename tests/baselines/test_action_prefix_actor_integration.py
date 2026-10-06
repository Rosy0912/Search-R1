"""Real DPActor control, explicit local CPU forward/monitor fixture only.

320 trajectories represent one rank's share of the global2560 rollout. We
exercise the actual wrapper, selection, safety probes, losses and Adam control.
The one-rank monitor is32; only validate_config's fixed global256 field is
translated in this CPU fixture. No distributed/CUDA/FSDP acceptance is claimed.
"""
import copy
import hashlib
import json
from types import MethodType

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.action_prefix_forward import ACTION_PREFIX_KEYS
from tests.baselines.test_action_prefix_forward import micro_fixture, PAD, EOS


def actor_fixture(monkeypatch, tmp_path, rows=320):
    monkeypatch.setattr(TensorDict, 'cuda', lambda self: self)
    monkeypatch.delenv('SEARCH_R1_BASELINE_COMMIT', raising=False)
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR', str(tmp_path))
    import search_r1.baselines.runtime_audit as audit
    monkeypatch.setattr(audit, 'enabled', lambda: False)
    import search_r1.baselines.policy_safety as safety_module
    original_validate = safety_module.validate_config
    # Local fixture only: actual target config remains monitor_first_global_rows256.
    def fixture_validate(control):
        translated = dict(control)
        assert translated['monitor_first_global_rows'] == 32
        translated['monitor_first_global_rows'] = 256
        return original_validate(translated)
    monkeypatch.setattr(safety_module, 'validate_config', fixture_validate)
    proof = dict(status='PASS', scope='GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION', rank_count=8,
                 restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
                 target_old_policy_kl=.1, negative_tail_limit=.1, dual_clip=3., kl_log_bound=30.)
    raw = json.dumps(proof).encode()
    proof_path = tmp_path / 'SYNTHETIC-CPU-CONTROL-FIXTURE-NOT-GPU-ACCEPTANCE.json'
    proof_path.write_bytes(raw)
    config = OmegaConf.create(dict(ppo_mini_batch_size=32, ppo_micro_batch_size=2, ppo_epochs=1,
        actor_token_identity='sampled_action_prefix_v1', use_dynamic_bsz=False, state_masking=True,
        use_kl_loss=True, kl_loss_type='low_var_kl', kl_loss_coef=.001, clip_ratio=.2,
        entropy_coeff=0., grad_clip=1., baseline_method='gigpo', joint_loss_scaling=True,
        complete_internal_state=True, policy_safety=dict(enabled=True, dual_clip=3., kl_log_bound=30.,
            monitor_first_global_rows=32, target_old_policy_kl=.1, negative_tail_limit=.1,
            calibration_evidence_path=str(proof_path), calibration_evidence_sha256=hashlib.sha256(raw).hexdigest())))
    trajectories = []
    for row in range(rows):
        actions = []
        for slot in range(row % 4 + 1):
            action = ([4, PAD, 5] if slot == 0 else [6] * (slot + 1))
            actions.append(([1, PAD, slot + 2], action, slot * 5))
        trajectories.append(actions)
    tensors = micro_fixture(trajectories, width=20)
    mask = tensors['loss_mask']
    tensors.update(input_ids=torch.cat((torch.ones(rows, 3, dtype=torch.long), tensors['responses']), 1),
        attention_mask=torch.ones(rows, 23, dtype=torch.long),
        position_ids=torch.arange(23).repeat(rows, 1),
        policy_weights=mask.double() / mask.sum(-1, keepdim=True).clamp_min(1),
        advantages=mask.double(), old_log_probs=torch.zeros(rows, 20, dtype=torch.float64),
        ref_log_prob=torch.zeros(rows, 20, dtype=torch.float64))
    data = DataProto.from_dict(tensors, meta_info=dict(temperature=1., baseline_outer_step=26,
        actor_token_identity='sampled_action_prefix_v1', pad_token_id=PAD, eos_token_id=EOS,
        micro_batch_size=4, use_dynamic_bsz=False))

    def new_actor():
        actor = object.__new__(DataParallelPPOActor)
        actor.config = copy.deepcopy(config)
        actor.actor_module = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            actor.actor_module.weight.zero_()
        actor.actor_optimizer = torch.optim.Adam(actor.actor_module.parameters(), lr=.001)
        actor.flat_calls, actor.boundary_calls, actor.raw_gradients = [], [], []
        original_sync = actor._safety_sync_invalid
        def sync(self, bad, device, identity, phase, writes, *args, **kwargs):
            self.boundary_calls.append(dict(bad=bool(bad), phase=phase, writes=writes,
                                           forwards=len(self.flat_calls)))
            return original_sync(bad, device, identity, phase, writes, *args, **kwargs)
        actor._safety_sync_invalid = MethodType(sync, actor)
        def flat(self, micro_batch, temperature):
            context = dict(getattr(self, '_action_prefix_forward_context', {}))
            nrows = micro_batch['responses'].shape[0]
            self.flat_calls.append(dict(context, rows=nrows, grad_enabled=torch.is_grad_enabled(),
                                       weight=float(self.actor_module.weight.detach())))
            # This is a declared differentiable CPU probability interface, not
            # actual model probabilities. Exact raw prefixes affect base logp.
            prefix_width = micro_batch['prompts'].shape[1]
            prefix = (micro_batch['prompts'] * micro_batch['attention_mask'][:, :prefix_width]).sum(-1)
            base = -prefix.double()[:, None] * .01 - micro_batch['responses'].double() * .002
            logp = base + self.actor_module.weight.reshape(1, 1)
            return logp * 0, logp
        actor._forward_flat_micro_batch = MethodType(flat, actor)  # keep actual new wrapper
        original_step = actor._optimizer_step
        def step(self, *args, **kwargs):
            self.raw_gradients.append(self.actor_module.weight.grad.detach().clone())
            return original_step(*args, **kwargs)
        actor._optimizer_step = MethodType(step, actor)
        return actor
    actor, reference = new_actor(), new_actor()
    selected = []
    original_select = DataProto.select
    def select(self, *args, **kwargs):
        selected.append(tuple(kwargs.get('batch_keys', ())))
        return original_select(self, *args, **kwargs)
    monkeypatch.setattr(DataProto, 'select', select)
    return actor, reference, data, selected


def assign_fixed_logps(actor, reference, data):
    data.batch['old_log_probs'] = actor.compute_log_prob(data)
    data.batch['ref_log_prob'] = reference.compute_log_prob(data)
    actor.flat_calls.clear(); actor.boundary_calls.clear()


def test_real_control_ten_adam160_gradient320_probes_and_raw_l_weights(monkeypatch, tmp_path):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path)
    assign_fixed_logps(actor, reference, data)
    fixed_old = data.batch['old_log_probs'].clone()
    rng = torch.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    metrics = actor.update_policy(data)
    assert metrics['actor/optimizer_steps'] == 10
    assert metrics['actor/skipped_empty_minibatches'] == 0
    assert int(next(iter(actor.actor_optimizer.state.values()))['step']) == 10
    assert len(actor.raw_gradients) == 10
    gradient = [row for row in actor.flat_calls if row['phase'] == 'gradient_action_prefix']
    before = [row for row in actor.flat_calls if row['phase'].startswith('before_action_prefix')]
    after = [row for row in actor.flat_calls if row['phase'].startswith('after_action_prefix')]
    assert len(gradient) == 160 and len(before) == len(after) == 160
    assert len(actor.flat_calls) == 480 and all(row['rows'] == 8 for row in actor.flat_calls)
    assert all(row['grad_enabled'] for row in gradient)
    assert all(not row['grad_enabled'] for row in before + after)
    records = [json.loads(line) for line in (tmp_path / 'policy-safety/rank-0.jsonl').read_text().splitlines()]
    assert len(records) == 20 and all(row['allowed'] for row in records)
    assert [row['actual_optimizer_writes'] for row in records if row['phase'] == 'after'] == list(range(1, 11))
    assert records[0]['statistics']['max_abs_token_delta'] == 0
    assert records[1]['statistics']['max_abs_token_delta'] > 0
    assert all(not row['monitor_identity_changed'] for row in records)
    torch.testing.assert_close(data.batch['old_log_probs'], fixed_old)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert torch.cuda.is_initialized() == cuda_before
    assert float(actor.actor_module.weight.detach()) > 0
    mask = data.batch['loss_mask']
    length = mask.sum(-1)
    expected_weights = mask / length[:, None]
    torch.testing.assert_close(data.batch['policy_weights'], expected_weights.double())
    # At the first update ratio=1, KL derivative=0, advantage=1. Full joint
    # broadcast yields -sum_s(length_s^2)/L per original trajectory, /32.
    action_lengths = data.batch['actor_action_lengths'][:32].double()
    expected_gradient = -(action_lengths.square().sum(-1) / length[:32]).sum() / 32
    torch.testing.assert_close(actor.raw_gradients[0].reshape(()), expected_gradient)
    assert all(set(ACTION_PREFIX_KEYS).issubset(keys) for keys in selected)


def test_old_and_reference_logprob_selection_keeps_every_prefix_key(monkeypatch, tmp_path):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path, rows=8)
    old, ref = actor.compute_log_prob(data), reference.compute_log_prob(data)
    assert len(selected) == 2
    assert all(set(ACTION_PREFIX_KEYS).issubset(keys) for keys in selected)
    assert len(actor.flat_calls) == len(reference.flat_calls) == 2
    assert all(row['rows'] == 16 and not row['grad_enabled'] for row in actor.flat_calls + reference.flat_calls)
    torch.testing.assert_close(old, ref)
    valid = data.batch['loss_mask'].bool()
    torch.testing.assert_close(old[~valid], torch.zeros_like(old[~valid]))
    # Every first action has sampler prefix[1,0,2], including real PAD, and
    # actor token4, irrespective of the canonical full-row concatenation.
    torch.testing.assert_close(old[:, 0], torch.full((8,), -.038, dtype=torch.float64))
    assert not actor.actor_optimizer.state and not reference.actor_optimizer.state


@pytest.mark.parametrize('failure', ['canonical_input', 'bad_prefix', 'partial_action_mask', 'missing_prefix_key'])
def test_provenance_or_prefix_failure_refuses_before_forward_and_zero_writes(monkeypatch, tmp_path, failure):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path)
    if failure == 'canonical_input':
        for key in ACTION_PREFIX_KEYS:
            data.batch.pop(key)
        data.meta_info.pop('actor_token_identity')
    elif failure == 'bad_prefix':
        data.batch['actor_action_prefix_lengths'][0, 0] = 0
    elif failure == 'partial_action_mask':
        data.batch['loss_mask'][0, 0] = 0
    else:
        data.batch.pop(ACTION_PREFIX_KEYS[0])
    with pytest.raises(RuntimeError, match='POLICY_SAFETY_REJECTED'):
        actor.update_policy(data)
    assert not actor.flat_calls and not actor.actor_optimizer.state and not actor.raw_gradients
    refused = [call for call in actor.boundary_calls if call['bad']]
    assert refused and all(call['writes'] == 0 and call['forwards'] == 0 for call in refused)
    if failure in ('bad_prefix', 'partial_action_mask'):
        assert len(refused) == 1 and refused[0]['phase'].endswith('_schema')


def test_bad_prefix_old_ref_computation_refuses_before_flat_forward(monkeypatch, tmp_path):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path, rows=8)
    data.batch['actor_action_output_offsets'][0, 0] = 20
    for instance in (actor, reference):
        with pytest.raises(RuntimeError, match='POLICY_SAFETY_REJECTED'):
            instance.compute_log_prob(data)
        assert not instance.flat_calls and not instance.actor_optimizer.state
        refused = [call for call in instance.boundary_calls if call['bad']]
        assert len(refused) == 1 and refused[0]['phase'] == 'log_prob_action_prefix_schema'


def test_generation_probe_exception_cannot_bypass_training_provenance(monkeypatch, tmp_path):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path, rows=8)
    data.meta_info.pop('actor_token_identity')
    data.meta_info['generation_actor_token_identity'] = 'sampled_completion_v1'
    for key in ACTION_PREFIX_KEYS:
        data.batch.pop(key)
    # step_ids remains, so the declared sampler-only probe exemption is false.
    with pytest.raises(RuntimeError, match='POLICY_SAFETY_REJECTED'):
        actor.compute_log_prob(data)
    assert not actor.flat_calls and not actor.actor_optimizer.state


@pytest.mark.parametrize('temperature', [0., float('nan'), True])
def test_invalid_temperature_reaches_schema_boundary_before_any_forward(monkeypatch, tmp_path, temperature):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path)
    data.meta_info['temperature'] = temperature
    with pytest.raises(RuntimeError, match='POLICY_SAFETY_REJECTED'):
        actor.update_policy(data)
    assert not actor.flat_calls and not actor.actor_optimizer.state and not actor.raw_gradients
    refused = [call for call in actor.boundary_calls if call['bad']]
    assert len(refused) == 1
    assert refused[0]['phase'] == 'before_action_prefix_micro0_schema'
    assert refused[0]['writes'] == 0 and refused[0]['forwards'] == 0
    records = list((tmp_path / 'policy-safety').glob('numeric-*.json'))
    assert len(records) == 1
    assert 'Invalid probability temperature' in json.loads(records[0].read_text())['details']


@pytest.mark.parametrize('failure', ['shape', 'device', 'nonfloat'])
def test_malformed_flat_output_reaches_outputs_boundary_after_one_forward(monkeypatch, tmp_path, failure):
    actor, reference, data, selected = actor_fixture(monkeypatch, tmp_path)
    original_flat = actor._forward_flat_micro_batch
    def malformed_flat(self, micro_batch, temperature):
        entropy, logp = original_flat(micro_batch, temperature)
        if failure == 'shape':
            entropy = entropy[:, :0]
        elif failure == 'device':
            # CPU-only meta tensor is an explicit wrong-device fixture. No
            # GPU allocation or contents need to be read to reject this schema.
            entropy = torch.empty(entropy.shape, dtype=entropy.dtype, device='meta')
        else:
            logp = logp.long()
        return entropy, logp
    actor._forward_flat_micro_batch = MethodType(malformed_flat, actor)
    with pytest.raises(RuntimeError, match='POLICY_SAFETY_REJECTED'):
        actor.update_policy(data)
    assert len(actor.flat_calls) == 1
    assert not actor.actor_optimizer.state and not actor.raw_gradients
    refused = [call for call in actor.boundary_calls if call['bad']]
    assert len(refused) == 1
    assert refused[0]['phase'] == 'before_action_prefix_micro0_outputs'
    assert refused[0]['writes'] == 0 and refused[0]['forwards'] == 1
    records = list((tmp_path / 'policy-safety').glob('numeric-*.json'))
    assert len(records) == 1
    assert 'Invalid flat output' in json.loads(records[0].read_text())['details']
