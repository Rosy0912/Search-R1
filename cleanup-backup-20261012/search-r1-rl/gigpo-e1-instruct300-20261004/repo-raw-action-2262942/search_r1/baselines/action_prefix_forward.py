"""Score complete sampled actions at their own actual sampler prefixes.

The caller still batches trajectories, not actions. One rank-local trajectory
micro-batch becomes four fixed slots per trajectory for ONE existing flat HF
forward. Results return to the original trajectory timeline; this module never
changes action advantages, joint ratios, trajectory weights, or collectives.

Distributed callers MUST prepare on every rank, synchronize any preparation
failure at their existing rejection boundary, then forward the prepared batch.
The convenience function is only appropriate without that synchronization need.
"""
from dataclasses import dataclass
import math
import numbers
from typing import Callable, Mapping

import torch


MAX_ACTIONS = 4
MAX_PREFIX = 4096
MAX_ACTION = 500
ACTION_PREFIX_KEYS = (
    'actor_action_prefix_ids', 'actor_action_tokens', 'actor_action_lengths',
    'actor_action_prefix_lengths', 'actor_action_output_offsets', 'actor_action_count',
)
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@dataclass(frozen=True)
class PreparedActionPrefixMicro:
    flat_micro: dict
    source_indices: torch.Tensor
    output_indices: torch.Tensor
    action_mask: torch.Tensor
    trajectory_count: int
    timeline_width: int
    action_width: int


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_tensor(micro, name, shape=None, device=None):
    value = micro.get(name)
    _require(isinstance(value, torch.Tensor), 'Missing tensor: ' + name)
    _require(value.dtype in _INTEGER_DTYPES, 'Noninteger tensor: ' + name)
    if shape is not None:
        _require(tuple(value.shape) == shape, 'Invalid shape: ' + name)
    if device is not None:
        _require(value.device == device, 'Mixed devices: ' + name)
    return value


def _support_tensor(value, shape, device, name):
    _require(isinstance(value, torch.Tensor), 'Missing tensor: ' + name)
    _require(tuple(value.shape) == shape and value.device == device, 'Invalid shape/device: ' + name)
    _require(bool(((value == 0) | (value == 1)).all()), 'Nonbinary mask: ' + name)
    return value.bool()


def prepare_action_prefix_micro(micro: Mapping, pad_id: int, eos_id: int) -> PreparedActionPrefixMicro:
    """Validate and construct flat inputs without a model call or collective.

    Stored prefixes/actions are right padded: the valid original IDs are in
    ``[:length]``. Offsets are absolute in ``responses`` (observations included).
    Inactive slots require lengths zero and offset -1. Active empty actions are
    permitted with an actual nonempty prefix. Real PAD/special IDs count as
    valid tokens according to lengths, never according to their token ID.
    """
    for name, value in [('pad_id', pad_id), ('eos_id', eos_id)]:
        _require(isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0,
                 'Invalid token ID: ' + name)
    responses = _integer_tensor(micro, 'responses')
    _require(responses.ndim == 2 and responses.shape[0] > 0, 'Invalid responses shape')
    ntraj, width = responses.shape
    device = responses.device
    prefix = _integer_tensor(micro, 'actor_action_prefix_ids', (ntraj, MAX_ACTIONS, MAX_PREFIX), device)
    tokens = _integer_tensor(micro, 'actor_action_tokens', (ntraj, MAX_ACTIONS, MAX_ACTION), device)
    lengths = _integer_tensor(micro, 'actor_action_lengths', (ntraj, MAX_ACTIONS), device)
    prefix_lengths = _integer_tensor(micro, 'actor_action_prefix_lengths', (ntraj, MAX_ACTIONS), device)
    offsets = _integer_tensor(micro, 'actor_action_output_offsets', (ntraj, MAX_ACTIONS), device)
    counts = _integer_tensor(micro, 'actor_action_count', (ntraj,), device)
    _require(bool(((counts >= 0) & (counts <= MAX_ACTIONS)).all()), 'Invalid action count')
    _require(bool(((lengths >= 0) & (lengths <= MAX_ACTION)).all()), 'Invalid action length')
    _require(bool(((prefix_lengths >= 0) & (prefix_lengths <= MAX_PREFIX)).all()), 'Invalid prefix length')

    # The small metadata conversion does not touch RNG or the model. All rank
    # errors occur here, before the caller's collective rejection boundary.
    count_list, len_list = counts.tolist(), lengths.tolist()
    prefix_list, offset_list = prefix_lengths.tolist(), offsets.tolist()
    covered = torch.zeros((ntraj, width), dtype=torch.bool, device=device)
    expected_steps = torch.full((ntraj, width), -1, dtype=torch.long, device=device)
    pairs = []
    max_prefix, max_action = 1, 1
    for trajectory in range(ntraj):
        previous_end = 0
        for slot in range(MAX_ACTIONS):
            length, prefix_length, offset = (len_list[trajectory][slot], prefix_list[trajectory][slot],
                                              offset_list[trajectory][slot])
            if slot >= count_list[trajectory]:
                _require(length == 0 and prefix_length == 0 and offset == -1,
                         'Inactive action metadata is not empty')
                continue
            _require(prefix_length > 0, 'Active action has an empty prefix')
            _require(0 <= offset <= width and offset + length <= width, 'Action exceeds response timeline')
            _require(offset >= previous_end, 'Overlapping or unordered action offsets')
            previous_end = offset + length
            actual_prefix, actual_action = prefix[trajectory, slot, :prefix_length], tokens[trajectory, slot, :length]
            _require(bool((actual_prefix >= 0).all()) and bool((actual_action >= 0).all()), 'Negative valid token ID')
            _require(torch.equal(actual_action.to(torch.long), responses[trajectory, offset:offset + length].to(torch.long)),
                     'Action tokens differ from the response timeline')
            covered[trajectory, offset:offset + length] = True
            expected_steps[trajectory, offset:offset + length] = slot
            pairs.append((trajectory, slot, prefix_length, length, offset))
            max_prefix, max_action = max(max_prefix, prefix_length), max(max_action, length)

    if 'loss_mask' in micro:
        support = _support_tensor(micro['loss_mask'], (ntraj, width), device, 'loss_mask')
        _require(torch.equal(support, covered), 'Loss mask does not cover complete sampled actions')
    if 'info_mask' in micro:
        info, attention = micro['info_mask'], micro.get('attention_mask')
        _require(isinstance(info, torch.Tensor) and info.ndim == 2 and info.shape[0] == ntraj and info.shape[1] >= width,
                 'Invalid info mask shape')
        info = _support_tensor(info, tuple(info.shape), device, 'info_mask')
        attention = _support_tensor(attention, tuple(info.shape), device, 'attention_mask')
        support = (info[:, -width:] & attention[:, -width:]) if width else info[:, :0]
        _require(torch.equal(support, covered), 'Info/attention mask does not cover complete sampled actions')
    if 'step_ids' in micro:
        steps = _integer_tensor(micro, 'step_ids', (ntraj, width), device)
        _require(torch.equal(steps.to(torch.long), expected_steps), 'Action step IDs do not match complete sampled actions')

    rows = ntraj * MAX_ACTIONS
    prompts = torch.full((rows, max_prefix), int(pad_id), dtype=torch.long, device=device)
    action_ids = torch.full((rows, max_action), int(pad_id), dtype=torch.long, device=device)
    prefix_attention = torch.zeros_like(prompts)
    action_attention = torch.zeros_like(action_ids)
    # EOS prefix makes every dummy varlen row nonempty. Its action support is
    # empty, so neither its probabilities nor a next-row rolled label are used.
    prompts[:, -1] = int(eos_id)
    prefix_attention[:, -1] = 1
    source_indices, output_indices = [], []
    for trajectory, slot, prefix_length, length, offset in pairs:
        row = trajectory * MAX_ACTIONS + slot
        prompts[row] = int(pad_id)
        prefix_attention[row] = 0
        prompts[row, -prefix_length:] = prefix[trajectory, slot, :prefix_length]
        prefix_attention[row, -prefix_length:] = 1
        action_ids[row, :length] = tokens[trajectory, slot, :length]
        action_attention[row, :length] = 1
        source_indices.extend(row * max_action + token for token in range(length))
        output_indices.extend(trajectory * width + offset + token for token in range(length))
    input_ids = torch.cat((prompts, action_ids), dim=-1)
    attention = torch.cat((prefix_attention, action_attention), dim=-1)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    positions = torch.where(attention.bool(), positions, 0)
    return PreparedActionPrefixMicro(
        flat_micro=dict(prompts=prompts, responses=action_ids, input_ids=input_ids,
                        attention_mask=attention, position_ids=positions),
        source_indices=torch.tensor(source_indices, dtype=torch.long, device=device),
        output_indices=torch.tensor(output_indices, dtype=torch.long, device=device),
        action_mask=action_attention.bool(), trajectory_count=ntraj, timeline_width=width, action_width=max_action)


def scatter_action_prefix_outputs(prepared: PreparedActionPrefixMicro, entropy: torch.Tensor,
                                  logp: torch.Tensor):
    """Map action-token probabilities back; observations/dummies are graph zeros.

    Empty slices connect zero outputs to the one flat forward's autograd graph
    without ``0 * NaN`` contaminating an empty action. Nonfinite real action
    scores stay visible to the caller's ordinary synchronized safety checks.
    """
    result = []
    shape = (prepared.trajectory_count * MAX_ACTIONS, prepared.action_width)
    device = prepared.flat_micro['input_ids'].device
    for name, values in [('entropy', entropy), ('logp', logp)]:
        _require(isinstance(values, torch.Tensor) and tuple(values.shape) == shape, 'Invalid flat output: ' + name)
        _require(values.device == device and values.is_floating_point(), 'Invalid flat output device/dtype: ' + name)
        values = values.reshape(-1)
        selected = values.index_select(0, prepared.source_indices)
        zero = values[:0].sum()
        output = zero.expand(prepared.trajectory_count * prepared.timeline_width).scatter_add(
            0, prepared.output_indices, selected)
        result.append(output.reshape(prepared.trajectory_count, prepared.timeline_width))
    return tuple(result)


def forward_prepared_action_prefix(prepared: PreparedActionPrefixMicro, temperature: float,
                                   forward_flat: Callable):
    """Call the unchanged original flat HF forward exactly once, then scatter."""
    _require(isinstance(temperature, numbers.Real) and math.isfinite(temperature) and temperature > 0,
             'Invalid probability temperature')
    entropy, logp = forward_flat(prepared.flat_micro, temperature)
    return scatter_action_prefix_outputs(prepared, entropy, logp)


def forward_action_prefix_micro(micro: Mapping, temperature: float, forward_flat: Callable,
                                pad_id: int, eos_id: int):
    """Convenience for non-distributed use; distributed callers separate prepare."""
    return forward_prepared_action_prefix(prepare_action_prefix_micro(micro, pad_id, eos_id), temperature, forward_flat)
