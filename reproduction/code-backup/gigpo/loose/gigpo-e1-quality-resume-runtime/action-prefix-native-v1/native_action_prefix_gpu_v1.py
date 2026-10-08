"""Official3B eight-rank raw-prefix packing native (synthetic inputs only).

One retained native outer plus its next-outer replay: three executed native
updates, zero formal training updates. No sampler/FAISS/quality/probability
acceptance. Healthy25 is the later calibration origin, NOT restored here.
"""
import os
physical_rank = int(os.environ['LOCAL_RANK'])
allocated = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(allocated) == 8
os.environ['CUDA_VISIBLE_DEVICES'] = allocated[physical_rank]
os.environ['LOCAL_RANK'] = '0'

import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MethodType
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import (offload_fsdp_param_and_grad, load_fsdp_param_and_grad,
                                  offload_fsdp_optimizer, load_fsdp_optimizer)
from search_r1.baselines import recovery
from search_r1.baselines.instruct300 import equal_state
from search_r1.baselines.action_prefix_forward import prepare_action_prefix_micro, scatter_action_prefix_outputs
from search_r1.baselines.objectives import weighted_policy_loss
from action_prefix_native_control_v1 import SCOPE, OFFICIAL_REVISION, validate_result

rank = int(os.environ['RANK'])
torch.cuda.set_device(0)
assert torch.cuda.device_count() == 1 and torch.__version__.startswith('2.4.')
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
assert dist.get_world_size() == 8
binding_path = Path(os.environ['ACTION_PREFIX_NATIVE_BINDING'])
binding = json.loads(binding_path.read_text())
assert recovery.file_sha256(binding_path) == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256']
output = Path(os.environ['NUMERIC_FIXTURE_OUTPUT'])
output.mkdir(parents=True, exist_ok=True)


def event(name, **details):
    with (output / f'progress-rank-{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=name, rank=rank, **details), allow_nan=False) + '\n')


config_record = binding['native_config_driver']
assert recovery.file_sha256(config_record['path']) == config_record['sha256']
driver = torch.load(config_record['path'], map_location='cpu', weights_only=False)
assert driver['source_revision'] == OFFICIAL_REVISION
config = OmegaConf.create(driver['config']['actor_rollout_ref'])
del driver
# Constructor-only official initialization, not trained25 weight recovery.
config.model.path = binding['official_model_path']
config.actor.actor_token_identity = 'sampled_action_prefix_v1'
config.actor.ppo_mini_batch_size = 256
config.actor.ppo_micro_batch_size = 16
config.actor.ppo_epochs = 1
config.actor.use_dynamic_bsz = False
config.actor.ulysses_sequence_parallel_size = 1
config.actor.complete_internal_state = True
config.actor.joint_loss_scaling = True
config.actor.optim.initial_logical_step = 0
config.actor.optim.lr = 1e-6
config.actor.optim.lr_warmup_steps = 57
config.actor.optim.total_training_steps = 300
config.model.use_remove_padding = True
config.model.enable_gradient_checkpointing = True
config.rollout.n = 1
config.actor.policy_safety = OmegaConf.create(dict(enabled=True, mode='native_fixture', dual_clip=3.,
    kl_log_bound=30., monitor_first_global_rows=256, target_old_policy_kl=100., negative_tail_limit=1.))
worker = ActorRolloutRefWorker(config, role='actor')
worker.init_model()
model, opt, scheduler, actor = (worker.actor_module_fsdp, worker.actor_optimizer,
                                worker.actor_lr_scheduler, worker.actor)
worker.rollout_sharding_manager = None
actor.rollout_sharding_manager = None
assert isinstance(model, FSDP) and actor.use_remove_padding and not actor.use_ulysses_sp
assert actor.config.ppo_mini_batch_size == 32 and actor.config.ppo_micro_batch_size == 2
assert worker.actor_model_config._attn_implementation == 'flash_attention_2'
assert worker.actor_model_config.num_hidden_layers == 36 and worker.actor_model_config.hidden_size == 2048
assert scheduler.last_epoch == 0 and not opt.state
pad, eos = worker.tokenizer.pad_token_id, worker.tokenizer.eos_token_id
assert type(pad) is int and type(eos) is int
os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(output / 'audit'),
    SEARCH_R1_BASELINE_RUN_ID='action-prefix-official3b-native',
    SEARCH_R1_BASELINE_DATA_IDENTITY='SYNTHETIC_PREFIX_NATIVE_NOT_COMMON_TRAINING_DATA',
    SEARCH_R1_BASELINE_CONFIG_SHA256=hashlib.sha256(OmegaConf.to_yaml(config).encode()).hexdigest())
event('official_constructor', official_path=config.model.path, revision=OFFICIAL_REVISION,
      initialized_from_official=True, healthy25_restored=False, actor_file=inspect.getfile(type(actor)))


def load():
    load_fsdp_param_and_grad(model, 0, load_grad=True)
    load_fsdp_optimizer(opt, 0)


def offload():
    offload_fsdp_param_and_grad(model, offload_grad=True)
    offload_fsdp_optimizer(opt)


def fixture(rows=320, maximum=False):
    width = 2000 if maximum else 20
    tensors = dict(responses=torch.full((rows, width), 100, dtype=torch.long),
        actor_action_prefix_ids=torch.full((rows, 4, 4096), pad, dtype=torch.long),
        actor_action_tokens=torch.full((rows, 4, 500), pad, dtype=torch.long),
        actor_action_lengths=torch.zeros(rows, 4, dtype=torch.long),
        actor_action_prefix_lengths=torch.zeros(rows, 4, dtype=torch.long),
        actor_action_output_offsets=torch.full((rows, 4), -1, dtype=torch.long),
        actor_action_count=torch.zeros(rows, dtype=torch.long),
        loss_mask=torch.zeros(rows, width), step_ids=torch.full((rows, width), -1, dtype=torch.long))
    for row in range(rows):
        count = 4 if maximum else row % 4 + 1
        tensors['actor_action_count'][row] = count
        for slot in range(count):
            plen, length, offset = (4096, 500, slot * 500) if maximum else (3 + slot, slot + 1, slot * 5)
            prefix = 100 + (torch.arange(plen) + row + slot + rank) % 300
            tokens = 100 + (torch.arange(length) + row + slot + rank * 2) % 300
            # Valid sampled PAD is retained and attended by actual length.
            prefix[1] = pad
            if length > 1:
                tokens[1] = pad
            tensors['actor_action_prefix_ids'][row, slot, :plen] = prefix
            tensors['actor_action_tokens'][row, slot, :length] = tokens
            tensors['actor_action_lengths'][row, slot] = length
            tensors['actor_action_prefix_lengths'][row, slot] = plen
            tensors['actor_action_output_offsets'][row, slot] = offset
            tensors['responses'][row, offset:offset + length] = tokens
            tensors['loss_mask'][row, offset:offset + length] = 1
            tensors['step_ids'][row, offset:offset + length] = slot
    mask = tensors['loss_mask']
    tensors.update(input_ids=torch.cat((torch.full((rows, 3), 101, dtype=torch.long), tensors['responses']), 1),
        attention_mask=torch.ones(rows, width + 3, dtype=torch.long),
        position_ids=torch.arange(width + 3).repeat(rows, 1), policy_weights=mask / mask.sum(-1, keepdim=True),
        advantages=torch.where(tensors['step_ids'] % 2 == 0, 1., -1.) * mask,
        old_log_probs=torch.zeros(rows, width), ref_log_prob=torch.zeros(rows, width))
    return DataProto.from_dict(tensors, meta_info=dict(temperature=1., baseline_outer_step=0,
        actor_token_identity='sampled_action_prefix_v1', pad_token_id=pad, eos_token_id=eos,
        micro_batch_size=4, use_dynamic_bsz=False)).to('cuda')


forward_calls = []
original_flat = actor._forward_flat_micro_batch


def observed_flat(self, micro_batch, temperature):
    context = getattr(self, '_action_prefix_forward_context', {})
    forward_calls.append(dict(phase=context.get('phase'), rows=len(micro_batch['responses']),
                              grad_enabled=torch.is_grad_enabled()))
    return original_flat(micro_batch, temperature)


actor._forward_flat_micro_batch = MethodType(observed_flat, actor)
import search_r1.baselines.objectives as objectives
original_denominator = objectives.row_denominator
denominators = []


def observed_denominator(rows, device):
    value = original_denominator(rows, device)
    assert rows == 32 and float(value) == 32.
    denominators.append(float(value))
    return value


objectives.row_denominator = observed_denominator
load()
model.train()
maximum = fixture(2, maximum=True)
actor._action_prefix_batch_keys(maximum)
actor._action_prefix_forward_context = dict(phase='packing_maximum', mini_index=-1, writes=0)
opt.zero_grad()
entropy, logp = actor._forward_micro_batch(maximum.batch, 1.)
assert tuple(logp.shape) == (2, 2000) and bool(torch.isfinite(logp).all())
loss = weighted_policy_loss(logp.detach(), logp, maximum.batch['advantages'], maximum.batch['loss_mask'],
                           maximum.batch['policy_weights'], .2, 32., maximum.batch['step_ids'], dual_clip=3.)[0]
loss.backward()
from search_r1.baselines.gradient_norm import clip_grad_norm64
maximum_grad = clip_grad_norm64(model, float('inf'))
assert bool(torch.isfinite(maximum_grad)) and float(maximum_grad) > 0
assert len(forward_calls) == 1 and forward_calls[0]['rows'] == 8 and not opt.state
opt.zero_grad(); del maximum, entropy, logp, loss
torch.cuda.empty_cache()
event('maximum_prefix4096_action500_8rows_forward_backward', gradient_norm=float(maximum_grad),
      max_allocated_bytes=torch.cuda.max_memory_allocated(), native_adam_writes=0)

# The shared old/ref globalmicro32 path uses rank4 trajectories -> 16 action
# rows. Its full-vocabulary no-grad entropy/logp peak is a separate real limit.
maximum_old = fixture(4, maximum=True)
forward_calls.clear()
old_maximum = actor.compute_log_prob(maximum_old)
assert tuple(old_maximum.shape) == (4, 2000) and bool(torch.isfinite(old_maximum).all())
assert len(forward_calls) == 1 and forward_calls[0]['rows'] == 16
assert forward_calls[0]['grad_enabled'] is False and not opt.state
event('old_ref_rank4_action16_maximum_no_grad_forward',
      max_allocated_bytes=torch.cuda.max_memory_allocated(), native_adam_writes=0,
      actual_frozen_reference_model_executed=False)
del maximum_old, old_maximum
forward_calls.clear()
torch.cuda.empty_cache()

# Actual officialHF outputs feed the independent packing/formula oracle. The
# probability leaf derivative proves scatter semantics; the real maximum and
# native outers above/below separately prove actual decoder parameter backward.
small = fixture(2)
actor._action_prefix_batch_keys(small)
prepared = prepare_action_prefix_micro(small.batch, pad, eos)
with torch.no_grad():
    flat_entropy, flat_logp = actor._forward_flat_micro_batch(prepared.flat_micro, 1.)
leaf = flat_logp.detach().double().requires_grad_()
_, mapped = scatter_action_prefix_outputs(prepared, flat_entropy.double(), leaf)
manual = leaf[:0].sum().expand(2, 20).clone()
for row in range(2):
    for slot in range(int(small.batch['actor_action_count'][row])):
        length = int(small.batch['actor_action_lengths'][row, slot])
        offset = int(small.batch['actor_action_output_offsets'][row, slot])
        manual[row, offset:offset + length] = leaf[row * 4 + slot, :length]
torch.testing.assert_close(mapped, manual, rtol=0, atol=0)
old = manual.detach() - small.batch['loss_mask'] * .001
got = weighted_policy_loss(old, mapped, small.batch['advantages'], small.batch['loss_mask'],
                          small.batch['policy_weights'], .2, 32., small.batch['step_ids'], dual_clip=3.)[0]
expected = leaf[:0].sum()
for row in range(2):
    total = small.batch['loss_mask'][row].sum()
    for slot in range(int(small.batch['actor_action_count'][row])):
        length = int(small.batch['actor_action_lengths'][row, slot])
        offset = int(small.batch['actor_action_output_offsets'][row, slot])
        delta = (leaf[row * 4 + slot, :length] - old[row, offset:offset + length]).sum()
        adv = small.batch['advantages'][row, offset]
        bounded = delta.clamp_max(math.log1p(.2)) if adv > 0 else delta.clamp(math.log1p(-.2), math.log(3.))
        expected = expected - adv * bounded.exp() * small.batch['policy_weights'][row, offset:offset + length].double().sum() / 32
torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
g1 = torch.autograd.grad(got, leaf, retain_graph=True)[0]
g2 = torch.autograd.grad(expected, leaf)[0]
torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)
assert bool((g1[~prepared.action_mask] == 0).all())
event('mapping_full_trajectory_formula_value_gradient_pass', scalar=float(got.detach()))
del small, prepared, leaf, mapped, manual, old, got, expected, g1, g2, flat_entropy, flat_logp
offload()


def update(label):
    load()
    batch = fixture()
    batch.meta_info['baseline_outer_step'] = scheduler.last_epoch + 1
    batch.batch['old_log_probs'] = actor.compute_log_prob(batch)
    # Synthetic KL reference only; this is NOT frozen-reference or engine proof.
    batch.batch['ref_log_prob'] = batch.batch['old_log_probs'] - batch.batch['loss_mask'] * .01
    forward_calls.clear(); denominators.clear()
    result = actor.update_policy(batch)
    gradient = [item for item in forward_calls if item['phase'] == 'gradient_action_prefix']
    before = [item for item in forward_calls if item['phase'].startswith('before_action_prefix')]
    after = [item for item in forward_calls if item['phase'].startswith('after_action_prefix')]
    assert result['actor/optimizer_steps'] == 10 and result['actor/skipped_empty_minibatches'] == 0
    assert len(gradient) == len(before) == len(after) == 160
    assert all(item['rows'] == 8 for item in forward_calls)
    assert all(item['grad_enabled'] for item in gradient) and not any(item['grad_enabled'] for item in before + after)
    assert denominators == [32.] * 10
    assert all(math.isfinite(float(value)) and float(value) > 0 for value in result['actor/grad_norm'])
    assert all(math.isfinite(float(value)) and float(value) <= 1.000001 for value in result['actor/grad_norm_after_clip'])
    scheduler.step()
    record = dict(label=label, optimizer_writes=10, gradient_micro_forwards=160,
        before_monitor_forwards=160, after_monitor_forwards=160, original_local_trajectory_rows=320,
        mini_denominators=list(denominators), scheduler=scheduler.last_epoch,
        adam_steps=sorted({int(slot['step']) for slot in opt.state.values()}), metrics=result)
    offload()
    event('native_outer_complete', label=label, scheduler=scheduler.last_epoch,
          actual_executed_adam=record['adam_steps'])
    return record


outer_records = [update('retained_native_outer1')]
assert scheduler.last_epoch == 1 and all(int(slot['step']) == 10 for slot in opt.state.values())
saved = recovery.worker_training_state(model, opt, scheduler, rank, 8, 1, None, complete_internal=True)
assert saved['complete_internal'] and saved['local_shards'] and saved['buffers']
receipt = recovery.atomic_torch_save(saved, output / f'state-after-native1-rank-{rank}.pt')
del saved
# Full native HF save is bounded to this new path and this single native point.
auxiliary_rng = recovery.rng_state()
worker.save_checkpoint(str(output / 'global_step_1'))
# save_checkpoint manually offloads; NCCL full-state gathering needs CUDA shards.
load()
# Compare saved full HF tensor values at this native point. Rank0 CPU loading
# is a semantic file roundtrip, not inference/parity on that CPU model.
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    full = model.state_dict()
if rank == 0:
    from transformers import AutoModelForCausalLM
    reloaded_hf = AutoModelForCausalLM.from_pretrained(str(output / 'global_step_1'), local_files_only=True,
                                                    torch_dtype=torch.float32)
    for key, tensor in reloaded_hf.state_dict().items():
        torch.testing.assert_close(tensor, full[key], rtol=0, atol=0)
    del reloaded_hf
del full
# CPU HF construction is auxiliary verification, not the native trajectory.
recovery.restore_rng(auxiliary_rng)
assert equal_state(auxiliary_rng, recovery.rng_state())
del auxiliary_rng
dist.barrier()
offload()
outer_records.append(update('expected_native_outer2'))
expected = recovery.worker_training_state(model, opt, scheduler, rank, 8, 2, None, complete_internal=True)
restored = torch.load(receipt['path'], map_location='cpu', weights_only=False)
assert recovery.load_worker_training_state(restored, model, opt, scheduler, rank, 8, None) == 1
assert equal_state(restored, recovery.worker_training_state(model, opt, scheduler, rank, 8, 1, None, True))
del restored
outer_records.append(update('replayed_native_outer2'))
replayed = recovery.worker_training_state(model, opt, scheduler, rank, 8, 2, None, complete_internal=True)
assert equal_state(expected, replayed), 'New prefix packing next-outer state replay differs'
del expected, replayed
assert scheduler.last_epoch == 2 and all(int(slot['step']) == 20 for slot in opt.state.values())
event('prefix_native_state_next_outer_replay_pass')

# Exact same production rejection boundary on one invalid rank. The bad shape
# is injected AFTER the real officialHF flat forward; no second forward/Adam.
refusals = {}
load()
for kind in ('temperature', 'output_shape'):
    batch = fixture()
    batch.batch['old_log_probs'] = actor.compute_log_prob(batch)
    batch.batch['ref_log_prob'] = batch.batch['old_log_probs'].clone()
    before_state = recovery.checkpoint_pause_identity(model, opt, scheduler, rank, None, True)
    forward_calls.clear()
    if kind == 'temperature' and rank == 5:
        batch.meta_info['temperature'] = 0.
    if kind == 'output_shape':
        def injected_shape(self, micro_batch, temperature):
            entropy, logp = observed_flat(self, micro_batch, temperature)
            return (entropy[:, :0] if rank == 5 else entropy), logp
        actor._forward_flat_micro_batch = MethodType(injected_shape, actor)
    try:
        actor.update_policy(batch)
        raise AssertionError('Single-rank malformed prefix path was admitted')
    except RuntimeError as error:
        assert 'POLICY_SAFETY_REJECTED' in str(error)
    finally:
        actor._forward_flat_micro_batch = MethodType(observed_flat, actor)
    assert before_state == recovery.checkpoint_pause_identity(model, opt, scheduler, rank, None, True)
    assert all(int(slot['step']) == 20 for slot in opt.state.values())
    count = torch.tensor(1, device='cuda')
    dist.all_reduce(count)
    assert int(count) == 8
    # Every rank participates in schema or output MAX, with local-invalid rank5.
    assert len(forward_calls) == (0 if kind == 'temperature' else 1)
    refusals[kind] = dict(all8_refused=True, extra_adam=0, flat_forwards=len(forward_calls))
    event('single_rank_refusal_pass', kind=kind, native_adam=20)
offload()
record = dict(status='PASS', native_scope=SCOPE, rank=rank, world=8,
    code_commit=binding['code_commit'], binding_sha256=recovery.file_sha256(binding_path),
    official_3b=True, official_model_path=config.model.path, official_revision=OFFICIAL_REVISION,
    native_initialization='OFFICIAL_MODEL_INITIALIZATION', native_actual_restore=False,
    formal_updates_contributed=0, actual_fsdp=True, flash_attention2=True, remove_padding=True,
    manual_offload=True, prefix_4096_action500_forward_backward=True,
    old_ref_rank4_action16_no_grad_maximum=True,
    mapping_formula_value_and_gradient_equal=True, state_save_restore_next_outer_equal=True,
    native_hf_semantic_reload_equal=True,
    outer_records=outer_records, scheduler_at_retained_boundary=2, adam_at_retained_boundary=20,
    state_receipt=receipt, hf_native_point=str(output / 'global_step_1'), rank5_refusals=refusals,
    real_sampler_executed=False, real_fixed_reference_executed=False, hf_engine_probability_acceptance=False,
    production_thresholds_selected=False, global_row_denominator=256, production_denominator_per_mini=32,
    max_allocated_bytes=torch.cuda.max_memory_allocated(), torch_version=torch.__version__,
    imported_prefix_module=inspect.getfile(prepare_action_prefix_micro))
validate_result(record, rank)
(output / f'rank-{rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
dist.barrier()
dist.destroy_process_group()
