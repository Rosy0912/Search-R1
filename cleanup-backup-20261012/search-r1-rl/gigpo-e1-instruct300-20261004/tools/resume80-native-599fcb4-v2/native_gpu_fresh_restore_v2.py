"""Official 3B constructor-only FSDP -> original full80 -> one native mini.

Two independently spawned process groups repeat this boundary. No rollout,
reference scoring, training data fetch or completed outer iteration is claimed.
"""
import os
physical_rank = int(os.environ['LOCAL_RANK'])
allocated = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(allocated) == 8
os.environ['CUDA_VISIBLE_DEVICES'] = allocated[physical_rank]
os.environ['LOCAL_RANK'] = '0'
import datetime, hashlib, json, math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from verl.utils.fsdp_utils import offload_fsdp_param_and_grad, load_fsdp_param_and_grad, offload_fsdp_optimizer, load_fsdp_optimizer
from search_r1.baselines import recovery
from search_r1.baselines.instruct300 import equal_state
from search_r1.baselines.instruct_resume import CHECKPOINT80, MANIFEST80
rank = int(os.environ['RANK'])
torch.cuda.set_device(0)
assert torch.cuda.device_count() == 1 and torch.__version__.startswith('2.4.')
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
assert dist.get_world_size() == 8
output = Path(os.environ['NUMERIC_FIXTURE_OUTPUT'])
output.mkdir(parents=True, exist_ok=True)
phase = os.environ['NUMERIC_FIXTURE_PHASE']
def event(name, **details):
    with (output / f'progress-{phase}-rank-{rank}.jsonl').open('a') as out:
        out.write(json.dumps(dict(event=name, rank=rank, phase=phase, **details)) + '\n')
manifest_path = Path(CHECKPOINT80) / 'training_state/manifest.json'
assert recovery.file_sha256(manifest_path) == MANIFEST80
# Exact continuation provenance: saved80 came from the accepted75 branch.
identity = json.loads(manifest_path.read_text())
assert identity['job_id'] == '3225'
assert identity['code_commit'] == 'e5b30d7cf35c30a83d07375733501392d6fa42f1'
assert identity['logical_step'] == 80 and identity['actual_new_updates'] == 5
assert identity['recovery_mode'] == 'full_state_instruct75'
manifest = json.loads(manifest_path.read_text())
driver_record = next(item for item in manifest['files'] if item['path'].endswith('/driver.pt'))
assert recovery.file_sha256(driver_record['path']) == driver_record['sha256']
driver = torch.load(driver_record['path'], map_location='cpu', weights_only=False)
config = OmegaConf.create(driver['config']['actor_rollout_ref'])
config.actor.joint_loss_scaling = True
revision = driver['source_revision']
assert revision == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
assert driver['logical_step'] == 80 and driver['actual_new_updates'] == 5
assert driver['sampler']['cursor'] == 40960
assert config.actor.ppo_mini_batch_size == 128 and config.actor.ppo_micro_batch_size == 16
worker = ActorRolloutRefWorker(config, role='actor')
worker.init_model()
model, opt, sch = worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler
assert model._is_root is None, 'Fixture already performed a forward and missed the fresh boundary'
manager = FSDPVLLMShardingManager(model, None, worker.actor_model_config, full_params=True,
    device_mesh=init_device_mesh('cuda', (8, 1), mesh_dim_names=('dp', 'tp')))
worker.rollout_sharding_manager = manager
worker.actor.rollout_sharding_manager = manager
state_record = next(item for item in manifest['files'] if item['path'].endswith(f'/rank_{rank}.pt'))
assert recovery.file_sha256(state_record['path']) == state_record['sha256']
state = torch.load(state_record['path'], map_location='cpu', weights_only=False)
event('fresh_constructor', local_shards_live=sum(hasattr(p, '_local_shard') for p in model.parameters()),
    local_shards_saved=len(state['local_shards']), buffer_keys_live=sorted(dict(model.named_buffers())),
    buffer_keys_saved=sorted(state['buffers']))
assert state['local_shards'] and state['complete_internal']
shas = {str(i):next(item['sha256'] for item in manifest['files'] if item['path'].endswith(f'/rank_{i}.pt')) for i in range(8)}
receipt = worker.load_training_state(CHECKPOINT80, shas, 80, 1600)
assert receipt['fsdp_initialization']['initialized']
assert receipt['fsdp_initialization']['tensor_values_rng_modes_unchanged']
restored = recovery.worker_training_state(model, opt, sch, rank, 8, 80, manager, True)
for key in state:
    if key != 'pause_identity':
        assert equal_state(state[key], restored[key]), 'Full restored state differs: ' + key
assert sch.last_epoch == 80 and sch.get_last_lr() == [1e-6]
assert all(float(slot['step']) == 1600 for slot in opt.state.values())
before = restored['pause_identity']
del state, restored, driver
# Must be a no-op on initialized roots, even after manual offload.
offload_fsdp_param_and_grad(model, offload_grad=True)
offload_fsdp_optimizer(opt)
from verl.utils.fsdp_utils import initialize_fsdp_before_full_restore
pause = recovery.checkpoint_pause_identity(model, opt, sch, rank, manager, True)
assert initialize_fsdp_before_full_restore(model) == dict(initialized=False, reason='already_initialized')
assert pause == recovery.checkpoint_pause_identity(model, opt, sch, rank, manager, True)
event('full80_restored', receipt=receipt, identity=pause)
load_fsdp_param_and_grad(model, 0, load_grad=True)
load_fsdp_optimizer(opt, 0)
actor = worker.actor
inputs = torch.randint(100, 1000, (16, 16), device='cuda')
responses = inputs[:, -8:]
attention = torch.ones_like(inputs)
position = torch.arange(16, device='cuda').expand(16, 16)
mask = torch.ones(16, 8, device='cuda')
mask[:, 1] = 0
mask[::7, 4:] = 0
ids = torch.where(mask.bool(), torch.arange(8, device='cuda') // 2, -1).expand(16, 8).clone()
with torch.no_grad():
    logp = torch.cat([actor._forward_micro_batch(dict(input_ids=inputs[i:i+4], responses=responses[i:i+4],
        attention_mask=attention[i:i+4], position_ids=position[i:i+4]), 1.)[1] for i in range(0, 16, 4)])
advantages = torch.randn(16, 4, device='cuda').repeat_interleave(2, dim=1)
batch = DataProto.from_dict(dict(input_ids=inputs, responses=responses, attention_mask=attention,
    position_ids=position, loss_mask=mask, step_ids=ids, old_log_probs=logp.clone(),
    ref_log_prob=logp-.01, advantages=advantages,
    policy_weights=mask/mask.sum(-1, keepdim=True)), meta_info={'temperature':1.}).to('cuda')
result = actor.update_policy(batch)
assert result['actor/optimizer_steps'] == 1 and result['actor/skipped_empty_minibatches'] == 0
assert result['actor/probe_micro_batches'] == [8]
assert all(math.isfinite(float(x)) and float(x)>0 for x in result['actor/grad_norm'])
assert all(float(slot['step']) == 1601 for slot in opt.state.values())
assert sch.last_epoch == 80  # This is one mini, not a completed outer update.
offload_fsdp_param_and_grad(model, offload_grad=True)
offload_fsdp_optimizer(opt)
after = recovery.checkpoint_pause_identity(model, opt, sch, rank, manager, True)
assert before['model_shards_sha256'] != after['model_shards_sha256']
assert before['optimizer_tensors_sha256'] != after['optimizer_tensors_sha256']
rng = recovery.rng_state()
python_numpy_sha = hashlib.sha256(repr(rng['python']).encode() + rng['numpy'][0].encode()
    + rng['numpy'][1].tobytes() + repr(rng['numpy'][2:]).encode()).hexdigest()
record = dict(status='PASS', phase=phase, rank=rank, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    source_manifest_sha256=MANIFEST80, source_rank_sha256=state_record['sha256'],
    official_model_path=config.model.path, official_revision=revision,
    actual_constructor_worker='ActorRolloutRefWorker(role=actor)', no_actual_rollout_reference_or_retrieval=True,
    fresh_initialization=receipt['fsdp_initialization'], full_saved_state_equal=True,
    restored_identity=before, next_mini_identity=after, python_numpy_rng_sha256=python_numpy_sha,
    optimizer_steps_before=1600, optimizer_steps_after=1601, scheduler=80,
    global_mini_rows=128, rank_micro_rows=2, native_mini_optimizer_steps=1,
    probe_micro_batches=8, grad_norms=result['actor/grad_norm'], actual_outer_updates=0)
# Real CUDA reason diagnostics must preserve the original refusal and RNG.
from search_r1.baselines.loss_scaling import active_joint_bound, describe_invalid_joint_probe
reason_checks = []
for case, reason in [('overflow', 'nonfinite_raw_joint_ratio'),
                     ('advantage', 'inconsistent_action_advantage'),
                     ('negative_id', 'negative_action_id'),
                     ('negative_weight', 'negative_weight')]:
    old = torch.zeros(1, 2, device='cuda')
    current = torch.zeros_like(old)
    adv = torch.ones_like(old)
    mask = torch.ones_like(old)
    weights = mask / 2
    ids = torch.zeros_like(old, dtype=torch.long)
    if case == 'overflow': current.fill_(400.)
    if case == 'advantage': adv[0, 1] = -1.
    if case == 'negative_id': ids.fill_(-1)
    if case == 'negative_weight': weights.fill_(-1)
    args = (old, current, adv, mask, weights, ids, .2)
    bound, invalid = active_joint_bound(*args)
    before_rng = recovery.rng_state()
    detail = describe_invalid_joint_probe(*args)
    after_bound, after_invalid = active_joint_bound(*args)
    assert invalid == 1 and after_invalid == invalid and reason in detail['reasons']
    assert torch.equal(bound, after_bound) and equal_state(before_rng, recovery.rng_state())
    reason_checks.append(case)
record['cuda_diagnostic_checks'] = reason_checks
(output / f'rank-{phase}-{rank}.json').write_text(json.dumps(record, indent=2))
event('native_mini_pass')
dist.barrier()
dist.destroy_process_group()
