# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _checkpoint_matched_stage(phase, error):
    """Decide a known local phase at the same boundary on every actor rank.

    Callers must never call this from an unknown FSDP/collective exception path.
    Collective failures here also propagate directly; this is not a recovery
    protocol for a broken process group, nor does it undo any completed writes.
    """
    device = torch.device('cuda', torch.cuda.current_device())
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if not failed.item():
        return
    # Fixed-size diagnostics prevent an exception string from allocating an
    # unbounded object collective. All ranks enter this only after the same MAX.
    try:
        diagnostic = (type(error).__name__ + ': ' + str(error)) if error is not None else ''
    except Exception:
        diagnostic = type(error).__name__ + ': diagnostic formatting failed'
    encoded = diagnostic.encode('utf-8', errors='replace')[:1024]
    local = torch.zeros(1024, dtype=torch.uint8, device=device)
    if encoded:
        local[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    rows = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(rows, local)
    errors = {rank: bytes(row.cpu().tolist()).rstrip(b'\0').decode('utf-8', errors='replace')
              for rank, row in enumerate(rows) if row.any().item()}
    raise RuntimeError(f'Matched checkpoint phase rejected: {phase}; errors={errors!r}; '
                       'automatic_rollback=False; collective_failure_recovery=False')


def _validate_checkpoint_state(state, module, optimizer, scheduler, rank, world_size,
                               sharding_manager, expected_step, expected_adam_steps,
                               require_complete):
    """Non-mutating counterpart of recovery.load_worker_training_state checks.

    Keep these checks aligned with that loader so all ranks validate before any
    rank copies parameters or loads Adam/scheduler/RNG. Constructor-only roots
    validate saved local shards against their real current parameter layout;
    the same checks run again after PyTorch initializes its own local shards.
    """
    import math
    from search_r1.baselines.recovery import parameter_layout, validate_rng
    if (not isinstance(state, dict) or state.get('schema') != 1
            or type(state.get('logical_step')) is not int
            or state['layout'] != parameter_layout(module, optimizer, rank, world_size)):
        raise ValueError('Training-state schema/progress/layout changed')
    if expected_step is not None and state['logical_step'] != expected_step:
        raise ValueError('Saved worker logical step differs from explicit restore source')
    if require_complete and state.get('complete_internal') is not True:
        raise ValueError('Fresh Instruct requires complete internal FSDP state')
    if require_complete and ('partial_update' in state or 'continuation_boundary' in state):
        raise ValueError('Partial diagnostic/continuation states cannot masquerade as complete checkpoints')
    live = dict(module.named_parameters())
    saved = state['model_shards']
    if saved.keys() != live.keys():
        raise ValueError('Model shard identity changed')

    def validate_tensors(sources, targets, label):
        if sources.keys() != targets.keys():
            raise ValueError('Incomplete checkpoint tensor coverage: ' + label)
        for name, target in targets.items():
            source = sources[name]
            if (not isinstance(source, torch.Tensor) or source.device.type != 'cpu'
                    or source.shape != target.shape or source.dtype != target.dtype
                    or not torch.isfinite(source).all()):
                raise ValueError('Invalid checkpoint tensor: ' + label + '/' + name)

    validate_tensors(saved, live, 'model_shards')
    if state.get('complete_internal', False):
        local = ({name: parameter._local_shard for name, parameter in live.items()
                  if hasattr(parameter, '_local_shard')} if module._is_root is not None
                 else live)
        validate_tensors(state['local_shards'], local, 'local_shards')
        validate_tensors(state['buffers'], dict(module.named_buffers()), 'buffers')
    saved_groups = state['optimizer']['param_groups']
    slots = state['optimizer']['state']
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError('Optimizer group count changed')
    identities = []
    if expected_adam_steps is not None and not slots:
        raise ValueError('Saved Adam slots are empty at an accepted update boundary')
    for saved_group, group in zip(saved_groups, optimizer.param_groups):
        if (len(saved_group['params']) != len(group['params'])
                or {key: value for key, value in saved_group.items() if key not in ('params', 'lr')}
                != {key: value for key, value in group.items() if key not in ('params', 'lr')}):
            raise ValueError('Optimizer parameter group/hyperparameters changed')
        for identity, parameter in zip(saved_group['params'], group['params']):
            identities.append(identity)
            slot = slots.get(identity)
            if slot is None:
                continue
            required = {'step', 'exp_avg', 'exp_avg_sq'}
            if group.get('amsgrad', False):
                required.add('max_exp_avg_sq')
            if set(slot) != required:
                raise ValueError('Unexpected AdamW state fields')
            step = slot['step']
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not torch.isfinite(step).all() or step.item() < 0
                    or step.item() != int(step.item())
                    or (expected_adam_steps is not None and step.item() != expected_adam_steps)):
                raise ValueError('Invalid or unexpected actual Adam update count')
            for key in required - {'step'}:
                value = slot[key]
                if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                        or value.shape != parameter.shape or value.dtype != parameter.dtype
                        or not torch.isfinite(value).all()):
                    raise ValueError('Invalid AdamW moment: ' + key)
    if len(set(identities)) != len(identities) or not set(slots).issubset(identities):
        raise ValueError('Invalid optimizer parameter identity mapping')
    schedule = state['scheduler']
    rates = [base * fn(state['logical_step']) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    if (schedule.get('last_epoch') != state['logical_step']
            or len(schedule.get('_last_lr', [])) != len(optimizer.param_groups)
            or schedule.get('base_lrs') != scheduler.base_lrs
            or schedule.get('_step_count') != state['logical_step'] + 1
            or schedule.get('_last_lr') != rates
            or [group.get('lr') for group in saved_groups] != rates
            or any(not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate < 0
                   for rate in schedule['_last_lr'])):
        raise ValueError('Invalid scheduler learning rate or position')
    validate_rng(state['rng'])
    if set(state['sharding_rng']) != {'gen_random_states', 'torch_random_states'}:
        raise ValueError('Sharding RNG field coverage changed')
    for key, value in state['sharding_rng'].items():
        current = getattr(sharding_manager, key, None)
        if value is None:
            if current is not None:
                raise ValueError('Missing saved sharding RNG')
        elif (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.device.type != 'cpu' or current is None or value.shape != current.shape):
            raise ValueError('Invalid sharding RNG')
    if 'runtime_metadata' in state:
        from search_r1.baselines.instruct300 import equal_state
        metadata = state['runtime_metadata']
        required = {'rank', 'rng', 'scheduler', 'module_modes', 'parameters', 'buffers',
                    'local_shards', 'optimizer_tensors', 'sharding_rng'}
        if 'fresh200_live_admission' in metadata:
            required.add('fresh200_live_admission')
        if 'fresh200_worker_ledger' in state:
            from search_r1.baselines.fresh200 import validate_worker_attempt_ledger
            validate_worker_attempt_ledger(state['fresh200_worker_ledger'])
            required.add('fresh200_worker_ledger')
            if (not equal_state(metadata.get('fresh200_worker_ledger'), state['fresh200_worker_ledger'])
                    or state['fresh200_worker_ledger']['completed_outer_updates'] != state['logical_step']
                    or (expected_adam_steps is not None and
                        state['fresh200_worker_ledger']['actual_adam_total'] != expected_adam_steps)):
                raise ValueError('Saved actual worker attempt ledger differs from the accepted boundary')
        if (not isinstance(metadata, dict) or set(metadata) != required or metadata['rank'] != rank
                or not equal_state(metadata['rng'], state['rng'])
                or not equal_state(metadata['scheduler'], state['scheduler'])
                or not equal_state(metadata['sharding_rng'], state['sharding_rng'])):
            raise ValueError('Saved complete runtime metadata disagrees with the actual rank state')
        modes = {name: child.training for name, child in module.named_modules()}
        if (metadata['module_modes'].keys() != modes.keys()
                or any(type(mode) is not bool for mode in metadata['module_modes'].values())):
            raise ValueError('Invalid saved module training-mode coverage')
        legal_devices = {'cpu'}
        if torch.cuda.is_initialized():
            legal_devices.add(str(torch.device('cuda', torch.cuda.current_device())))
        parameter_ids = {id(parameter): identity for saved_group, group in zip(saved_groups, optimizer.param_groups)
                         for identity, parameter in zip(saved_group['params'], group['params'])}
        saved_moments = {str(index) + '/' + key: value for index, parameter in enumerate(module.parameters())
                         for key, value in sorted(slots.get(parameter_ids[id(parameter)], {}).items())
                         if isinstance(value, torch.Tensor)}
        for key, sources in [('parameters', saved), ('buffers', state['buffers']),
                             ('local_shards', state['local_shards']), ('optimizer_tensors', saved_moments)]:
            if metadata[key].keys() != sources.keys():
                raise ValueError('Invalid saved placement field coverage: ' + key)
            for name, source in sources.items():
                placement = metadata[key][name]
                if (set(placement) != {'device', 'dtype', 'shape'} or placement['device'] not in legal_devices
                        or placement['dtype'] != str(source.dtype) or placement['shape'] != list(source.shape)):
                    raise ValueError('Invalid saved CPU/current-device placement: ' + key + '/' + name)


def _validate_checkpoint_runtime_placement(worker, metadata):
    """Only the original configured CPU/current-CUDA offload layout is legal."""
    cuda_device = str(torch.device('cuda', torch.cuda.current_device()))
    parameter_device = 'cpu' if worker._is_offload_param else cuda_device
    for key in ('parameters', 'local_shards'):
        if any(value['device'] != parameter_device for value in metadata[key].values()):
            raise ValueError('Saved parameter placement differs from the original offload configuration')
    if worker._is_offload_optimizer:
        if any(value['device'] != 'cpu' for value in metadata['optimizer_tensors'].values()):
            raise ValueError('Saved Adam placement differs from the original CPU offload configuration')
    else:
        for name, value in metadata['optimizer_tensors'].items():
            if not name.endswith('/step') and value['device'] != cuda_device:
                raise ValueError('Saved Adam moment is outside its original current-CUDA configuration')


def _restore_checkpoint_runtime_metadata(worker, metadata):
    """Restore saved placement with the existing FSDP offload helpers/modes."""
    live_protocol = worker.config.actor.get('fresh200_live_protocol') is True
    if ('fresh200_live_admission' in metadata) != live_protocol:
        raise ValueError('Saved and live worker Fresh200 admission protocols differ')
    _validate_checkpoint_runtime_placement(worker, metadata)
    module = worker.actor_module_fsdp
    if worker._is_offload_param:
        offload_fsdp_param_and_grad(module, offload_grad=worker._is_offload_grad)
    else:
        load_fsdp_param_and_grad(module, device_id=torch.cuda.current_device(), load_grad=worker._is_offload_grad)
    for name, buffer in module.named_buffers():
        buffer.data = buffer.data.to(metadata['buffers'][name]['device'])
    if worker._is_offload_optimizer:
        offload_fsdp_optimizer(worker.actor_optimizer)
    else:
        moments_need_loading = any(str(value.device) != metadata['optimizer_tensors'][str(index) + '/' + key]['device']
            for index, parameter in enumerate(module.parameters())
            for key, value in worker.actor_optimizer.state.get(parameter, {}).items()
            if isinstance(value, torch.Tensor) and key != 'step')
        if moments_need_loading:
            load_fsdp_optimizer(worker.actor_optimizer, device_id=torch.cuda.current_device())
        # Native Adam's non-capturable scalar step can legitimately remain on
        # CPU while moments are on GPU. The existing helper loads all tensors;
        # preserve that saved scalar placement after loading the real moments.
        for index, parameter in enumerate(module.parameters()):
            step = worker.actor_optimizer.state.get(parameter, {}).get('step')
            if isinstance(step, torch.Tensor):
                device = metadata['optimizer_tensors'][str(index) + '/step']['device']
                if str(step.device) != device:
                    worker.actor_optimizer.state[parameter]['step'] = step.to(device)
    # Direct assignments preserve mixed child modes instead of recursively
    # overwriting children through a parent train() call.
    for name, child in module.named_modules():
        child.training = metadata['module_modes'][name]
    if live_protocol:
        from search_r1.baselines.fresh200_live import restore_worker_admission_state
        restore_worker_admission_state(worker, metadata['fresh200_live_admission'])


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = optim_config.get("lr_warmup_steps", None)
            if num_warmup_steps is None:
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            if type(num_warmup_steps) is not int or not 0 <= num_warmup_steps <= total_steps:
                raise ValueError("Warmup must be a nonnegative integer outer-update count")

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            initial_step = self.config.actor.optim.get('initial_logical_step', 0) if self._is_actor else 0
            if initial_step:
                from search_r1.baselines.recovery import rebuild_scheduler_at_logical_step
                rebuild_scheduler_at_logical_step(self.actor_lr_scheduler, int(initial_step))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)
            if self.config.actor.get('baseline_method') == 'opsd':
                from search_r1.baselines.actor import DistillationActor
                self.actor = DistillationActor(self.config.actor, self.actor_module_fsdp,
                                               self.actor_optimizer, self.tokenizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()
            if self._is_actor:
                self.actor.rollout_sharding_manager = self.rollout_sharding_manager

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.ref.get('model_path', self.config.model.path),
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')
        if data.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1':
            data.meta_info.update(pad_token_id=self.tokenizer.pad_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id)

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            attempt_receipt = None
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                from search_r1.baselines.policy_safety import PolicySafetyRejected
                try:
                    if self.config.actor.get('fresh200_protocol', False) and 'fresh200_attempt_id' in data.meta_info:
                        metrics, attempt_receipt = self._fresh200_update_attempt(data)
                    else:
                        metrics = self.actor.update_policy(data=data)
                except PolicySafetyRejected as rejection:
                    if self.config.actor.get('preserve_partial_policy_safety_state', False):
                        from search_r1.baselines.partial_update import preserve_worker
                        try:
                            preserve_worker(self, data, rejection)
                        except BaseException as preservation_error:
                            # Retain the refusal and explicitly report a failed snapshot.
                            # Unknown unsynchronized failures never enter a new FSDP collective.
                            rejection.record['partial_state_preservation_error'] = repr(preservation_error)
                            try:
                                from search_r1.baselines.recovery import atomic_json_save
                                from pathlib import Path
                                outer = data.meta_info['baseline_outer_step']
                                from search_r1.baselines.partial_update import _folder
                                folder = _folder(outer, data.meta_info.get('fresh200_attempt_id'))
                                atomic_json_save(dict(status='PARTIAL_WORKER_STATE_SAVE_FAILED', rank=self.rank,
                                    pid=os.getpid(), outer_step=outer, attempt_id=data.meta_info.get('fresh200_attempt_id'),
                                    job_id=os.environ.get('SLURM_JOB_ID'),
                                    run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                                    source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                                    config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                                    rejection=rejection.record, preservation_error=repr(preservation_error)),
                                    folder / f'rank_{self.rank}-save-failed.json')
                            except BaseException as diagnostic_error:
                                rejection.record['partial_state_failure_diagnostic_write_error'] = repr(diagnostic_error)
                    raise
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            if attempt_receipt is not None:
                from search_r1.baselines.fresh200 import record_worker_attempt
                record_worker_attempt(self, attempt_receipt)
            elif metrics.get('actor/optimizer_steps', 0) > 0:
                self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            if attempt_receipt is not None:
                import numpy as np
                output = DataProto(non_tensor_batch={'fresh200_attempt_receipts':
                    np.asarray([attempt_receipt], dtype=object)}, meta_info={'metrics': metrics})
            else:
                output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    def _fresh200_update_attempt(self, data):
        """Resume only the next mini after a known matched numeric/gradient refusal."""
        from search_r1.baselines.policy_safety import PolicySafetyRejected
        from search_r1.baselines.partial_update import (continue_worker_after_rejection,
            rejection_disposition, _adam_steps)
        meta = data.meta_info
        prior = getattr(self, 'fresh200_worker_ledger', dict(attempts=0,
            completed_outer_updates=0, actual_adam_total=0))
        error = None
        try:
            if (meta['fresh200_attempt_id'] != prior['attempts'] + 1
                    or meta['fresh200_completed_before'] != prior['completed_outer_updates']
                    or meta['fresh200_adam_total_before'] != prior['actual_adam_total']
                    or self.actor_lr_scheduler.last_epoch != prior['completed_outer_updates']):
                raise ValueError('Attempt identity does not follow the same live worker ledger')
            if _adam_steps({'optimizer': self.actor_optimizer.state_dict()}) != prior['actual_adam_total']:
                raise ValueError('Worker ledger differs from actual Adam slots before attempt')
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('attempt/entry_actual_counters', error)
        self.actor._fresh200_attempt_id = meta['fresh200_attempt_id']
        self.actor._fresh200_completed_before = meta['fresh200_completed_before']
        self.actor._fresh200_attempt_adam_writes = 0
        self.actor._fresh200_attempt_gradient_micros = 0
        self.actor._fresh200_attempt_audit = None
        self.actor._fresh200_written_mini = None
        meta.update(fresh200_attempt_adam_writes=0, fresh200_processed_opportunities=0,
                    fresh200_skipped_opportunities=0)
        refusals, abandoned, processed = [], False, 0
        while processed < 20:
            try:
                metrics = self.actor.update_policy(data=data)
                processed = int(metrics['actor/opportunities_processed'])
                break
            except PolicySafetyRejected as rejection:
                disposition = rejection_disposition(rejection.record)
                if disposition == 'fail_closed':
                    raise  # Config/entry failures cannot authorize a fresh continuation.
                refusal = continue_worker_after_rejection(self, data, rejection, len(refusals) + 1)
                refusals.append(refusal)
                processed = rejection.record['mini_index'] + 1
                actual_writes = self.actor._fresh200_attempt_adam_writes
                if actual_writes != rejection.record['actual_optimizer_writes']:
                    raise RuntimeError('Actual successful optimizer calls differ from matched refusal count')
                meta.update(fresh200_attempt_adam_writes=actual_writes,
                    fresh200_processed_opportunities=processed,
                    fresh200_skipped_opportunities=processed - actual_writes)
                abandoned = disposition == 'abandon'
                if abandoned or processed == 20:
                    metrics = dict(actor_placeholder=0)
                    audit = self.actor._fresh200_attempt_audit
                    if audit is not None:
                        audit.finish(actual_writes, processed - actual_writes,
                                     complete=not abandoned and actual_writes > 0)
                    break
                # Only matched refusal reaches here. Unknown original model,
                # optimizer, Ray or collective errors escape without another MAX.
        writes = self.actor._fresh200_attempt_adam_writes
        micros = self.actor._fresh200_attempt_gradient_micros
        countable = processed == 20 and writes > 0 and not abandoned
        error, total = None, None
        try:
            self.actor_optimizer.zero_grad(set_to_none=True)
            tensors = list(self.actor_module_fsdp.parameters()) + list(self.actor_module_fsdp.buffers())
            tensors += [p._local_shard for p in self.actor_module_fsdp.parameters() if hasattr(p, '_local_shard')]
            tensors += [value for slot in self.actor_optimizer.state.values()
                        for value in slot.values() if isinstance(value, torch.Tensor)]
            if (any(p.grad is not None for p in self.actor_module_fsdp.parameters())
                    or not all(bool(torch.isfinite(value).all()) for value in tensors)):
                raise ValueError('Attempt endpoint has pending gradients or nonfinite model/Adam')
            total = _adam_steps({'optimizer': self.actor_optimizer.state_dict()})
            if total != meta['fresh200_adam_total_before'] + writes or not 0 <= writes <= processed <= 20:
                raise ValueError('Actual Adam slots differ from successful optimizer calls')
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('attempt/finite_actual_endpoint', error)
        signature = torch.tensor([meta['fresh200_attempt_id'], processed, writes, micros, total,
            int(abandoned), int(countable)], dtype=torch.int64, device='cuda')
        low, high = signature.clone(), signature.clone()
        torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX)
        if not torch.equal(low, high):
            raise RuntimeError('Actual eight-rank attempt accounting differs')
        receipt = dict(rank=self.rank, pid=os.getpid(), attempt_id=meta['fresh200_attempt_id'],
            completed_before=meta['fresh200_completed_before'], opportunities_processed=processed,
            actual_successful_adam_writes=writes, skipped_opportunities=processed - writes,
            gradient_micro_batches=micros, abandoned=abandoned, counted_complete=countable,
            actual_adam_total=total, scheduler_step=self.actor_lr_scheduler.last_epoch,
            safe_to_continue=True, refusals=refusals)
        metrics.update({'actor/optimizer_steps': writes, 'actor/opportunities_processed': processed,
            'actor/skipped_opportunities': processed - writes, 'actor/attempt_counted_complete': int(countable),
            'actor/actual_gradient_micro_batches': micros, 'actor/skipped_empty_minibatches': 0})
        return metrics, receipt

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """mostly copying from generate_sequences"""
        data = data.to('cuda')

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        data.meta_info.update(meta_info)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            old_log_probs = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': old_log_probs})
            output = self.ulysses_sharding_manager.postprocess_data(output)
            
        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output = self.ulysses_sharding_manager.preprocess_data(output)
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')
        if data.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1':
            data.meta_info.update(pad_token_id=self.tokenizer.pad_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id)

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fresh_initialization_identity(self):
        """Read actual initialized actor/reference metadata before first validation."""
        import json
        from pathlib import Path
        from search_r1.baselines.fresh200 import REVISION
        from search_r1.baselines.recovery import tensor_digest
        actor = self._is_actor
        path = self.config.model.path if actor else self.config.ref.get('model_path', self.config.model.path)
        inventory = json.loads((Path(path) / 'VERIFIED-MODEL-MANIFEST.json').read_text())
        if inventory['repo'] != 'Qwen/Qwen2.5-3B-Instruct' or inventory['revision'] != REVISION:
            raise RuntimeError('Actual worker did not load the pinned official Instruct model')
        module = self.actor_module_fsdp if actor else self.ref_module_fsdp
        result = dict(rank=self.rank, pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            role='actor' if actor else 'reference', origin_kind='OFFICIAL_MODEL_INITIALIZATION',
            official_revision=REVISION, model_path=path,
            model_shards_sha256=tensor_digest(module.named_parameters()),
            official_bf16_parameter_sha256=tensor_digest((name, parameter.detach().to('cpu', dtype=torch.bfloat16))
                for name, parameter in module.named_parameters()),
            buffers_sha256=tensor_digest(module.named_buffers()))
        if actor:
            result.update(optimizer_state_entries=len(self.actor_optimizer.state),
                optimizer_populated_slots=sum(bool(slot) for slot in self.actor_optimizer.state.values()),
                scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,
                scheduler_step_count=self.actor_lr_scheduler._step_count,
                learning_rates=self.actor_lr_scheduler.get_last_lr(),
                local_mini_batch_size=self.config.actor.ppo_mini_batch_size,
                local_micro_batch_size=self.config.actor.ppo_micro_batch_size,
                reference_kl_coefficient=self.config.actor.kl_loss_coef)
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def checkpoint_pause_identity(self):
        assert self._is_actor
        from search_r1.baselines.recovery import checkpoint_pause_identity
        return checkpoint_pause_identity(self.actor_module_fsdp, self.actor_optimizer,
                                         self.actor_lr_scheduler, self.rank,
                                         self.rollout_sharding_manager,
                                         self.config.actor.get("complete_internal_state", False))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def checkpoint_runtime_metadata(self):
        """Read complete small state, including RNG absent from tensor digests."""
        assert self._is_actor
        from search_r1.baselines.recovery import cpu_snapshot, rng_state
        module = self.actor_module_fsdp
        placement = lambda values: {name: dict(device=str(value.device), dtype=str(value.dtype),
                                               shape=list(value.shape)) for name, value in values}
        result = dict(rank=self.rank, rng=cpu_snapshot(rng_state()),
            scheduler=cpu_snapshot(self.actor_lr_scheduler.state_dict()),
            module_modes={name: child.training for name, child in module.named_modules()},
            parameters=placement(module.named_parameters()), buffers=placement(module.named_buffers()),
            local_shards=placement((name, parameter._local_shard) for name, parameter in module.named_parameters()
                                   if hasattr(parameter, '_local_shard')),
            optimizer_tensors=placement((str(index) + '/' + key, value)
                for index, parameter in enumerate(module.parameters())
                for key, value in sorted(self.actor_optimizer.state.get(parameter, {}).items())
                if isinstance(value, torch.Tensor)),
            sharding_rng={key: cpu_snapshot(getattr(self.rollout_sharding_manager, key, None))
                          for key in ('gen_random_states', 'torch_random_states')})
        if self.config.actor.get('fresh200_protocol', False):
            result['fresh200_worker_ledger'] = cpu_snapshot(getattr(self, 'fresh200_worker_ledger',
                dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1', attempts=0,
                     completed_outer_updates=0, actual_adam_total=0, history=[])))
        if self.config.actor.get('fresh200_live_protocol') is True:
            from search_r1.baselines.fresh200_live import worker_admission_state
            result['fresh200_live_admission'] = cpu_snapshot(worker_admission_state(self))
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_training_state(self, local_path, expected_rank_shas, expected_logical_step=None, expected_optimizer_steps=None):
        assert self._is_actor
        from search_r1.baselines.recovery import file_sha256, load_worker_training_state
        path = os.path.join(local_path, 'training_state', f'rank_{self.rank}.pt')
        state, error = None, None
        try:
            if file_sha256(path) != expected_rank_shas[str(self.rank)]:
                raise RuntimeError('Training-state SHA differs from frozen checkpoint manifest')
            state = torch.load(path, map_location='cpu', weights_only=False)
            _validate_checkpoint_state(state, self.actor_module_fsdp, self.actor_optimizer,
                self.actor_lr_scheduler, self.rank, self.world_size, self.rollout_sharding_manager,
                expected_logical_step, expected_optimizer_steps,
                self.config.actor.get('complete_internal_state', False))
            if self.config.actor.get('fresh200_protocol', False) and 'runtime_metadata' not in state:
                raise ValueError('Fresh200 state lacks complete live runtime metadata')
            if self.config.actor.get('fresh200_protocol', False) and 'fresh200_worker_ledger' not in state:
                raise ValueError('Fresh200 state lacks actual attempt/write accounting')
            if ('fresh200_live_admission' in state.get('runtime_metadata', {})) != (
                    self.config.actor.get('fresh200_live_protocol') is True):
                raise ValueError('Saved and live worker Fresh200 admission protocols differ')
            if 'runtime_metadata' in state:
                _validate_checkpoint_runtime_placement(self, state['runtime_metadata'])
            if state.get('complete_internal', False):
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                module = self.actor_module_fsdp
                if not isinstance(module, FSDP) or module._is_root is False:
                    raise ValueError('Full restore requires the actual FSDP root')
                if module._is_root is None:
                    if torch.__version__.split('+')[0].split('.')[:2] != ['2', '4']:
                        raise ValueError('Constructor FSDP initialization requires reviewed Torch 2.4')
                    if any(p.device != torch.device('cuda', torch.cuda.current_device())
                           for p in module.parameters()):
                        raise ValueError('Constructor FSDP root is not on its original CUDA device')
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('load/local_read_and_validate', error)
        initialization = None
        if state.get('complete_internal', False):
            from verl.utils.fsdp_utils import initialize_fsdp_before_full_restore
            # PyTorch initialization is outside local-error handling: an unknown
            # collective failure must terminate the job, not start a new reduce.
            initialization = initialize_fsdp_before_full_restore(self.actor_module_fsdp)
        error = None
        try:
            _validate_checkpoint_state(state, self.actor_module_fsdp, self.actor_optimizer,
                self.actor_lr_scheduler, self.rank, self.world_size, self.rollout_sharding_manager,
                expected_logical_step, expected_optimizer_steps,
                self.config.actor.get('complete_internal_state', False))
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('load/initialized_local_validation', error)
        error, step = None, None
        try:
            # This loader only copies local shards/buffers and restores local
            # Adam/scheduler/RNG. It performs no FSDP/distributed collective.
            step = load_worker_training_state(
                state, self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler,
                self.rank, self.world_size, self.rollout_sharding_manager)
            if 'runtime_metadata' in state:
                _restore_checkpoint_runtime_metadata(self, state['runtime_metadata'])
            if 'fresh200_worker_ledger' in state:
                from search_r1.baselines.recovery import cpu_snapshot
                self.fresh200_worker_ledger = cpu_snapshot(state['fresh200_worker_ledger'])
            from search_r1.baselines.instruct300 import equal_state
            actual_runtime = self.checkpoint_runtime_metadata()
            if (not equal_state(actual_runtime['rng'], state['rng'])
                    or not equal_state(actual_runtime['scheduler'], state['scheduler'])
                    or not equal_state(actual_runtime['sharding_rng'], state['sharding_rng'])
                    or ('runtime_metadata' in state
                        and not equal_state(actual_runtime, state['runtime_metadata']))):
                raise RuntimeError('Actual worker RNG/scheduler/modes/placement readback differs from saved state')
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('load/local_state_copy', error)
        del state
        torch.distributed.barrier()
        return dict(rank=self.rank, logical_step=step, fsdp_initialization=initialization,
                    actual_rank_pt_reload=True, matched_local_stages=True,
                    actual_complete_runtime_readback=True,
                    automatic_rollback=False, collective_failure_recovery=False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        error, before_save, staged_runtime = None, None, None
        try:
            if self.config.actor.get('fresh200_protocol', False):
                before_save = (self.checkpoint_runtime_metadata(), self.checkpoint_pause_identity())
            if self._is_offload_param:
                load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                         device_id=torch.cuda.current_device(),
                                         load_grad=self._is_offload_grad)
            if self.config.actor.get('fresh200_protocol', False):
                staged_runtime = self.checkpoint_runtime_metadata()
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('save/local_parameter_staging', error)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # Unknown full-state collective errors propagate directly. Only phases
        # below that are known local IO/computation enter matched error reduces.
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        error, hf_receipt = None, None
        try:
            if self.rank == 0:
                print(f'Saving actor checkpoint to {local_path}')
                os.makedirs(local_path, exist_ok=True)
                self.actor_module.save_pretrained(local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(local_path)
                if (self.config.actor.get('fresh200_protocol', False)
                        and self.config.actor.get('human_nonblocking_diagnostics') is not True):
                    from search_r1.baselines.instruct300 import hf_export_identity, reload_hf_checkpoint_cpu
                    exported = hf_export_identity(state_dict, self.actor_module.config, self.tokenizer)
                    # The child holds the FP32 HF model; release the ~12 GB
                    # full export first instead of holding two full CPU models.
                    del state_dict
                    hf_receipt = reload_hf_checkpoint_cpu(local_path, exported)
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path)
        except Exception as local_error:
            error = local_error
        # Every rank releases its export even when rank0's local write failed.
        if 'state_dict' in locals():
            del state_dict
        _checkpoint_matched_stage('save/rank0_hf_write_and_cpu_reload', error)
        torch.distributed.barrier()
        receipt, error = None, None
        try:
            if self.config.actor.optim.get('save_training_state', False):
                from search_r1.baselines.recovery import atomic_torch_save, worker_training_state
                if before_save is not None:
                    from search_r1.baselines.instruct300 import equal_state
                    if (not equal_state(staged_runtime, self.checkpoint_runtime_metadata())
                            or before_save[1] != self.checkpoint_pause_identity()):
                        raise RuntimeError('HF save/reload changed live model/Adam/scheduler/buffers/RNG/PID')
                logical_step = int(os.path.basename(local_path).removeprefix('global_step_'))
                state = worker_training_state(
                    self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler,
                    self.rank, self.world_size, logical_step, self.rollout_sharding_manager,
                    self.config.actor.get("complete_internal_state", False))
                if self.config.actor.get('fresh200_protocol', False):
                    # Normal save temporarily stages offloaded parameters on
                    # GPU. Bind the placement returned to the caller, not that
                    # temporary export placement, while retaining all real RNG.
                    state['runtime_metadata'] = before_save[0]
                    state['fresh200_worker_ledger'] = before_save[0]['fresh200_worker_ledger']
                    _validate_checkpoint_state(state, self.actor_module_fsdp, self.actor_optimizer,
                        self.actor_lr_scheduler, self.rank, self.world_size, self.rollout_sharding_manager,
                        logical_step, state['fresh200_worker_ledger']['actual_adam_total'], require_complete=True)
                receipt = atomic_torch_save(state, os.path.join(local_path, 'training_state',
                                                               f'rank_{self.rank}.pt'))
                receipt['pause_identity'] = state['pause_identity']
                if hf_receipt is not None:
                    receipt['hf_loader_roundtrip'] = hf_receipt
                del state
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('save/all_rank_training_state_write', error)
        torch.distributed.barrier()
        error = None
        try:
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
            if before_save is not None:
                from search_r1.baselines.instruct300 import equal_state
                if (not equal_state(before_save[0], self.checkpoint_runtime_metadata())
                        or before_save[1] != self.checkpoint_pause_identity()):
                    raise RuntimeError('Completed checkpoint save changed original worker runtime state')
        except Exception as local_error:
            error = local_error
        _checkpoint_matched_stage('save/local_parameter_offload', error)
        return receipt


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            forward_prefetch=False)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
