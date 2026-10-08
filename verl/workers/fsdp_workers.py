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
        if self.config.actor.get('fresh200_enabled', False):
            from search_r1.baselines.fresh200_contract import verify_official_model_assets
            from search_r1.baselines.policy_safety import consensus
            ok=True
            if self.rank==0:
                try:
                    verify_official_model_assets(local_path)
                except Exception as error:
                    print('FRESH200_OFFICIAL_MODEL_REJECTED='+repr(error),flush=True)
                    ok=False
            if not consensus(ok):
                raise ValueError('Fresh200 official local model assets rejected')

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        model_revision = self.config.model.get('revision') if self.config.actor.get('fresh200_enabled', False) else None
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, revision=model_revision)

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
                                                                trust_remote_code=trust_remote_code, revision=model_revision)
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
            from search_r1.baselines.full_state import resolve_warmup_steps
            num_warmup_steps = resolve_warmup_steps(optim_config)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            if self.config.actor.get('fresh200_enabled', False):
                from torch.optim.lr_scheduler import LambdaLR
                from search_r1.baselines.fresh200_contract import fresh200_lr_factor
                actor_lr_scheduler = LambdaLR(actor_optimizer, fresh200_lr_factor)
            else:
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

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
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

        if self._is_actor and self.config.actor.get('fresh200_enabled', False):
            import json
            from search_r1.baselines.fresh200_native_evidence import tensor_digest,native_process,file_sha,capture_official_alignment
            from search_r1.baselines.fresh200_contract import critical_source_hashes
            from pathlib import Path
            directory=Path(self.config.actor.fresh200_fault_dir);directory_error=None
            from search_r1.baselines.policy_safety import consensus
            try:directory.mkdir(parents=True,exist_ok=True)
            except OSError as error:directory_error=str(error)
            if not consensus(directory_error is None):raise ValueError('Collective native initial state directory failed')
            initial_fresh=(len(self.actor_optimizer.state)==0 and self.actor_lr_scheduler.last_epoch==0
                           and self.world_size==8 and all(group['lr']==1e-6/57 for group in self.actor_optimizer.param_groups))
            if not consensus(initial_fresh): raise RuntimeError('Fresh200 official initial actor requires Adam0/scheduler0/initialLR/8ranks')
            official_alignment=capture_official_alignment(self,directory)
            initial_error=None
            try:
                path=directory/f'native-initial-rank_{self.rank}.pt'
                metadata_path=directory/f'native-initial-rank_{self.rank}.json'
                if path.exists() or metadata_path.exists():raise ValueError('Refusing to overwrite immutable native initial state')
                parameters={name:param.detach().cpu().clone() for name,param in self.actor_module_fsdp.named_parameters()}
                process=native_process(self.rank,self.world_size)
                self._fresh200_native_process=process
                payload=dict(model_parameters=parameters,optimizer=self.actor_optimizer.state_dict(),
                    scheduler=self.actor_lr_scheduler.state_dict(),native_process=process,source_sha256=critical_source_hashes())
                torch.save(payload,path)
                self._fresh200_initial_state = dict(path=str(path.resolve()),sha256=file_sha(path),
                    official_model_manifest_sha256='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44',
                    model_tensor_sha256={name:tensor_digest(param) for name,param in parameters.items()},official_alignment=official_alignment,
                    optimizer_state_entries=len(self.actor_optimizer.state),scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,
                    lr=self.actor_lr_scheduler.get_last_lr(),native_process=process,source_sha256=critical_source_hashes(),
                    status='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE')
                metadata_path.write_text(json.dumps(self._fresh200_initial_state,indent=2,allow_nan=False))
                del parameters,payload
            except Exception as error:
                initial_error=str(error)
                print('FRESH200_NATIVE_INITIAL_CAPTURE_FAULT='+repr(error),flush=True)
            if not consensus(initial_error is None):raise ValueError('Collective native initial state/provenance capture failed')
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def capture_fresh200_diagnostic(self, data: DataProto, global_mini, role, directory):
        assert self._is_actor
        from search_r1.baselines.fresh200_native_evidence import run_diagnostic_update
        if self._is_offload_param:
            load_fsdp_param_and_grad(self.actor_module_fsdp,torch.cuda.current_device(),load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(self.actor_optimizer,torch.cuda.current_device())
        try:
            with self.ulysses_sharding_manager:
                data=self.ulysses_sharding_manager.preprocess_data(data.to('cuda'))
                try:
                    event=run_diagnostic_update(self,data,global_mini,role,directory)
                except Exception as error:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    import traceback,json
                    from pathlib import Path
                    event=dict(getattr(self,'_diagnostic_last_report',{}))
                    event.update(status='DIAGNOSTIC_PARTIAL_FAULT',role=role,global_mini_batch_size=global_mini,
                        rank=self.rank,optimizer_writes=getattr(self,'_diagnostic_write_count',0),
                        formal_outer_updates=0,scheduler_outer_steps=self.actor_lr_scheduler.last_epoch,
                        stop_reason=str(error),raw_exception=traceback.format_exc())
                    print('FRESH200_DIAGNOSTIC_RANK_FAULT_JSON='+json.dumps(event,default=str),flush=True)
                    try:
                        Path(directory,f'rank_{self.rank}_diagnostic_ledger.json').write_text(json.dumps(event,indent=2,default=str))
                    except OSError:pass
                import torch.distributed as dist
                events=[None]*self.world_size;dist.all_gather_object(events,event)
                return DataProto(meta_info={'fresh200_diagnostic_events':events})
        finally:
            if self._is_offload_param:offload_fsdp_param_and_grad(self.actor_module_fsdp,offload_grad=self._is_offload_grad)
            if self._is_offload_optimizer:offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to(torch.device('cuda', torch.cuda.current_device()))

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda(device=torch.cuda.current_device())

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # Validate reporting topology before any optimizer update.
            aggregate_bapo_metrics = (self.config.actor.get('baseline_method') == 'bapo'
                                      and (self.config.actor.get('loss_reduction') == 'sequence_mean' or self.config.actor.get('baseline_aligned', False)))
            if aggregate_bapo_metrics:
                from search_r1.baselines.actor_metrics import validate_actor_metric_topology
                validate_actor_metric_topology(self.ulysses_sequence_parallel_size,
                                               self.config.rollout.tensor_model_parallel_size)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                lr_before = self.actor_lr_scheduler.get_last_lr()[0]
                fresh200 = self.config.actor.get('fresh200_enabled', False)
                try:
                    metrics = self.actor.update_policy(data=data)
                except Exception as error:
                    if not fresh200 or not isinstance(getattr(error, 'event', None), dict):
                        if fresh200:
                            import traceback
                            from pathlib import Path
                            fault = Path(self.config.actor.fresh200_fault_dir)
                            fault.mkdir(parents=True, exist_ok=True)
                            (fault / ('rank_%d-outer_%d.raw.log' % (self.rank, data.meta_info['actual_outer']))).write_text(traceback.format_exc())
                        raise
                    metrics = {'fresh200_update_state': error.event,
                               'actor/optimizer_steps': error.event['optimizer_steps'],
                               'fresh200_policy_monitor': getattr(error, 'reports', [])}
                rank_updates = None
                if fresh200:
                    from search_r1.baselines.fresh200_contract import validate_rank_updates
                    import torch.distributed as dist
                    event = dict(metrics.pop('fresh200_update_state'))
                    event['policy_monitor'] = metrics.pop('fresh200_policy_monitor', [])
                    event['rank'] = self.rank
                    rank_updates = [None] * self.world_size
                    dist.all_gather_object(rank_updates, event)
                    from search_r1.baselines.policy_safety import agree_document
                    decision_keys=('status','update_complete','attempt_complete','optimizer_steps',
                        'expected_optimizer_steps','optimizer_opportunities_total','processed_opportunities',
                        'skipped_opportunities','unprocessed_opportunities','skipped_indices','skipped_reasons',
                        'postabort','actual_outer','attempt_id')
                    self._fresh200_last_update_event=event
                    fatal_update=any(item.get('status')=='INFRASTRUCTURE_FAULT' for item in rank_updates)
                    decision={key:event.get(key) for key in decision_keys}
                    if not fatal_update:
                        if not agree_document(decision,group=self.actor_module_fsdp.process_group):
                            raise RuntimeError('All-rank skip/write decision mismatch; scheduler refused')
                        validate_rank_updates(rank_updates, data.meta_info['actual_outer'])
                    from pathlib import Path
                    import json
                    fault = Path(self.config.actor.fresh200_fault_dir)
                    from search_r1.baselines.policy_safety import persist_rank_update_event
                    path=fault / ('rank_%d-outer_%d-attempt_%s.json' % (self.rank,data.meta_info['actual_outer'],data.meta_info.get('attempt_id','missing')))
                    event,fatal_events=persist_rank_update_event(path,event,self.actor_module_fsdp.process_group)
                    if fatal_events is not None:
                        rank_updates=fatal_events;fatal_update=True
                        self._fresh200_last_update_event=event
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            if fresh200:
                from search_r1.baselines.fresh200_contract import validate_rank_updates
                if not fatal_update and validate_rank_updates(rank_updates, data.meta_info['actual_outer']):
                    self.actor_lr_scheduler.step()
            elif metrics.get('actor/optimizer_steps', 0) > 0:
                self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr
            metrics['actor/lr_before_outer'] = lr_before
            metrics['actor/scheduler_outer_steps'] = self.actor_lr_scheduler.last_epoch
            print('ACTOR_RANK_UPDATE_JSON=' + __import__('json').dumps(dict(rank=self.rank, lr_before=lr_before, lr_after=lr, scheduler_last_epoch=self.actor_lr_scheduler.last_epoch, optimizer_steps=metrics.get('actor/optimizer_steps'), metrics=metrics), default=lambda x:x.item(), allow_nan=False), flush=True)

            log_gpu_memory_usage('After update policy', logger=logger)

            # Preserve ACTOR_RANK_UPDATE_JSON above as the original local
            # record. Every returned DataProto now carries the same global
            # reporting metrics, so concat retaining first metadata is safe.
            if fresh200 and fatal_update:
                # Local complete/partial commit counts can differ on an infrastructure fault.
                # Their exact truth lives in rank_updates; never average/replicate that metric.
                metrics={'actor/lr':lr,'actor/scheduler_outer_steps':self.actor_lr_scheduler.last_epoch}
            elif aggregate_bapo_metrics:
                from search_r1.baselines.actor_metrics import aggregate_sequence_mean_actor_metrics
                metrics = aggregate_sequence_mean_actor_metrics(
                    metrics, self.ulysses_sequence_parallel_size,
                    self.config.rollout.tensor_model_parallel_size)
            output = DataProto(meta_info={'metrics': metrics, **({'fresh200_rank_updates': rank_updates} if fresh200 else {})})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """mostly copying from generate_sequences"""
        data = data.to(torch.device('cuda', torch.cuda.current_device()))

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda(device=torch.cuda.current_device())
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        data.meta_info.update(meta_info)
        fresh200=getattr(self.config, 'actor', {}).get('fresh200_enabled',False)
        old_policy_meta={}
        if fresh200:
            old_policy_meta=dict(micro_batch_size=self.config.rollout.log_prob_micro_batch_size,
                temperature=self.config.rollout.temperature,use_dynamic_bsz=self.config.rollout.log_prob_use_dynamic_bsz,
                max_token_len=self.config.rollout.log_prob_max_token_len_per_gpu)
            data.meta_info.update(old_policy_meta)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            old_log_probs = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': old_log_probs},meta_info=old_policy_meta)
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
        prompts = prompts.to(torch.device('cuda', torch.cuda.current_device()))
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda(device=torch.cuda.current_device())
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

        data = data.to(torch.device('cuda', torch.cuda.current_device()))

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
    def load_training_state(self, manifest, witness_directory):
        """Fresh-process same-topology restore; historical generation RNG is absent."""
        if self.config.actor.get('fresh200_enabled', False):
            raise ValueError('Fresh200 forbids restoration of old model/Adam/scheduler')
        import json,os
        from pathlib import Path
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig
        from search_r1.baselines.full_state import rng_state,restore_rng,state_digest
        from search_r1.baselines.recovery import validate_native,exact_file,OLD_RUN,write_witness,require,process_identity
        require(self._is_actor and self.world_size==8,'Actual eight-rank actor required')
        entry=next(x for x in manifest['ranks']if x['rank']==self.rank)
        path=exact_file(entry['state'],OLD_RUN);report=json.loads(exact_file(entry['report'],OLD_RUN).read_text())
        if self._is_offload_param:
            load_fsdp_param_and_grad(self.actor_module_fsdp,torch.cuda.current_device(),load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(self.actor_optimizer,torch.cuda.current_device())
        try:
            with FSDP.state_dict_type(self.actor_module_fsdp,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
                error=None
                try:
                    loaded=torch.load(path,map_location='cpu',weights_only=False)
                    validate_native(loaded,report,self.rank,self.world_size)
                except Exception as exc:
                    error=f'rank{self.rank}: {type(exc).__name__}: {exc}'
                errors=[None]*self.world_size
                torch.distributed.all_gather_object(errors,error)
                require(not any(errors),'Native input validation rejected on all ranks: '+str(errors))
                self.actor_module_fsdp.load_state_dict(loaded['model'],strict=True)
                self.actor_optimizer.load_state_dict(loaded['optimizer'])
                self.actor_lr_scheduler.load_state_dict(loaded['scheduler'])
                restore_rng(loaded['rng'])
                actual=dict(version=2,model_state_dict_type='SHARDED_STATE_DICT',rank=self.rank,world_size=self.world_size,step=25,
                            model=self.actor_module_fsdp.state_dict(),optimizer=self.actor_optimizer.state_dict(),
                            scheduler=self.actor_lr_scheduler.state_dict(),rng=rng_state(torch.cuda.current_device()))
                require(state_digest(actual)==report['state_digest'],'Fresh-process model/Adam/scheduler/RNG reload differs')
            require(os.getpid()!=report['pid'],'Fresh-process PID did not change')
            require(os.environ.get('SLURM_JOB_ID')==str(manifest['target_job_id'])and os.environ.get('BAPO_RUN_ID')==manifest['target_run_root'],'Actual rank job/run mismatch')
            witness=dict(status='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK',rank=self.rank,world_size=self.world_size,
                step=25,next_step=26,old_pid=report['pid'],new_pid=os.getpid(),process=process_identity(),
                runtime_module_path=str(Path(__file__).resolve()),runtime_module_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),old_state_sha256=entry['state']['sha256'],
                distributed_backend=torch.distributed.get_backend(),model_state_dict_type=actual['model_state_dict_type'],
                model_dtensor_entries=sum(hasattr(t,'device_mesh')and hasattr(t,'to_local')for t in actual['model'].values()),
                model_class=self.actor_module.__class__.__name__,
                model_config={k:getattr(self.actor_model_config,k)for k in ('model_type','hidden_size','num_hidden_layers','vocab_size','architectures')},
                state_digest=state_digest(actual),scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,
                optimizer_state_entries=len(self.actor_optimizer.state),lr=self.actor_lr_scheduler.get_last_lr(),
                target_source_commit=manifest['target_source_commit'],target_run_root=manifest['target_run_root'],
                actual_env={k:os.environ.get(k)for k in ('SLURM_JOB_ID','BAPO_RUN_ID','CUDA_VISIBLE_DEVICES')},
                sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',
                reconstructed_generation_rng_digest=state_digest(self.rollout_sharding_manager.generation_rng_state()))
            write_witness(Path(witness_directory)/f'rank_{self.rank}.json',witness)
            print('BAPO_RECOVERY_RANK_JSON='+json.dumps(witness,allow_nan=False),flush=True)
            torch.distributed.barrier()
            return witness
        finally:
            if self._is_offload_param:offload_fsdp_param_and_grad(self.actor_module_fsdp,offload_grad=self._is_offload_grad)
            if self._is_offload_optimizer:offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_training_state(self, local_path, step, roundtrip=False, state_status=None):
        """Per-rank same-topology SHARDED/DTensor model plus native Adam/RNG."""
        import json
        from pathlib import Path
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig
        from search_r1.baselines.full_state import rng_state, restore_rng, state_digest, atomic_save
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(self.actor_module_fsdp, torch.cuda.current_device(), load_grad=self._is_offload_grad)
        directory=Path(local_path);directory.mkdir(parents=True,exist_ok=True)
        try:
            with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.SHARDED_STATE_DICT,
                                     ShardedStateDictConfig(offload_to_cpu=True)):
                payload=dict(version=3,model_state_dict_type='SHARDED_STATE_DICT',rank=self.rank,world_size=self.world_size,step=step,
                    model=self.actor_module_fsdp.state_dict(), optimizer=self.actor_optimizer.state_dict(),
                    scheduler=self.actor_lr_scheduler.state_dict(), rng=rng_state(torch.cuda.current_device()),
                    generation_rng=self.rollout_sharding_manager.generation_rng_state())
                if self.config.actor.get('fresh200_diagnostic_only',False):
                    payload.update(diagnostic_optimizer_writes=getattr(self,'_diagnostic_write_count',0),formal_outer_updates=0,scheduler_outer_steps=self.actor_lr_scheduler.last_epoch)
                if self.config.actor.get('fresh200_enabled',False):
                    payload['latest_rank_event']=getattr(self,'_fresh200_last_update_event',None)
                    payload['attempt_id']=None if payload['latest_rank_event'] is None else payload['latest_rank_event'].get('attempt_id')
                if state_status is not None:
                    payload['update_status']=state_status
                    payload['attempted_outer']=step
                    payload['previous_complete_outer']=step-1 if state_status in ('PARTIAL','ABORTED','SKIPPED') else step
                before=state_digest(payload)
                generation_before=state_digest(payload['generation_rng'])
                after_digest=before;generation_after=generation_before
                path=directory/('rank_%d.pt'%self.rank);atomic_save(payload,path)
                loaded=torch.load(path,map_location='cpu',weights_only=False)
                if state_digest(loaded)!=before:raise ValueError('Serialized full state changed')
                if roundtrip:
                    self.actor_module_fsdp.load_state_dict(loaded['model'],strict=True)
                    self.actor_optimizer.load_state_dict(loaded['optimizer'])
                    self.actor_lr_scheduler.load_state_dict(loaded['scheduler'])
                    restore_rng(loaded['rng'])
                    self.rollout_sharding_manager.load_generation_rng_state(loaded['generation_rng'])
                    after=dict(version=3,model_state_dict_type='SHARDED_STATE_DICT',rank=self.rank,world_size=self.world_size,step=step,
                        model=self.actor_module_fsdp.state_dict(),optimizer=self.actor_optimizer.state_dict(),
                        scheduler=self.actor_lr_scheduler.state_dict(),rng=rng_state(torch.cuda.current_device()),
                        generation_rng=self.rollout_sharding_manager.generation_rng_state())
                    if self.config.actor.get('fresh200_enabled',False):
                        after.update(latest_rank_event=payload.get('latest_rank_event'),attempt_id=payload.get('attempt_id'))
                    if state_status is not None:
                        after.update(update_status=state_status,attempted_outer=step,previous_complete_outer=step-1 if state_status in ('PARTIAL','ABORTED','SKIPPED') else step)
                    after_digest=state_digest(after);generation_after=state_digest(after['generation_rng'])
                    if after_digest!=before:raise ValueError('Full state reload changed model/Adam/scheduler/RNG')
            from search_r1.baselines.checkpoint_gate import digest
            report=dict(status='EXACT_FULL_STATE_RELOAD_OK' if roundtrip else 'FULL_STATE_SERIALIZED_OK',
                rank=self.rank,world_size=self.world_size,step=step,path=str(path),sha256=digest(path),
                model_state_dict_type='SHARDED_STATE_DICT',
                model_dtensor_entries=sum(hasattr(t,'device_mesh') and hasattr(t,'to_local') for t in loaded['model'].values()),
                state_digest=before,scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,
                optimizer_state_entries=len(self.actor_optimizer.state),pid=os.getpid(),native_state_version=3,
                roundtrip=bool(roundtrip),reloaded_state_digest=after_digest,
                generation_rng_before_digest=generation_before,generation_rng_after_digest=generation_after,
                generation_rng_topology=dict(version=loaded['generation_rng']['version'],
                    torch_random_states_numel=loaded['generation_rng']['torch_random_states'].numel(),
                    torch_random_states_dtype=str(loaded['generation_rng']['torch_random_states'].dtype),
                    torch_random_states_shape=list(loaded['generation_rng']['torch_random_states'].shape),
                    gen_random_states_dtype=None if loaded['generation_rng']['gen_random_states']is None else str(loaded['generation_rng']['gen_random_states'].dtype),
                    gen_random_states_shape=None if loaded['generation_rng']['gen_random_states']is None else list(loaded['generation_rng']['gen_random_states'].shape),
                    gen_random_states_numel=None if loaded['generation_rng']['gen_random_states']is None else loaded['generation_rng']['gen_random_states'].numel(),
                    device_mesh=None if self.rollout_sharding_manager.device_mesh is None else dict(
                        shape=list(self.rollout_sharding_manager.device_mesh.mesh.shape),
                        names=list(self.rollout_sharding_manager.device_mesh.mesh_dim_names),
                        ranks=self.rollout_sharding_manager.device_mesh.mesh.cpu().tolist())))
            if self.config.actor.get('fresh200_enabled',False):
                report.update(attempt_id=payload.get('attempt_id'),latest_rank_event=payload.get('latest_rank_event'))
            if state_status in ('PARTIAL','ABORTED','SKIPPED'):
                report.update(status=state_status+'_CURRENT_STATE_SAVED',attempted_outer=step,previous_complete_outer=step-1)
            (directory/('rank_%d.json'%self.rank)).write_text(json.dumps(report,indent=2,allow_nan=False))
            print('FULL_STATE_RANK_JSON='+json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
            torch.distributed.barrier()
            return report
        finally:
            if self._is_offload_param:
                offload_fsdp_param_and_grad(self.actor_module_fsdp,offload_grad=self._is_offload_grad)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)



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
