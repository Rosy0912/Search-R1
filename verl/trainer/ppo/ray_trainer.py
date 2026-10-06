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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


def _baseline_validation(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        controller = getattr(self, 'baseline_controller', None)
        if controller is None or controller.ranker is None:
            return function(self, *args, **kwargs)
        old = controller.ranker.validation
        controller.ranker.validation = True
        try:
            return function(self, *args, **kwargs)
        finally:
            controller.ranker.validation = old
    return wrapped


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(torch.where(response_mask.bool(), data.batch['old_log_probs'], 0.),
                                    torch.where(response_mask.bool(), data.batch['ref_log_prob'], 0.),
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
        self.baseline_controller = None
        if self.config.get('baselines') is not None and self.config.baselines.method != 'grpo':
            from search_r1.baselines.controller import Controller, validate_config
            validate_config(self.config)
            self.baseline_controller = Controller(self)
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        e1 = (getattr(self.config.trainer, 'e1_gate_enabled', False)
              or getattr(self.config.trainer, 'fresh200_enabled', False))
        self.e1_sampler = None
        if e1:
            from search_r1.baselines.full_state import StatefulSampler
            self.e1_sampler = StatefulSampler(self.train_dataset, None,
                                               self.config.data.shuffle_train_dataloader)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader if not e1 else False,
                                           sampler=self.e1_sampler,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        if getattr(self.config.trainer, 'e1_gate_enabled', False):
            if (total_training_steps != 301 or self.config.trainer.save_freq != 25
                    or self.config.trainer.get('acceptance_step') != 5
                    or self.config.actor_rollout_ref.actor.optim.get('num_warmup_steps') != 114
                    or self.config.trainer.n_gpus_per_node != 8 or self.config.trainer.nnodes != 1
                    or self.config.algorithm.adv_estimator != 'grpo'
                    or self.config.baselines.method != 'bapo'
                    or self.config.trainer.get('smoke_only', False)):
                raise ValueError('E1 requires actual300 (steps301), warmup114, save25/5, BAPO formal run')
            if not self.config.trainer.get('e1_gate_receipt'):
                raise ValueError('E1 requires an independent fifth-outer release receipt path')
        if getattr(self.config.trainer, 'fresh200_enabled', False):
            from search_r1.baselines.fresh200_contract import validate_fresh_config
            validate_fresh_config(OmegaConf.to_container(self.config, resolve=True))
        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _prepare_final_validation(self):
        """Switch only terminal evaluation to the complete held-out data."""
        files=self.config.data.get('final_val_files')
        if not files:
            return
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.val_dataset=RLHFDataset(
            parquet_files=files, tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat',False),
            truncation='error')
        self.val_dataloader=DataLoader(
            dataset=self.val_dataset, batch_size=self.config.data.val_batch_size,
            shuffle=False, drop_last=False, collate_fn=collate_fn)
        print(f'Final validation dataset size: {len(self.val_dataset)}',flush=True)

    @_baseline_validation
    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        qa_rows = []

        def collect_qa(batch):
            if not self.config.get('baselines'):
                return
            from search_r1.baselines.evaluation import batch_answer_metrics
            qa_rows.extend(batch_answer_metrics(batch, self.tokenizer))

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            force_final_answer=self.config.get('force_final_answer', True),
        )

        if self.config.trainer.get('fresh200_enabled',False):
            gen_config.preserve_sampling_tokens=True
        validation_snapshot=f'{os.environ.get("BAPO_RUN_ID","validation")}:eval:{uuid.uuid4().hex}:outer_{getattr(self,"global_steps",0)}'

        def bind_fresh200_eval(gen_batch):
            if self.config.trainer.get('fresh200_enabled',False):
                gen_batch.meta_info.update(preserve_sampling_tokens=True,fresh200_evaluation=True,
                    probability_role='greedy_model_logprob_not_sampling_policy',policy_snapshot_id=validation_snapshot)
                gen_batch.non_tensor_batch['fresh200_row_id']=np.array([str(uuid.uuid4()) for _ in range(len(gen_batch))],dtype=object)

        # Agent config preparation
        validation_manager_class = LLMGenerationManager
        validation_controller = getattr(self, 'baseline_controller', None)
        if validation_controller is not None and validation_controller.method == 'bapo':
            from search_r1.baselines.bapo_validation import BapoValidationManager
            validation_manager_class = BapoValidationManager
        generation_manager = validation_manager_class(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )
        controller = getattr(self, 'baseline_controller', None)
        if controller is not None and controller.ranker is not None:
            from search_r1.baselines.controller import ranked_validation_manager
            generation_manager = ranked_validation_manager(generation_manager, controller)

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                bind_fresh200_eval(test_gen_batch)

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                collect_qa(test_batch)
                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                bind_fresh200_eval(test_gen_batch)
                with _timer('step', timing_raw):
                    # Raw mode must retain the full input tensor aligned with its original mask.
                    # Generation removes padding and rejects actual overlong context.
                    first_input_ids = (test_gen_batch.batch['input_ids'] if gen_config.preserve_sampling_tokens
                                       else test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:]).clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        if key not in ('sampling_log_probs','old_log_probs'):
                            test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    collect_qa(test_batch)
                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            metric_dict[f'val/evaluated_count/{data_source}'] = len(rewards)

        if validation_manager_class is not LLMGenerationManager:
            metric_dict.update({'val/cost/' + key: value for key, value in generation_manager.cost.items()})
            metric_dict.update({'val/bapo/' + key: value for key, value in generation_manager.answer_metrics().items()})
        if qa_rows:
            from search_r1.baselines.evaluation import aggregate_answer_metrics
            metric_dict.update(aggregate_answer_metrics(qa_rows))
        if self.config.get('baselines'):
            print('BASELINE_EVALUATION_METRICS_JSON=' + json.dumps(metric_dict, sort_keys=True), flush=True)
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resume_manifest = None
        if self.config.trainer.get('fresh200_enabled', False):
            from search_r1.baselines.fresh200_native_evidence import bind_shared_inputs
            self.fresh200_shared_input_bindings=bind_shared_inputs(self.config.baselines.holdout.manifest_path,
                dict(train=self.config.data.train_files,validation=self.config.data.val_files,
                     final=self.config.data.final_val_files,stage=self.config.baselines.bapo.stage_validation_files))
            from search_r1.baselines.fresh200_contract import validate_fresh_config
            validate_fresh_config(OmegaConf.to_container(self.config, resolve=True))
        if self.config.trainer.get('resume_full_state'):
            from search_r1.baselines.recovery import validate_manifest,require
            import subprocess
            from pathlib import Path
            source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            require(source==self.config.baselines.algorithm_source_commit and not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Recovery source changed/dirty')
            run=Path(os.environ['BAPO_RUN_ID']).resolve()
            witness=Path(self.config.trainer.resume_witness_dir)
            require(witness.is_absolute()and witness==witness.resolve()and run in witness.parents and not witness.exists(),'Fresh owned recovery witness directory required')
            self.resume_manifest=validate_manifest(self.config.trainer.resume_full_state,OmegaConf.to_container(self.config,resolve=True),source,run,self.config.trainer.resume_permission)
            self.resume_witness_directory=witness
        elif self.config.trainer.get('resume_verify_only') or self.config.trainer.get('resume_permission'):
            raise ValueError('Recovery flags require complete manifest')
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        if self.resume_manifest is not None:
            from search_r1.baselines.recovery import require
            reports=self.actor_rollout_wg.load_training_state(self.resume_manifest,str(self.resume_witness_directory))
            require(len(reports)==8 and {r['rank']for r in reports}==set(range(8)) and all(r['status']=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'for r in reports),'Recovery rank witnesses incomplete')
            self.resume_rank_reports=reports
            if self.config.trainer.resume_verify_only:
                self._verify_recovery_native_v3()

    def _verify_recovery_native_v3(self):
        """Real production save/reload only for independently permitted GPU verification."""
        from pathlib import Path
        import json
        from search_r1.baselines.recovery import require
        require(self.resume_manifest is not None and self.config.trainer.resume_verify_only is True,'Only resume verify-only may roundtrip native v3')
        run=Path(self.resume_manifest['target_run_root']);path=run/'recovery-native-v3-roundtrip'
        require(run.is_absolute()and run==run.resolve()and not path.exists(),'Fresh owned v3 roundtrip output required')
        reports=self.actor_rollout_wg.save_training_state(str(path),step=25,roundtrip=True)
        require(len(reports)==8 and {x['rank']for x in reports}==set(range(8))and all(x['status']=='EXACT_FULL_STATE_RELOAD_OK'and x['native_state_version']==3 and x['roundtrip']is True and x['state_digest']==x['reloaded_state_digest']and x['generation_rng_before_digest']==x['generation_rng_after_digest']for x in reports),'Production v3 roundtrip witnesses incomplete')
        self.resume_v3_reports=reports
        for report in reports:print('BAPO_RECOVERY_V3_RANK_JSON='+json.dumps(report,allow_nan=False),flush=True)
        return reports

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        if getattr(self, 'baseline_controller', None) is not None:
            self.baseline_controller.save(self.global_steps)
        if getattr(self.config.trainer, 'fresh200_enabled', False):
            self._save_fresh200_state(actor_local_path, 'COMPLETE', getattr(self, 'fresh200_rank_updates', []))
        elif getattr(self.config.trainer, 'e1_gate_enabled', False):
            self._save_e1_full_state(actor_local_path)

    def _save_fresh200_state(self, actor_local_path, status, rank_updates):
        from pathlib import Path
        from search_r1.baselines.full_state import rng_state, restore_rng, state_digest, atomic_save
        from search_r1.baselines.fresh200_contract import outer_contract
        step = self.global_steps
        directory = Path(self.config.trainer.default_local_dir) / 'fresh200_state' / (f'outer_{step}_{status}')
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'all-rank-write-ledger.json').write_text(json.dumps(rank_updates, indent=2, default=str))
        reports = self.actor_rollout_wg.save_training_state(str(directory / 'ranks'), step,
                                                          roundtrip=(status == 'COMPLETE' and step == 5), state_status=status)
        if len(reports) != 8 or {r['rank'] for r in reports} != set(range(8)):
            raise ValueError('Fresh200 checkpoint lacks all eight current-state shards')
        controller = self.baseline_controller
        payload = dict(version=1, protocol='BAPO_FRESH200_V1', status=status, driver_pid=os.getpid(),
            global_step=step, next_global_step=step + (status == 'COMPLETE'), epoch=self.e1_epoch,
            sampler=self.e1_sampler.state_dict(), rng=rng_state(), bapo=controller.stage.state_dict(),
            actual_outer_updates=controller.actual_outer_updates, rank_write_ledger=rank_updates,
            outer_contract=outer_contract(step))
        path=directory/'driver.pt'; atomic_save(payload,path)
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        if state_digest(loaded)!=state_digest(payload):
            raise ValueError('Fresh200 driver serialization mismatch')
        if status == 'COMPLETE' and step == 5:
            self.e1_sampler.load_state_dict(loaded['sampler']); restore_rng(loaded['rng'])
            controller.stage.load_state_dict(loaded['bapo'])
            controller.actual_outer_updates=loaded['actual_outer_updates']
            after=dict(loaded)
            after.update(sampler=self.e1_sampler.state_dict(),rng=rng_state(),
                bapo=controller.stage.state_dict(),actual_outer_updates=controller.actual_outer_updates)
            if state_digest(after)!=state_digest(loaded):
                raise ValueError('Fresh200 driver sampler/RNG/BAPO reload mismatch')
        from search_r1.baselines.checkpoint_gate import digest
        (directory/'driver-reload.json').write_text(json.dumps(dict(
            status=('EXACT_DRIVER_RELOAD_OK' if step==5 else 'DRIVER_SERIALIZED_OK') if status=='COMPLETE' else 'PARTIAL_CURRENT_STATE_SAVED',
            step=step, pid=os.getpid(), path=str(path.resolve()), sha256=digest(path),
            state_digest=state_digest(payload)), indent=2))
        # A PARTIAL directory is diagnostic current state, never a final/full-state completion marker.
        (directory/'status.json').write_text(json.dumps(dict(status=status, actual_outer=step,
            completed_outer_updates=controller.actual_outer_updates, retry_allowed=False,
            automatic_rollback_allowed=False), indent=2))
        self.fresh200_state_directory=directory; self.fresh200_state_reports=reports
        return directory

    def _fresh200_partial_exit(self, rank_updates):
        from pathlib import Path
        directory=Path(self.config.trainer.default_local_dir)/'diagnostics'/f'outer_{self.global_steps}_PARTIAL'
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'all-rank-write-ledger.json').write_text(json.dumps(rank_updates,indent=2,default=str))
        self.actor_rollout_wg.save_checkpoint(str(directory/'actor'), None)
        self._save_fresh200_state(str(directory/'actor'), 'PARTIAL', rank_updates)
        raise RuntimeError('Fresh200 PARTIAL: current writes saved; no outer increment, scheduler step, automatic rollback or retry')

    def _fresh200_pause_after_five(self):
        from pathlib import Path
        import subprocess
        from search_r1.baselines.fresh200_gate import write_manifest, wait_for_release
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if source!=self.config.baselines.algorithm_source_commit:
            raise ValueError('Fresh200 source-start differs from current source')
        run=Path(os.environ['BAPO_RUN_ID']).resolve()
        source_record=run/'source-commit.txt'
        if source_record.read_text().strip()!=source:
            raise ValueError('Fresh200 source marker differs')
        paths=[source_record]
        base=Path(self.config.trainer.default_local_dir)
        for directory in (base/'actor'/'global_step_5',base/'algorithm'/'global_step_5',self.fresh200_state_directory):
            paths.extend(p for p in directory.rglob('*') if p.is_file())
        manifest=write_manifest(paths,self.fresh200_state_directory,OmegaConf.to_container(self.config,resolve=True),source,run,5,self.fresh200_state_reports)
        wait_for_release(Path(self.config.trainer.fresh200_gate_receipt),manifest,Path.cwd())

    def _save_e1_full_state(self, actor_local_path):
        from pathlib import Path
        from search_r1.baselines.full_state import rng_state, restore_rng, state_digest, atomic_save
        step=self.global_steps
        reports=self.actor_rollout_wg.save_training_state(os.path.join(actor_local_path,'training_state'),
                                                          step, roundtrip=(step==5))
        world=self.config.trainer.n_gpus_per_node*self.config.trainer.nnodes
        if len(reports)!=world or {r['rank'] for r in reports}!=set(range(world)):
            raise ValueError('Missing/duplicate full-state rank reports')
        directory=Path(self.config.trainer.default_local_dir)/'full_state'/f'global_step_{step}'
        controller=self.baseline_controller
        payload=dict(version=1,driver_pid=os.getpid(),global_step=step,next_global_step=step+1,epoch=self.e1_epoch,
            sampler=self.e1_sampler.state_dict(),rng=rng_state(),bapo=controller.stage.state_dict(),
            actual_outer_updates=controller.actual_outer_updates)
        before=state_digest(payload);path=directory/'driver.pt';atomic_save(payload,path)
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        if state_digest(loaded)!=before:raise ValueError('Driver state serialization mismatch')
        if step==5:
            self.e1_sampler.load_state_dict(loaded['sampler']);restore_rng(loaded['rng'])
            controller.stage.load_state_dict(loaded['bapo'])
            controller.actual_outer_updates=loaded['actual_outer_updates']
            after=dict(version=1,driver_pid=os.getpid(),global_step=self.global_steps,next_global_step=self.global_steps+1,
                epoch=self.e1_epoch,sampler=self.e1_sampler.state_dict(),rng=rng_state(),
                bapo=controller.stage.state_dict(),actual_outer_updates=controller.actual_outer_updates)
            if state_digest(after)!=before:raise ValueError('Driver sampler/RNG/BAPO reload mismatch')
        self.e1_rank_reports=reports;self.e1_state_directory=directory
        (directory/'driver-reload.json').write_text(json.dumps(dict(
            status='EXACT_DRIVER_RELOAD_OK' if step==5 else 'DRIVER_SERIALIZED_OK',step=step,
            state_digest=before,pid=os.getpid(),path=str(path.resolve()),
            sha256=__import__('search_r1.baselines.checkpoint_gate',fromlist=['digest']).digest(path),
            sampler_cursor=self.e1_sampler.cursor),indent=2))
        if step != 5:
            # Publish completeness only after all eight native shards, driver,
            # algorithm and HF export have been persisted and hashed.
            from search_r1.baselines.checkpoint_gate import write_manifest,digest
            from search_r1.baselines.recovery import write_witness,require
            run=Path(os.environ['BAPO_RUN_ID']).resolve()
            source=(run/'source-commit.txt').read_text().strip()
            require(source==self.config.baselines.algorithm_source_commit,'Checkpoint source mismatch')
            base=Path(self.config.trainer.default_local_dir)
            paths=[run/'source-commit.txt']
            for child in (base/'actor'/f'global_step_{step}',base/'algorithm'/f'global_step_{step}',directory):
                paths.extend(p for p in child.rglob('*')if p.is_file()and p.name not in ('manifest.json','checkpoint-complete.json'))
            require(not any(str(p).endswith('.tmp')for p in paths),'Incomplete temporary checkpoint files')
            manifest=write_manifest(paths,directory,OmegaConf.to_container(self.config,resolve=True),source,run,step,reports)
            write_witness(directory/'checkpoint-complete.json',dict(status='FULL_STATE_CHECKPOINT_COMPLETE',
                step=step,actual_outer_updates=controller.actual_outer_updates,source_commit=source,
                manifest_path=str(manifest),manifest_sha256=digest(manifest),world_size=world,
                generation_rng_saved=True,recovery_lineage=None if getattr(self,'resume_manifest',None) is None else
                dict(old_run_root=self.resume_manifest['old_run_root'],old_step=25,
                     manifest_path=self.config.trainer.resume_full_state,manifest_sha256=digest(self.config.trainer.resume_full_state))))


    def _e1_pause_after_five(self):
        from pathlib import Path
        import subprocess
        from search_r1.baselines.checkpoint_gate import write_manifest, wait_for_release
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if source!=self.config.baselines.algorithm_source_commit:
            raise ValueError('E1 source-start configuration differs from current source')
        run=Path(os.environ['BAPO_RUN_ID']).resolve()
        source_record=run/'source-commit.txt'
        if source_record.read_text().strip()!=source or source_record.resolve().parent!=run:
            raise ValueError('Run-start E1 source differs from current source')
        receipt=Path(self.config.trainer.e1_gate_receipt).resolve()
        if run.parent.parent not in receipt.parents:raise ValueError('E1 release outside owned ROOT')
        base=Path(self.config.trainer.default_local_dir)
        paths=[source_record]
        for directory in (base/'actor'/'global_step_5',base/'algorithm'/'global_step_5',self.e1_state_directory):
            paths.extend(p for p in directory.rglob('*') if p.is_file())
        config=OmegaConf.to_container(self.config,resolve=True)
        manifest=write_manifest(paths,self.e1_state_directory,config,source,run,5,self.e1_rank_reports)
        wait_for_release(receipt,manifest,Path.cwd())

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if getattr(self,'resume_manifest',None) is None and self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return val_metrics

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            force_final_answer=self.config.get('force_final_answer', True),
        )

        if self.config.trainer.get('fresh200_enabled', False):
            gen_config.preserve_sampling_tokens = True

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # Construct the first iterator before restoring CPU RNG: DataLoader
        # iterator creation consumes a base_seed even with num_workers=0.
        resume_iterator = None
        start_epoch = 0
        if getattr(self,'resume_manifest',None) is not None:
            from pathlib import Path
            from search_r1.baselines.recovery import validate_driver,write_witness,require,process_identity
            from search_r1.baselines.full_state import restore_rng,state_digest
            loaded=torch.load(self.resume_manifest['driver']['path'],map_location='cpu',weights_only=False)
            validate_driver(loaded,self.resume_manifest)
            self.e1_sampler.load_state_dict(loaded['sampler'])
            self.baseline_controller.stage.load_state_dict(loaded['bapo'])
            self.baseline_controller.actual_outer_updates=loaded['actual_outer_updates']
            self.global_steps=loaded['next_global_step'];start_epoch=loaded['epoch'];self.e1_epoch=start_epoch
            resume_iterator=iter(self.train_dataloader)
            restore_rng(loaded['rng'])
            require(state_digest(self.e1_sampler.state_dict())==state_digest(loaded['sampler']),'Recovered sampler differs')
            from search_r1.baselines.full_state import rng_state
            require(state_digest(rng_state())==state_digest(loaded['rng']),'Recovered driver RNG differs')
            witness=dict(status='EXACT_CROSS_JOB_DRIVER_RELOAD_OK',old_pid=loaded['driver_pid'],new_pid=os.getpid(),
                process=process_identity(),runtime_module_path=str(Path(__file__).resolve()),
                runtime_module_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
                loaded_state_digest=state_digest(loaded),step=25,next_step=self.global_steps,actual_outer_updates=25,
                epoch=start_epoch,sampler_cursor=self.e1_sampler.cursor,
                restored_sampler_digest=state_digest(self.e1_sampler.state_dict()),saved_sampler_digest=state_digest(loaded['sampler']),
                restored_rng_digest=state_digest(rng_state()),saved_rng_digest=state_digest(loaded['rng']),
                no_generation_or_optimizer_update=bool(self.config.trainer.resume_verify_only),bapo=self.baseline_controller.stage.state_dict(),
                target_source_commit=self.resume_manifest['target_source_commit'],target_run_root=self.resume_manifest['target_run_root'],
                target_job_id=self.resume_manifest['target_job_id'],sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',
                verify_only=self.config.trainer.resume_verify_only,rank_witnesses=self.resume_rank_reports)
            require(witness['new_pid']!=witness['old_pid'],'New driver PID did not change')
            write_witness(self.resume_witness_directory/'driver.json',witness)
            print('BAPO_RECOVERY_DRIVER_JSON='+json.dumps(witness,allow_nan=False),flush=True)
            if self.config.trainer.resume_verify_only:
                return witness
        # Only complete continuation contributes to the unchanged total300.
        for epoch in range(start_epoch,self.config.trainer.total_epochs):
            self.e1_epoch = epoch
            iterator=resume_iterator if resume_iterator is not None else iter(self.train_dataloader)
            resume_iterator=None
            for batch_dict in iterator:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if self.config.trainer.get('fresh200_enabled', False):
                    gen_batch.meta_info.update(preserve_sampling_tokens=True,recompute_log_prob=False,
                        policy_snapshot_id=f'{os.environ["BAPO_RUN_ID"]}:before_outer_{self.global_steps}')
                    gen_batch.non_tensor_batch['fresh200_row_id'] = np.array([str(uuid.uuid4()) for _ in range(len(gen_batch))],dtype=object)

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = (gen_batch.batch['input_ids'] if gen_config.preserve_sampling_tokens
                                           else gen_batch.batch['input_ids'][:, -gen_config.max_start_length:]).clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            if self.baseline_controller is None:
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    initial_input_ids=first_input_ids,
                                )
                            else:
                                final_gen_batch_output, sources = self.baseline_controller.generate(
                                    generation_manager, gen_batch, batch, metrics)
                                from search_r1.baselines.rollout import select_rows
                                batch = select_rows(batch, sources)

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            if key != 'sampling_log_probs':
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        if self.config.trainer.get('fresh200_enabled', False):
                            from search_r1.llm_agent.sampling_tokens import audit_batch
                            final_gen_batch_output.meta_info['sampling_context_audit'] = audit_batch(final_gen_batch_output)
                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key not in ('old_log_probs', 'sampling_log_probs'):
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.trainer.get('fresh200_enabled', False):
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                            metrics['critic/kl_coeff'] = 0.0
                        elif self.baseline_controller is not None and self.baseline_controller.method in ('topd', 'opsd'):
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        elif not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        if self.baseline_controller is None:
                            batch = compute_advantage(batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n)
                        else:
                            batch = self.baseline_controller.compute_advantages(batch, metrics)

                    if self.config.trainer.get('fresh200_diagnostic_only',False):
                        from pathlib import Path
                        directory=Path(self.config.trainer.default_local_dir)/'native-diagnostics'
                        directory.mkdir(parents=True,exist_ok=True)
                        batch,metrics=self._create_loss_mask(batch,metrics)
                        batch.meta_info['fresh200_shared_input_bindings']=self.fresh200_shared_input_bindings
                        role=self.config.trainer.fresh200_diagnostic_role
                        mini=self.config.trainer.fresh200_diagnostic_global_mini
                        try:
                            diagnostic=self.actor_rollout_wg.capture_fresh200_diagnostic(batch,mini,role,str(directory))
                            events=diagnostic.meta_info['fresh200_diagnostic_events']
                            (directory/'all-rank-diagnostic-ledger.json').write_text(json.dumps(events,indent=2,default=str))
                        except Exception:
                            import traceback
                            (directory/'driver-fault.raw.log').write_text(traceback.format_exc())
                            self.actor_rollout_wg.save_training_state(str(directory/'FAULT_CURRENT_STATE'),0,roundtrip=False,state_status='DIAGNOSTIC_FAULT')
                            raise
                        diagnostic_fault=any(e.get('stop_reason') is not None or e.get('status')=='DIAGNOSTIC_PARTIAL_FAULT' for e in events)
                        state_tag='DIAGNOSTIC_PARTIAL' if diagnostic_fault else 'DIAGNOSTIC_ONLY'
                        reports=self.actor_rollout_wg.save_training_state(str(directory/'DIAGNOSTIC_FINAL_STATE'),0,roundtrip=False,state_status=state_tag)
                        (directory/'diagnostic-state-reports.json').write_text(json.dumps(reports,indent=2,default=str))
                        (directory/'status.json').write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY_NOT_PRODUCTION',
                            role=role,global_mini_batch_size=mini,formal_outer_updates=0,scheduler_outer_steps=0,
                            optimizer_writes_per_rank=[e['optimizer_writes'] for e in events],
                            diagnostic_status=[e['status'] for e in events],release_allowed=False),indent=2))
                        from search_r1.baselines.full_state import atomic_save,rng_state
                        atomic_save(dict(status=state_tag,formal_outer_updates=0,sampler=self.e1_sampler.state_dict(),
                            rng=rng_state(),bapo=self.baseline_controller.stage.state_dict(),rank_write_ledger=events),directory/'driver-diagnostic-state.pt')
                        if diagnostic_fault:
                            raise RuntimeError('Diagnostic PARTIAL: all available rank writes/current state retained; no formal increment or release')
                        return {'status':'DIAGNOSTIC_ONLY_NOT_PRODUCTION','formal_outer_updates':0}

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            if self.config.trainer.get('fresh200_enabled', False):
                                from search_r1.baselines.fresh200_contract import outer_contract
                                batch.meta_info.update(outer_contract(self.global_steps))
                                from search_r1.baselines.policy_safety import tensor_identity
                                batch.meta_info['rollout_identity'] = tensor_identity(batch.batch)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            if self.config.trainer.get('fresh200_enabled', False):
                                from search_r1.baselines.fresh200_contract import validate_rank_updates
                                self.fresh200_rank_updates = actor_output.meta_info['fresh200_rank_updates']
                                if not validate_rank_updates(self.fresh200_rank_updates, self.global_steps):
                                    self._fresh200_partial_exit(self.fresh200_rank_updates)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        if getattr(self, 'baseline_controller', None) is not None:
                            self.baseline_controller.after_actor(batch, metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                    if (getattr(self, 'baseline_controller', None) is not None
                            and metrics.get('actor/optimizer_steps', 0) > 0):
                        with _timer('stage_testing', timing_raw):
                            metrics.update(self.baseline_controller.stage_validation(self.global_steps))

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps % self.config.trainer.save_freq == 0 or
                             self.global_steps == getattr(self.config.trainer, 'acceptance_step', 0) or
                             self.global_steps + 1 >= self.total_training_steps):
                        # Save the final updated policy before the counter advances.
                        # Final validation below must have a matching saved model,
                        # even when the last update is not a save-frequency boundary.
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                if (getattr(self, 'baseline_controller', None) is not None
                        and self.baseline_controller.method == 'bapo'):
                    print('BAPO_TRAIN_METRICS_JSON=' + json.dumps(
                        dict(step=self.global_steps, metrics=metrics), sort_keys=True,
                        default=lambda value: value.item(), allow_nan=False), flush=True)

                if self.config.trainer.get('fresh200_enabled', False) and self.global_steps == 5:
                    self._fresh200_pause_after_five()
                if getattr(self.config.trainer, 'e1_gate_enabled', False) and self.global_steps == 5:
                    self._e1_pause_after_five()
                completed_step = self.global_steps
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    if self.config.trainer.get('fresh200_enabled', False):
                        if completed_step != 200 or self.baseline_controller.actual_outer_updates != 200:
                            raise ValueError('Fresh200 must complete exactly 200 actual outer updates')
                        self.global_steps = completed_step
                    if getattr(self.config.trainer, 'e1_gate_enabled', False):
                        if completed_step != 300 or self.baseline_controller.actual_outer_updates != 300:
                            raise ValueError('E1 must end at exactly actual outer300')
                        self.global_steps = completed_step
                    # perform validation after training
                    if self.val_reward_fn is not None and not getattr(self.config.trainer, 'smoke_only', False):
                        self._prepare_final_validation()
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        print('SEARCH_R1_FINAL_METRICS_JSON=' + json.dumps(val_metrics, sort_keys=True), flush=True)
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        if getattr(getattr(self, 'baseline_controller', None), 'method', None) == 'bapo':
            metrics['bapo/mask_binary_errors'] = int(((loss_mask != 0) & (loss_mask != 1)).sum())
            steps = batch.batch['step_ids']
            metrics['bapo/mask_on_observation_tokens'] = int((loss_mask.bool() & (steps < 0) & response_mask.bool()).sum())
            metrics['bapo/model_tokens_missing_loss'] = int((~loss_mask.bool() & (steps >= 0) & response_mask.bool()).sum())
            metrics['bapo/mask_on_padding_tokens'] = int((loss_mask.bool() & ~response_mask.bool()).sum())

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
