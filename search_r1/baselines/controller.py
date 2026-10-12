"""Opt-in hooks for the reviewed Ray trainer; no changes to baseline inputs."""
import json
import os
import random
from collections import Counter
from pathlib import Path
import torch
import numpy as np
from . import advantages
from .rollout import SearchRollout, select_rows
from .bapo import BoundaryStage, BoundaryReward, decode_answers, ordinary_answer_f1, resample_rows


METHODS = ('grpo', 'gigpo', 'tree_grpo', 'bapo', 'opsd', 'topd', 'cosearch')


def option(settings,key,default=None):
    getter=getattr(settings,'get',None)
    return getter(key,default) if callable(getter) else getattr(settings,key,default)


def canonical_paths(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    return {str(Path(path).expanduser().resolve()) for path in paths}


def question_keys(dataset, prompt_key='prompt'):
    def plain(value):
        if hasattr(value, 'tolist'):
            return plain(value.tolist())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    return {json.dumps(plain(prompt), sort_keys=True) for prompt in dataset.dataframe[prompt_key]}


def phase_validation_frame(frame, count, seed):
    """Fixed within-holdout phase subset, leaving complete teacher data intact."""
    if count < 1:
        raise ValueError('Phase validation question count must be positive')
    return frame.sample(n=min(int(count), len(frame)), random_state=int(seed)).reset_index(drop=True)


def validate_behavior_mu_config(config):
    """Explicit Tree-only correction; never infer opt-in from captured scalars."""
    actor = config.actor_rollout_ref.actor
    safety = option(actor, 'policy_safety', {})
    correction = option(safety, 'sampling_probability_correction', {})
    tree = option(config.baselines, 'tree', {})
    behavior = option(tree, 'behavior_mu', {})
    enabled = option(behavior, 'enabled', False)
    actor_enabled = option(correction, 'enabled', False)
    if type(enabled) is not bool or type(actor_enabled) is not bool or enabled != actor_enabled:
        raise ValueError('Explicit Tree behavior-mu and actor correction boolean opt-ins must match')
    if not enabled:
        return
    protocol = 'tree_behavior_mu_centered_ppo_v1'
    rollout = config.actor_rollout_ref.rollout
    preservation = option(tree, 'token_preservation', {})
    if (config.baselines.method != 'tree_grpo' or option(safety, 'enabled', False) is not True
            or option(behavior, 'protocol') != protocol or option(correction, 'protocol') != protocol
            or option(preservation, 'enabled', False) is not True
            or option(preservation, 'protocol') != 'tree_sampled_prefix_legacy_semantics_v1'
            or option(rollout, 'tree_raw_emission', False) is not True):
        raise ValueError('Behavior-mu requires explicit guarded Tree raw-token/native-emission protocol')
    fresh = option(config.baselines, 'training_protocol') == 'tree_fresh200_five_leaf_v1'
    batch, mini = (512, 128) if fresh else (640, 384)
    if fresh:
        from .tree_fresh200 import validate_fresh200_config
        validate_fresh200_config(config)
    if (config.max_turns != 4 or config.force_final_answer is not False
            or (config.data.max_start_length, config.data.max_prompt_length,
                config.data.max_response_length, config.data.max_obs_length) != (2048, 4096, 500, 500)
            or (tree.roots, tree.leaves) != (2, 3)
            or config.data.train_batch_size != batch or actor.ppo_mini_batch_size != mini
            or actor.ppo_micro_batch_size != 16 or actor.ppo_epochs != 1
            or actor.optim.lr != 1e-6
            or (config.trainer.n_gpus_per_node, config.trainer.nnodes) != (8, 1)):
        raise ValueError('Behavior-mu candidate must preserve the authorized Tree recovery protocol')
    required = dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                    presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)
    if any(option(rollout, key, value) != value for key, value in required.items()):
        raise ValueError('Behavior-mu candidate requires the original plain temperature-one sampling support')


def validate_config(config):
    validate_behavior_mu_config(config)
    method = config.baselines.method
    if method not in METHODS:
        raise ValueError(f'Unknown baseline method {method}')
    if config.actor_rollout_ref.actor.strategy != 'fsdp' or not config.do_search:
        raise ValueError('New algorithms currently require FSDP Search-R1')
    if config.actor_rollout_ref.rollout.n != 1 or not config.actor_rollout_ref.actor.state_masking:
        raise ValueError('Use rollout.n=1 and action state masking')
    if config.actor_rollout_ref.actor.ppo_epochs < 1:
        raise ValueError('Inner epochs must be positive')
    actor = config.actor_rollout_ref.actor
    world = config.trainer.n_gpus_per_node * config.trainer.nnodes
    sp = actor.ulysses_sequence_parallel_size
    if min(world, sp, actor.ppo_micro_batch_size, actor.ppo_mini_batch_size) < 1 or world % sp:
        raise ValueError('Positive GPU/SP/batch counts and world divisible by SP required')
    dp = world // sp
    if actor.ppo_micro_batch_size % dp or actor.ppo_mini_batch_size % dp:
        raise ValueError('Global actor mini/micro batches must be divisible by DP ranks')
    if actor.ppo_mini_batch_size % actor.ppo_micro_batch_size:
        raise ValueError('Actor minibatch must be divisible by microbatch')
    if config.data.train_batch_size % world:
        raise ValueError('Training question batch must be divisible by GPU world size')
    if method == 'gigpo':
        retained_budget = config.max_turns * (config.data.max_response_length + config.data.max_obs_length)
        if config.force_final_answer:
            retained_budget += config.data.max_response_length + config.data.max_obs_length
        if retained_budget > config.data.max_prompt_length:
            raise ValueError('GiGPO response cap can truncate the action timeline; reduce turn/response/observation budget')
    if method in ('opsd', 'topd'):
        actor = config.actor_rollout_ref.actor
        if actor.use_kl_loss or actor.entropy_coeff or config.algorithm.kl_ctrl.kl_coef:
            raise ValueError('Pure distillation requires reference KL and entropy coefficients zero')
        if not config.baselines.distillation.teacher_path:
            raise ValueError('Provide an explicit frozen teacher checkpoint')
        if method == 'topd' and (canonical_paths(config.baselines.distillation.teacher_path) == canonical_paths(config.actor_rollout_ref.model.path)
                                 and not config.baselines.distillation.privileged_answers):
            raise ValueError('Identical non-privileged teacher gives zero initial TOP-D signal')
        if method == 'opsd' and not config.baselines.distillation.privileged_answers:
            raise ValueError('OPSD requires teacher-only training-answer context')
        if method == 'opsd' and (actor.ulysses_sequence_parallel_size != 1 or actor.use_dynamic_bsz):
            raise ValueError('OPSD currently requires SP=1 and fixed microbatches')
        if method == 'opsd' and actor.ppo_micro_batch_size != dp:
            raise ValueError('OPSD starts at microbatch 1 per rank; use global microbatch equal to DP rank count')
        if config.baselines.distillation.chunk_size < 1:
            raise ValueError('Distillation chunk_size must be positive')
    if method == 'tree_grpo':
        tree = config.baselines.tree
        from .tree_fresh200 import leaf_quotas
        quotas = leaf_quotas(tree)
        preservation=option(tree,'token_preservation', {})
        enabled=option(preservation,'enabled', False)
        if enabled:
            from .tree_raw_tokens import PROTOCOL, require
            require(option(preservation,'protocol') == PROTOCOL and config.actor_rollout_ref.rollout.get('tree_raw_emission', False),
                    'Tree token preservation/native metadata configuration differs')
            require(config.max_turns==4 and config.force_final_answer is False and
                    (config.data.max_start_length,config.data.max_prompt_length,
                     config.data.max_response_length,config.data.max_obs_length)==(2048,4096,500,500) and
                    (tree.roots,tree.leaves)==(2,3), 'Fixed Tree search budgets/caps changed')
        if config.actor_rollout_ref.rollout.n_agent != tree.roots:
            raise ValueError('Tree roots must equal rollout.n_agent')
        if min(tree.roots, tree.leaves, tree.expansions, tree.rounds) < 1:
            raise ValueError('Tree budgets must be positive')
        if max(quotas) > 1 + tree.expansions * tree.rounds:
            raise ValueError('Insufficient expansions for selected leaves')
    if config.actor_rollout_ref.rollout.get('tree_raw_emission', False) and not (
            method == 'tree_grpo' and option(option(config.baselines.tree,'token_preservation',{}),'enabled',False)):
        raise ValueError('Native raw emission metadata only accepts the explicitly opted-in Tree protocol')
    if method == 'bapo':
        holdout = config.baselines.bapo.stage_validation_files
        if config.baselines.bapo.stage_validation_freq < 1:
            raise ValueError('BAPO stage validation frequency must be positive')
        if config.baselines.bapo.stage_validation_num < 1:
            raise ValueError('BAPO stage validation question count must be positive')
        evaluation_paths = canonical_paths(config.data.val_files)
        if config.data.get('final_val_files'):
            evaluation_paths |= canonical_paths(config.data.final_val_files)
        if not holdout or canonical_paths(holdout) & evaluation_paths:
            raise ValueError('BAPO requires independent stage_validation_files; never use final test sets')
    if method == 'cosearch':
        ranker = config.baselines.ranker
        if not ranker.model_path:
            raise ValueError('CoSearch requires an auxiliary ranker checkpoint')
        if ranker.keep != config.retriever.topk:
            raise ValueError('Controlled comparison must preserve actor observation topk')
        if (ranker.candidates < ranker.keep or ranker.min_group_size < 2 or ranker.temperature <= 0
                or min(ranker.batch_size, ranker.epochs, ranker.max_new_tokens) < 1
                or not 0 < ranker.clip_ratio < 1 or ranker.grad_clip <= 0
                or not 0 <= ranker.similarity <= 1):
            raise ValueError('Invalid ranker candidate/group/temperature settings')
    if config.algorithm.adv_estimator != 'grpo':
        raise ValueError('Select grpo as the critic-free outer driver; baseline hook supplies advantages')


def weighted_validation(metrics):
    total = count = 0
    for key, value in metrics.items():
        if key.startswith('val/test_score/'):
            source = key.split('/')[-1]
            n = metrics['val/evaluated_count/' + source]
            total += value * n
            count += n
    if not count:
        raise ValueError('Empty stage validation')
    return total / count


class Controller:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config.baselines
        self.tree_rng = random.Random(self.config.seed) if self.config.method == 'tree_grpo' else None
        torch.manual_seed(self.config.seed)
        self.method = self.config.method
        self.stage = BoundaryStage(patience=self.config.bapo.patience)
        self.ranker = self.teacher = None
        if self.method == 'bapo':
            trainer.reward_fn = BoundaryReward(trainer.tokenizer, self.stage, self.config.bapo.alpha,
                                               self.config.bapo.bonus)
            from torch.utils.data import DataLoader
            from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
            dataset = RLHFDataset(self.config.bapo.stage_validation_files, trainer.tokenizer,
                                 prompt_key=trainer.config.data.prompt_key,
                                 max_prompt_length=trainer.config.data.max_prompt_length,
                                 filter_prompts=True, return_raw_chat=False, truncation='error')
            prompt_key = trainer.config.data.prompt_key
            keys = question_keys(dataset, prompt_key)
            if keys & (question_keys(trainer.val_dataset, prompt_key) | question_keys(trainer.train_dataset, prompt_key)):
                raise ValueError('BAPO stage holdout overlaps training or final-test questions')
            final_files = trainer.config.data.get('final_val_files')
            if final_files:
                final_dataset = RLHFDataset(final_files, trainer.tokenizer,
                    prompt_key=prompt_key, max_prompt_length=trainer.config.data.max_prompt_length,
                    filter_prompts=True, return_raw_chat=False, truncation='error')
                if keys & question_keys(final_dataset, prompt_key):
                    raise ValueError('BAPO stage holdout overlaps full final-test questions')
            # User-directed 64-question phase checks. Keep the complete holdout
            # file and teacher selection unchanged; select one fixed subset per run.
            dataset.dataframe = phase_validation_frame(dataset.dataframe,
                self.config.bapo.stage_validation_num, self.config.seed)
            self.stage_dataset = dataset
            self.stage_loader = DataLoader(dataset, batch_size=trainer.config.data.val_batch_size,
                                           shuffle=False, drop_last=False, collate_fn=collate_fn)
        elif self.method == 'cosearch':
            from .cosearch import GenerativeRanker
            self.ranker = GenerativeRanker(self.config.ranker)
            trainer.reward_fn = self.main_f1_reward
        elif self.method == 'topd':
            from .distillation import FrozenTeacher
            self.teacher = FrozenTeacher(self.config.distillation.teacher_path, trainer.tokenizer, 'cpu')
        if self.method=='tree_grpo' and option(option(self.config.tree,'token_preservation',{}),'enabled',False):
            from .tree_raw_tokens import tokenizer_identity, tokenizer_validation_scope
            preservation=option(self.config.tree,'token_preservation',{})
            self._tree_raw_identity=tokenizer_identity(trainer.tokenizer,option(preservation,'tokenizer_sha256'))
            reward_fn=trainer.reward_fn
            def scoped_reward(*args,**kwargs):
                with tokenizer_validation_scope(trainer.tokenizer,self._tree_raw_identity,
                        _context=getattr(self,'_tree_validation_context',None)) as context:
                    self._tree_validation_context=context
                    return reward_fn(*args,**kwargs)
            trainer.reward_fn=scoped_reward

    def generate(self, manager, gen_batch, qa_batch, metrics):
        preservation=(option(self.config.tree,'token_preservation',{}) if self.method=='tree_grpo' else {})
        if not option(preservation,'enabled',False):
            return self._generate(manager,gen_batch,qa_batch,metrics)
        from .tree_raw_tokens import tokenizer_identity, tokenizer_validation_scope
        identity=getattr(self,'_tree_raw_identity',None)
        if identity is None:
            identity=tokenizer_identity(manager.tokenizer,option(preservation,'tokenizer_sha256'))
            self._tree_raw_identity=identity
        # A fresh outer has no proof cache from its predecessor. The default
        # reward reader and actor audit reuse only this completed outer's proofs.
        with tokenizer_validation_scope(manager.tokenizer,identity) as context:
            self._tree_validation_context=context
            return self._generate(manager,gen_batch,qa_batch,metrics)

    def _generate(self, manager, gen_batch, qa_batch, metrics):
        preservation=(option(self.config.tree,'token_preservation',{}) if self.method=='tree_grpo' else {})
        preservation=preservation if option(preservation,'enabled',False) else None
        behavior = option(self.config.tree, 'behavior_mu', {}) if self.method == 'tree_grpo' else {}
        behavior = behavior if option(behavior, 'enabled', False) else None
        engine = SearchRollout(manager, self.config.gigpo.anchor_mode, self.ranker, self.config.seed,
                               self.config.context_alignment, token_preservation=preservation,
                               behavior_mu=behavior)
        if self.tree_rng is not None:
            # Keep parent/leaf sampling continuous across outer updates while
            # constructing a fresh engine and cost ledger for each rollout.
            engine.rng = self.tree_rng
        if self.ranker:
            self.ranker.question_uids = qa_batch.non_tensor_batch['index'].tolist()
            self.ranker.validation = False
            self.ranker.events.clear()
        tree = self.config.tree if self.method == 'tree_grpo' else None
        if tree is not None and option(tree, 'branch_sampling') in ('branpo_tail_v1', 'random_nested_backtrack_v1'):
            def training_rewards(states):
                return self.tree_training_rewards(engine, qa_batch, states,
                    scoring_only=option(tree, 'branch_sampling') == 'random_nested_backtrack_v1')
            engine.reward_callback = training_rewards
        audit_dir = os.environ.get('SEARCH_R1_TREE_AUDIT_DIR')
        if tree is not None and audit_dir:
            from .tree_audit import TreeAudit
            self.tree_audit_sequence = getattr(self, 'tree_audit_sequence', 0) + 1
            engine.audit = TreeAudit(audit_dir, os.environ.get('SEARCH_R1_BASELINE_OUTPUT'),
                                     self.trainer.global_steps, self.tree_audit_sequence,
                                     qa_batch.non_tensor_batch['index'].tolist())
            engine.audit.report['algorithm_source_commit'] = getattr(self.config, 'algorithm_source_commit', None)
            engine.audit.report['base_commit'] = getattr(self.config, 'base_commit', None)
            engine.audit.report['attempt_id'] = getattr(self.trainer, 'training_attempts', None)
        output, sources = engine.run(gen_batch, tree)
        if engine.audit is not None:
            self.latest_tree_audit = engine.audit
        costs = engine.cost.copy()
        if self.method == 'bapo' and not self.stage.early:
            states = engine.last_states
            targets = [r['ground_truth']['target'] for r in qa_batch.non_tensor_batch['reward_model']]
            uids = qa_batch.non_tensor_batch['index']
            for _ in range(self.config.bapo.resample_attempts):
                rows = resample_rows(decode_answers(output, self.trainer.tokenizer, require_legal=True), targets, uids, self.stage)
                if not rows:
                    break
                engine.run(select_rows(gen_batch, rows))
                for row, state in zip(rows, engine.last_states):
                    state.source = row
                    states[row] = state
                for key, value in engine.cost.items():
                    costs[key] += value
                output, sources = engine.compose(states)
        metrics.update({'rollout/' + k: v for k, v in costs.items()})
        branches = [s.branch_metadata for s in engine.last_states if s.branch_metadata.get('continuation', 0)]
        if branches:
            for key in ('action_depth', 'backoff_actions', 'inherited_response_tokens', 'inherited_observation_tokens'):
                metrics['tree_sampling/' + key + '_mean'] = float(np.mean([b[key] for b in branches]))
            metrics['tree_sampling/continuations'] = len(branches)
            metrics['tree_sampling/inherited_prefix_verified_fraction'] = sum(b['inherited_prefix_verified'] for b in branches)/len(branches)
            metrics['tree_sampling/same_reward_branch_fraction'] = sum(b['same_reward_as_original'] for b in branches)/len(branches)
            metrics['tree_sampling/same_reward_branch_count'] = sum(b['same_reward_as_original'] for b in branches)
            metrics['tree_sampling/different_reward_branch_count'] = sum(not b['same_reward_as_original'] for b in branches)
            for depth in range(self.trainer.config.max_turns+1):
                metrics[f'tree_sampling/action_depth_{depth}_count'] = sum(b['action_depth'] == depth for b in branches)
            metrics['tree_sampling/earliest_state_clamp_count'] = sum(b['backoff_clamped'] for b in branches)
        return output, sources

    def tree_training_rewards(self, engine, qa_batch, states, scoring_only=False):
        # Build the selected training representation or a candidate scoring-only view.
        # No targets or rewards enter model generation contexts.
        scored, sources = engine.reward_batch(states) if scoring_only else engine.compose(states)
        for key, values in qa_batch.non_tensor_batch.items():
            if key not in scored.non_tensor_batch:
                scored.non_tensor_batch[key] = values[sources]
        return self.trainer.reward_fn(scored).sum(-1).detach().cpu().tolist()

    def audit_actor_batch(self, batch):
        if self.method=='tree_grpo' and option(option(self.config.tree,'token_preservation',{}),'enabled',False):
            from .tree_raw_tokens import FIELD,require,semantic_spans,tokenizer_validation_scope
            require(FIELD in batch.non_tensor_batch,'Opted-in actor batch lost raw/semantic provenance')
            first=batch[0].non_tensor_batch[FIELD]['raw_ledger'][0]['action']['tokenizer_identity']
            with tokenizer_validation_scope(self.trainer.tokenizer,first,
                    _context=getattr(self,'_tree_validation_context',None)) as context:
                self._tree_validation_context=context
                contexts={}
                for i in range(len(batch)):semantic_spans(batch[i],self.trainer.tokenizer,contexts)
        if self.method != 'tree_grpo' or not os.environ.get('SEARCH_R1_TREE_AUDIT_DIR'):
            return
        audit = getattr(self, 'latest_tree_audit', None)
        if audit is None:
            raise ValueError('Actor Tree audit has no completed rollout evidence')
        audit.policy_batch(batch, self.trainer.global_steps)

    def main_f1_reward(self, data):
        answers = decode_answers(data, self.trainer.tokenizer, require_legal=True)
        result = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        width = result.shape[1]
        mask = data.batch['info_mask'][:, -width:] * data.batch['attention_mask'][:, -width:]
        for i, answer in enumerate(answers):
            targets = data.non_tensor_batch['reward_model'][i]['ground_truth']['target']
            score = -.2 if answer is None else max(0., ordinary_answer_f1(answer, targets))
            positions = torch.where(mask[i].bool())[0]
            if len(positions):
                result[i, positions[-1]] = score
        return result

    def compute_advantages(self, data, metrics):
        width = data.batch['responses'].shape[1]
        mask = data.batch['attention_mask'][:, -width:] * data.batch['info_mask'][:, -width:]
        uids = data.non_tensor_batch['uid']
        rewards = data.batch['token_level_rewards']
        # Terminal base EM can fall on final search observation. For action estimators,
        # assign the unchanged scalar to the last retained actor token.
        if self.method in ('gigpo', 'tree_grpo'):
            outcome = rewards.sum(-1)
            rewards = torch.zeros_like(rewards)
            for i in range(len(data)):
                positions = torch.where(mask[i].bool())[0]
                if len(positions):
                    rewards[i, positions[-1]] = outcome[i]
            data.batch['token_level_rewards'] = rewards
        if self.method == 'gigpo':
            g = self.config.gigpo
            adv = advantages.gigpo(rewards, mask, data.batch['step_ids'], data.non_tensor_batch['anchors'],
                                    uids, g.gamma, g.omega, g.normalize)
            returns = adv.clone()
            counts = Counter((uid, anchor) for uid, row in zip(uids, data.non_tensor_batch['anchors']) for anchor in row)
            total = sum(counts.values())
            metrics['gigpo/shared_anchor_coverage'] = sum(n for n in counts.values() if n > 1) / max(1, total)
        elif self.method == 'tree_grpo':
            adv = advantages.tree_grpo(rewards, mask, uids, data.non_tensor_batch['tree_uid'])
            metrics.update(advantages.tree_variance_metrics(outcome, uids,
                data.non_tensor_batch['tree_uid'], data.non_tensor_batch.get('tree_branch_metadata'),
                estimator_score=rewards.sum(-1)))
            metrics['tree_variance/empty_loss_row_count'] = int((mask.sum(-1) == 0).sum())
            returns = adv.clone()
        elif self.method == 'topd':
            setting = self.config.distillation
            scored = []
            for i in range(len(data)):
                targets = [data.non_tensor_batch['reward_model'][i]['ground_truth']['target']] if setting.privileged_answers else None
                scored.append(self.teacher.log_probs(data.batch['prompts'][i:i+1], data.batch['responses'][i:i+1],
                              data.batch['attention_mask'][i:i+1, -width:],
                              self.trainer.config.actor_rollout_ref.rollout.temperature, targets, setting.chunk_size))
            teacher_logp = torch.cat(scored).to(rewards.device)
            adv, returns, dense = advantages.topd(teacher_logp, data.batch['old_log_probs'], mask, uids, self.config.topd.alpha)
            data.batch['token_level_rewards'] = dense
            metrics.update({'teacher/scored_tokens_cumulative': self.teacher.scored_tokens,
                            'teacher/parameters': self.teacher.parameter_count})
        elif self.method == 'opsd':
            # Pure distribution loss; outcome scores are reporting only.
            adv = returns = torch.zeros_like(rewards)
        else:
            adv = advantages.grouped_center(rewards.sum(-1), list(uids))[:, None] * mask
            returns = adv.clone()
            if self.method == 'bapo':
                from .bapo import is_idk
                answers = decode_answers(data, self.trainer.tokenizer, require_legal=True)
                idk_count = sum(map(is_idk, answers))
                metrics['bapo/idk_ratio'] = idk_count / max(1, len(answers))
                metrics['bapo/answered_coverage'] = sum(a is not None and not is_idk(a) for a in answers) / max(1, len(answers))
                metrics['bapo/early_stage'] = int(self.stage.early)
        data.batch['advantages'], data.batch['returns'] = adv, returns
        if self.method in ('gigpo', 'tree_grpo', 'topd', 'cosearch'):
            from .objectives import attach_policy_weights
            attach_policy_weights(data, self.method)
        return data

    def after_actor(self, data, metrics):
        if self.ranker:
            answers = decode_answers(data, self.trainer.tokenizer, require_legal=True)
            outcomes = {}
            for i, uid in enumerate(data.non_tensor_batch['trajectory_uid']):
                targets = data.non_tensor_batch['reward_model'][i]['ground_truth']['target']
                outcomes[uid] = (float(data.batch['token_level_scores'][i].sum()), targets, answers[i] is not None)
            metrics.update(self.ranker.update(outcomes))

    def stage_validation(self, step=None):
        if self.method != 'bapo':
            return {}
        if step is not None and step % self.config.bapo.stage_validation_freq:
            return {}
        t = self.trainer
        old_reward = t.val_reward_fn
        from verl.trainer.main_ppo import RewardManager
        t.val_reward_fn = RewardManager(t.tokenizer, 0)
        old_loader, old_dataset = t.val_dataloader, t.val_dataset
        try:
            t.val_dataloader, t.val_dataset = self.stage_loader, self.stage_dataset
            score = weighted_validation(t._validate())
        finally:
            t.val_dataloader, t.val_dataset = old_loader, old_dataset
            t.val_reward_fn = old_reward
        self.stage.observe(score)
        return {'bapo/stage_validation': score, 'bapo/early_stage': int(self.stage.early),
                'bapo/stalled': self.stage.stalled}

    def save(self, step):
        directory = os.path.join(self.trainer.config.trainer.default_local_dir, 'algorithm', f'global_step_{step}')
        os.makedirs(directory, exist_ok=True)
        if self.ranker:
            self.ranker.save(os.path.join(directory, 'ranker'))
        state = dict(method=self.method, global_step=step, bapo=self.stage.state_dict(),
                     base_commit=self.config.base_commit,
                     algorithm_source_commit=self.config.algorithm_source_commit,
                     teacher_path=self.config.distillation.teacher_path,
                     resume_supported=False)
        temporary = os.path.join(directory, 'state.json.tmp')
        with open(temporary, 'w') as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
        os.replace(temporary, os.path.join(directory, 'state.json'))


def ranked_validation_manager(manager, controller):
    """Validation uses the trained ranker, with no teacher/gold/oracle at inference."""
    class Adapter:
        timing_raw = None
        def run_llm_loop(self, gen_batch, initial_input_ids):
            engine = SearchRollout(manager, ranker=controller.ranker)
            result, _ = engine.run(gen_batch)
            return result
    return Adapter()


def token_preserving_validation_manager(manager, controller):
    """Same greedy one-trajectory validation, with the Tree semantic projection."""
    class Adapter:
        timing_raw = None
        def run_llm_loop(self, gen_batch, initial_input_ids):
            settings=controller.config.tree.token_preservation
            engine=SearchRollout(manager,context_alignment='error',token_preservation=settings)
            result,_=engine.run(gen_batch)
            return result
    return Adapter()
