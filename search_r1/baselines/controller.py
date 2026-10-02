"""Opt-in hooks for the reviewed Ray trainer; no changes to baseline inputs."""
import json
import os
from collections import Counter
from pathlib import Path
import torch
from . import advantages
from .rollout import SearchRollout, select_rows
from .bapo import BoundaryStage, BoundaryReward, decode_answers, ordinary_answer_f1, resample_rows


METHODS = ('grpo', 'gigpo', 'tree_grpo', 'bapo', 'opsd', 'topd', 'cosearch')


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


def validate_config(config):
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
        if config.actor_rollout_ref.rollout.n_agent != tree.roots:
            raise ValueError('Tree roots must equal rollout.n_agent')
        if min(tree.roots, tree.leaves, tree.expansions, tree.rounds) < 1:
            raise ValueError('Tree budgets must be positive')
        if tree.leaves > 1 + tree.expansions * tree.rounds:
            raise ValueError('Insufficient expansions for selected leaves')
    if method == 'bapo':
        holdout = config.baselines.bapo.stage_validation_files
        if config.baselines.bapo.stage_validation_freq < 1:
            raise ValueError('BAPO stage validation frequency must be positive')
        if not holdout or canonical_paths(holdout) & canonical_paths(config.data.val_files):
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

    def generate(self, manager, gen_batch, qa_batch, metrics):
        engine = SearchRollout(manager, self.config.gigpo.anchor_mode, self.ranker, self.config.seed,
                               self.config.context_alignment)
        if self.ranker:
            self.ranker.question_uids = qa_batch.non_tensor_batch['index'].tolist()
            self.ranker.validation = False
            self.ranker.events.clear()
        tree = self.config.tree if self.method == 'tree_grpo' else None
        output, sources = engine.run(gen_batch, tree)
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
        return output, sources

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
                     base_commit='b251ba6b503217a59cd3395c1326d7b66b602989',
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
