"""Explicit fresh-200 experiment contract, independent of historical recovery."""
import os

PROTOCOL = 'tree_fresh200_five_leaf_v1'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_SHA256 = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'


def tree_option(tree, key, default=None):
    getter = getattr(tree, 'get', None)
    return getter(key, default) if callable(getter) else getattr(tree, key, default)


def leaf_quotas(tree):
    """Legacy rectangular trees or explicit per-root allocation, never drop extras."""
    quotas = tree_option(tree, 'leaf_quotas')
    if quotas is None:
        return (int(tree.leaves),) * int(tree.roots)
    quotas = tuple(quotas)
    if (len(quotas) != tree.roots or any(type(q) is not int or q < 1 for q in quotas)
            or max(quotas) != tree.leaves):
        raise ValueError('Invalid explicit Tree per-root leaf quotas')
    return quotas


def is_fresh200(config):
    return config.baselines.get('training_protocol') == PROTOCOL


def validate_fresh200_config(config):
    if config.baselines.tree.get('branch_sampling') == 'recursive_uniform_v1':
        return validate_recursive_config(config)
    actor, rollout, ref = (config.actor_rollout_ref[k] for k in ('actor', 'rollout', 'ref'))
    tree, holdout, trainer = config.baselines.tree, config.baselines.holdout, config.trainer
    safety = actor.get('policy_safety', {})
    if tree.get('branch_sampling') == 'branpo_tail_v1':
        if tree.get('continuation_budget_per_question') != 3:
            raise ValueError('BranPO-inspired sampling requires exactly three finite continuations per question')
    elif tree.get('branch_sampling') is not None:
        raise ValueError('Unknown Tree branch sampling protocol')
    correction = safety.get('sampling_probability_correction', {})
    from .tree_guard_continue import PROTOCOL as CONTINUE_PROTOCOL
    if actor.get('guard_continue_protocol') != CONTINUE_PROTOCOL or safety.get('target_old_policy_kl') != .2:
        raise ValueError('Fresh200 requires explicit fresh-rollout guard continuation and sampled-policy KL 0.2')
    if (safety.get('enabled') is not True or safety.get('dual_clip') != 3.
            or tree.get('token_preservation', {}).get('enabled') is not True
            or tree.get('behavior_mu', {}).get('enabled') is not True
            or correction.get('enabled') is not True
            or correction.get('protocol') != 'tree_behavior_mu_centered_ppo_v1'
            or rollout.get('tree_raw_emission') is not True):
        raise ValueError('Fresh200 requires the explicit guarded sampled-mu correction, without an uncorrected fallback')
    if (not is_fresh200(config) or config.baselines.method != 'tree_grpo'
            or config.baselines.get('model_revision') != REVISION
            or config.baselines.get('model_repository') != 'Qwen/Qwen2.5-3B-Instruct'
            or config.actor_rollout_ref.model.path != ref.model_path
            or holdout.input_mode != 'prebuilt_manifest' or holdout.manifest_sha256 != DATA_SHA256):
        raise ValueError('Fresh200 model/split/protocol identity changed')
    if ((tree.roots, tree.rounds, tree.expansions, tree.leaves) != (2, 1, 2, 3)
            or leaf_quotas(tree) != (3, 2) or rollout.n_agent != 2 or rollout.n != 1
            or type(rollout.get('best_of')) is not int or rollout.best_of != 1
            or config.data.train_batch_size != 512
            or (actor.ppo_mini_batch_size, actor.ppo_micro_batch_size, actor.ppo_epochs) != (128, 16, 1)
            or (rollout.log_prob_micro_batch_size, ref.log_prob_micro_batch_size) != (32, 32)
            or (trainer.n_gpus_per_node, trainer.nnodes) != (8, 1)
            or actor.ulysses_sequence_parallel_size != 1 or actor.use_dynamic_bsz):
        raise ValueError('Fresh200 five-leaf/batch/DP20-call contract changed')
    if (actor.optim.lr != 1e-6 or actor.optim.lr_warmup_steps != 57
            or actor.optim.lr_warmup_steps_ratio != 0 or actor.optim.total_training_steps != 200
            or actor.entropy_coeff != 0 or actor.kl_loss_coef != 0
            or actor.kl_loss_type != 'low_var_kl' or actor.clip_ratio != .2 or actor.grad_clip != 1
            or config.algorithm.kl_ctrl.kl_coef != 0
            or actor.get('fresh200_protocol') != PROTOCOL):
        raise ValueError('Fresh200 objective/57-warmup/scheduler contract changed')
    if (trainer.total_training_steps != 201 or trainer.actual_update_limit != 200
            or trainer.save_freq != 50 or trainer.acceptance_step != 5
            or not trainer.acceptance_gate_enabled or trainer.test_freq != 5
            or config.max_turns != 4 or config.force_final_answer is not False
            or (config.data.max_start_length, config.data.max_prompt_length,
                config.data.max_response_length, config.data.max_obs_length) != (2048, 4096, 500, 500)):
        raise ValueError('Fresh200 iteration/checkpoint/search contract changed')


def refuse_recovery_environment(environ=None):
    environ = os.environ if environ is None else environ
    keys = [key for key in environ if key.startswith('TREE_RECOVERY_')]
    if keys:
        raise ValueError('Fresh200 forbids historical recovery environment: '+', '.join(sorted(keys)))
    from .tree_current_chain_resume import validate_entry
    validate_entry(environ=environ)


def validate_fresh_sampler(sampler):
    # The checkpoint sampler creates epoch zero's order only on first iteration.
    if sampler.cursor != 0 or sampler.epoch != -1 or sampler.order is not None:
        raise RuntimeError('Fresh200 sampler was initialized or consumed before the fresh boundary')


def validate_completed_outer(metrics, outer):
    """Credit twenty processed opportunities, actual writes >=1, one scheduler advance."""
    opportunities = metrics.get('actor/optimizer_opportunities_processed')
    writes = metrics.get('actor/optimizer_steps')
    if opportunities is not None:
        if opportunities != 20 or writes is None or not 1 <= writes <= 20 or writes != int(writes):
            raise RuntimeError('Fresh200 incomplete opportunities or zero actual optimizer writes')
        expected = {'actor/optimizer_opportunities_skipped': 20-writes,
                    'actor/scheduler_step_before_outer': outer-1, 'actor/scheduler_step_after_outer': outer}
    else:
        expected = {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0,
                'actor/scheduler_step_before_outer': outer - 1,
                'actor/scheduler_step_after_outer': outer}
    if any(metrics.get(k) != value for k, value in expected.items()):
        raise RuntimeError('Fresh200 incomplete actual optimizer/scheduler outer; writes are retained')


def validate_recursive_config(config):
    actor,rollout=config.actor_rollout_ref.actor,config.actor_rollout_ref.rollout
    tree,trainer=config.baselines.tree,config.trainer
    if ((tree.roots,tree.leaves,tree.continuation_budget_per_question)!=(1,5,10)
        or leaf_quotas(tree)!=(5,) or rollout.n_agent!=1 or rollout.n!=1
        or (actor.ppo_mini_batch_size,actor.ppo_micro_batch_size,actor.ppo_epochs)!=(128,16,1)
        or config.data.train_batch_size!=512 or trainer.actual_update_limit!=200
        or trainer.total_training_steps!=201 or trainer.n_gpus_per_node!=8
        or trainer.test_freq!=5 or trainer.save_freq!=50
        or trainer.acceptance_gate_enabled or actor.optim.lr!=1e-6
        or actor.optim.lr_warmup_steps!=57 or actor.optim.total_training_steps!=200
        or not actor.get('recursive_document_objective')
        or actor.entropy_coeff!=0 or actor.kl_loss_coef!=0 or config.algorithm.kl_ctrl.kl_coef!=0
        or actor.clip_ratio!=.2 or actor.grad_clip!=1
        or config.baselines.model_revision!=REVISION or config.max_turns!=4
        or config.force_final_answer or config.actor_rollout_ref.model.path!=config.actor_rollout_ref.ref.model_path):
        raise ValueError('Recursive document experiment A contract changed')
