"""The new formal200 contract, distinct from immutable constant300 diagnostics."""
import copy
import hashlib
from pathlib import Path

from reproduction.common8.policy_calibration_plan import MATHEMATICS, canonical_bytes

CONTRACT_KIND = 'fresh200_warmup57'
TARGET_UPDATES = 200
WARMUP_STEPS = 57
SAVE_STEPS = (5, 50, 100, 150, 200)
PROBE_ROUNDS = (0, 50, 100, 150)
MEASUREMENT_ROLE = 'warmup57_measurement'
FORMAL_MATHEMATICS = copy.deepcopy(MATHEMATICS)
FORMAL_MATHEMATICS.update(target_actual_outer_updates=TARGET_UPDATES,
    lr_warmup_steps=WARMUP_STEPS,
    learning_rate_schedule='linear_fixed_outer_warmup_then_constant',
    entropy_coefficient=0.0, reference_kl_coefficient=0.0)


def consumed_lr(step):
    assert type(step) is int and 1 <= step <= TARGET_UPDATES
    return 1e-6 * min((step - 1) / WARMUP_STEPS, 1.0)


def assert_scientific_run(run, *, expected_role=None):
    assert run['profile'] == 'common8' and run['contract_kind'] == CONTRACT_KIND
    role=run.get('policy_calibration_role','production')
    assert role in ('production',MEASUREMENT_ROLE), 'Wrong new scientific execution role'
    assert expected_role is None or role==expected_role
    if role==MEASUREMENT_ROLE:
        assert type(run['measurement_outer_count']) is int and run['measurement_outer_count']==2
    assert type(run['completed_updates']) is int and run['completed_updates'] == 0
    assert type(run['target_updates']) is int and run['target_updates'] == TARGET_UPDATES
    assert type(run['acceptance_update']) is int and run['acceptance_update'] == 5
    assert run['initial_model'] == run['reference_model']
    assert run['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    budget = run['budget']
    assert budget['contract_kind'] == CONTRACT_KIND and budget['profile']=='common8'
    assert budget['model_id']=='Qwen/Qwen2.5-3B-Instruct' and budget['model_revision']==run['model_revision']
    for name, value in FORMAL_MATHEMATICS.items():
        assert canonical_bytes(budget[name]) == canonical_bytes(value), 'Formal budget differs: ' + name
    assert budget['common_manifest_sha256'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    return run


def assert_formal_run(run):
    return assert_scientific_run(run,expected_role='production')


def verify_scientific_config(document, run, base):
    """Validate actual resolved fields, paths and bound source, not a label alone."""
    assert_scientific_run(run)
    base = Path(base).resolve()
    actor = document['actor_rollout_ref']['actor']
    reference = document['actor_rollout_ref']['ref']
    rollout = document['actor_rollout_ref']['rollout']
    def exact(part, values, label):
        for name, value in values.items():
            actual = part[name]
            equal = (type(actual) in (int, float) and actual == value) if type(value) is float else (
                canonical_bytes(actual) == canonical_bytes(value))
            assert equal, label + ' differs: ' + name
    exact(actor, dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1,
        grad_clip=1.0, clip_ratio=0.2, state_masking=True, use_kl_loss=True,
        entropy_coeff=0.0, kl_loss_coef=0.0, kl_loss_type='low_var_kl',
        use_dynamic_bsz=False, ulysses_sequence_parallel_size=1), 'Actor mathematics')
    exact(actor['optim'], dict(lr=1e-6, lr_warmup_steps_ratio=0.0,
        lr_warmup_steps=WARMUP_STEPS, warmup_style='constant'), 'Optimizer mathematics')
    for part in (actor, reference):
        exact(part, dict(compute_dtype='float32', allow_tf32=False), 'Actor/reference precision')
        assert part['fsdp_config']['model_dtype'] == 'fp32'
    assert all(actor['fsdp_config']['mixed_precision'][name] == 'fp32'
               for name in ('param_dtype', 'reduce_dtype', 'buffer_dtype'))
    exact(rollout, dict(n=1, n_agent=5, temperature=1.0, top_p=1.0,
        log_prob_micro_batch_size=32, dtype='float32', tensor_model_parallel_size=1,
        name='vllm', gpu_memory_utilization=0.4), 'Rollout mathematics')
    exact(reference, dict(log_prob_micro_batch_size=32), 'Reference micro batch')
    exact(document['algorithm'], dict(adv_estimator='grpo', no_think_rl=False), 'Advantages')
    exact(document['data'], dict(train_batch_size=512, val_batch_size=64,
        max_prompt_length=4096, max_response_length=500, max_start_length=2048,
        max_obs_length=500, train_data_num=None, val_data_num=None,
        shuffle_train_dataloader=True), 'Data budget')
    assert document['data']['train_files'] == str(base / 'data/train.parquet')
    assert document['data']['val_files'] == str(base / 'data/stage.parquet')
    assert document['actor_rollout_ref']['model']['path'] == run['initial_model']
    exact(document['actor_rollout_ref']['model'], dict(enable_gradient_checkpointing=True,
        use_remove_padding=False), 'Model execution graph')
    exact(actor['fsdp_config'], dict(param_offload=True,grad_offload=True,
        optimizer_offload=True), 'Actor FSDP graph')
    assert reference['fsdp_config']['param_offload'] is True
    exact(document['trainer'], dict(n_gpus_per_node=8, nnodes=1,
        total_training_steps=TARGET_UPDATES + 1, total_epochs=15, save_freq=50,
        test_freq=50, acceptance_step=5, val_only=False, val_before_train=True), 'Training horizon')
    assert document['trainer']['default_local_dir'] == run['checkpoint_dir']
    assert Path(run['checkpoint_dir']).resolve() == base / 'checkpoints/search-g1-common8'
    assert document['max_turns'] == 4 and document['do_search'] is True
    exact(document['retriever'], dict(url='http://127.0.0.1:18010/retrieve', topk=3), 'Retrieval')
    policy = actor['policy_safety']
    exact(policy, dict(enabled=True, dual_clip=3.0, target_old_policy_kl=0.01,
        kl_log_bound=30.0, negative_tail_limit=None), 'Safeguard candidates')
    assert policy['calibration_manifest'] == str(base / 'logs/policy-safety-calibration.json')
    repo = Path(__file__).resolve().parents[2]
    for name in ('reproduction/common8/recipe.json', 'reproduction/common8/budget.json',
                 'verl/trainer/config/ppo_trainer.yaml'):
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == run['source_sha256'][name]
    return dict(contract_kind=CONTRACT_KIND,execution_role=run.get('policy_calibration_role','production'),
        mathematical_config_sha256=hashlib.sha256(canonical_bytes(FORMAL_MATHEMATICS)).hexdigest(),
        resolved_config_sha256=hashlib.sha256(canonical_bytes(document)).hexdigest(),
        source_map_sha256=hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest(),
        source_snapshot_semantics='bound disk source hashes; not proof of loaded modules')


def verify_formal_config(document,run,base):
    assert_formal_run(run)
    return verify_scientific_config(document,run,base)
