"""Manifest-bound BF16 execution; legacy diagnostic manifests remain FP32."""

BF16_PROFILE = 'bf16_policy_fp32_master'
SAMPLING_JOURNAL_POLICY = 'sampling_journal_no_extra_hf_forward'


def bf16_policy(run):
    profile = run.get('precision_profile')
    assert profile in (None, BF16_PROFILE), 'Unknown policy precision profile'
    if profile == BF16_PROFILE:
        assert run.get('execution_backend') == 'ssh_local' and run.get('training_world_size') == 4
        return True
    return False


def overrides(run):
    if not bf16_policy(run):
        return {}
    return {
        'actor_rollout_ref.actor.compute_dtype': 'bfloat16',
        'actor_rollout_ref.actor.allow_tf32': 'null',
        'actor_rollout_ref.ref.compute_dtype': 'bfloat16',
        'actor_rollout_ref.ref.allow_tf32': 'null',
        '+actor_rollout_ref.actor.fsdp_config.model_dtype': 'fp32',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype': 'bf16',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype': 'fp32',
        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype': 'fp32',
        '+actor_rollout_ref.ref.fsdp_config.model_dtype': 'fp32',
        'actor_rollout_ref.rollout.dtype': 'bfloat16',
    }


def sampling_journal_only(run):
    policy = run.get('behavior_probability_policy')
    assert policy in (None, SAMPLING_JOURNAL_POLICY), 'Unknown behavior probability policy'
    if policy == SAMPLING_JOURNAL_POLICY:
        assert bf16_policy(run), 'Sampling-journal-only mode requires the explicit BF16 profile'
        return True
    return False


def verify(document, run):
    parts = document['actor_rollout_ref']
    mixed = parts['actor']['fsdp_config']['mixed_precision']
    use_bf16 = bf16_policy(run)
    for role in ('actor', 'ref'):
        part = parts[role]
        assert part['compute_dtype'] == ('bfloat16' if use_bf16 else 'float32')
        assert part['allow_tf32'] is (None if use_bf16 else False)
        assert part['fsdp_config']['model_dtype'] == 'fp32'
    assert mixed['param_dtype'] == ('bf16' if use_bf16 else 'fp32')
    assert mixed['reduce_dtype'] == mixed['buffer_dtype'] == 'fp32'
    assert parts['rollout']['dtype'] == ('bfloat16' if use_bf16 else 'float32')
    return True
