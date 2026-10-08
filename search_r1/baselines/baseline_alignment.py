"""Strict Search-R1 mechanics profile; BAPO reward/controller remain separate."""
import hashlib
import json
from pathlib import Path


def read_bound_json(path, expected_sha):
    raw = Path(path).read_bytes()
    if len(expected_sha or '') != 64 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Aligned input manifest SHA256 differs')
    return json.loads(raw)


def validate_aligned_config(config):
    actor = config['actor_rollout_ref']['actor']
    rollout = config['actor_rollout_ref']['rollout']
    trainer = config['trainer']
    required = [config.get('baseline_aligned') is True, actor.get('baseline_aligned') is True,
                actor.get('fresh200_enabled') is True, trainer.get('fresh200_enabled') is True,
                actor.get('use_kl_loss') is True, actor.get('kl_loss_coef') == .001,
                actor.get('kl_loss_type') == 'low_var_kl',
                actor.get('ppo_mini_batch_size') == 128, actor.get('ppo_micro_batch_size') == 16,
                actor.get('ppo_epochs') == 1, actor.get('clip_ratio') == .2,
                actor.get('grad_clip') == 1., actor.get('entropy_coeff') == 0.,
                actor.get('loss_reduction') == 'token_mean',
                actor.get('policy_safety', {}).get('dual_clip') is None,
                actor.get('policy_safety', {}).get('skip_on_policy_guard') is False,
                actor['optim'].get('num_warmup_steps') == 57, actor['optim'].get('lr') == 1e-6,
                config['algorithm']['kl_ctrl'].get('kl_coef') == 0.,
                config['data'].get('train_batch_size') == 512, rollout.get('n_agent') == 5,
                config['data'].get('train_data_num') is None, config['data'].get('val_data_num') is None,
                trainer.get('fresh200_diagnostic_only') is False,
                config['baselines'].get('method') == 'bapo',
                config['actor_rollout_ref']['model'].get('revision') == 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
                config.get('force_final_answer') is True, config.get('max_turns') == 4,
                config['retriever'].get('topk') == 3,
                trainer.get('n_gpus_per_node') == 8, trainer.get('nnodes') == 1,
                trainer.get('total_training_steps') == 201,
                config['baselines']['holdout'].get('input_mode') == 'baseline_aligned_manifest']
    required.extend(not trainer.get(k) for k in ('resume_full_state', 'resume_permission', 'resume_from', 'resume_path', 'resume_verify_only'))
    if not all(required):
        raise ValueError('Aligned profile requires HF-old PPO, actor-only KL, mini128/20, no Dual-Clip or finite-update SKIP')
    return 200, 57


def bind_frozen_hf(data):
    """Do not overwrite HF old LP with native sampler LP on retokenized text."""
    if 'frozen_hf_log_probs' in data.batch:
        raise ValueError('Frozen HF denominator already bound')
    old = data.batch['old_log_probs']
    if not old.is_floating_point():
        raise ValueError('HF denominator must remain floating point')
    data.batch['frozen_hf_log_probs'] = old.detach().clone()
    data.meta_info.update(baseline_aligned=True, ppo_denominator_role='frozen_preupdate_hf_actor',
                          behavior_policy_role='raw_sampler_diagnostic_only',
                          frozen_hf_probability_role='ppo_denominator')


def prepare_aligned_inputs(config):
    from .baseline_alignment_data import verify_prepared_manifest, verify_stage_files
    settings = config.baselines.holdout
    manifest = read_bound_json(settings.manifest_path, settings.manifest_sha256)
    actual = verify_prepared_manifest(manifest, config.data.train_files, config.data.val_files,
                                     config.data.final_val_files)
    if actual['splits']['validation']['rows'] != 512 or actual['splits']['test']['rows'] != 51713:
        raise ValueError('Aligned shared dev512 and final51713 required')
    stage = verify_stage_files(manifest, config.baselines.bapo.stage_validation_files,
                              config.data.train_files, config.data.val_files, config.data.final_val_files)
    if stage['stage']['rows'] != 64:
        raise ValueError('Aligned BAPO stage must contain64 questions')
    return dict(input_mode='baseline_aligned_manifest', manifest=manifest,
                manifest_path=str(Path(settings.manifest_path).resolve()),
                manifest_sha256=settings.manifest_sha256, stage=stage,
                content_identity={'sha256': actual['splits']['train']['semantic_sha256']},
                train_rows=actual['splits']['train']['rows'], holdout_rows=actual['splits']['validation']['rows'])
