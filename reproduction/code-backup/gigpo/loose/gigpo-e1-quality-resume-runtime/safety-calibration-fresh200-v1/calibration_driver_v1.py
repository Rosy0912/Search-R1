"""Two real official-fresh outers, independent of formal200 progress.

The actual trainer APIs keep GiGPO math, raw action mapping and state groups.
First existing sampling is observed and independently accepted before any Adam.
No fit(), historical restore, synthetic native or threshold invention occurs.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from calibration_contract_v1 import (SCOPE, DATA, REVISION, initial_fields, require,
    validate_alignment, admit_alignment, validate_probability_receipts, summarize_guards,
    validate_probability_approval, validate_action_probability_checks)
from calibration_state_v1 import state_digest, validate_pause_workers, assert_complete_pause_unchanged


def json_save(path, value):
    from search_r1.baselines.recovery import atomic_json_save
    atomic_json_save(value, path)


def validate_actual_flash_backend(rows, binding):
    """An environment request or missing fallback log is insufficient for admission."""
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
            and len({row['pid'] for row in rows}) == 8, 'Missing actual eight-worker backend identity')
    for row in rows:
        require(row['scope'] == 'ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY'
                and row['source_commit'] == binding['code_commit']
                and row['binding_sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                'Backend evidence identity differs')
        require(row['requested_backend'] == 'FLASH_ATTN'
                and row['runner_backend_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionBackend'
                and row['runner_backend_name'] == 'flash-attn', 'Actual vLLM backend fell back or differs')
        require(row['model_dtype'] == 'torch.bfloat16' and row['cache_dtype'] == 'auto'
                and row['block_size'] == 16 and row['model_sliding_window'] is None,
                'Actual Flash dtype/cache/block/sliding configuration differs')
        layers = row['attention_layers']
        require(len(layers) == 36 and len({layer['name'] for layer in layers}) == 36,
                'Missing actual full-model attention implementation metadata')
        for layer in layers:
            require(layer['implementation_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionImpl'
                    and layer['head_size'] == 128 and layer['num_heads'] == 16
                    and layer['num_kv_heads'] == 2 and layer['sliding_window'] == [-1, -1],
                    'Actual attention implementation differs or silently falls back')
        require(row['observation_forward_calls'] == 0 and row['observation_tensor_value_reads'] == 0
                and row['actual_vllm_runner_positions_observed'] is False
                and row['probability_accepted'] is False, 'Backend metadata scope overstated')
    return dict(status='PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY',
                expected_backend='FLASH_ATTN', rank_count=8, layers_per_rank=36,
                actual_observations=rows, probability_accepted=False)


def observe_official_zero(trainer, run):
    from search_r1.baselines.fresh200 import record_initialization
    trainer.global_steps = 0
    trainer.actual_outer_updates = 0
    record_initialization(trainer)
    proof = trainer.fresh200_initial_identity
    require(proof['status'] == 'ACTUAL_FRESH_OFFICIAL_ZERO_STATE', 'Actual fresh initialization absent')
    workers = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(len(workers) == 8 and {r['rank'] for r in workers} == set(range(8))
            and all(r['scheduler_step'] == 0 and r['learning_rates'] == [0.0] for r in workers),
            'Actual zero initial scheduler/LR differs')
    # Empty inspected Adam dictionary slots are legal; actual initial identity
    # explicitly forbids populated step/moments. No load_training_state at0.
    value = dict(scope=SCOPE, **initial_fields(), actual_initialization=proof,
        workers=workers, phase0_state_reload_performed=False, formal_updates_contributed=0)
    json_save(run / 'actual-official-zero-initialization.json', value)
    return value


def create_trainer(config):
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from calibration_worker_v1 import CalibrationWorker
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    mapping = {Role.ActorRollout: 'global_pool', Role.RefPolicy: 'global_pool'}
    manager = ResourcePoolManager(resource_pool_spec={'global_pool': [8]}, mapping=mapping)
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer,
        role_worker_mapping={role: ray.remote(CalibrationWorker) for role in mapping},
        resource_pool_manager=manager, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=RewardManager(tokenizer, 0), val_reward_fn=RewardManager(tokenizer, 1))
    from metadata_colocation_v1 import scoped_metadata_colocation
    with scoped_metadata_colocation():
        trainer.init_workers()
    require(len(trainer.train_dataset) == 169028 and len(trainer.val_dataset) == 512,
            'Actual common dataset row count differs')
    require(not trainer.use_critic and trainer.use_reference_policy, 'Unexpected algorithm execution path')
    return trainer


def evaluate(trainer, phase, run):
    trainer.actor_rollout_wg.calibration_phase(phase + '512')
    full = trainer._validate()
    require(full.get('val/qa_count/all') == 512 and full.get('val/qa_count/nq') == 256
            and full.get('val/qa_count/hotpotqa') == 256, 'Incomplete same immutable512 validation')
    value = dict(scope=SCOPE, phase=phase, logical_step=trainer.global_steps,
                 retained_training_updates_contributed=0, full512=full, extra_stage64_evaluation=False)
    json_save(run / f'{phase}-evaluation.json', value)
    return value


def install_training_prefix_observer(run, outer):
    from search_r1.baselines.rollout import SearchRollout
    original = SearchRollout.advance
    turns = []

    def observed(engine, states, meta):
        before = [dict(source=s.source, step_id=len(s.anchors), generation_prefix=list(s.rolling),
                       canonical_context_prefix=list(s.prompt + s.response),
                       prior_action_count=len(s.action_prefix_records)) for s in states]
        value = original(engine, states, meta)
        for row, state in zip(before, states):
            require(len(state.action_prefix_records) == row['prior_action_count'] + 1, 'Missing new raw action provenance')
            item = copy.deepcopy(state.action_prefix_records[-1])
            require(item['generation_prefix'] == row['generation_prefix'], 'Actual source prefix changed')
            row['raw_actor_record'] = item
            row['retained_action_tokens'] = item['actor_action_tokens']
            row['canonical_retained_action_tokens'] = item['canonical_action_tokens']
        turns.append(before)
        json_save(run / ('training-prefix-turns.json' if outer == 1 else 'training-prefix-turns2.json'), turns)
        return value

    SearchRollout.advance = observed
    return original, turns


def driver_complete_pause_state(trainer, batch):
    """The real post-sampling state. This function neither fetches data nor restores RNG."""
    from search_r1.baselines.recovery import cpu_snapshot, rng_state
    iterator = trainer._calibration_training_iterator
    controller = trainer.baseline_controller
    start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(schema=1, pid=os.getpid(), start_ticks=start,
        global_steps=trainer.global_steps, actual_outer_updates=trainer.actual_outer_updates,
        rng=cpu_snapshot(rng_state()), sampler=cpu_snapshot(trainer.recovery_sampler.state_dict()),
        data_iterator=dict(base_seed=iterator._base_seed, num_yielded=iterator._num_yielded),
        kl_controller=cpu_snapshot(vars(trainer.kl_ctrl)),
        algorithm_state=dict(method='gigpo', stage=cpu_snapshot(controller.stage.state_dict()),
            parameters=OmegaConf.to_container(trainer.config.baselines.gigpo, resolve=True),
            groups='stateless_per_rollout_with_step_ids_and_advantage_broadcast',
            base_commit=controller.config.base_commit,
            algorithm_source_commit=controller.config.algorithm_source_commit),
        config=OmegaConf.to_container(trainer.config, resolve=True),
        common_data_identity=copy.deepcopy(trainer.instruct_data_identity),
        validation_trace_serial=getattr(trainer, 'validation_trace_serial', None),
        quality_health=copy.deepcopy(getattr(trainer, 'quality_health', None)),
        fixed_batch=dict(tensors=cpu_snapshot(dict(batch.batch.items())),
                         non_tensors=copy.deepcopy(batch.non_tensor_batch), metadata=copy.deepcopy(batch.meta_info)))


def capture_post_sampling_pause(trainer, batch, run, material):
    """Bind all live states after genuine sampling, not the pre-rollout checkpoint RNG."""
    from search_r1.baselines.recovery import atomic_torch_save, file_sha256
    worker_states = trainer.actor_rollout_wg.calibration_complete_pause_state()
    workers = validate_pause_workers([row['identity'] for row in worker_states])
    driver = driver_complete_pause_state(trainer, batch)
    require(driver['sampler']['cursor'] == 512 and driver['global_steps'] == 1
            and driver['actual_outer_updates'] == 0, 'Wrong actual post-sampling driver state; zero Adam')
    raw = atomic_torch_save(dict(workers=worker_states, driver=driver), run / 'post-sampling-pause-state.pt')
    summary = dict(scope=SCOPE, status='POST_SAMPLING_BEFORE_FIRST_ADAM', workers=workers,
        worker_pause_sha256=state_digest(workers), driver_complete_state_sha256=state_digest(driver),
        driver_pid=driver['pid'], driver_start_ticks=driver['start_ticks'], data_cursor=512,
        scheduler=0, actual_adam_writes=0, raw=raw)
    summary_path = run / 'post-sampling-pause-state.json'
    json_save(summary_path, summary)
    manifest = json.loads(material.read_text())
    manifest['post_sampling_pause'] = summary
    manifest['files'] += [raw, dict(path=str(summary_path), bytes=summary_path.stat().st_size,
                                  sha256=file_sha256(summary_path))]
    json_save(material, manifest)
    return workers, driver, summary


def wait_probability_acceptance(trainer, batch, run, binding, material, records, alignment, cross_checks):
    """Same workers, Adam and sampled batch stay live; next action is review or controlled failure."""
    from search_r1.baselines.recovery import file_sha256
    admit_alignment(alignment)
    # Repeated probes must already be neutral, before advertising a reviewable
    # pause. Structural identities never borrow a numerical tolerance.
    require(all(row['repeated_probability_probe_complete_state_unchanged']
                and row['repeated_probe_identity_before'] == row['repeated_probe_identity_after'] for row in records),
            'HF probability probe changed actual model/internal/Adam/scheduler/RNG state; zero Adam')
    workers, driver, pause = capture_post_sampling_pause(trainer, batch, run, material)
    identity = dict(code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                    job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
                    native_binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                    material_manifest_sha256=file_sha256(material),
                    raw_alignment_schema='RAW_ACTION_PREFIX_ALIGNMENT_V1',
                    actual_complete_hf_inputs_observed=True, actual_hf_roles=['old', 'current', 'reference'],
                    actual_vllm_runner_positions_observed=True,
                    post_sampling_worker_identities=workers,
                    post_sampling_worker_pause_sha256=pause['worker_pause_sha256'],
                    post_sampling_driver_complete_state_sha256=pause['driver_complete_state_sha256'],
                    post_sampling_pause_raw_sha256=pause['raw']['sha256'],
                    actual_normalizer_inputs_observed=True,
                    decisive_normalizer_evidence_sha256=file_sha256(run / 'actual-decisive-normalizer-evidence.json'))
    approval_path = run / 'probability-identity-approval.json'
    json_save(run / 'probability-identity-ready.json', dict(scope=SCOPE, status='WAITING_BEFORE_FIRST_ADAM',
                identity=identity, material_manifest=str(material), approval_path=str(approval_path),
                post_sampling_pause=pause, original_adam=0, scheduler=0, retained_training_updates_contributed=0,
                actual_adam_writes=0, data_cursor=512, sampled_trajectories=2560))
    deadline = time.monotonic() + binding['probability_review_wait_seconds']
    while not approval_path.is_file():
        require(time.monotonic() < deadline, 'Probability identity review wait expired before any Adam')
        time.sleep(1)
    approval = json.loads(approval_path.read_text())
    admitted = validate_probability_approval(approval, identity, binding)
    # SHA all original bytes once more after the human-independent review interval.
    require(file_sha256(material) == identity['material_manifest_sha256'], 'Probability review material changed')
    manifest = json.loads(material.read_text())
    for record in manifest['files']:
        path = Path(record['path'])
        require(path.stat().st_size == record['bytes'] and file_sha256(path) == record['sha256'],
                'Raw probability evidence changed while waiting')
    validate_probability_receipts(records, admitted)
    validate_action_probability_checks(cross_checks, admitted)
    after_worker_states = trainer.actor_rollout_wg.calibration_complete_pause_state()
    after_workers = [row['identity'] for row in after_worker_states]
    after_driver = driver_complete_pause_state(trainer, batch)
    unchanged = assert_complete_pause_unchanged(workers, after_workers, driver, after_driver)
    # Save both actual snapshots for independent inspection. No restore occurs.
    from search_r1.baselines.recovery import atomic_torch_save
    receipt = atomic_torch_save(dict(workers=after_worker_states, driver=after_driver),
                               run / 'post-approval-pause-state.pt')
    json_save(run / 'post-approval-pause-state.json', dict(unchanged, raw=receipt,
        worker_identities=after_workers, approval_sha256=file_sha256(approval_path)))
    trainer._calibration_approved_pause = (workers, driver)
    json_save(run / 'probability-identity-admitted.json', dict(status='PASS', scope=SCOPE,
                identity=identity, approval_sha256=file_sha256(approval_path),
                actor_recompute_abs_tolerance=admitted['actor_recompute_abs_tolerance'],
                actor_vllm_abs_tolerance=admitted['actor_vllm_abs_tolerance'], actual_adam_writes_at_admission=0,
                retained_training_updates_contributed=0, complete_pause_unchanged=unchanged))
    return admitted


def probability_and_alignment(trainer, run, turns, binding, batch):
    """Capture strict raw support, actual HF inputs and live state before any Adam."""
    from search_r1.baselines.recovery import file_sha256
    from search_r1.baselines.action_prefix_forward import prepare_action_prefix_micro
    from raw_hf_observation_v1 import unpack_complete_action_calls
    from raw_alignment_v1 import collect_raw_actions
    def checked_load(path, receipt):
        require(path.stat().st_size == receipt['bytes'] and file_sha256(path) == receipt['sha256'], 'Raw HF/sampler evidence changed')
        return torch.load(path, map_location='cpu', weights_only=False)
    raw_turns, paths = {}, [run / 'actual-vllm-backend-identity.json', run / 'actual-rollout-2560.pt', run / 'training-prefix-turns.json']
    for serial, rows in enumerate(turns):
        for rank in range(8):
            path = run / f'probabilities/train/call-{serial}-rank-{rank}.json'
            record = json.loads(path.read_text())
            require(record['source_commit'] == binding['code_commit'] and record['rank'] == rank and record['serial'] == serial,
                    'Perturn actual identity differs')
            raw_path = Path(record['raw']['path'])
            raw = checked_load(raw_path, record['raw'])
            require(raw['pid'] == record['pid'] and raw['source_commit'] == record['source_commit']
                    and raw['binding_sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                    'Raw sampling PT and receipt process/source identity differ')
            raw['probe_rng_unchanged'] = record['repeated_probability_probe_rng_unchanged']
            raw['probe_complete_unchanged'] = record['repeated_probability_probe_complete_state_unchanged']
            raw_turns[(serial, rank)] = raw
            paths += [path, raw_path]
    roles = {role: {} for role in ('old', 'current', 'reference')}
    for role in roles:
        for rank in range(8):
            path = run / f'probabilities/complete-actions/{role}-rank-{rank}.json'
            record = json.loads(path.read_text())
            require(record['role'] == role and record['rank'] == rank and record['code_commit'] == binding['code_commit']
                    and record['binding_sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'], 'Complete HF role/source binding differs')
            raw_path = Path(record['raw']['path'])
            raw = checked_load(raw_path, record['raw'])
            require(raw['role'] == role and raw['rank'] == rank and raw['code_commit'] == binding['code_commit']
                    and raw['binding_sha256'] == record['binding_sha256'] and raw['pid'] == record['pid'], 'Complete HF raw role/source identity differs')
            items = unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro,
                pad_id=binding['pad_token_id'], eos_id=binding['eos_token_id'])
            require(not (roles[role].keys() & items.keys()), 'Duplicate actual HF source/step mapping')
            roles[role].update(items)
            paths += [path, raw_path]
        if role != 'old':
            for rank in range(8):
                path = run / f'probabilities/complete-actions/{role}-neutral-rank-{rank}.json'
                record = json.loads(path.read_text())
                require(record['rank'] == rank and record['actual_adam_writes'] == 0, 'Wrong neutral observation phase')
                require(record['complete_state_unchanged'] if role == 'current' else record['fixed_reference_unchanged'],
                        'HF probability observation changed real actor/reference state; zero Adam')
                paths.append(path)
    material = run / 'probability-material-manifest.json'
    def save_material(extra=()):
        json_save(material, dict(scope=SCOPE, schema='RAW_ACTION_PROBABILITY_MATERIAL_V1',
            code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], restore_manifest_sha256=None, initialization=initial_fields(),
            actor_token_identity='sampled_action_prefix_v1', actual_hf_roles=['old', 'current', 'reference'],
            files=[dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path)) for path in paths + list(extra)]))
    save_material()
    try:
        alignment, records, checks, actions = collect_raw_actions(turns, raw_turns, roles, dict(batch.batch.items()),
            [int(value) for value in batch.non_tensor_batch['audit_trajectory_row']],
            question_ids=batch.non_tensor_batch['audit_question_index'].tolist(), pad=binding['pad_token_id'],
            eos=binding['eos_token_id'], code_commit=binding['code_commit'],
            binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
    except Exception as error:
        json_save(run / 'raw-identity-structural-rejection.json', dict(scope=SCOPE, status='FAILED_BEFORE_FIRST_ADAM',
            error=repr(error), actual_adam_writes=0, production_thresholds_selected=False))
        raise
    alignment_path = run / 'actual-probability-alignment.json'
    check_path = run / 'full-old-versus-sampled-prefix.json'
    json_save(alignment_path, dict(scope=SCOPE, alignment=alignment, probability_receipts=records,
                                  actions=actions, retained_training_updates_contributed=0))
    json_save(check_path, dict(scope=SCOPE, schema='RAW_COMPLETE_HF_ACTION_CHECKS_V1', checks=checks,
        complete_raw_2560=str(run / 'actual-rollout-2560.pt'), actual_hf_roles=['old', 'current', 'reference']))
    from decisive_normalizer_v1 import collect_decisive
    decisive = collect_decisive(turns, raw_turns, roles['old'])
    decisive_path = run / 'actual-decisive-normalizer-evidence.json'
    json_save(decisive_path, decisive)
    save_material([alignment_path, check_path, decisive_path])
    require(binding['calibration_phase'] == 'probability_probe_then_wait', 'No unreviewed raw Adam mode')
    wait_probability_acceptance(trainer, batch, run, binding, material, records, alignment, checks)
    return alignment


def one_outer(trainer, run, binding, training_iterator, outer):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_data_metrics
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_r1.baselines.rollout import SearchRollout, select_rows
    from search_r1.baselines.trajectory_records import RolloutTrace, record_scored_batch
    from search_r1.baselines.recovery import atomic_torch_save
    config = trainer.config
    trainer.global_steps = outer
    trainer.actor_rollout_wg.calibration_phase('train' if outer == 1 else 'train2')
    batch = DataProto.from_single_dict(next(training_iterator))
    require(len(batch) == 512 and trainer.recovery_sampler.cursor == outer * 512, 'Wrong actual fresh diagnostic data fetch')
    batch = batch.repeat(repeat_times=5, interleave=True)
    batch.non_tensor_batch['audit_trajectory_row'] = np.arange(2560, dtype=object)
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    manager = LLMGenerationManager(trainer.tokenizer, trainer.actor_rollout_wg,
        GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
                         max_response_length=500, max_obs_length=500, num_gpus=8,
                         no_think_rl=config.algorithm.no_think_rl, search_url=config.retriever.url,
                         topk=3, force_final_answer=config.force_final_answer))
    manager.rollout_trace = RolloutTrace(run / 'trajectories', outer, batch.non_tensor_batch['index'].tolist(), 5)
    metrics = {}
    original, turns = install_training_prefix_observer(run, outer)
    try:
        generated, sources = trainer.baseline_controller.generate(manager, gen_batch, batch, metrics)
    finally:
        SearchRollout.advance = original
    require(len(generated) == 2560, 'Incomplete actual512x5 rollout')
    batch = select_rows(batch, sources)
    for key in generated.batch.keys():
        generated.batch[key] = generated.batch[key].long()
    require(generated.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1', 'Not raw action source')
    generated.non_tensor_batch['audit_trajectory_row'] = np.asarray(sources, dtype=object)
    generated.non_tensor_batch['audit_question_index'] = batch.non_tensor_batch['index'].copy()
    generated.meta_info['raw_calibration_observation'] = outer == 1
    with torch.no_grad():
        generated = generated.union(trainer.actor_rollout_wg.compute_log_prob(generated))
    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
    batch = batch.union(generated)
    trainer._balance_batch(batch, metrics=metrics)
    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
    for key in batch.batch.keys():
        if key != 'old_log_probs':
            batch.batch[key] = batch.batch[key].long()
    batch = batch.union(trainer.ref_policy_wg.compute_ref_log_prob(batch))
    # A separate actual same-state complete-action repeat, not inferred current logits.
    if outer == 1:
        trainer.actor_rollout_wg.calibration_raw_current_log_prob(copy.deepcopy(batch))
    batch.batch['token_level_scores'] = trainer.reward_fn(batch)
    require(config.actor_rollout_ref.actor.use_kl_loss, 'Original fixed reference loss missing')
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
    batch = trainer.baseline_controller.compute_advantages(batch, metrics)
    batch, metrics = trainer._create_loss_mask(batch, metrics)
    batch.meta_info['baseline_outer_step'] = outer
    record_scored_batch(run / 'trajectories', outer, batch, trainer.tokenizer)
    atomic_torch_save(dict(tensors={k: v.detach().cpu() for k, v in batch.batch.items()},
                          non_tensors=batch.non_tensor_batch, metadata=batch.meta_info,
                          data_cursor=trainer.recovery_sampler.state_dict(), scope=SCOPE), run / ('actual-rollout-2560.pt' if outer == 1 else 'actual-rollout2-2560.pt'))
    alignment = None
    if outer == 1:
        try:
            alignment = probability_and_alignment(trainer, run, turns, binding, batch)
        except BaseException as error:
            # Complete original official shards/empty Adam/RNG are retained as
            # failure evidence, not represented as an accepted resume checkpoint.
            records = trainer.actor_rollout_wg.calibration_preserve_zero_adam()
            driver = atomic_torch_save(driver_complete_pause_state(trainer, batch),
                run / 'probability-rejection-state/driver.pt')
            json_save(run / 'probability-rejection-state/result.json', dict(status='REFUSED_BEFORE_FIRST_ADAM',
                actual_adam_writes=0, retained_training_updates_contributed=0,
                workers=records, driver=driver, error=repr(error), exact_resume_accepted=False))
            raise
    from search_r1.baselines.runtime_audit import driver_outer
    driver_outer(batch, outer, 'gigpo')
    if outer == 1:
        # Check again immediately before the first update RPC, including any
        # intervening audit operation and still-current fixed rollout bytes.
        before_workers, before_driver = trainer._calibration_approved_pause
        live = trainer.actor_rollout_wg.calibration_complete_pause_state()
        unchanged = assert_complete_pause_unchanged(before_workers, [row['identity'] for row in live],
                                                   before_driver, driver_complete_pause_state(trainer, batch))
        json_save(run / 'immediately-before-first-adam.json', unchanged)
    before_update = trainer.actor_rollout_wg.checkpoint_pause_identity()
    expected_lr = 1e-6 * (float(outer - 1) / 57.)
    require(all(r['scheduler_step'] == outer - 1 and r['learning_rates'] == [expected_lr]
                for r in before_update), 'Actual outer scheduler/LR boundary differs')
    try:
        result = trainer.actor_rollout_wg.update_actor(batch)
    except BaseException as error:
        from search_r1.baselines.partial_update import preserve_driver
        preserve_driver(trainer, batch, error)
        raise
    metrics.update(reduce_metrics(result.meta_info['metrics']))
    require(metrics['actor/optimizer_steps'] == 20 and metrics['actor/skipped_empty_minibatches'] == 0,
            'One calibration outer requires exactly twenty actual Adam writes')
    metrics.update(compute_data_metrics(batch, use_critic=False))
    trainer.actual_outer_updates = outer  # Only the isolated artifact, never retained goal progress.
    after_update = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(all(r['scheduler_step'] == outer for r in after_update), 'Scheduler must advance once per complete outer')
    json_save(run / f'calibration-update{outer}.json', dict(scope=SCOPE, calibration_actual_outer_updates=outer,
        retained_training_updates_contributed=0, logical_label=outer, optimizer_steps=20,
        data_cursor=outer*512, alignment=alignment, metrics=metrics,
        actual_lr_before_update=expected_lr, worker_before=before_update, worker_after=after_update))
    return metrics


def reload_saved_complete_state(trainer, run):
    """Real rank model/Adam40/scheduler2/RNG and driver/GiGPO/data reload."""
    from search_r1.baselines.instruct300 import check_manifest, equal_state
    from search_r1.baselines.recovery import file_sha256, rng_state, restore_rng
    manifest = trainer.instruct_manifest
    check_manifest(manifest)
    files = manifest['identity']['files']
    saved = [r['pause_identity'] for r in files if 'pause_identity' in r]
    before = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(before == saved and len(before) == 8, 'Actual saved eight-worker identity differs')
    shas = {str(r['pause_identity']['rank']): r['sha256'] for r in files if 'pause_identity' in r}
    loaded = trainer.actor_rollout_wg.load_training_state(str(run / 'checkpoints/actor/global_step_2'), shas, 2, 40)
    after = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(before == after and len(loaded) == 8 and all(r['logical_step'] == 2 for r in loaded),
            'Actual two-outer rank state reload differs')
    receipt = next(r for r in files if r['path'].endswith('/driver.pt'))
    require(file_sha256(receipt['path']) == receipt['sha256'], 'Driver state SHA differs')
    driver = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    require(driver['logical_step'] == driver['actual_new_updates'] == 2
            and driver['sampler']['cursor'] == 1024 and driver['source_revision'] == REVISION
            and driver['config'] == OmegaConf.to_container(trainer.config, resolve=True)
            and equal_state(driver['rng'], rng_state()), 'Saved actual driver identity differs')
    old_sampler = trainer.recovery_sampler.state_dict()
    trainer.recovery_sampler.load_state_dict(driver['sampler'])
    require(equal_state(old_sampler, trainer.recovery_sampler.state_dict()), 'Real data replay differs')
    before_ctrl = copy.deepcopy(vars(trainer.kl_ctrl))
    vars(trainer.kl_ctrl).clear(); vars(trainer.kl_ctrl).update(driver['kl_controller'])
    require(equal_state(before_ctrl, vars(trainer.kl_ctrl)), 'KL controller roundtrip differs')
    algorithm = driver['algorithm_state']
    algorithm_receipt = next(r for r in files if '/algorithm/' in r['path'])
    require(algorithm == json.loads(Path(algorithm_receipt['path']).read_text()), 'Saved GiGPO state differs')
    stage = trainer.baseline_controller.stage
    before_stage = copy.deepcopy(stage.state_dict())
    stage.load_state_dict(algorithm['bapo'])
    require(equal_state(before_stage, stage.state_dict()), 'Actual GiGPO controller state roundtrip differs')
    restore_rng(driver['rng'])
    require(equal_state(driver['rng'], rng_state()), 'Actual driver RNG reload differs')
    value = dict(scope=SCOPE, status='ACTUAL_TWO_OUTER_COMPLETE_STATE_RELOADED',
        manifest=manifest, workers=loaded, same_worker_identity=True, logical_step=2,
        actual_optimizer_steps_per_rank=40, scheduler_step=2, data_cursor=1024,
        phase0_state_reload_performed=False, retained_training_updates_contributed=0)
    json_save(run / 'actual-two-outer-state-roundtrip.json', value)
    return value


def main_task(config, binding):
    run = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    require(os.environ.get('VLLM_ATTENTION_BACKEND') == 'FLASH_ATTN', 'Ray backend environment differs')
    trainer = create_trainer(config)
    backend_rows = trainer.actor_rollout_wg.calibration_backend_identity()
    json_save(run / 'actual-vllm-backend-identity.json', dict(status='OBSERVED_UNREVIEWED',
        actual_observations=backend_rows, probability_accepted=False))
    json_save(run / 'actual-vllm-backend-identity.json', validate_actual_flash_backend(backend_rows, binding))
    from search_r1.baselines.holdout import _validate_prebuilt
    trainer.instruct_data_identity = _validate_prebuilt(config)
    initial = observe_official_zero(trainer, run)
    reference = trainer.ref_policy_wg.calibration_reference_identity()
    require(len(reference) == 8 and all(r['finite'] for r in reference), 'Nonfinite actual fixed reference')
    baseline = evaluate(trainer, 'baseline', run)
    baseline_after = trainer.actor_rollout_wg.checkpoint_pause_identity()
    for old, now in zip(initial['workers'], baseline_after):
        require(old['pid'] == now['pid'] and all(old[key] == now[key] for key in
            ('model_shards_sha256', 'local_shards_sha256', 'optimizer_tensors_sha256', 'buffers_sha256',
             'scheduler_step', 'learning_rates')), 'Baseline changed actor/model/Adam/scheduler')
    require(trainer.recovery_sampler.cursor == 0 and reference == trainer.ref_policy_wg.calibration_reference_identity(),
            'Baseline advanced training data or fixed reference')
    # Validation naturally consumes RNG. Preserve the actual trajectory, do not
    # reset official model/scheduler/RNG/data or reconstruct a historical restore.
    iterator = iter(trainer.train_dataloader)
    trainer._calibration_training_iterator = iterator
    metrics = [one_outer(trainer, run, binding, iterator, outer) for outer in (1, 2)]
    updated = evaluate(trainer, 'updated', run)
    require(reference == trainer.ref_policy_wg.calibration_reference_identity(), 'Fixed official reference changed')
    trainer._save_checkpoint()
    before_reload = trainer.actor_rollout_wg.checkpoint_pause_identity()
    for old, now in zip(initial['workers'], before_reload):
        require(old['pid'] == now['pid'] and old['model_shards_sha256'] != now['model_shards_sha256']
                and old['optimizer_tensors_sha256'] != now['optimizer_tensors_sha256']
                and now['scheduler_step'] == 2, 'Missing actual cumulative nonzero two-outer parameter/Adam changes')
    roundtrip = reload_saved_complete_state(trainer, run)
    rank_records = [list(map(json.loads, (run / f'audit/policy-safety/rank-{rank}.jsonl').read_text().splitlines()))
                    for rank in range(8)]
    from calibration_audit_v1 import validate_calibration_update, validate_calibration_numeric
    from safety_audit_v1 import validate_guard
    update_checks = []
    for rank in range(8):
        rows = [json.loads(line) for line in (run / f'audit/{rank}.jsonl').read_text().splitlines()]
        actors = [row for row in rows if row['kind'] == 'actor_outer']
        require(len(actors) == 2 and [row['outer_step'] for row in actors] == [1, 2], 'Unexpected diagnostic outers')
        update_checks.append([validate_calibration_update(row, os.environ['SEARCH_R1_BASELINE_COMMIT'],
            os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'], os.environ['SEARCH_R1_BASELINE_RUN_ID'], rank, outer)
            for outer, row in enumerate(actors, 1)])
        numeric = [json.loads(line) for line in (run / f'audit/numeric/micro-{rank}.jsonl').read_text().splitlines()]
        require(len(numeric) == 320, 'Missing actual320 micro loss evidence')
        for row in numeric: validate_calibration_numeric(row)
        for row in rank_records[rank]: validate_guard(row, 'calibration', os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
    json_save(run / 'calibration-actual-actor-audit.json', dict(scope=SCOPE, rank_count=8, checks=update_checks))
    observations = summarize_guards(rank_records)
    json_save(run / 'calibration-observations.json', observations)
    require(updated['full512']['val/f1/all'] >= baseline['full512']['val/f1/all'] / 2,
            'Two-outer same512 output quality collapsed; no production thresholds')
    return dict(status='CALIBRATION_OBSERVED_UNREVIEWED', scope=SCOPE, **initial_fields(),
        retained_training_updates_contributed=0, calibration_actual_outer_updates=2,
        calibration_actual_outer_budget=2, actual_optimizer_steps_per_rank=40,
        actual_gradient_micro_per_rank=320, optimizer_steps_before=0, optimizer_steps_after=40,
        scheduler_after=2, data_cursor=1024, baseline=baseline, updated=updated,
        metrics=metrics, manifest=trainer.instruct_manifest, full_state_reloaded=True, roundtrip=roundtrip,
        production_thresholds_not_selected=True, target_old_policy_kl=None,
        negative_tail_limit=None, candidate_old_policy_kl=.01, observations=observations)


if __name__ == '__main__':
    config = OmegaConf.load(sys.argv[1])
    binding = json.loads(Path(sys.argv[2]).read_text())
    env = {key: value for key, value in os.environ.items() if key.startswith(('SEARCH_R1_BASELINE_', 'GIGPO_POLICY_'))}
    env.update(PYTHONPATH=os.environ['PYTHONPATH'], CPATH=os.environ['CPATH'],
               SEARCH_R1_PYTHON_HEADER_CPATH=os.environ['SEARCH_R1_PYTHON_HEADER_CPATH'],
               VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    require(os.environ.get('VLLM_ATTENTION_BACKEND') == 'FLASH_ATTN', 'Driver backend environment differs')
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': env})
    try:
        value = ray.get(ray.remote(num_cpus=1)(main_task).options(runtime_env={'env_vars': env}).remote(config, binding))
        json_save(Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'driver-result.json', value)
        print(json.dumps(value, allow_nan=False), flush=True)
    finally:
        ray.shutdown()
