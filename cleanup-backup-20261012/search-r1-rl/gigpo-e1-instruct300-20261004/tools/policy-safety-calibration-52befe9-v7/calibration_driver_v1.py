"""One real Search-E1 outer from complete healthy25, isolated from the retained training chain.

Frozen trainer/controller/actor APIs perform generation, reward, GiGPO advantage
and ten Adam writes. This driver only supplies explicit calibration scope,
records probability identity and refuses before Adam if sampling prefixes differ.
It never calls trainer.fit(), writes a first-five approval or selects thresholds.
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
from calibration_contract_v1 import (SCOPE, CHECKPOINT, MANIFEST, DATA, require,
    validate_alignment, admit_alignment, validate_probability_receipts, summarize_guards,
    validate_probability_approval, validate_action_probability_checks)
from calibration_state_v1 import state_digest, validate_pause_workers, assert_complete_pause_unchanged


def json_save(path, value):
    from search_r1.baselines.recovery import atomic_json_save
    atomic_json_save(value, path)


def restore25(trainer):
    """Exact saved source state; only explicitly approved mini/protection controls differ."""
    from search_r1.baselines import recovery
    from search_r1.baselines.instruct300 import check_manifest, equal_state, REVISION
    from search_r1.baselines.quality_resume import migration_diff, selected_origin
    path = Path(CHECKPOINT) / 'training_state/manifest.json'
    require(recovery.file_sha256(path) == MANIFEST, 'Healthy25 manifest changed')
    identity = json.loads(path.read_text())
    check_manifest(dict(path=str(path), sha256=MANIFEST, identity=identity))
    origin = selected_origin(trainer.config)
    require(origin['checkpoint'] == CHECKPOINT and origin['manifest'] == MANIFEST and origin['step'] == 25,
            'Not independently selected healthy25')
    require(identity['logical_step'] == 25 and identity['job_id'] == '3185'
            and identity['recovery_mode'] == 'fresh_official_instruct', 'Source history changed')
    files = identity['files']
    record = next(r for r in files if r['path'].endswith('/driver.pt'))
    driver = torch.load(record['path'], map_location='cpu', weights_only=False)
    require(driver['schema'] == 1 and driver['logical_step'] == 25 and driver['actual_new_updates'] == 25
            and driver['source_revision'] == REVISION and driver['source'] == trainer.config.actor_rollout_ref.model.path
            and driver['sampler']['cursor'] == 12800 and driver['sampler']['epoch'] == 0
            and driver['sampler']['size'] == 169028, 'Incomplete source driver state')
    # Reuse the frozen migration validator on all existing controls. Remove only
    # the new protection dictionary to avoid pretending calibration is production.
    current = OmegaConf.to_container(trainer.config, resolve=True)
    protection = current['actor_rollout_ref']['actor'].pop('policy_safety')
    changes = migration_diff(driver['config'], current)
    require(protection['mode'] == 'calibration' and protection['enabled'] is True
            and protection['dual_clip'] == 3. and protection['kl_log_bound'] == 30.
            and protection['monitor_first_global_rows'] == 256
            and protection['target_old_policy_kl'] is None and protection['negative_tail_limit'] is None,
            'Calibration introduced additional mathematical/configuration changes')
    algorithm = driver['algorithm_state']
    algorithm_record = next(r for r in files if '/algorithm/' in r['path'])
    require(algorithm == json.loads(Path(algorithm_record['path']).read_text()), 'Algorithm state changed')
    controller = trainer.baseline_controller
    require(algorithm['method'] == 'gigpo' and algorithm['global_step'] == 25
            and algorithm['resume_supported'] and algorithm['gigpo_parameters'] == OmegaConf.to_container(trainer.config.baselines.gigpo, resolve=True)
            and algorithm['loss_reduction'] == 'sequence_mean'
            and algorithm['groups'] == 'stateless_per_rollout_with_step_ids_and_advantage_broadcast'
            and algorithm['base_commit'] == controller.config.base_commit
            and algorithm['algorithm_source_commit'] == controller.config.algorithm_source_commit
            and algorithm['teacher_path'] == controller.config.distillation.teacher_path, 'GiGPO definition changed')
    rank_records = [r for r in files if 'pause_identity' in r]
    shas = {str(r['pause_identity']['rank']): r['sha256'] for r in rank_records}
    loaded = trainer.actor_rollout_wg.load_training_state(CHECKPOINT, shas, 25, 500)
    now = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(len(loaded) == len(now) == 8 and {r['rank'] for r in now} == set(range(8))
            and len({r['pid'] for r in now}) == 8, 'Missing actual eight-rank restore')
    saved = {r['pause_identity']['rank']: r['pause_identity'] for r in rank_records}
    for row in now:
        require({k: v for k, v in row.items() if k not in ('pid', 'start_ticks')} ==
                {k: v for k, v in saved[row['rank']].items() if k not in ('pid', 'start_ticks')},
                'Actual model/buffer/Adam/scheduler/RNG restore differs')
    trainer.recovery_sampler.load_state_dict(driver['sampler'])
    require(equal_state(trainer.recovery_sampler.state_dict(), driver['sampler']), 'Data cursor/permutation differs')
    require(vars(trainer.kl_ctrl).keys() == driver['kl_controller'].keys(), 'KL controller changed')
    vars(trainer.kl_ctrl).update(driver['kl_controller'])
    controller.stage.load_state_dict(algorithm['bapo'])
    require(controller.stage.state_dict() == algorithm['bapo'], 'Controller state differs')
    recovery.restore_rng(driver['rng'])
    require(equal_state(recovery.rng_state(), driver['rng']), 'Driver RNG differs')
    trainer.global_steps = 25
    trainer.actual_outer_updates = 0
    trainer.instruct_restore_manifest = dict(path=str(path), sha256=MANIFEST, identity=identity)
    return dict(scope=SCOPE, restore_checkpoint=CHECKPOINT, restore_manifest_sha256=MANIFEST,
                logical_step=25, original_adam=500, scheduler=25, cursor=12800, workers=now,
                authorized_config_diff=changes, complete_saved_state_equal=True)


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
    from search_r1.baselines.instruct300 import validate_stage64
    trainer.actor_rollout_wg.calibration_phase(phase + '512')
    full = trainer._validate()
    require(full.get('val/qa_count/all') == 512 and full.get('val/qa_count/nq') == 256
            and full.get('val/qa_count/hotpotqa') == 256, 'Incomplete fixed512 validation')
    trainer.actor_rollout_wg.calibration_phase(phase + '64')
    stage = validate_stage64(trainer)
    require(stage.get('val/qa_count/all') == 64, 'Incomplete common64 validation')
    value = dict(scope=SCOPE, phase=phase, logical_step=trainer.global_steps,
                 retained_training_updates_contributed=0, full512=full, stage64=stage)
    json_save(run / f'{phase}-evaluation.json', value)
    return value


def install_training_prefix_observer(run):
    from search_r1.baselines.rollout import SearchRollout
    original = SearchRollout.advance
    turns = []

    def observed(engine, states, meta):
        before = [dict(source=s.source, step_id=len(s.anchors), generation_prefix=list(s.rolling),
                       scoring_prefix=list(s.prompt + s.response)) for s in states]
        value = original(engine, states, meta)
        for row, state in zip(before, states):
            row['retained_action_tokens'] = [token for token, action in zip(state.response, state.steps)
                                             if action == row['step_id']]
        turns.append(before)
        json_save(run / 'training-prefix-turns.json', turns)
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
    require(driver['sampler']['cursor'] == 13312 and driver['global_steps'] == 26
            and driver['actual_outer_updates'] == 0, 'Wrong actual post-sampling driver state; zero Adam')
    raw = atomic_torch_save(dict(workers=worker_states, driver=driver), run / 'post-sampling-pause-state.pt')
    summary = dict(scope=SCOPE, status='POST_SAMPLING_BEFORE_FIRST_ADAM', workers=workers,
        worker_pause_sha256=state_digest(workers), driver_complete_state_sha256=state_digest(driver),
        driver_pid=driver['pid'], driver_start_ticks=driver['start_ticks'], data_cursor=13312,
        scheduler=25, actual_adam_writes=0, raw=raw)
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
                    post_sampling_worker_identities=workers,
                    post_sampling_worker_pause_sha256=pause['worker_pause_sha256'],
                    post_sampling_driver_complete_state_sha256=pause['driver_complete_state_sha256'],
                    post_sampling_pause_raw_sha256=pause['raw']['sha256'])
    approval_path = run / 'probability-identity-approval.json'
    json_save(run / 'probability-identity-ready.json', dict(scope=SCOPE, status='WAITING_BEFORE_FIRST_ADAM',
                identity=identity, material_manifest=str(material), approval_path=str(approval_path),
                post_sampling_pause=pause, original_adam=500, scheduler=25, retained_training_updates_contributed=0,
                actual_adam_writes=0, data_cursor=13312, sampled_trajectories=2560))
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
    from search_r1.baselines.recovery import file_sha256
    records = []
    per_turn_old = {}
    per_turn_metadata = {}
    by_source = {i: dict(source=i, actions=[]) for i in range(2560)}
    pad = binding['pad_token_id']
    for serial, rows in enumerate(turns):
        tensors = []
        for rank in range(8):
            path = run / f'probabilities/train/call-{serial}-rank-{rank}.json'
            record = json.loads(path.read_text())
            require(record['rank'] == rank and record['serial'] == serial, 'Wrong sampled turn rank identity')
            require(file_sha256(record['raw']['path']) == record['raw']['sha256'], 'Raw sampling probabilities changed')
            projected = copy.deepcopy(record)
            projected['raw_generated_token_diagnostics'] = {key: record[key] for key in
                ('valid_tokens', 'nonfinite_effective_count', 'actor_old_max_abs_delta', 'actor_vllm_max_abs_delta')}
            projected.update(valid_tokens=0, nonfinite_effective_count=0,
                             actor_old_max_abs_delta=0., actor_vllm_max_abs_delta=0.,
                             effective_probability_scope='RETAINED_ACTOR_ACTION_PREFIX_EXCLUDES_DISCARDED_AND_PADDED_ROWS')
            records.append(projected)
            tensors.append(torch.load(record['raw']['path'], map_location='cpu', weights_only=False))
        maximum_width = max(r['sampled_ids'].shape[1] for r in tensors)
        all_ids = torch.cat([torch.nn.functional.pad(r['sampled_ids'],
                             (0, maximum_width - r['sampled_ids'].shape[1]), value=pad) for r in tensors])
        all_prompts = torch.cat([r['tensors']['prompts'] for r in tensors])
        all_old = torch.cat([r['tensors']['old_log_probs'] for r in tensors])
        all_positions = torch.cat([r['tensors']['position_ids'] for r in tensors])
        all_attention = torch.cat([r['tensors']['attention_mask'] for r in tensors])
        all_sampled_logp = torch.cat([torch.nn.functional.pad(r['sampled_log_probs'],
                                 (0, maximum_width - r['sampled_log_probs'].shape[1])) for r in tensors])
        all_repeated = torch.cat([torch.nn.functional.pad(r['actor_repeated_log_probs'],
                             (0, all_old.shape[1] - r['actor_repeated_log_probs'].shape[1])) for r in tensors])
        engine_prompts = [row for raw in tensors for row in raw['actual_engine_prompt_token_ids']]
        require(len(all_ids) >= len(rows) and len(all_ids) - len(rows) < 8, 'Unexpected generation padding coverage')
        for index, row in enumerate(rows):
            require(all_prompts[index][all_prompts[index].ne(pad)].tolist() == row['generation_prefix'],
                    'Worker raw prompt differs from driver action context')
            row['sampled_tokens'] = all_ids[index].tolist()  # Never delete a middle sampled token.
            row['engine_prompt_token_ids'] = engine_prompts[index]
            count = len(row['retained_action_tokens'])
            # Generated tails beyond the retained action and duplicated GPU padding
            # are diagnostic only; they have no actor loss mass and cannot set admission.
            rank = index // tensors[0]['tensors']['responses'].shape[0]
            projected = records[serial * 8 + rank]
            old = all_old[index, :count].double()
            repeated = all_repeated[index, :count].double()
            sampling = all_sampled_logp[index, :count].double()
            finite = torch.isfinite(old) & torch.isfinite(repeated) & torch.isfinite(sampling)
            projected['valid_tokens'] += count
            projected['nonfinite_effective_count'] += int((~finite).sum())
            if bool(finite.any()):
                projected['actor_old_max_abs_delta'] = max(projected['actor_old_max_abs_delta'],
                    float((old - repeated)[finite].abs().max()))
                projected['actor_vllm_max_abs_delta'] = max(projected['actor_vllm_max_abs_delta'],
                    float((old - sampling)[finite].abs().max()))
            by_source[row['source']]['actions'].append(row)
            per_turn_old[(row['source'], row['step_id'])] = all_old[index]
            per_turn_metadata[(row['source'], row['step_id'])] = dict(
                positions=all_positions[index], attention=all_attention[index],
                prompt_width=all_prompts.shape[1], sampled_log_probs=all_sampled_logp[index])
    alignment = validate_alignment(list(by_source.values()))
    json_save(run / 'actual-probability-alignment.json', dict(scope=SCOPE, alignment=alignment,
              probability_receipts=records, actions=list(by_source.values()), retained_training_updates_contributed=0))
    # Check that the scored batch's effective action tokens exactly match observed ones.
    by_row = {int(row): i for i, row in enumerate(batch.non_tensor_batch['audit_trajectory_row'])}
    probability_cross_checks = []
    for source, value in by_source.items():
        row = by_row[source]
        for action in value['actions']:
            selected = batch.batch['step_ids'][row].eq(action['step_id']) & batch.batch['loss_mask'][row].bool()
            require(batch.batch['responses'][row][selected].tolist() == action['retained_action_tokens'],
                    'GiGPO effective action differs from sampling identity')
            full_old = batch.batch['old_log_probs'][row][selected].double()
            sampled_prefix_old = per_turn_old[(source, action['step_id'])][:len(full_old)].double()
            require(torch.isfinite(full_old).all() and torch.isfinite(sampled_prefix_old).all(),
                    'Nonfinite full-trajectory sampling identity')
            delta = float((full_old - sampled_prefix_old).abs().max()) if len(full_old) else 0.
            meta = per_turn_metadata[(source, action['step_id'])]
            action_size = len(full_old)
            actual_prompt_mask = meta['attention'][:meta['prompt_width']].bool()
            actual_prompt_positions = meta['positions'][:meta['prompt_width']][actual_prompt_mask].tolist()
            sample_positions = meta['positions'][meta['prompt_width']:][:action_size].tolist()
            sample_mask = meta['attention'][meta['prompt_width']:][:action_size].tolist()
            score_positions = batch.batch['position_ids'][row, -batch.batch['responses'].shape[1]:][selected].tolist()
            score_mask = batch.batch['loss_mask'][row][selected].tolist()
            action['sampling_action_position_ids'] = sample_positions
            action['scoring_action_position_ids'] = score_positions
            action['sampling_action_mask'] = sample_mask
            action['scoring_action_mask'] = score_mask
            natural_positions = list(range(len(action['generation_prefix'])))
            position_identity = (actual_prompt_positions == natural_positions and sample_positions == score_positions
                and sample_positions == list(range(len(natural_positions), len(natural_positions) + action_size)))
            mask_identity = sample_mask == score_mask == [1] * action_size
            sampled_logp = meta['sampled_log_probs'][:action_size].double()
            probability_cross_checks.append(dict(source=source, step_id=action['step_id'],
                effective_tokens=len(full_old), full_trajectory_vs_sample_prefix_old_max_abs=delta,
                full_vs_prefix_old_joint_sum_delta=float((full_old-sampled_prefix_old).sum()),
                full_vs_vllm_joint_sum_delta=float((full_old-sampled_logp).sum()),
                prefix_old_vs_vllm_joint_sum_delta=float((sampled_prefix_old-sampled_logp).sum()),
                same_position_ids=position_identity, same_effective_mask=mask_identity,
                engine_prefix_matches_generation=action['engine_prompt_token_ids']==action['generation_prefix']))
    alignment.update(position_ids_mismatch_count=sum(not r['same_position_ids'] for r in probability_cross_checks),
                     effective_mask_mismatch_count=sum(not r['same_effective_mask'] for r in probability_cross_checks),
                     engine_prefix_mismatch_count=sum(not r['engine_prefix_matches_generation'] for r in probability_cross_checks))
    json_save(run / 'actual-probability-alignment.json', dict(scope=SCOPE, alignment=alignment,
              probability_receipts=records, actions=list(by_source.values()), retained_training_updates_contributed=0))
    json_save(run / 'full-old-versus-sampled-prefix.json', dict(scope=SCOPE, checks=probability_cross_checks,
                complete_raw_2560=str(run / 'actual-rollout-2560.pt'), raw_turn_directory=str(run / 'probabilities/train')))
    evidence_paths = [run / 'actual-rollout-2560.pt', run / 'training-prefix-turns.json',
                      run / 'actual-probability-alignment.json', run / 'full-old-versus-sampled-prefix.json']
    evidence_paths += [Path(r['raw']['path']) for r in records]
    evidence_paths += sorted((run / 'probabilities/train').glob('call-*-rank-*.json'))
    material = run / 'probability-material-manifest.json'
    json_save(material, dict(scope=SCOPE, code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
              restore_manifest_sha256=MANIFEST, files=[dict(path=str(path), bytes=path.stat().st_size,
              sha256=file_sha256(path)) for path in evidence_paths]))
    if binding['calibration_phase'] == 'probability_probe_only':
        return alignment
    if binding['calibration_phase'] == 'probability_probe_then_wait':
        wait_probability_acceptance(trainer, batch, run, binding, material, records, alignment, probability_cross_checks)
        return alignment
    # Preserve all diagnostics first. Do not hide context/retokenization mismatches.
    admit_alignment(alignment)
    validate_probability_receipts(records, binding)
    validate_action_probability_checks(probability_cross_checks, binding)
    return alignment


def one_outer(trainer, run, binding, training_iterator):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_data_metrics
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_r1.baselines.rollout import SearchRollout, select_rows
    from search_r1.baselines.trajectory_records import RolloutTrace, record_scored_batch
    from search_r1.baselines.recovery import atomic_torch_save
    config = trainer.config
    trainer.global_steps = 26
    trainer.actor_rollout_wg.calibration_phase('train')
    batch = DataProto.from_single_dict(next(training_iterator))
    require(len(batch) == 512 and trainer.recovery_sampler.cursor == 13312, 'Wrong actual resumed data fetch')
    batch = batch.repeat(repeat_times=5, interleave=True)
    batch.non_tensor_batch['audit_trajectory_row'] = np.arange(2560, dtype=np.int64)
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    manager = LLMGenerationManager(trainer.tokenizer, trainer.actor_rollout_wg,
        GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
                         max_response_length=500, max_obs_length=500, num_gpus=8,
                         no_think_rl=config.algorithm.no_think_rl, search_url=config.retriever.url,
                         topk=3, force_final_answer=config.force_final_answer))
    manager.rollout_trace = RolloutTrace(run / 'trajectories', 26, batch.non_tensor_batch['index'].tolist(), 5)
    metrics = {}
    original, turns = install_training_prefix_observer(run)
    try:
        generated, sources = trainer.baseline_controller.generate(manager, gen_batch, batch, metrics)
    finally:
        SearchRollout.advance = original
    require(len(generated) == 2560, 'Incomplete actual512x5 rollout')
    batch = select_rows(batch, sources)
    for key in generated.batch.keys():
        generated.batch[key] = generated.batch[key].long()
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
    batch.batch['token_level_scores'] = trainer.reward_fn(batch)
    require(config.actor_rollout_ref.actor.use_kl_loss, 'Original fixed reference loss missing')
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
    batch = trainer.baseline_controller.compute_advantages(batch, metrics)
    batch, metrics = trainer._create_loss_mask(batch, metrics)
    batch.meta_info['baseline_outer_step'] = 26
    record_scored_batch(run / 'trajectories', 26, batch, trainer.tokenizer)
    atomic_torch_save(dict(tensors={k: v.detach().cpu() for k, v in batch.batch.items()},
                          non_tensors=batch.non_tensor_batch, metadata=batch.meta_info,
                          data_cursor=trainer.recovery_sampler.state_dict(), scope=SCOPE), run / 'actual-rollout-2560.pt')
    alignment = probability_and_alignment(trainer, run, turns, binding, batch)
    if binding['calibration_phase'] == 'probability_probe_only':
        json_save(run / 'probability-probe-only.json', dict(scope=SCOPE,
            status='PROBABILITY_OBSERVATIONS_UNREVIEWED_NOT_CALIBRATED', alignment=alignment,
            retained_training_updates_contributed=0, actual_adam_writes=0,
            consumed_diagnostic_questions=512, sampled_trajectories=2560, data_cursor=13312,
            metrics=metrics, target_old_policy_kl=None, negative_tail_limit=None))
        return None
    from search_r1.baselines.runtime_audit import driver_outer
    driver_outer(batch, 26, 'gigpo')
    if binding['calibration_phase'] == 'probability_probe_then_wait':
        # Check again immediately before the first update RPC, including any
        # intervening audit operation and still-current fixed rollout bytes.
        before_workers, before_driver = trainer._calibration_approved_pause
        live = trainer.actor_rollout_wg.calibration_complete_pause_state()
        unchanged = assert_complete_pause_unchanged(before_workers, [row['identity'] for row in live],
                                                   before_driver, driver_complete_pause_state(trainer, batch))
        json_save(run / 'immediately-before-first-adam.json', unchanged)
    result = trainer.actor_rollout_wg.update_actor(batch)
    metrics.update(reduce_metrics(result.meta_info['metrics']))
    require(metrics['actor/optimizer_steps'] == 10 and metrics['actor/skipped_empty_minibatches'] == 0,
            'One calibration outer requires exactly ten actual Adam writes')
    metrics.update(compute_data_metrics(batch, use_critic=False))
    trainer.actual_outer_updates = 1  # Only the isolated artifact, never retained goal progress.
    json_save(run / 'calibration-update.json', dict(scope=SCOPE, calibration_actual_outer_updates=1,
        retained_training_updates_contributed=0, logical_label=26, optimizer_steps=10,
        data_cursor=13312, alignment=alignment, metrics=metrics))
    return metrics


def main_task(config, binding):
    run = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    trainer = create_trainer(config)
    from search_r1.baselines.holdout import _validate_prebuilt
    trainer.instruct_data_identity = _validate_prebuilt(config)
    # Exactly the frozen fit boundary: iterator creation consumes Torch base_seed;
    # create before restoration, and never fetch until after the second full25 restore.
    training_iterator = iter(trainer.train_dataloader)
    trainer._calibration_training_iterator = training_iterator
    first = restore25(trainer)
    json_save(run / 'restore25-before-baseline.json', first)
    reference = trainer.ref_policy_wg.calibration_reference_identity()
    require(len(reference) == 8 and all(r['finite'] for r in reference), 'Nonfinite fixed reference')
    baseline = evaluate(trainer, 'baseline', run)
    require(baseline['full512']['val/f1/all'] >= binding['healthy_baseline_f1_floor'], 'Selected checkpoint no longer healthy')
    second = restore25(trainer)  # Reset model/Adam/scheduler/RNG/data after validation.
    require([r['pid'] for r in first['workers']] == [r['pid'] for r in second['workers']], 'Restore replaced workers')
    json_save(run / 'restore25-before-rollout.json', second)
    metrics = one_outer(trainer, run, binding, training_iterator)
    if binding['calibration_phase'] == 'probability_probe_only':
        after = trainer.actor_rollout_wg.checkpoint_pause_identity()
        require(all(r['scheduler_step'] == 25 for r in after), 'Diagnostic probe advanced scheduler')
        # Sampling legitimately consumes RNG. Model/buffer/Adam must remain unchanged.
        for before, now in zip(second['workers'], after):
            require(all(before[key] == now[key] for key in ('model_shards_sha256', 'optimizer_tensors_sha256',
                        'local_shards_sha256', 'buffers_sha256')), 'Probability probe modified model/Adam/buffers')
        require(reference == trainer.ref_policy_wg.calibration_reference_identity(), 'Fixed reference changed')
        return dict(status='PROBABILITY_OBSERVATIONS_UNREVIEWED_NOT_CALIBRATED', scope=SCOPE,
            retained_training_updates_contributed=0, calibration_actual_outer_updates=0,
            restored_logical=25, optimizer_steps_before=500, optimizer_steps_after=500,
            scheduler_after=25, consumed_diagnostic_data_cursor=13312, actual_adam_writes=0,
            baseline=baseline, target_old_policy_kl=None, negative_tail_limit=None,
            production_thresholds_not_selected=True, model_adam_reference_unchanged=True)
    updated = evaluate(trainer, 'updated', run)
    reference_after = trainer.ref_policy_wg.calibration_reference_identity()
    require(reference == reference_after, 'Fixed official reference changed')
    trainer.quality_health = dict(calibration_scope=SCOPE, baseline512=baseline['full512'], updated512=updated['full512'])
    trainer._save_checkpoint()
    manifest = trainer.instruct_manifest
    from search_r1.baselines.instruct300 import check_manifest
    check_manifest(manifest)
    shas = {str(r['pause_identity']['rank']): r['sha256'] for r in manifest['identity']['files'] if 'pause_identity' in r}
    before = trainer.actor_rollout_wg.checkpoint_pause_identity()
    for initial, current in zip(second['workers'], before):
        require(initial['pid'] == current['pid'] and initial['model_shards_sha256'] != current['model_shards_sha256']
                and initial['optimizer_tensors_sha256'] != current['optimizer_tensors_sha256']
                and current['scheduler_step'] == 26, 'Missing same-worker finite nonzero model/Adam continuation')
    trainer.actor_rollout_wg.load_training_state(str(run / 'checkpoints/actor/global_step_26'), shas, 26, 510)
    after = trainer.actor_rollout_wg.checkpoint_pause_identity()
    require(before == after, 'Actual isolated saved-state reload changed tensors/Adam/scheduler/RNG')
    rank_records = [list(map(json.loads, (run / f'audit/policy-safety/rank-{rank}.jsonl').read_text().splitlines())) for rank in range(8)]
    from calibration_audit_v1 import validate_calibration_update, validate_calibration_numeric
    from safety_audit_v1 import validate_guard
    update_checks = []
    for rank in range(8):
        rows = [json.loads(line) for line in (run / f'audit/{rank}.jsonl').read_text().splitlines()]
        actors = [row for row in rows if row['kind'] == 'actor_outer']
        require(len(actors) == 1, 'Calibration cannot contain unreported extra actor outer')
        update_checks.append(validate_calibration_update(actors[0], os.environ['SEARCH_R1_BASELINE_COMMIT'],
                             os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'], os.environ['SEARCH_R1_BASELINE_RUN_ID'], rank))
        numeric = [json.loads(line) for line in (run / f'audit/numeric/micro-{rank}.jsonl').read_text().splitlines()]
        require(len(numeric) == 160, 'Missing actual gradient-micro loss evidence')
        for row in numeric:
            validate_calibration_numeric(row)
        for row in rank_records[rank]:
            validate_guard(row, 'calibration', os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
    json_save(run / 'calibration-actual-actor-audit.json', dict(scope=SCOPE, rank_count=8, checks=update_checks))
    observations = summarize_guards(rank_records)
    json_save(run / 'calibration-observations.json', observations)
    require(updated['full512']['val/f1/all'] >= baseline['full512']['val/f1/all'] / 2,
            'One-outer output quality collapsed; do not derive production thresholds')
    require(updated['stage64']['val/f1/all'] >= baseline['stage64']['val/f1/all'] / 2
            and metrics['critic/rewards/mean'] >= .01, 'Stage/reward quality collapse; preserve and diagnose')
    return dict(status='CALIBRATION_OBSERVED_UNREVIEWED', scope=SCOPE,
                retained_training_updates_contributed=0, calibration_actual_outer_updates=1,
                restored_logical=25, isolated_logical=26, optimizer_steps_before=500,
                optimizer_steps_after=510, scheduler_after=26, data_cursor=13312,
                baseline=baseline, updated=updated, manifest=manifest, full_state_reloaded=True,
                production_thresholds_not_selected=True, target_old_policy_kl=None,
                negative_tail_limit=None, observations=observations)


if __name__ == '__main__':
    config = OmegaConf.load(sys.argv[1])
    binding = json.loads(Path(sys.argv[2]).read_text())
    env = {key: value for key, value in os.environ.items() if key.startswith(('SEARCH_R1_BASELINE_', 'GIGPO_POLICY_'))}
    env.update(PYTHONPATH=os.environ['PYTHONPATH'], CPATH=os.environ['CPATH'],
               SEARCH_R1_PYTHON_HEADER_CPATH=os.environ['SEARCH_R1_PYTHON_HEADER_CPATH'])
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': env})
    try:
        value = ray.get(ray.remote(num_cpus=1)(main_task).options(runtime_env={'env_vars': env}).remote(config, binding))
        json_save(Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'driver-result.json', value)
        print(json.dumps(value, allow_nan=False), flush=True)
    finally:
        ray.shutdown()
