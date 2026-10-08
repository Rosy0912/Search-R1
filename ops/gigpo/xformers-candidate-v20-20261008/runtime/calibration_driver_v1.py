"""One official-zero trainer and live iterator, retained first5 through completed200.

The actual trainer APIs keep GiGPO math, raw action mapping and state groups.
Required training forwards retain raw sampled action structure; diagnostics are background only.
No fit(), historical restore, synthetic native or threshold invention occurs.
"""
import copy
import json
import os
if (os.environ.get('CUDA_LAUNCH_BLOCKING') != '0'
        or os.environ.get('VLLM_ATTENTION_BACKEND') != 'XFORMERS'):
    raise RuntimeError('XFORMERS/asynchronous CUDA policy must be inherited before driver imports')
import sys
import time
from pathlib import Path
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from calibration_contract_v1 import (SCOPE, LIVE_SCOPE, DATA, REVISION, initial_fields, require,
    validate_alignment, admit_alignment, validate_probability_receipts, summarize_guards,
    validate_probability_approval, validate_action_probability_checks)
from calibration_state_v1 import (state_digest, validate_pause_workers, assert_complete_pause_unchanged,
                                  validate_bf16_accumulation_identity)
from continuation_timing_v10 import observe_time, publish_timing


def json_save(path, value):
    from search_r1.baselines.recovery import atomic_json_save
    atomic_json_save(value, path)


def sampler_json_metadata(sampler):
    """JSON observation only; PT snapshots/reload retain the exact typed state."""
    state = sampler.state_dict()
    generator = state['generator']
    require(isinstance(generator, torch.Tensor) and generator.dtype == torch.uint8
            and generator.device.type == 'cpu' and generator.ndim == 1,
            'Actual sampler generator must remain its CPU byte state')
    complete_state_sha256 = state_digest(state)
    state['generator'] = dict(type='torch.Tensor', dtype=str(generator.dtype),
        shape=list(generator.shape), typed_state_sha256=state_digest(generator), metadata_only=True)
    return dict(state, complete_state_sha256=complete_state_sha256, metadata_only=True)


def validate_actual_xformers_backend(rows, binding):
    """An environment request or missing fallback log is insufficient for admission."""
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
            and len({row['pid'] for row in rows}) == 8, 'Missing actual eight-worker backend identity')
    for row in rows:
        validate_bf16_accumulation_identity(row, role='actor_rollout')
        require(row['cuda_launch_blocking']['value'] == '0'
                and row['cuda_launch_blocking']['vllm_attention_backend'] == 'XFORMERS'
                and row['cuda_launch_blocking']['backend_present_before_torch_import'] is True
                and row['cuda_launch_blocking']['present_before_torch_import'] is True
                and row['cuda_launch_blocking']['constructor_pid'] == row['pid'],
                'Actual vLLM process lost its pre-initialization XFORMERS/asynchronous policy')
        require(row['executor_class'] == 'verl.third_party.vllm.vllm_v_0_6_3.spmd_gpu_executor.SPMDGPUExecutor'
                and row['executor_worker_class'] == 'verl.third_party.vllm.vllm_v_0_6_3.worker.Worker',
                'BF16 process readback requires the actual in-process SPMD vLLM worker')
        require(row['scope'] == 'ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY'
                and row['source_commit'] == binding['code_commit']
                and row['binding_sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
                'Backend evidence identity differs')
        require(row['requested_backend'] == binding['vllm_attention_backend'] == 'XFORMERS'
                and row['runner_backend_class'] == 'vllm.attention.backends.xformers.XFormersBackend'
                and row['runner_backend_name'] == 'xformers', 'Actual vLLM backend fell back or differs')
        require(row['model_dtype'] == 'torch.bfloat16' and row['cache_dtype'] == 'auto'
                and row['block_size'] == 16 and row['model_sliding_window'] is None,
                'Actual XFORMERS dtype/cache/block/sliding configuration differs')
        layers = row['attention_layers']
        require(len(layers) == 36 and len({layer['name'] for layer in layers}) == 36,
                'Missing actual full-model attention implementation metadata')
        for layer in layers:
            require(layer['implementation_class'] == 'vllm.attention.backends.xformers.XFormersImpl'
                    and layer['head_size'] == 128 and layer['num_heads'] == 16
                    and layer['num_kv_heads'] == 2 and layer['sliding_window'] is None,
                    'Actual attention implementation differs or silently falls back')
        require(row['observation_forward_calls'] == 0 and row['observation_tensor_value_reads'] == 0
                and row['actual_vllm_runner_positions_observed'] is False
                and row['probability_accepted'] is False, 'Backend metadata scope overstated')
    return dict(status='PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY',
                expected_backend='XFORMERS', rank_count=8, layers_per_rank=36,
                actual_observations=rows, probability_accepted=False)


def observe_bf16_accumulation(trainer, run, stage):
    """Read both real all-rank roles at idle boundaries; never reset a live flag."""
    identities = {}
    for role, group in (('actor_rollout', trainer.actor_rollout_wg), ('ref', trainer.ref_policy_wg)):
        rows = worker_call(trainer, 'bf16_accumulation/' + stage + '/' + role,
                           group.calibration_bf16_accumulation_identity)
        require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
                and len({row['pid'] for row in rows}) == 8, 'Missing actual eight-rank BF16 role readback')
        for row in rows:
            validate_bf16_accumulation_identity(row, role=role)
            policy = row['cuda_launch_blocking']
            require(policy['value'] == '0' and policy['present_before_torch_import'] is True
                    and policy['vllm_attention_backend'] == 'XFORMERS'
                    and policy['backend_present_before_torch_import'] is True
                    and policy['constructor_pid'] == row['pid'] and policy['worker_role'] == role,
                    'Actual role lost the pre-initialization XFORMERS/asynchronous policy')
        identities[role] = sorted(rows, key=lambda row: row['rank'])
    previous = getattr(trainer, '_calibration_bf16_accumulation', None)
    require(previous is None or previous == identities,
            'BF16 policy or worker process changed; in-process repair is forbidden')
    trainer._calibration_bf16_accumulation = copy.deepcopy(identities)
    json_save(run / ('actual-bf16-accumulation-' + stage + '.json'), dict(
        status='ACTUAL_ALL8_BOTH_ROLES_FALSE', stage=stage, actual_roles=identities,
        cuda_launch_blocking_status='ACTUAL_ALL8_BOTH_ROLES_0_BEFORE_INITIALIZATION',
        vllm_attention_backend_status='ACTUAL_ALL8_BOTH_ROLES_XFORMERS_BEFORE_INITIALIZATION',
        live_flag_reset_permitted=False, probability_accepted=False))
    return identities


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
        runtime_numerical_execution=copy.deepcopy(trainer._calibration_bf16_accumulation),
        workers=workers, phase0_state_reload_performed=False, formal_updates_contributed=0)
    json_save(run / 'actual-official-zero-initialization.json', value)
    return value


def create_trainer(config):
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from calibration_worker_v1 import CalibrationActorRolloutRefWorker
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    mapping = {Role.ActorRollout: 'global_pool', Role.RefPolicy: 'global_pool'}
    manager = ResourcePoolManager(resource_pool_spec={'global_pool': [8]}, mapping=mapping)
    require(config.trainer.val_before_train is False, 'User-authorized historical baseline reuse must skip initial validation')
    # Frozen source checks its former baseline requirement while constructing
    # the sampler. Only that constructor compatibility bit uses the old value;
    # the actual saved/trainer/worker config remains False and fit is never used.
    constructor_config = copy.deepcopy(config)
    constructor_config.trainer.val_before_train = True
    trainer = RayPPOTrainer(config=constructor_config, tokenizer=tokenizer,
        role_worker_mapping={role: ray.remote(CalibrationActorRolloutRefWorker) for role in mapping},
        resource_pool_manager=manager, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=RewardManager(tokenizer, 0), val_reward_fn=RewardManager(tokenizer, 1))
    trainer.config = config
    from metadata_colocation_v1 import scoped_metadata_colocation
    with scoped_metadata_colocation():
        trainer.init_workers()
    require(len(trainer.train_dataset) == 169028 and len(trainer.val_dataset) == 512,
            'Actual common dataset row count differs')
    require(not trainer.use_critic and trainer.use_reference_policy, 'Unexpected algorithm execution path')
    return trainer


def evaluate(trainer, phase, run):
    trainer.actor_rollout_wg.calibration_phase(phase + '512')
    with observe_time(trainer, phase + '512_evaluation', 'evaluation'):
        full = trainer._validate()
    require(full.get('val/qa_count/all') == 512 and full.get('val/qa_count/nq') == 256
            and full.get('val/qa_count/hotpotqa') == 256, 'Incomplete same immutable512 validation')
    value = dict(scope=SCOPE, phase=phase, logical_step=trainer.global_steps,
                 retained_training_updates_contributed=trainer.actual_outer_updates,
                 full512=full, extra_stage64_evaluation=False)
    json_save(run / f'{phase}-evaluation.json', value)
    return value


def install_training_prefix_observer(run, outer, progress):
    from search_r1.baselines.rollout import SearchRollout
    original = SearchRollout.advance
    turns, errors = [], []
    progress['prefix_observer_errors'] = errors

    def observed(engine, states, meta):
        # Retain actual live references; snapshotting happens only on refusal.
        progress['active_rollout_engine'] = engine
        progress['active_rollout_states'] = states
        before = []
        try:
            before = [dict(source=state.source, step_id=len(state.anchors), generation_prefix=list(state.rolling),
                canonical_context_prefix=list(state.prompt + state.response),
                prior_action_count=len(state.action_prefix_records)) for state in states]
        except BaseException as error:
            errors.append('prefix_before: ' + repr(error))
        progress['active_turn_prefix_before'] = before
        value = original(engine, states, meta)  # Driver observer never skips an original advance.
        try:
            require(len(before) == len(states), 'Missing prefix observer setup')
            for row, state in zip(before, states):
                require(len(state.action_prefix_records) == row['prior_action_count'] + 1, 'Missing new raw action provenance')
                item = copy.deepcopy(state.action_prefix_records[-1])
                require(item['generation_prefix'] == row['generation_prefix'], 'Actual source prefix changed')
                row.update(raw_actor_record=item, retained_action_tokens=item['actor_action_tokens'],
                    canonical_retained_action_tokens=item['canonical_action_tokens'])
            turns.append(before)
            json_save(run / ('training-prefix-turns.json' if outer == 1 else f'training-prefix-turns-attempt{outer}.json'), turns)
        except BaseException as error:
            errors.append('prefix_after: ' + repr(error))
        return value

    SearchRollout.advance = observed
    return original, turns, errors


def driver_complete_pause_state(trainer, batch=None, *, local_only=False):
    """The real post-sampling state. This function neither fetches data nor restores RNG."""
    from search_r1.baselines.recovery import cpu_snapshot, rng_state
    iterator = trainer._calibration_training_iterator
    from search_r1.baselines.fresh200 import driver_flow_state
    from search_r1.baselines.fresh200_live import driver_admission_state
    if local_only:
        admission = dict(payload=copy.deepcopy(getattr(trainer, 'fresh200_live_admission', None)),
            last_known_role_admissions=copy.deepcopy(getattr(trainer, 'fresh200_live_role_admissions', None)),
            actual_roles_requeried=False, saved_after_unknown_failure=True,
            audit_environment={key: os.environ.get(key) for key in
                ('SEARCH_R1_BASELINE_CONFIG_SHA256', 'GIGPO_POLICY_NATIVE_BINDING', 'GIGPO_POLICY_NATIVE_BINDING_SHA256')})
    else:
        admission = driver_admission_state(trainer)
    driver_flow = driver_flow_state(trainer)
    controller = trainer.baseline_controller
    start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(schema=1, pid=os.getpid(), start_ticks=start,
        global_steps=trainer.global_steps, actual_outer_updates=trainer.actual_outer_updates,
        rng=cpu_snapshot(rng_state()), sampler=cpu_snapshot(trainer.recovery_sampler.state_dict()),
        data_iterator=dict(base_seed=iterator._base_seed, num_yielded=iterator._num_yielded),
        fresh200_driver_flow=cpu_snapshot(driver_flow),
        kl_controller=cpu_snapshot(vars(trainer.kl_ctrl)),
        algorithm_state=dict(method='gigpo', stage=cpu_snapshot(controller.stage.state_dict()),
            parameters=OmegaConf.to_container(trainer.config.baselines.gigpo, resolve=True),
            groups='stateless_per_rollout_with_step_ids_and_advantage_broadcast',
            base_commit=controller.config.base_commit,
            algorithm_source_commit=controller.config.algorithm_source_commit),
        config=OmegaConf.to_container(trainer.config, resolve=True),
        runtime_numerical_execution=copy.deepcopy(trainer._calibration_bf16_accumulation),
        fresh200_live_admission=admission,
        common_data_identity=copy.deepcopy(trainer.instruct_data_identity),
        validation_trace_serial=driver_flow['validation_trace_serial'],
        quality_health=copy.deepcopy(getattr(trainer, 'quality_health', None)),
        fixed_batch=None if batch is None else dict(tensors=cpu_snapshot(dict(batch.batch.items())),
                         non_tensors=copy.deepcopy(batch.non_tensor_batch), metadata=copy.deepcopy(batch.meta_info)))


def worker_call(trainer, stage, operation):
    """Driver stage label only; actual dispatch/await callbacks own safety state."""
    progress = getattr(trainer, '_calibration_progress', None)
    if progress is not None:
        progress['stage'] = stage
    return operation()


def capture_post_sampling_pause(trainer, batch, run, material):
    """Bind all live states after genuine sampling, not the pre-rollout checkpoint RNG."""
    from search_r1.baselines.recovery import atomic_torch_save, file_sha256
    numerical_execution = observe_bf16_accumulation(trainer, run, 'post-sampling-before-first-adam')
    worker_states = worker_call(trainer, 'post_sampling_complete_pause',
                               trainer.actor_rollout_wg.calibration_complete_pause_state)
    for state in worker_states:
        policy = validate_bf16_accumulation_identity(dict(pid=state['identity']['pid'],
            bf16_accumulation=state['metadata']['bf16_accumulation']), role='actor_rollout')
        require(policy == numerical_execution['actor_rollout'][state['identity']['rank']]['bf16_accumulation'],
                'Complete pause and all-rank BF16 readback differ')
    workers = validate_pause_workers([row['identity'] for row in worker_states])
    driver = driver_complete_pause_state(trainer, batch)
    require(driver['sampler']['cursor'] == 512 and driver['global_steps'] == 1
            and driver['actual_outer_updates'] == 0, 'Wrong actual post-sampling driver state; zero Adam')
    raw = atomic_torch_save(dict(workers=worker_states, driver=driver), run / 'post-sampling-pause-state.pt')
    summary = dict(scope=SCOPE, status='POST_SAMPLING_BEFORE_FIRST_ADAM', workers=workers,
        runtime_numerical_execution=numerical_execution,
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
                    actual_hf_role_micro_batch_sizes={'old': 4, 'current': 2, 'reference': 4},
                    actual_dropout_neutral_train_eval_verified=True,
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
    observe_bf16_accumulation(trainer, run, 'post-probability-review-before-first-adam')
    after_worker_states = worker_call(trainer, 'post_approval_complete_pause',
                                     trainer.actor_rollout_wg.calibration_complete_pause_state)
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
    with observe_time(trainer, 'first_raw_probability_alignment_normalizer_hashing', 'first_probability_probe'):
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
                validate_bf16_accumulation_identity(record, role='actor_rollout')
                require(record['source_commit'] == binding['code_commit'] and record['rank'] == rank and record['serial'] == serial,
                        'Perturn actual identity differs')
                raw_path = Path(record['raw']['path'])
                raw = checked_load(raw_path, record['raw'])
                validate_bf16_accumulation_identity(raw, role='actor_rollout')
                require(raw['bf16_accumulation'] == record['bf16_accumulation'],
                        'Perturn raw and receipt BF16 process readback differ')
                for key in ('repeated_probe_state_before', 'repeated_probe_state_after'):
                    state = raw[key]
                    policy = validate_bf16_accumulation_identity(dict(pid=state['identity']['pid'],
                        bf16_accumulation=state['metadata']['bf16_accumulation']), role='actor_rollout')
                    require(policy == raw['bf16_accumulation'], 'Perturn repeated state BF16 readback differs')
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
                worker_role = 'ref' if role == 'reference' else 'actor_rollout'
                validate_bf16_accumulation_identity(record, role=worker_role)
                require(record['role'] == role and record['rank'] == rank and record['code_commit'] == binding['code_commit']
                        and record['binding_sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'], 'Complete HF role/source binding differs')
                raw_path = Path(record['raw']['path'])
                raw = checked_load(raw_path, record['raw'])
                validate_bf16_accumulation_identity(raw, role=worker_role)
                require(raw['bf16_accumulation'] == record['bf16_accumulation'],
                        'Complete HF raw and receipt BF16 process readback differ')
                require(raw['role'] == role and raw['rank'] == rank and raw['code_commit'] == binding['code_commit']
                        and raw['binding_sha256'] == record['binding_sha256'] and raw['pid'] == record['pid'], 'Complete HF raw role/source identity differs')
                require(raw['micro_batch_size'] == {'old': 4, 'current': 2, 'reference': 4}[role]
                        and raw['dropout_neutral_train_eval_verified'] is True
                        and raw['actual_dropout_config'].get('attention_dropout') == 0.
                        and all(value == 0. for value in raw['actual_dropout_config'].values())
                        and all(value == 0. for value in raw['actual_dropout_modules'].values()),
                        'Native probe lost current2/old4/ref4 or actual dropout identity')
                items = unpack_complete_action_calls(raw, prepare=prepare_action_prefix_micro,
                    pad_id=binding['pad_token_id'], eos_id=binding['eos_token_id'])
                require(not (roles[role].keys() & items.keys()), 'Duplicate actual HF source/step mapping')
                roles[role].update(items)
                paths += [path, raw_path]
            if role != 'old':
                for rank in range(8):
                    path = run / f'probabilities/complete-actions/{role}-neutral-rank-{rank}.json'
                    record = json.loads(path.read_text())
                    validate_bf16_accumulation_identity(record, role=worker_role)
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
    with observe_time(trainer, 'probability_independent_wait_and_pause_readback', 'first_probability_review'):
        admitted = wait_probability_acceptance(trainer, batch, run, binding, material, records, alignment, checks)
    trainer._calibration_probability_binding = copy.deepcopy(admitted)
    return alignment


def _preserve_first_refusal(trainer, run, progress, error):
    from search_r1.baselines.recovery import atomic_torch_save, cpu_snapshot, rng_state
    from calibration_failure_v2 import preserve_first_refusal
    return preserve_first_refusal(trainer, run, progress, error,
        driver_snapshot=lambda trainer, batch: driver_complete_pause_state(trainer, batch, local_only=True),
        atomic_torch_save=atomic_torch_save,
        json_save=json_save, cpu_snapshot=cpu_snapshot, rng_state=rng_state)


def one_outer(trainer, run, binding, training_iterator, outer):
    progress = dict(stage='attempt_setup', outer=outer, attempt_id=None,
        actual_adam_total_before=None, workers_safe_to_snapshot=True,
        first_adam_possible=False, batch=None)
    trainer._calibration_progress = progress
    try:
        from search_r1.baselines.fresh200 import initialize_attempt_ledger
        ledger = initialize_attempt_ledger(trainer)
        progress.update(attempt_id=ledger['attempts'] + 1, actual_adam_total_before=ledger['actual_adam_total'])
        require(outer == ledger['completed_outer_updates'] + 1, 'Attempt must target the next uncompleted outer')
        from worker_rpc_boundary_v4 import track_first_outer_rpcs
        with track_first_outer_rpcs(trainer, progress):
            with observe_time(trainer, 'complete_attempt_wall', 'attempt_total_including_observation'):
                value = _one_outer_body(trainer, run, binding, training_iterator, outer, progress)
            publish_timing(trainer, run, value['outcome']['counted_complete'])
            return value
    except BaseException as error:
        if progress['attempt_id'] == 1 and not progress['first_adam_possible']:
            try:
                _preserve_first_refusal(trainer, run, progress, error)
            except BaseException as preservation_error:
                # A damaged observer/save cannot replace the original refusal.
                try:
                    json_save(run / 'first-refusal-preservation-error.json', dict(original_error=repr(error),
                        preservation_error=repr(preservation_error), actual_cursor=trainer.recovery_sampler.cursor,
                        missing_ranks=list(range(8)), exact_resume_accepted=False, automatic_resume_allowed=False))
                except BaseException:
                    pass
        else:
            # Unknown execute/await errors retain local driver and actual resolved
            # rank outputs only. This path never starts a recovery RPC/collective.
            try:
                from calibration_failure_v5 import preserve_attempt_failure
                preserve_attempt_failure(trainer, run, progress, error,
                    driver_snapshot=lambda trainer, batch: driver_complete_pause_state(trainer, batch, local_only=True),
                    json_save=json_save)
            except BaseException as preservation_error:
                try:
                    json_save(run / f'attempt{progress["attempt_id"]}-preservation-error.json',
                        dict(original_error=repr(error), preservation_error=repr(preservation_error),
                            worker_recovery_rpc_dispatched=False, automatic_resume_allowed=False))
                except BaseException:
                    pass
        raise
    finally:
        trainer._calibration_progress = None


def _one_outer_body(trainer, run, binding, training_iterator, outer, progress):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_data_metrics
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_r1.baselines.rollout import SearchRollout, select_rows
    from search_r1.baselines.trajectory_records import RolloutTrace, record_scored_batch
    from search_r1.baselines.recovery import atomic_torch_save
    from search_r1.baselines.fresh200 import begin_attempt, consume_attempt, record_epoch_end
    config = trainer.config
    trainer.global_steps = outer
    observe_bf16_accumulation(trainer, run, 'before-training-batch-' + str(outer))
    progress['stage'] = 'next_training_batch'
    try:
        fetched = next(training_iterator)
    except StopIteration:
        record_epoch_end(trainer)  # Actual exhausted sampler; never infer512×completed.
        training_iterator = iter(trainer.train_dataloader)
        trainer._calibration_training_iterator = training_iterator
        fetched = next(training_iterator)
    progress['fetched_batch'] = fetched
    batch = DataProto.from_single_dict(fetched)
    progress['batch'] = batch
    require(len(batch) == 512, 'Wrong actual fresh diagnostic batch size')
    attempt_id = begin_attempt(trainer, batch)
    require(attempt_id == progress['attempt_id'], 'Real source attempt identity differs')
    first_sampling = attempt_id == 1
    require(binding.get('human_nonblocking_diagnostics') is True,
            'Actual fresh attempt requires bound human nonblocking authorization')
    # train2 selects the unchanged source generation/old/reference APIs without
    # diagnostic repeats, current probes or full-logit capture on any attempt.
    trainer.actor_rollout_wg.calibration_phase('train2')
    batch = batch.repeat(repeat_times=5, interleave=True)
    progress['batch'] = batch
    batch.non_tensor_batch['audit_trajectory_row'] = np.arange(2560, dtype=object)
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    manager = LLMGenerationManager(trainer.tokenizer, trainer.actor_rollout_wg,
        GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
                         max_response_length=500, max_obs_length=500, num_gpus=8,
                         no_think_rl=config.algorithm.no_think_rl, search_url=config.retriever.url,
                         topk=3, force_final_answer=config.force_final_answer))
    manager.rollout_trace = RolloutTrace(run / 'trajectories', attempt_id, batch.non_tensor_batch['index'].tolist(), 5)
    metrics = {}
    original, turns, prefix_errors = install_training_prefix_observer(run, attempt_id, progress)
    progress['turns'] = turns
    progress['stage'] = 'rollout_driver_local_setup'
    try:
        with observe_time(trainer, 'generation_and_retrieval_client',
                'first_rollout_including_identity_observation' if first_sampling else 'ordinary_training'):
            generated, sources = trainer.baseline_controller.generate(manager, gen_batch, batch, metrics)
    finally:
        SearchRollout.advance = original
    # Only the actual frozen Ray collect-entry callback proves successful all-rank await.
    progress.update(generated=generated, sources=sources, stage='rollout_local_schema')
    require(not prefix_errors, 'Training prefix observer failed: ' + repr(prefix_errors))
    require(len(generated) == 2560, 'Incomplete actual512x5 rollout')
    batch = select_rows(batch, sources)
    progress['batch'] = batch
    for key in generated.batch.keys():
        generated.batch[key] = generated.batch[key].long()
    require(generated.meta_info.get('actor_token_identity') == 'sampled_action_prefix_v1', 'Not raw action source')
    generated.non_tensor_batch['audit_trajectory_row'] = np.asarray(sources, dtype=object)
    generated.non_tensor_batch['audit_question_index'] = batch.non_tensor_batch['index'].copy()
    generated.meta_info['raw_calibration_observation'] = False
    progress['stage'] = 'complete_old_hf'
    with torch.no_grad():
        with observe_time(trainer, 'old_policy_recompute',
                'first_probability_probe_with_required_recompute' if first_sampling else 'ordinary_training'):
            old_output = trainer.actor_rollout_wg.compute_log_prob(generated)
    generated = generated.union(old_output)
    progress['generated'] = generated
    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
    batch = batch.union(generated)
    progress['batch'] = batch
    trainer._balance_batch(batch, metrics=metrics)
    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
    for key in batch.batch.keys():
        if key != 'old_log_probs':
            batch.batch[key] = batch.batch[key].long()
    progress['stage'] = 'complete_reference_hf'
    with observe_time(trainer, 'reference_recompute',
            'first_probability_probe_with_required_recompute' if first_sampling else 'ordinary_training'):
        reference_output = trainer.ref_policy_wg.compute_ref_log_prob(batch)
    batch = batch.union(reference_output)
    progress['batch'] = batch
    with observe_time(trainer, 'advantage_mask_and_rollout_evidence'):
        progress['stage'] = 'advantage_mask_and_evidence'
        batch.batch['token_level_scores'] = trainer.reward_fn(batch)
        require(config.actor_rollout_ref.actor.use_kl_loss, 'Original fixed reference loss missing')
        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
        batch = trainer.baseline_controller.compute_advantages(batch, metrics)
        batch, metrics = trainer._create_loss_mask(batch, metrics)
        progress['batch'] = batch
        batch.meta_info['baseline_outer_step'] = outer
        record_scored_batch(run / 'trajectories', attempt_id, batch, trainer.tokenizer)
        atomic_torch_save(dict(tensors={k: v.detach().cpu().clone() for k, v in batch.batch.items()},
                              non_tensors=batch.non_tensor_batch, metadata=batch.meta_info,
                              data_cursor=trainer.recovery_sampler.state_dict(), scope=SCOPE,
                              attempt_id=attempt_id, completed_outer_label=outer),
            run / ('actual-rollout-2560.pt' if first_sampling else f'actual-rollout-attempt{attempt_id}-2560.pt'))
    alignment = dict(status='HUMAN_NONBLOCKING_DIAGNOSTICS', independent_review_status='PENDING_BACKGROUND',
                     actual_probability_accepted=False, probability_tolerances_selected=False)
    if first_sampling:
        json_save(run / 'human-nonblocking-diagnostics.json', dict(alignment,
            job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
            source_commit=binding['code_commit'], binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
            diagnostic_authorization_scope=binding['diagnostic_authorization_scope'],
            external_diagnostic_review_blocks_training=False,
            training_evidence=str(run / 'actual-rollout-2560.pt'),
            extra_probability_forward_calls=0, hf_semantic_test_status='PENDING_BACKGROUND',
            negative_tail_threshold_status='PENDING_BACKGROUND'))
    from search_r1.baselines.runtime_audit import driver_outer
    driver_outer(batch, outer, 'gigpo')
    progress['stage'] = 'immediately_before_first_adam'
    if first_sampling:
        # Check again immediately before the first update RPC, including any
        # intervening audit operation and still-current fixed rollout bytes.
        observe_bf16_accumulation(trainer, run, 'immediately-before-first-adam')
        json_save(run / 'immediately-before-first-adam.json', dict(
            status='HUMAN_NONBLOCKING_DIAGNOSTICS', actual_probability_accepted=False,
            independent_review_status='PENDING_BACKGROUND', actual_adam_writes=0,
            diagnostic_authorization_scope=binding['diagnostic_authorization_scope']))
    before_update_states = worker_call(trainer, 'pre_update_scheduler_identity',
                                       trainer.actor_rollout_wg.calibration_complete_pause_state)
    before_update = [row['identity'] for row in before_update_states]
    expected_lr = 1e-6 * min(1., float(outer - 1) / 57.)
    require(len(before_update) == 8 and {r['rank'] for r in before_update} == set(range(8))
            and len({r['pid'] for r in before_update}) == 8
            and all(r['scheduler_step'] == outer - 1 and r['learning_rates'] == [expected_lr]
                for r in before_update), 'Actual outer scheduler/LR boundary differs')
    progress['stage'] = 'update_actor_driver_dispatch'
    with observe_time(trainer, 'actor_update_and_original_guards'):
        result = trainer.actor_rollout_wg.update_actor(batch)
    progress['stage'] = 'consume_actual_eight_rank_attempt_receipts'
    outcome = consume_attempt(trainer, result)
    progress['attempt_outcome'] = outcome
    from search_r1.baselines.partial_update import save_attempt_driver
    driver_receipt = save_attempt_driver(trainer, batch, outcome)
    metrics.update(reduce_metrics(result.meta_info['metrics']))
    attempt = outcome['attempt']
    require(metrics['actor/optimizer_steps'] == attempt['actual_successful_adam_writes']
            and metrics['actor/opportunities_processed'] == attempt['opportunities_processed']
            and metrics['actor/actual_gradient_micro_batches'] == attempt['gradient_micro_batches'],
            'Metrics differ from real eight-rank opportunity/write receipts')
    metrics.update(compute_data_metrics(batch, use_critic=False))
    if outcome['counted_complete']:
        trainer.baseline_controller.after_actor(batch, metrics)
    after_states = trainer.actor_rollout_wg.calibration_complete_pause_state()
    from attempt_accounting_v5 import validate_actual_worker_ledgers
    validate_actual_worker_ledgers(after_states, outcome['ledger'])
    json_save(run / f'calibration-attempt{attempt_id}.json', dict(scope=SCOPE,
        calibration_actual_outer_updates=trainer.actual_outer_updates,
        retained_training_updates_contributed=trainer.actual_outer_updates,
        config_epoch=attempt['config_epoch'], config_sha256=attempt['config_sha256'],
        logical_label=outer, attempt_id=attempt_id,
        counted_complete=outcome['counted_complete'], actual_opportunities_processed=attempt['opportunities_processed'],
        actual_successful_adam_writes=attempt['actual_successful_adam_writes'],
        actual_gradient_micro_batches=attempt['gradient_micro_batches'],
        data_cursor=trainer.recovery_sampler.cursor, sampler=sampler_json_metadata(trainer.recovery_sampler),
        alignment=alignment, metrics=metrics, outcome=outcome, driver_receipt=driver_receipt,
        actual_lr_before_update=expected_lr, timings=copy.deepcopy(progress.get('timings', [])), worker_before=before_update,
        worker_after=[row['identity'] for row in after_states], automatic_rollback=False))
    return dict(metrics=metrics, outcome=outcome, driver_receipt=driver_receipt)


def reload_saved_complete_state(trainer, run, expected_completed=None, endpoint_name=None):
    """Compare actual complete typed state around real same-worker reload.

    The endpoint supplement binds every typed driver field and full worker
    metadata. Fresh200 source driver-flow helpers persist and actually reload
    iterator/validation progress in the same live iterator; no new data is drawn.
    """
    from search_r1.baselines.instruct300 import check_manifest, equal_state, _live_gigpo_state
    from search_r1.baselines.recovery import file_sha256, rng_state, restore_rng, atomic_torch_save
    from search_r1.baselines.fresh200 import driver_flow_state, restore_driver_flow
    from attempt_accounting_v5 import actual_attempt_totals, validate_actual_worker_ledgers
    ledger = copy.deepcopy(trainer.fresh200_attempt_ledger)
    totals = actual_attempt_totals(ledger, expected_completed)
    completed = ledger['completed_outer_updates']
    endpoint_name = endpoint_name or ('completed' + str(completed))
    manifest = trainer.instruct_manifest
    check_manifest(manifest)
    files = manifest['identity']['files']
    saved = [record['pause_identity'] for record in files if 'pause_identity' in record]
    before_workers = trainer.actor_rollout_wg.calibration_complete_pause_state()
    before_workers = sorted(before_workers, key=lambda row: row['identity']['rank'])
    require(len(before_workers) == 8 and {row['identity']['rank'] for row in before_workers} == set(range(8)),
            'Missing full typed endpoint worker state')
    worker_ledgers = validate_actual_worker_ledgers(before_workers, ledger)
    require([dict((key, value) for key, value in row['identity'].items() if key != 'complete_state_sha256')
             for row in before_workers] == sorted(saved, key=lambda row: row['rank']),
            'Actual saved eight-worker pause identity differs')
    before_driver = driver_complete_pause_state(trainer)
    supplemental = atomic_torch_save(dict(schema='FRESH200_ENDPOINT_COMPLETE_TYPED_STATE_V2',
        workers=before_workers, driver=before_driver, source_manifest_sha256=file_sha256(manifest['path'])),
        run / (endpoint_name + '-complete-state-before-reload.pt'))
    require(file_sha256(supplemental['path']) == supplemental['sha256'], 'Endpoint supplement SHA differs')
    saved_full = torch.load(supplemental['path'], map_location='cpu', weights_only=False)
    require(equal_state(saved_full['workers'], before_workers) and equal_state(saved_full['driver'], before_driver),
            'Endpoint supplement differs from live full state')
    shas = {str(record['pause_identity']['rank']): record['sha256'] for record in files if 'pause_identity' in record}
    checkpoint = str(Path(manifest['path']).parent.parent)
    loaded = trainer.actor_rollout_wg.load_training_state(checkpoint, shas,
        ledger['completed_outer_updates'], ledger['actual_adam_total'])
    after_workers = sorted(trainer.actor_rollout_wg.calibration_complete_pause_state(), key=lambda row: row['identity']['rank'])
    require(equal_state(before_workers, after_workers) and len(loaded) == 8
            and all(row['logical_step'] == completed for row in loaded), 'Full typed worker endpoint reload differs')
    receipt = next(record for record in files if record['path'].endswith('/driver.pt'))
    require(file_sha256(receipt['path']) == receipt['sha256'], 'Driver state SHA differs')
    driver = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    require(driver['logical_step'] == driver['actual_new_updates'] == completed
            and equal_state(driver['sampler'], trainer.recovery_sampler.state_dict()) and driver['source_revision'] == REVISION
            and driver['config'] == OmegaConf.to_container(trainer.config, resolve=True)
            and equal_state(driver['rng'], rng_state()), 'Saved actual driver identity differs')
    require(equal_state(driver['fresh200_driver_flow']['attempt_ledger'], ledger),
            'Saved complete driver lacks the actual attempt/write ledger')
    trainer.recovery_sampler.load_state_dict(driver['sampler'])
    vars(trainer.kl_ctrl).clear(); vars(trainer.kl_ctrl).update(copy.deepcopy(driver['kl_controller']))
    algorithm = driver['algorithm_state']
    algorithm_receipt = next(record for record in files if '/algorithm/' in record['path'])
    require(algorithm == json.loads(Path(algorithm_receipt['path']).read_text())
            and algorithm == _live_gigpo_state(trainer, completed), 'Saved GiGPO state differs from actual live algorithm')
    trainer.baseline_controller.stage.load_state_dict(copy.deepcopy(algorithm['bapo']))
    require(algorithm == _live_gigpo_state(trainer, completed), 'Actual GiGPO algorithm reload differs')
    endpoint_driver = saved_full['driver']
    require(equal_state(driver['fresh200_driver_flow'], endpoint_driver['fresh200_driver_flow'])
            and equal_state(driver['fresh200_driver_flow'], driver_flow_state(trainer)),
            'Saved source driver flow differs from actual endpoint')
    flow_reload = restore_driver_flow(trainer, driver['fresh200_driver_flow'])
    from search_r1.baselines.fresh200_live import restore_driver_admission_state
    restore_driver_admission_state(trainer, driver['fresh200_live_admission'])
    trainer.global_steps = endpoint_driver['global_steps']
    trainer.actual_outer_updates = endpoint_driver['actual_outer_updates']
    if endpoint_driver['quality_health'] is not None:
        trainer.quality_health = copy.deepcopy(endpoint_driver['quality_health'])
    restore_rng(driver['rng'])  # Natural post-updated512 RNG, including Python/NumPy.
    after_driver = driver_complete_pause_state(trainer)
    require(equal_state(before_driver, after_driver), 'Full typed driver/iterator/trace/algorithm/RNG endpoint differs')
    after_receipt = atomic_torch_save(dict(workers=after_workers, driver=after_driver),
                                     run / (endpoint_name + '-complete-state-after-reload.pt'))
    value = dict(scope=LIVE_SCOPE, status='ACTUAL_COMPLETE_TYPED_STATE_RELOADED',
        manifest=manifest, supplemental_before=supplemental, supplemental_after=after_receipt,
        worker_complete_before_sha256=state_digest(before_workers), worker_complete_after_sha256=state_digest(after_workers),
        driver_complete_before_sha256=state_digest(before_driver), driver_complete_after_sha256=state_digest(after_driver),
        comparison_fields='all_typed_worker_and_driver_fields_in_original_H100_PT',
        complete_rng_families=['Python', 'NumPy', 'TorchCPU', 'TorchCUDA', 'sharding_manager'],
        iterator_and_validation_trace_live_reloaded=True, driver_flow_reload=flow_reload,
        actual_algorithm_live_reloaded=True,
        workers=loaded, same_worker_identity=True, logical_step=completed, **totals,
        actual_worker_ledgers=worker_ledgers, actual_attempt_ledger=ledger,
        scheduler_step=completed, data_cursor=driver['sampler']['cursor'], data_epoch=driver['sampler']['epoch'],
        phase0_state_reload_performed=False, retained_training_updates_contributed=completed,
        realized_admission=copy.deepcopy(trainer.fresh200_live_admission))
    json_save(run / (endpoint_name + '-state-roundtrip.json'), value)
    return value


def main_task(config, binding):
    run = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    require(binding['cuda_launch_blocking'] == os.environ.get('CUDA_LAUNCH_BLOCKING') == '0',
            'Ray task asynchronous CUDA environment differs')
    require(binding['vllm_attention_backend'] == os.environ.get('VLLM_ATTENTION_BACKEND') == 'XFORMERS',
            'Ray backend environment differs')
    trainer = create_trainer(config)
    observe_bf16_accumulation(trainer, run, 'before-official-zero-and-first-batch')
    backend_rows = trainer.actor_rollout_wg.calibration_backend_identity()
    json_save(run / 'actual-vllm-backend-identity.json', dict(status='OBSERVED_UNREVIEWED',
        actual_observations=backend_rows, probability_accepted=False))
    json_save(run / 'actual-vllm-backend-identity.json', validate_actual_xformers_backend(backend_rows, binding))
    from search_r1.baselines.holdout import _validate_prebuilt
    trainer.instruct_data_identity = _validate_prebuilt(config)
    fsdp_initialization = trainer.actor_rollout_wg.calibration_initialize_actor_fsdp_before_zero()
    json_save(run / 'actual-native-fsdp-before-zero.json', fsdp_initialization)
    require(len(fsdp_initialization) == 8
            and {row['rank'] for row in fsdp_initialization} == set(range(8))
            and len({row['pid'] for row in fsdp_initialization}) == 8,
            'Official zero native FSDP initialization requires all eight actual workers')
    initial = observe_official_zero(trainer, run)
    reference = trainer.ref_policy_wg.calibration_reference_identity()
    require(len(reference) == 8 and {row['rank'] for row in reference} == set(range(8))
            and all(r['finite'] for r in reference), 'Nonfinite or incomplete actual fixed reference')
    for row in reference:
        validate_bf16_accumulation_identity(row, role='ref')
    from search_r1.baselines.recovery import file_sha256
    reuse = binding['initial_baseline_policy']
    require(reuse['status'] == 'REUSED_USER_AUTHORIZED_BASELINE' and reuse['parent_job'] == '3332'
            and reuse['new_run_evaluation_forward_calls'] == 0
            and config.trainer.val_before_train is False, 'Explicit historical baseline reuse binding required')
    for row in (reuse['historical_evaluation'], reuse['historical_state_comparison']):
        path = Path(row['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
                and file_sha256(path) == row['sha256'], 'Historical baseline provenance changed')
    historical = json.loads(Path(reuse['historical_evaluation']['path']).read_bytes())
    full = historical['full512']
    require(historical['phase'] == 'baseline' and historical['logical_step'] == 0
            and historical['retained_training_updates_contributed'] == 0
            and full['val/qa_count/all'] == 512 and full['val/qa_count/nq'] == full['val/qa_count/hotpotqa'] == 256
            and full['val/f1/all'] == .2429115394461968 and full['val/em/all'] == .18359375,
            'Wrong historical official512 baseline')
    baseline = dict(scope=SCOPE, phase='baseline', logical_step=0, retained_training_updates_contributed=0,
        status='REUSED_USER_AUTHORIZED_BASELINE', historical=True, actual_new_run_evaluation=False,
        new_run_evaluation_forward_calls=0, new_run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        new_run_binding_sha256=os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'],
        new_run_actual_initialization=initial['actual_initialization'], provenance=reuse,
        full512=copy.deepcopy(full), extra_stage64_evaluation=False)
    json_save(run / 'reused-user-authorized-baseline.json', baseline)
    require(trainer.recovery_sampler.cursor == 0 and reference == trainer.ref_policy_wg.calibration_reference_identity(),
            'Initialization advanced training data or fixed reference')
    # No initial validation RNG is consumed; one live iterator for first5 through200.
    from live_continuation_v10 import run_counted_training
    return run_counted_training(trainer, run, binding, initial, reference, baseline)


if __name__ == '__main__':
    config = OmegaConf.load(sys.argv[1])
    binding = json.loads(Path(sys.argv[2]).read_text())
    env = {key: value for key, value in os.environ.items() if key.startswith(('SEARCH_R1_BASELINE_', 'GIGPO_POLICY_'))}
    env.update(PYTHONPATH=os.environ['PYTHONPATH'], CPATH=os.environ['CPATH'],
               SEARCH_R1_PYTHON_HEADER_CPATH=os.environ['SEARCH_R1_PYTHON_HEADER_CPATH'],
               VLLM_ATTENTION_BACKEND='XFORMERS', CUDA_LAUNCH_BLOCKING='0')
    require(os.environ.get('VLLM_ATTENTION_BACKEND') == 'XFORMERS', 'Driver backend environment differs')
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': env})
    try:
        value = ray.get(ray.remote(num_cpus=1)(main_task).options(runtime_env={'env_vars': env}).remote(config, binding))
        json_save(Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'driver-result.json', value)
        print(json.dumps(value, allow_nan=False), flush=True)
    finally:
        ray.shutdown()
