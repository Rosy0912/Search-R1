"""CPU-verifiable, opt-in fresh200 identity and update boundary contract."""
from pathlib import Path

PROTOCOL = 'BAPO_FRESH200_V1'
SAVE_OUTERS = (5, 50, 100, 150, 200)


def outer_contract(actual_outer):
    if type(actual_outer) is not int or not 1 <= actual_outer <= 200:
        raise ValueError('Fresh200 actual_outer must be integer 1..200')
    mini = 128 if actual_outer < 50 else 256
    return dict(protocol=PROTOCOL, actual_outer=actual_outer,
                global_mini_batch_size=mini, rank_mini_batch_size=mini // 8,
                expected_optimizer_steps=2560 // mini, world_size=8,
                global_micro_batch_size=16, rank_micro_batch_size=2)


def validate_fresh_config(config):
    trainer=config['trainer']; ar=config['actor_rollout_ref']; actor=ar['actor']; optim=actor['optim']
    required=[trainer.get('fresh200_enabled') is True, trainer.get('fresh200_gate_enabled') is True,
              trainer.get('e1_gate_enabled') is False, trainer.get('total_training_steps') == 201,
              trainer.get('acceptance_step') == 5, trainer.get('save_freq') == 50, trainer.get('n_gpus_per_node') == 8,
              trainer.get('nnodes') == 1, trainer.get('smoke_only') is False,
              optim.get('num_warmup_steps') == 57, optim.get('lr') == 1e-6,
              list(optim.get('betas',[.9,.999]))==[.9,.999],optim.get('weight_decay',.01)==.01,
              optim.get('lr_warmup_steps_ratio') == 0.0, optim.get('warmup_style') == 'constant',
              actor.get('entropy_coeff') == 0.0, actor.get('use_kl_loss') is False,
              actor.get('kl_loss_coef') == 0.0, actor.get('grad_clip') == 1.0,
              actor.get('clip_ratio') == 0.2, actor.get('loss_reduction') == 'sequence_mean',
              actor.get('ppo_mini_batch_size') == 128, actor.get('ppo_micro_batch_size') == 16,
              actor.get('ppo_epochs') == 1, actor.get('use_dynamic_bsz') is False,
              ar['rollout'].get('n_agent') == 5, ar['rollout'].get('tensor_model_parallel_size') == 1,
              ar['rollout'].get('log_prob_micro_batch_size') == 32,
              ar['ref'].get('log_prob_micro_batch_size') == 32,
              config['data'].get('train_batch_size') == 512, config['baselines'].get('method') == 'bapo',
              config.get('max_turns') == 4, config.get('force_final_answer') is False,
              config['data'].get('max_start_length') == 2048,
              config['data'].get('max_prompt_length') == 4096,
              config['data'].get('max_response_length') == 500,
              config['data'].get('max_obs_length') == 500,
              config['algorithm'].get('adv_estimator') == 'grpo',
              config['algorithm']['kl_ctrl'].get('kl_coef') == 0.0,
              trainer.get('fresh200_gate_receipt') is not None,
              actor.get('fresh200_enabled') is True, actor.get('policy_safety',{}).get('enabled') is True,
              actor.get('policy_safety',{}).get('threshold_origin') == 'user_fixed',
              actor.get('policy_safety',{}).get('target_old_policy_kl') == .2,
              actor.get('policy_safety',{}).get('negative_tail_limit') == .2,
              actor.get('policy_safety',{}).get('skip_on_policy_guard') is True,
              ar['model'].get('revision') == 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
              not ar['model'].get('override_config'),
              config['baselines']['holdout'].get('manifest_sha256') == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
              config['baselines']['bapo'].get('stage_validation_size') == 64,
              isinstance(ar['model'].get('path'),str) and Path(ar['model']['path']).is_absolute(),
              trainer.get('fresh200_official_model') == 'Qwen/Qwen2.5-3B-Instruct']
    if not all(required):
        raise ValueError('Fresh200 requires official fresh Instruct, actual200/horizon201, warmup57, variable 20/10 Adam and independent gate')
    diagnostic=trainer.get('fresh200_diagnostic_only',False)
    if type(diagnostic) is not bool or actor.get('fresh200_diagnostic_only',False) is not diagnostic:
        raise ValueError('Driver/worker diagnostic-only identity mismatch')
    if not diagnostic and (trainer.get('fresh200_diagnostic_role') is not None or trainer.get('fresh200_diagnostic_global_mini') is not None):
        raise ValueError('Production fresh200 forbids diagnostic role/layout flags')
    if diagnostic:
        if (trainer.get('fresh200_diagnostic_only') is not True
                or trainer.get('fresh200_diagnostic_role') not in ('discovery','confirmation')
                or trainer.get('fresh200_diagnostic_global_mini') not in (128,256)
                or actor.get('diagnostic_max_writes') not in (1,2560//trainer.get('fresh200_diagnostic_global_mini',128))
                or actor.get('diagnostic_probe_lr')!=1e-6):
            raise ValueError('Diagnostic bootstrap requires explicit role/layout and hard diagnostic write cap')
    for key in ('resume_full_state', 'resume_permission', 'resume_verify_only', 'resume_from', 'resume_path'):
        if trainer.get(key):
            raise ValueError('Fresh200 forbids old model/Adam/scheduler restoration: '+key)
    return 200, 57


def _valid_skip_event(event):
    total=event.get('expected_optimizer_steps');writes=event.get('optimizer_steps')
    processed=event.get('processed_opportunities');skipped=event.get('skipped_opportunities')
    unprocessed=event.get('unprocessed_opportunities')
    if (any(type(x) is not int or x < 0 for x in (total,writes,processed,skipped,unprocessed))
            or total not in (10,20) or processed != writes+skipped
            or processed+unprocessed != total or event.get('optimizer_opportunities_total') != total
            or type(event.get('optimizer_opportunities_total')) is not int
            or type(event.get('actual_outer')) is not int or not 1 <= event['actual_outer'] <= 200
            or type(event.get('attempt_id')) is not int or event['attempt_id'] < 1
            or type(event.get('attempt_complete')) is not bool
            or type(event.get('update_complete')) is not bool or type(event.get('postabort')) is not bool):
        return False
    status=event.get('status')
    if status in ('COMPLETE','COMPLETE_WITH_SKIPS'):
        if (event['postabort'] or not event['attempt_complete'] or not event['update_complete']
                or processed != total or writes < 1 or unprocessed != 0):return False
        return (skipped == 0 and writes == total) if status == 'COMPLETE' else skipped > 0
    if event['update_complete']:return False
    if status == 'SKIPPED':
        return writes == 0 and not event['postabort'] and event['attempt_complete'] == (processed == total)
    if status == 'ABORTED':
        return event['postabort'] and not event['attempt_complete']
    return False


def scheduler_may_step(event):
    return _valid_skip_event(event) and event.get('status') in ('COMPLETE','COMPLETE_WITH_SKIPS')


def validate_rank_updates(events, actual_outer):
    contract=outer_contract(actual_outer);expected=contract['expected_optimizer_steps']
    if (len(events) != 8 or any(type(e.get('rank')) is not int for e in events)
            or {e.get('rank') for e in events} != set(range(8))):
        raise ValueError('Fresh200 all-rank write ledger missing/duplicate/invalid rank type')
    decision_keys=('actual_outer','attempt_id','status','optimizer_steps','expected_optimizer_steps',
                   'optimizer_opportunities_total','processed_opportunities','skipped_opportunities',
                   'unprocessed_opportunities','attempt_complete','update_complete','postabort')
    first={key:events[0].get(key) for key in decision_keys}
    for event in events:
        if (not _valid_skip_event(event) or event['actual_outer'] != actual_outer
                or event['expected_optimizer_steps'] != expected):
            raise ValueError('Fresh200 invalid opportunity/write/completion identity')
        if {key:event.get(key) for key in decision_keys} != first:
            raise ValueError('Fresh200 rank decisions/counts differ; scheduler admission rejected')
    return scheduler_may_step(events[0])


def verify_fixed_policy_safety_artifact(safety_config):
    """Bind explicit human thresholds; never label them empirically calibrated."""
    import hashlib,json,subprocess,re
    def need(ok,message):
        if not ok:raise ValueError(message)
    need(safety_config.get('threshold_origin') == 'user_fixed', 'Explicit user-fixed threshold origin required')
    need(safety_config.get('enabled') is True and safety_config.get('dual_clip') == 3,
         'BAPO safety and Dual-Clip c3 required')
    need(safety_config.get('target_old_policy_kl') == .2 and safety_config.get('negative_tail_limit') == .2,
         'Actual fixed KL/tail thresholds must both equal user0.2')
    need(safety_config.get('skip_on_policy_guard') is True, 'Policy guard must skip and continue')
    path = Path(safety_config.get('fixed_threshold_artifact') or '')
    need(path.is_absolute() and path.is_file() and path.resolve() == path and not path.is_symlink(),
         'Canonical immutable fixed-threshold document required')
    raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    need(sha == safety_config.get('fixed_threshold_artifact_sha256'), 'Fixed threshold artifact hash mismatch')
    doc = json.loads(raw); source = safety_config.get('source_commit')
    repo = Path(__file__).resolve().parents[2]
    need(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source), 'Exact new source commit required')
    need(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == source,
         'Actual runtime source differs from fixed-threshold binding')
    need(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),
         'Actual fixed-threshold source is dirty')
    need(doc.get('protocol') == 'BAPO_FRESH200_USER_FIXED_SAFETY_V1'
         and doc.get('scope') == 'USER_FIXED_THRESHOLDS_NOT_NATIVE_ACCEPTANCE_NOT_EMPIRICAL_CALIBRATION'
         and doc.get('source_commit') == source and doc.get('source_sha256') == critical_source_hashes(),
         'Fixed threshold provenance/source hash mismatch')
    need(doc.get('thresholds') == {'dual_clip':3,'sampled_old_policy_kl':.2,'negative_advantage_tail_fraction':.2,
                                 'negative_tail_ratio_boundary':3,'kl_log_bound':30.,'on_guard':'skip_and_continue'},
         'Fixed threshold mathematical definition differs')
    need(safety_config.get('kl_log_bound') == 30., 'Fixed finite exponent bound differs')
    need(doc.get('monitor_scopes') == ['full_frozen_rollout','full_current_minibatch'], 'Full policy monitor scopes required')
    need(doc.get('model_revision') == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
         and doc.get('data_manifest_sha256') == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
         'Fixed threshold model/data binding differs')
    need(isinstance(doc.get('human_goal_sha256'),str) and re.fullmatch('[0-9a-f]{64}',doc['human_goal_sha256']),
         'Bound human objective required')
    return .2, sha


def verify_policy_safety_artifact(safety_config):
    """Admit only hashed BAPO discovery plus separate confirmation raw monitors.

    CPU integrity proves bindings, never substitutes for the two CUDA experiments.
    The candidate KL and tail rules remain empirical, not guaranteed bounds.
    """
    if safety_config.get("threshold_origin") == "user_fixed":
        return verify_fixed_policy_safety_artifact(safety_config)
    import hashlib
    import json
    import math
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def require(ok, message):
        if not ok: raise ValueError(message)
    path=Path(safety_config.get('calibration_manifest') or '')
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Immutable absolute BAPO calibration manifest required')
    sha=digest(path)
    require(sha == safety_config.get('calibration_sha256'), 'Calibration hash mismatch')
    artifact=json.loads(path.read_text())
    repo=Path(__file__).resolve().parents[2]
    import subprocess
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==safety_config.get('source_commit'), 'Actual runtime commit differs from calibration')
    source_sha256=critical_source_hashes()
    identity=dict(protocol=PROTOCOL, algorithm='bapo', objective='sequence_mean',
        model_repo='Qwen/Qwen2.5-3B-Instruct', model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        data_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
        source_commit=safety_config.get('source_commit'), source_sha256=source_sha256, dual_clip=3,
        kl_log_bound=float(safety_config['kl_log_bound']), target_old_policy_kl=float(safety_config['target_old_policy_kl']),
        monitor_scopes=['full_frozen_rollout','full_current_minibatch'], trajectory_count=2560, world_size=8,
        optimizer_contract=dict(base_lr=1e-6,initial_lr=1e-6/57,diagnostic_probe_lr=1e-6,betas=[.9,.999],weight_decay=.01,grad_clip=1.,ppo_epochs=1,clip_ratio=.2))
    require(isinstance(identity['source_commit'],str) and len(identity['source_commit'])==40, 'Actual calibration source commit required')
    require(artifact.get('identity')==identity, 'BAPO calibration identity / actual thresholds mismatch')
    records={}; run_ids=[]; native_bindings={}; all_layout_runs=[]
    for role in ('discovery','confirmation'):
        evidence=artifact.get(role,{})
        raw=Path(evidence.get('raw_path',''))
        require(raw.is_absolute() and raw.is_file() and not raw.is_symlink() and digest(raw)==evidence.get('raw_sha256'), 'Missing or changed '+role+' raw BAPO evidence')
        record=json.loads(raw.read_text())
        require(record.get('identity')==identity and record.get('role')==role, role+' actual experiment identity mismatch')
        layout_runs=record.get('layout_run_roots',{})
        require(set(layout_runs)=={'128','256'} and all(isinstance(x,str) and Path(x).is_absolute() for x in layout_runs.values()),
                role+' requires two actual independent fresh diagnostic layout runs')
        layout_runs={int(k):v for k,v in layout_runs.items()}; all_layout_runs.extend(layout_runs.values())
        native_bindings[role]=dict(raw_path=str(raw),raw_sha256=evidence['raw_sha256'],samples=[])
        run_id=record.get('run_id');require(isinstance(run_id,str) and run_id, 'Actual separate run identity missing');run_ids.append(run_id)
        samples=record.get('pre_adam_monitors',[])
        require(samples and len(samples)==(20+10)*4 and
                {(s.get('global_mini_batch_size'),s.get('scope'),s.get('phase'),s.get('diagnostic_write_index')) for s in samples}==
                {(mini,scope,phase,index) for mini in (128,256) for scope in ('full_frozen_rollout','full_current_minibatch') for phase in ('pre_adam','post_adam') for index in range(2560//mini)},
                role+' must cover both 20/10 Adam schedules and full rollout/current-mini scopes')
        for sample in samples:
            mini=sample['global_mini_batch_size']
            expected_rows=2560 if sample.get('scope')=='full_frozen_rollout' else mini
            require(sample.get('expected_optimizer_steps')==2560//mini
                    and sample.get('trajectory_count')==expected_rows and sample.get('phase') in ('pre_adam','post_adam'), 'BAPO monitor scope/count/phase mismatch')
            for field in ('negative_tail_weight_fraction','sampled_old_policy_kl'):
                require(type(sample.get(field)) in (int,float) and math.isfinite(sample[field]), 'Nonfinite calibration monitor')
            require(0<=sample['negative_tail_weight_fraction']<=1, 'Invalid weighted negative tail')
            from .fresh200_native_evidence import verify_native_sample
            replay,proof=verify_native_sample(sample,identity,role,layout_runs)
            native_bindings[role]['samples'].append(dict(scope=sample['scope'],phase=sample['phase'],diagnostic_write_index=sample['diagnostic_write_index'],global_mini_batch_size=mini,replay=replay,proof=proof))
        records[role]=samples
    require(run_ids[0]!=run_ids[1] and len(set(all_layout_runs))==4, 'Both scopes/layouts and confirmation require four separate fresh native runs')
    derived=max(s['negative_tail_weight_fraction'] for s in records['discovery'])
    require(0<=derived<1 and artifact.get('negative_tail_limit')==derived, 'Tail limit must derive from actual discovery weighted-tail maximum')
    require(all(s['negative_tail_weight_fraction']<=derived and s['sampled_old_policy_kl']<=identity['target_old_policy_kl']
                for s in records['confirmation']), 'Separate BAPO confirmation rejects candidate thresholds')
    require(safety_config.get('negative_tail_limit') in (None,derived), 'Actual tail threshold differs from calibration')
    from .fresh200_native_evidence import verify_independent_review
    bindings=dict(identity=identity,discovery=native_bindings['discovery'],confirmation=native_bindings['confirmation'],
                  negative_tail_limit=derived)
    verify_independent_review(artifact.get('independent_review',{}),safety_config,bindings)
    return derived,sha


def verify_sampling_policy_binding(data, safety_config):
    """Compare actual transported sampler log-probs to unchanged HF old policy."""
    import hashlib
    import torch
    if data.meta_info.get('fresh200_evaluation') is True or data.meta_info.get('probability_role')=='greedy_model_logprob_not_sampling_policy':
        raise ValueError('Greedy evaluation model LP cannot bind a training old policy')
    snapshot=data.meta_info.get('policy_snapshot_id')
    audit=data.meta_info.get('sampling_context_audit')
    if not isinstance(snapshot,str) or not snapshot or not audit:
        raise ValueError('Actual sampler snapshot/context proof missing')
    from search_r1.llm_agent.sampling_tokens import audit_batch
    actual_audit=audit_batch(data)
    if actual_audit.get('context_verified') is not True or actual_audit.get('policy_snapshot_id')!=snapshot:
        raise ValueError('Actual sampling turn/context audit rejected')
    journals=data.non_tensor_batch.get('fresh200_turns',[])
    if len(journals)==0 or any(turn.get('effective_stop_provenance',{}).get('probability_role')!='stochastic_sampling_policy' for row in journals for turn in row):
        raise ValueError('Actual stochastic sampler proof required; greedy journals cannot bind training')
    batch=data.batch
    sampled=batch['sampling_log_probs'];mask=batch['sampling_mask'].bool();old=batch['old_log_probs']
    if sampled.shape!=old.shape or mask.shape!=old.shape or not mask.any():
        raise ValueError('Selected sampler policy shape/coverage mismatch')
    loss_mask=batch.get('loss_mask')
    if loss_mask is None or not torch.equal(loss_mask.bool(), mask):
        raise ValueError('Sampler supervised mask differs from actor policy mask')
    if not torch.isfinite(sampled[mask]).all():
        raise ValueError('Actual selected sampler probability transport nonfinite')
    # A NaN must not hide finite parity errors or invalid configured tolerance.
    atol=float(safety_config.get('sampling_logprob_atol', 1e-3))
    finite_old=torch.isfinite(old)
    comparable=mask & finite_old
    if not 0<atol<=1e-3 or not torch.allclose(old[comparable], sampled[comparable], atol=atol, rtol=0.):
        raise ValueError('Actual selected sampler and unchanged HF old policy differ')
    if not finite_old[mask].all():
        from search_r1.baselines.policy_safety import NonfiniteSamplingPolicy
        raise NonfiniteSamplingPolicy('Actual unchanged HF old-policy selected log-probabilities nonfinite')
    return hashlib.sha256(snapshot.encode()).hexdigest()


def fresh200_lr_factor(completed_outers):
    if type(completed_outers) is not int or not 0 <= completed_outers <= 200:
        raise ValueError('Invalid completed outer scheduler counter')
    # The next actual outer update uses 1/57..57/57, then constant.
    return min(1., (completed_outers + 1) / 57.)


def verify_official_model_assets(model_path):
    import json
    import hashlib
    root=Path(model_path)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('Fresh200 requires verified local official model assets')
    manifest=root/'VERIFIED-MODEL-MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44':
        raise ValueError('Fresh200 official model manifest differs from verified origin')
    doc=json.loads(manifest.read_text())
    if doc.get('repo')!='Qwen/Qwen2.5-3B-Instruct' or doc.get('revision')!='aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Fresh200 official model repo/revision mismatch')
    names=set()
    for entry in doc['files']:
        member=root/entry['file']
        if member.resolve().parent!=root.resolve() or member.name in names or not member.is_file():
            raise ValueError('Official model member invalid/missing/duplicate')
        names.add(member.name)
        h=hashlib.sha256()
        with member.open('rb') as handle:
            for block in iter(lambda:handle.read(1048576),b''):h.update(block)
        if member.stat().st_size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
            raise ValueError('Official model bytes changed')
    if not {'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}<=names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('Official complete model/tokenizer inventory required')
    actual={p.name for p in root.iterdir() if p.is_file() and p.name!='VERIFIED-MODEL-MANIFEST.json'}
    if actual!=names:
        raise ValueError('Unmanifested/missing official model assets')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def critical_source_hashes():
    import hashlib
    root=Path(__file__).resolve().parents[2]
    names=('search_r1/baselines/objectives.py','search_r1/baselines/policy_safety.py','search_r1/baselines/gradient_norm.py','search_r1/baselines/optimizer_transaction.py',
        'search_r1/baselines/fresh200_contract.py','search_r1/baselines/fresh200_native_evidence.py',
        'search_r1/baselines/fresh200_driver.py','search_r1/baselines/fresh200_gate.py','search_r1/baselines/main.py',
        'verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/ray_trainer.py',
        'search_r1/llm_agent/generation.py','search_r1/llm_agent/sampling_tokens.py',
        'search_r1/llm_agent/sampling_trace.py','search_r1/llm_agent/sampling_capture.py',
        'verl/workers/rollout/vllm_rollout/vllm_rollout.py','search_r1/baselines/controller.py',
        'search_r1/baselines/advantages.py','search_r1/baselines/bapo.py','search_r1/baselines/bapo_validation.py',
        'search_r1/baselines/rollout.py','verl/trainer/main_ppo.py',
        'verl/trainer/config/ppo_bapo_fresh200.yaml','verl/trainer/config/ppo_baselines.yaml',
        'verl/trainer/config/ppo_trainer.yaml','verl/utils/dataset/rl_dataset.py',
        'verl/utils/tokenizer.py','verl/utils/torch_functional.py','verl/utils/model.py','verl/utils/fsdp_utils.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
