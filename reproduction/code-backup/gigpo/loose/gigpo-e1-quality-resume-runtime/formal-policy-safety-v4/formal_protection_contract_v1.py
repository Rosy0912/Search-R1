"""Local production control draft. No GPU, READY creation, submission or threshold selection."""
import hashlib
import gzip
import json
import math
from pathlib import Path
import re

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
MODEL = str(ROOT / 'assets/qwen2.5-3b-instruct-aa8e725')
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CHECKPOINT = str(ROOT / 'runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25')
MANIFEST = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
CALIBRATION_SCOPE = 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'


def require(value, reason):
    if not value: raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read_bound(record):
    require(isinstance(record, dict), 'Required actual proof is null/missing')
    path = Path(record['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing/nonoriginal proof')
    require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Actual proof bytes/SHA changed')
    return json.loads(path.read_text())


def validate_closure(records):
    require(isinstance(records, list) and records, 'Empty actual proof closure')
    require(len({r['path'] for r in records}) == len(records), 'Duplicate proof closure paths')
    for record in records:
        path = Path(record['path'])
        require(path.is_absolute() and path.is_file() and path.stat().st_size == record['bytes']
                and sha(path) == record['sha256'], 'Actual closure bytes/SHA changed')


def validate_actual_native(native, source, guard):
    """No transfer of3267/source52 acceptance to a future compatible-prefix source."""
    require(isinstance(native, dict) and native.get('status') == 'PASS', 'No actual eight-GPU native acceptance')
    require(native['code_commit'] == source and native['restore_step'] == 25
            and native['restore_manifest_sha256'] == MANIFEST and native['guard_identity'] == guard,
            'Native source/restore/actualguard identity differs')
    def original(path_key, sha_key, compressed=False):
        path = Path(native[path_key]); require(path.is_absolute() and path.is_file()
            and not path.is_symlink() and sha(path) == native[sha_key], 'Actual native original SHA changed')
        raw = path.read_bytes()
        return json.loads(gzip.decompress(raw) if compressed else raw)
    result = original('result_path', 'result_sha256')
    review = original('review_path', 'review_sha256')
    terminal = original('terminal_path', 'terminal_sha256')
    physical = original('physical_payload_path', 'physical_payload_sha256', compressed=True)
    original_binding = original('native_binding_path', 'native_binding_sha256')
    from formal_native_evidence_v1 import validate_native_payload
    validate_native_payload(native, result, review, original_binding, guard, terminal, physical,
                            source, native['native_helpers'])
    job = str(native['job_id'])
    if source == '52befe968c856162d1bc79cd0171bb38309f4f9e':
        require(job == '3267', 'This source requires actual accepted native3267')


def validate_production_artifact(artifact, source, config):
    require(artifact['status'] == 'PASS' and artifact['scope'] == CALIBRATION_SCOPE
            and artifact['rank_count'] == 8 and artifact['source_commit'] == source
            and artifact['restore_manifest_sha256'] == MANIFEST, 'No same-source healthy25 calibration')
    for key in ('dual_clip', 'kl_log_bound', 'target_old_policy_kl', 'negative_tail_limit'):
        require(artifact[key] == config[key], 'Calibration/configuration differs: ' + key)
    kl, tail = config['target_old_policy_kl'], config['negative_tail_limit']
    require(isinstance(kl, (int, float)) and not isinstance(kl, bool) and math.isfinite(kl) and kl > 0
            and isinstance(tail, (int, float)) and not isinstance(tail, bool) and math.isfinite(tail) and 0 <= tail <= 1,
            'Unaccepted/null/invalid production thresholds')
    proofs = artifact['actual_proofs']
    required = {'probability', 'sampler_position_semantics', 'calibration', 'quality', 'state', 'terminal', 'threshold_derivation'}
    require(set(proofs) == required, 'Missing/extra actual production acceptance roles')
    accepted = {key: read_bound(proofs[key]) for key in required}
    identity = artifact['actual_identity']
    require(identity['source_commit'] == source and identity['common_manifest_sha256'] == DATA
            and identity['restore_manifest_sha256'] == MANIFEST and identity['node'] == 'h100-8'
            and identity['official_model_revision'] == REVISION, 'Calibration provenance differs')
    for role, proof in accepted.items():
        require(proof['status'] == 'PASS' and proof['source_commit'] == source
                and proof['actual_identity'] == identity, 'Independent ' + role + ' belongs to different actual run')
    probability = accepted['probability']
    require(probability['prefix_mismatch_count'] == probability['token_mismatch_count'] == 0
            and probability['hf_position_and_mask_identity'] is True
            and probability['actual_trajectories'] == 2560 and probability['post_sampling_live_pause_verified'] is True,
            'Actual sampler/HF probability identity not accepted')
    position = accepted['sampler_position_semantics']
    require(position['scope'] == 'INSTALLED_RUNNER_DEFAULT_POSITION_CONTRACT_AND_ACTUAL_LITERAL_PREFIX'
            and position['sampler_runner_positions_observed'] is False
            and position['actual_engine_literal_prefix_observed'] is True
            and position['default_causal_positions_and_attention_semantics_independently_verified'] is True
            and position['actual_runner_model_config_and_active_branch_verified'] is True
            and position['active_position_branch'] == 'NON_MROPE_DEFAULT_ABSOLUTE_CAUSAL'
            and sha(position['installed_runner_path']) == position['installed_runner_sha256'],
            'Missing exact installed backend position contract evidence; no runner tensor was captured')
    from formal_probability_v1 import validate_tolerances
    validate_tolerances(probability['probability_tolerances'])
    require(accepted['calibration']['actual_adam_per_rank'] == 10
            and accepted['calibration']['actual_gradient_micro_per_rank'] == 160
            and accepted['calibration']['retained_training_updates_contributed'] == 0,
            'Calibration counts fabricated/formal contribution')
    require(accepted['quality']['before_after_fixed512_and64'] is True
            and accepted['quality']['raw_text_actions_rewards_bound'] is True
            and accepted['quality']['healthy_restored25_baseline'] is True
            and accepted['quality']['after_calibration_no_confirmed_quality_collapse'] is True
            and accepted['quality']['actual_qa_counts'] ==
                dict(before512=512, before64=64, after512=512, after64=64),
            'Calibration output health not independently accepted')
    require(accepted['state']['all8_worker_complete_state_actual_reload'] is True
            and accepted['state']['full_hf_semantic_reload'] is True
            and accepted['state']['driver_saved_state_physical_verified'] is True,
            'Actual worker/HF or saveddriver state missing')
    # Do not claim driver26 replay: this is an explicitly separate, optional fact.
    require(isinstance(accepted['state']['driver26_actual_replayed'], bool), 'Driver replay truth value omitted')
    require(accepted['terminal']['slurm_completed_0_0'] is True
            and accepted['terminal']['owned_cleanup_empty'] is True
            and accepted['terminal']['job_and_scope_cgroup_removed'] is True,
            'Calibration terminal/resources not independently accepted')
    require(accepted['threshold_derivation']['target_old_policy_kl'] == kl
            and accepted['threshold_derivation']['negative_tail_limit'] == tail
            and accepted['threshold_derivation']['raw_monitor_statistics_bound'] is True,
            'Thresholds do not have actual GPU calibration derivation')


def validate_same_calibration_flash_backend(binding,artifact,actual_source):
    """Same actual calibration run, not a Flash import/CPU/static review."""
    from formal_backend_v1 import CALIBRATION_RUNTIME_SHA,validate_actual_flash_backend
    backend=binding['rollout_backend']
    require(backend['requested']=='FLASH_ATTN'
        and backend['calibration_runtime_manifest_sha256']==CALIBRATION_RUNTIME_SHA
        and backend['gate_before_original_fit_restore_baseline_adam'] is True,
        'Production backend does not inherit exact calibrationv9 propagation/gate')
    accepted=read_bound(backend['calibration_backend_acceptance'])
    require(accepted['status']=='PASS'
        and accepted['scope']=='ACTUAL_CALIBRATION_EIGHT_RANK_FLASH_BACKEND_METADATA'
        and accepted['source_commit']==actual_source and accepted['actual_identity']==artifact['actual_identity']
        and accepted['calibration_runtime_manifest_sha256']==CALIBRATION_RUNTIME_SHA
        and accepted['all8_runner_and36_attention_layers'] is True,
        'No same-source/run independently accepted actual8 Flash backend original')
    raw=read_bound(accepted['actual_backend_record'])
    require(raw['status']=='PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY'
        and raw['probability_accepted'] is False,
        'Backend metadata may not pretend it alone establishes probability parity')
    checked=validate_actual_flash_backend(raw['actual_observations'],binding,
        artifact['actual_identity']['binding_sha256'])
    require(raw==checked,'Actual accepted backend metadata differs from strict complete8x36 validation')
    return accepted


def validate_hook_startup_review(hook,review,source,artifact,native):
    """Startup physical/static scope; first actual hooks acceptance is gate30."""
    require(review['status']=='PASS' and review['source_commit']==source
        and review['scope']=='GIGPO_EXTERNAL_FORMAL_HOOK_STATIC_AND_PHYSICAL_READY'
        and review['helper_manifest_sha256']==hook['helper_manifest_sha256']
        and review['actual_native_acceptance_sha256']==native['review_sha256']
        and review['actual_calibration_acceptance_sha256']==artifact['actual_proofs']['calibration']['sha256']
        and review['metadata_only_hooks_and_first_adam_collective_static_verified'] is True
        and review['expected_rollout_backend']=='FLASH_ATTN'
        and review['extra_diagnostic_adam_required'] is False
        and review['actual_formal_GPU_hook_accepted'] is False
        and hook['actual_formal_hook_acceptance_stage']=='SAME_PROCESS26_TO30_BEFORE31',
        'Startup static review cannot waive native/calibration or fabricate actual formal hooks acceptance')


def admit_binding(binding, actual_source):
    require(binding['status'] == 'REVIEWED_FOR_PRODUCTION', 'Draft is not productionREADY')
    require(re.fullmatch(r'[0-9a-f]{40}', str(actual_source)) and binding['code_commit'] == actual_source,
            'Actual accepted source is null/different')
    validate_closure(binding['files']); validate_closure(binding['evidence'])
    require(binding['node'] == 'h100-8' and binding['restore_checkpoint'] == CHECKPOINT
            and binding['restore_manifest_sha256'] == MANIFEST and binding['common_manifest_sha256'] == DATA,
            'Wrong formal source checkpoint/node/data')
    policy = binding['policy_safety']
    require(policy['enabled'] is True and policy['mode'] == 'production'
            and policy['dual_clip'] == 3. and policy['kl_log_bound'] == 30.
            and policy['monitor_first_global_rows'] == 256, 'Wrong protection execution scope')
    artifact = read_bound(binding['production_artifact'])
    require(policy['calibration_evidence_path'] == binding['production_artifact']['path']
            and policy['calibration_evidence_sha256'] == binding['production_artifact']['sha256'],
            'Actor did not bind actual production artifact')
    validate_production_artifact(artifact, actual_source, policy)
    validate_same_calibration_flash_backend(binding,artifact,actual_source)
    validate_actual_native(binding['native_acceptance'], actual_source, binding['guard_identity'])
    for item in binding['guard_identity'].values():
        require(sha(item['path']) == item['sha256'], 'Actual accepted guard/module changed')
    require(binding['formal_budget'] == dict(retained25=25, new_outer275=275, stop300=300,
            mini256=256, adam_per_outer10=10, micro_per_outer160=160, warmup57=57), 'Formal budget differs')
    hook = binding['probability_identity']
    require(hook['scope'] == 'EVERY_REAL_OUTER_BEFORE_FIRST_ADAM'
            and hook['actual_global_trajectories'] == 2560 and hook['actual_rank_count'] == 8
            and hook['prefix_token_position_mask_mismatch_limit'] == 0,
            'Missing every-outer exact structural hook')
    probability = read_bound(hook['acceptance'])
    require(hook['acceptance'] == artifact['actual_proofs']['probability']
            and hook['tolerances'] == probability['probability_tolerances'],
            'Hook did not use independently accepted same-source probability evidence')
    require(hook['pad_token_id'] == probability['actual_pad_token_id']
            and hook['eos_token_id'] == probability['actual_eos_token_id'],
            'Hook tokenizer identity differs from actual official probability proof')
    startup_review=read_bound(hook['startup_static_physical_review'])
    validate_hook_startup_review(hook,startup_review,actual_source,artifact,binding['native_acceptance'])
    return artifact
