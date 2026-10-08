"""Pure CPU binding/result schema for a new limited official3B GPU native.

No READY/source SHA is invented here. The root must bind the finalized source,
complete installed environment/assets/helpers and an independent native review.
"""
from pathlib import Path
import re


SCOPE = 'RAW_ACTION_PREFIX_PACKING_OFFICIAL_3B_NATIVE_ONLY'
OFFICIAL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_record_closures(files, evidence):
    """Keep raw-path uniqueness and exact identity across references/aliases.

    A full environment may bind two symlink names for the same physical file.
    That is allowed only when their byte count and SHA agree. Neither a repeated
    raw path inside a list nor a conflicting identity may be deduplicated away.
    """
    by_raw_path, by_resolved_path = {}, {}
    for label, records in (('files', files), ('evidence', evidence)):
        require(isinstance(records, list) and records, 'Missing bound ' + label)
        seen = set()
        for item in records:
            require(isinstance(item, dict) and set(item) >= {'path', 'bytes', 'sha256'},
                    'Incomplete bound identity in ' + label)
            raw = item['path']
            require(isinstance(raw, str) and Path(raw).is_absolute() and '..' not in Path(raw).parts,
                    'Invalid bound path in ' + label)
            require(raw not in seen, 'Duplicate raw bound path in ' + label + ': ' + raw)
            seen.add(raw)
            require(type(item['bytes']) is int and item['bytes'] >= 0
                    and re.fullmatch(r'[0-9a-f]{64}', str(item['sha256'])) is not None,
                    'Invalid bound bytes/SHA in ' + label)
            identity = {key: item[key] for key in ('path', 'bytes', 'sha256')}
            require(raw not in by_raw_path or by_raw_path[raw] == identity,
                    'Conflicting cross-list bound identity: ' + raw)
            by_raw_path[raw] = identity
            physical = str(Path(raw).resolve())
            contents = (item['bytes'], item['sha256'])
            require(physical not in by_resolved_path or by_resolved_path[physical] == contents,
                    'Conflicting resolved alias identity: ' + physical)
            by_resolved_path[physical] = contents
    return True


def validate_binding(value, helpers, sha):
    require(value.get('status') == 'REVIEWED_FOR_NATIVE_ONLY', 'Native independent review missing')
    require(value.get('validation_scope') == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY', 'Wrong safety fixture scope')
    require(value.get('native_scope') == SCOPE, 'Wrong action-prefix native scope')
    require(re.fullmatch(r'[0-9a-f]{40}', str(value.get('code_commit'))) is not None, 'Final source commit missing')
    require(value.get('official_revision') == OFFICIAL_REVISION, 'Official model revision changed')
    require(value.get('native_initialization') == 'OFFICIAL_MODEL_INITIALIZATION', 'Not official initialization')
    require(value.get('native_actual_restore') is False, 'Native cannot claim healthy25 restore')
    # Existing policy_safety native authorization names the future calibration
    # origin. These fields do NOT claim this constructor-only native loaded it.
    require(value.get('restore_step') == 25 and value.get('restore_manifest_sha256') ==
            '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
            and value.get('restore_checkpoint') == '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25',
            'Future calibration origin identity missing')
    require(value.get('formal_updates_contributed') == 0, 'Native cannot contribute training updates')
    require(value.get('allowed_safety_modes') == ['native_fixture'], 'Production/calibration mode forbidden')
    require(value.get('native_test_threshold_sets') == [[100., 1.]], 'Wrong declared test-only thresholds')
    require(value.get('expected_optimizer_steps_per_outer') == 10
            and value.get('expected_gradient_micro_batches_per_rank') == 160
            and value.get('global_row_denominator') == 256
            and value.get('production_denominator_per_mini') == 32, 'Trajectory budget changed')
    require(value.get('gpu_count') == 8 and value.get('node') == 'h100-8', 'Wrong node/GPU scope')
    require(value.get('production_probability_tolerance') is None
            and value.get('production_thresholds') is None, 'Native must not select production thresholds')
    helpers = Path(helpers).resolve()
    require(Path(value.get('helpers', '')).resolve() == helpers, 'Isolated helper path mismatch')
    files = value.get('files')
    require(isinstance(files, list) and files, 'Full file closure missing')
    evidence = value.get('evidence')
    validate_record_closures(files, evidence)
    for name in ('native_run_safety_v1.py', 'native_action_prefix_gpu_v1.py',
                 'action_prefix_native_control_v1.py', 'test_action_prefix_native_control_v1.py',
                 'native_safety_v1.sbatch', 'owned_runtime_resume_v1.py'):
        path = helpers / name
        entries = [item for item in files if Path(item['path']) == path]
        require(len(entries) == 1 and sha(path) == entries[0]['sha256'], 'New helper omitted/changed: ' + name)
    driver = value.get('native_config_driver', {})
    require(isinstance(driver.get('path'), str) and driver['path'].endswith('/driver.pt'), 'Actual config driver missing')
    require(re.fullmatch(r'[0-9a-f]{64}', str(driver.get('sha256'))) is not None, 'Config driver SHA missing')
    require(any(item['path'] == driver['path'] and item['sha256'] == driver['sha256'] for item in files),
            'Config driver not in full closure')
    require(sha(driver['path']) == driver['sha256'], 'Config driver changed')
    require(isinstance(value.get('official_model_path'), str) and value['official_model_path'].startswith('/mnt/data/'),
            'Official model path missing')
    require(isinstance(evidence, list) and evidence, 'Independent evidence missing')
    for item in evidence:
        require(sha(item['path']) == item['sha256'], 'Independent evidence changed')
    return True


def validate_result(record, rank):
    require(record.get('status') == 'PASS' and record.get('rank') == rank
            and record.get('world') == 8, 'Incomplete actual rank result')
    require(record.get('native_scope') == SCOPE and record.get('native_actual_restore') is False
            and record.get('formal_updates_contributed') == 0, 'Native scope inflated')
    require(record.get('official_revision') == OFFICIAL_REVISION and record.get('official_3b') is True,
            'Not actual official3B decoder')
    require(record.get('actual_fsdp') is True and record.get('flash_attention2') is True
            and record.get('remove_padding') is True and record.get('manual_offload') is True,
            'Actual execution path missing')
    require(record.get('prefix_4096_action500_forward_backward') is True, 'Maximum packing not executed')
    require(record.get('old_ref_rank4_action16_no_grad_maximum') is True, 'Old/ref rank4 action16 maximum not executed')
    require(record.get('mapping_formula_value_and_gradient_equal') is True, 'New mapping/formula oracle missing')
    require(record.get('native_hf_semantic_reload_equal') is True, 'Native HF save/reload missing')
    require(record.get('state_save_restore_next_outer_equal') is True, 'Next outer replay differs')
    rows = record.get('outer_records', [])
    require(len(rows) == 3 and all(row['optimizer_writes'] == 10 and row['gradient_micro_forwards'] == 160
            and row['before_monitor_forwards'] == 160 and row['after_monitor_forwards'] == 160
            and row['original_local_trajectory_rows'] == 320 and row['mini_denominators'] == [32.] * 10 for row in rows),
            'Actual trajectory control counts differ')
    require(record.get('scheduler_at_retained_boundary') == 2
            and record.get('adam_at_retained_boundary') == 20, 'Native state boundary count differs')
    refuse = record.get('rank5_refusals', {})
    require(set(refuse) == {'temperature', 'output_shape'} and all(item.get('all8_refused') is True
            and item.get('extra_adam') == 0 for item in refuse.values()), 'Single-rank refusal scope incomplete')
    require(record.get('hf_engine_probability_acceptance') is False
            and record.get('production_thresholds_selected') is False, 'Native misrepresented as probability acceptance')
    return True
