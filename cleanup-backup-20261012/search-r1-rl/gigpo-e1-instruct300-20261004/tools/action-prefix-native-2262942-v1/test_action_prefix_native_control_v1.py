"""New pure CPU control counterexamples; no native/model/GPU execution."""
import ast
import copy
import hashlib
from pathlib import Path
import pytest
from action_prefix_native_control_v1 import SCOPE, OFFICIAL_REVISION, validate_binding, validate_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_fixture(tmp_path):
    names = ('native_run_safety_v1.py', 'native_action_prefix_gpu_v1.py', 'action_prefix_native_control_v1.py',
             'test_action_prefix_native_control_v1.py', 'native_safety_v1.sbatch', 'owned_runtime_resume_v1.py')
    files = []
    for name in names:
        path = tmp_path / name; path.write_text('explicit CPU closure fixture')
        files.append(dict(path=str(path), sha256=sha(path)))
    driver = tmp_path / 'driver.pt'; driver.write_bytes(b'CPU fixture, not a real checkpoint')
    files.append(dict(path=str(driver), sha256=sha(driver)))
    evidence = tmp_path / 'static-review.json'; evidence.write_text('CPU fixture, not actual review')
    return dict(status='REVIEWED_FOR_NATIVE_ONLY', validation_scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY',
        native_scope=SCOPE, code_commit='a' * 40, official_revision=OFFICIAL_REVISION,
        native_initialization='OFFICIAL_MODEL_INITIALIZATION', native_actual_restore=False,
        restore_step=25, restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
        restore_checkpoint='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25',
        formal_updates_contributed=0, allowed_safety_modes=['native_fixture'], native_test_threshold_sets=[[100., 1.]],
        expected_optimizer_steps_per_outer=10, expected_gradient_micro_batches_per_rank=160,
        global_row_denominator=256, production_denominator_per_mini=32, gpu_count=8, node='h100-8',
        production_probability_tolerance=None, production_thresholds=None, helpers=str(tmp_path),
        files=files, native_config_driver=dict(path=str(driver), sha256=sha(driver)),
        official_model_path='/mnt/data/explicit-official-model-fixture',
        evidence=[dict(path=str(evidence), sha256=sha(evidence))])


def result_fixture():
    outer = dict(optimizer_writes=10, gradient_micro_forwards=160, before_monitor_forwards=160,
                 after_monitor_forwards=160, original_local_trajectory_rows=320, mini_denominators=[32.] * 10)
    return dict(status='PASS', rank=5, world=8, native_scope=SCOPE, native_actual_restore=False,
        formal_updates_contributed=0, official_revision=OFFICIAL_REVISION, official_3b=True,
        actual_fsdp=True, flash_attention2=True, remove_padding=True, manual_offload=True,
        prefix_4096_action500_forward_backward=True, old_ref_rank4_action16_no_grad_maximum=True,
        mapping_formula_value_and_gradient_equal=True,
        native_hf_semantic_reload_equal=True, state_save_restore_next_outer_equal=True,
        outer_records=[copy.deepcopy(outer) for _ in range(3)], scheduler_at_retained_boundary=2,
        adam_at_retained_boundary=20, rank5_refusals={name:dict(all8_refused=True, extra_adam=0)
            for name in ('temperature', 'output_shape')}, hf_engine_probability_acceptance=False,
        production_thresholds_selected=False)


def test_complete_explicit_cpu_binding_fixture_passes(tmp_path):
    assert validate_binding(binding_fixture(tmp_path), tmp_path, sha)


@pytest.mark.parametrize('field,value', [('status', 'READY'), ('code_commit', None), ('official_revision', 'main'),
    ('native_actual_restore', True), ('formal_updates_contributed', 1), ('allowed_safety_modes', ['production']),
    ('expected_optimizer_steps_per_outer', 20), ('expected_gradient_micro_batches_per_rank', 80),
    ('global_row_denominator', 32), ('production_denominator_per_mini', 256), ('node', 'a800'),
    ('gpu_count', 4), ('production_probability_tolerance', .01), ('production_thresholds', {'kl': .01}),
    ('restore_step', 80), ('restore_manifest_sha256', '0' * 64)])
def test_changed_scope_budget_or_unbound_acceptance_refused(tmp_path, field, value):
    binding = binding_fixture(tmp_path); binding[field] = value
    with pytest.raises(ValueError):
        validate_binding(binding, tmp_path, sha)


@pytest.mark.parametrize('failure', ['missing_helper', 'duplicate_helper', 'changed_helper', 'driver_unbound', 'changed_review'])
def test_exact_new_helper_driver_review_closure(tmp_path, failure):
    value = binding_fixture(tmp_path)
    if failure == 'missing_helper':
        value['files'].pop(0)
    elif failure == 'duplicate_helper':
        value['files'].append(copy.deepcopy(value['files'][0]))
    elif failure == 'changed_helper':
        Path(value['files'][0]['path']).write_text('changed')
    elif failure == 'driver_unbound':
        value['files'].pop()
    else:
        Path(value['evidence'][0]['path']).write_text('changed review')
    with pytest.raises(ValueError):
        validate_binding(value, tmp_path, sha)


def test_result_complete_fixture_only():
    assert validate_result(result_fixture(), 5)


@pytest.mark.parametrize('failure', ['tiny_model', 'skipped_maximum', 'wrong_micro', 'skipped_probe',
    'extra_adam_after_refusal', 'missing_allrank_refusal', 'claimed_probability_acceptance', 'claimed_restore',
    'wrong_state_boundary', 'missing_hf_reload'])
def test_actual_result_cannot_inflate_or_omit_scope(failure):
    value = result_fixture()
    if failure == 'tiny_model': value['official_3b'] = False
    elif failure == 'skipped_maximum': value['prefix_4096_action500_forward_backward'] = False
    elif failure == 'wrong_micro': value['outer_records'][0]['gradient_micro_forwards'] = 80
    elif failure == 'skipped_probe': value['outer_records'][1]['after_monitor_forwards'] = 0
    elif failure == 'extra_adam_after_refusal': value['rank5_refusals']['temperature']['extra_adam'] = 1
    elif failure == 'missing_allrank_refusal': value['rank5_refusals']['output_shape']['all8_refused'] = False
    elif failure == 'claimed_probability_acceptance': value['hf_engine_probability_acceptance'] = True
    elif failure == 'claimed_restore': value['native_actual_restore'] = True
    elif failure == 'wrong_state_boundary': value['adam_at_retained_boundary'] = 30
    else: value['native_hf_semantic_reload_equal'] = False
    with pytest.raises(ValueError):
        validate_result(value, 5)


def test_script_static_scope_has_no_old_regression_or_tiny_or_sampler_run():
    folder = Path(__file__).parent
    wrapper = (folder / 'native_run_safety_v1.py').read_text()
    gpu = (folder / 'native_action_prefix_gpu_v1.py').read_text()
    script = (folder / 'native_safety_v1.sbatch').read_text()
    ast.parse(wrapper); ast.parse(gpu)
    assert 'tests/baselines' not in wrapper and 'native_gpu_safety_oracle' not in wrapper
    assert 'native_gpu_safety_restore' not in wrapper and 'Qwen2Config(' not in gpu
    assert '--exclusive' in script and '--no-requeue' in script and '--nodelist=h100-8' in script
    assert 'validate_inherited_gpu_leases(8)' in wrapper and 'owned.cleanup()' in wrapper
    assert 'NUMERIC_FIXTURE_PHASE' not in wrapper
    assert gpu.count("update('") == 3
    assert 'native_actual_restore=False' in gpu and 'formal_updates_contributed=0' in gpu


def test_hf_auxiliary_validation_restores_all_rng_without_ignoring_replay_fields():
    text = (Path(__file__).parent / 'native_action_prefix_gpu_v1.py').read_text()
    before = text.index('auxiliary_rng = recovery.rng_state()')
    construction = text.index('reloaded_hf = AutoModelForCausalLM.from_pretrained')
    restore = text.index('recovery.restore_rng(auxiliary_rng)')
    check = text.index('assert equal_state(auxiliary_rng, recovery.rng_state())')
    expected = text.index("update('expected_native_outer2')")
    assert before < construction < restore < check < expected
    assert "assert equal_state(expected, replayed)" in text


def test_post_save_full_state_gather_loads_cuda_shards_first():
    text = (Path(__file__).parent / 'native_action_prefix_gpu_v1.py').read_text()
    save = text.index("worker.save_checkpoint(str(output / 'global_step_1'))")
    gather = text.index('with FSDP.state_dict_type(model')
    assert '\nload()\n' in text[save:gather]


def test_new_rank4_old_ref_maximum_uses_real_compute_log_prob_original_path():
    text = (Path(__file__).parent / 'native_action_prefix_gpu_v1.py').read_text()
    assert 'maximum_old = fixture(4, maximum=True)' in text
    assert 'old_maximum = actor.compute_log_prob(maximum_old)' in text
    assert "forward_calls[0]['rows'] == 16" in text
    assert "forward_calls[0]['grad_enabled'] is False" in text


@pytest.mark.parametrize('flag', [False, None])
def test_new_old_ref_maximum_must_be_actual_result(flag):
    result = result_fixture()
    result['old_ref_rank4_action16_no_grad_maximum'] = flag
    with pytest.raises(ValueError):
        validate_result(result, 5)
