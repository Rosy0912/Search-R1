"""Local synthetic schema fixtures only; no actual future job or stage proof."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
EVAL_REPO = WORK / 'opsd-final200-paired-eval-source'
NATIVE = HERE.parent / 'c5bf-native-cpu-actual-originals-v1/raw/c5bf-native-cpu-preparation-once-v1/native-cpu-preparation.json'


def byte_module(path, name):
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


c = byte_module(HERE / 'paired_eval_contract.py', 'fixture_paired_eval_contract')
legacy = byte_module(EVAL_REPO / 'search_r1/baselines/corrected_evaluation.py', 'fixture_frozen_legacy')


def pin(label):
    return hashlib.sha256(('CPU_SYNTHETIC_FIXTURE_NOT_ACTUAL:' + label).encode()).hexdigest()


def ref(label):
    return {'path': '/cpu-synthetic-fixture/' + label, 'sha256': pin(label)}


def assign(config, key, value):
    parts = key.split('.')
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = copy.deepcopy(value)


def resolved(role):
    config = {}
    for key, value in c.BUDGET.items():
        if key != 'native_validation_sampling':
            assign(config, key, value)
    extra = {'trainer.val_only': True, 'trainer.val_before_train': True,
        'trainer.total_training_steps': 0, 'trainer.smoke_only': False,
        'baselines.method': 'opsd', 'baselines.holdout_eval_only': False,
        'baselines.opsd_e1.enabled': False, 'baselines.opsd_e1.gate_required': False,
        'baselines.opsd_e1.profile': None, 'baselines.corrected_evaluation.enabled': False,
        'baselines.final200_evaluation.enabled': True, 'baselines.local_runtime_only': True,
        'baselines.local_runtime_host': 'h100-10', 'data.val_data_num': None,
        'data.return_raw_chat': True, 'actor_rollout_ref.rollout.n_agent': 1,
        'actor_rollout_ref.actor.use_kl_loss': False, 'actor_rollout_ref.actor.kl_loss_coef': 0,
        'actor_rollout_ref.actor.entropy_coeff': 0, 'algorithm.kl_ctrl.kl_coef': 0,
        'baselines.distillation.teacher_path': c.BASE, 'baselines.distillation.teacher_device': 'cuda',
        'actor_rollout_ref.model.path': c.MODELS[role]}
    for key, value in extra.items():
        assign(config, key, value)
    for name in c.INACTIVE:
        assign(config, 'baselines.opsd_e1.' + name, None)
    return config


def fixture():
    # This identity exists only inside local test memory, never in the blocked plan.
    job = '990001'
    value = dict(kind='ROOT_QUALIFIED_PAIRED_FINAL200_STAGE_CONSTRUCTION_INPUTS',
        fixture_scope='CPU_SYNTHETIC_NOT_REAL_ORIGINAL_OR_SIGNATURE',
        source_commit=c.EVAL, training_source_commit=c.TRAIN,
        allocation=dict(job_id=job, step_id='0', node='h100-10', uid=0, pid=4321,
            start_ticks=12345, cgroup='1:freezer:/slurm_h100-10/uid_0/job_' + job + '/step_0\n',
            gpu_uuids=['GPU-00000000-0000-0000-0000-00000000000' + str(n) for n in range(8)]),
        root_pre_gpu_review=ref('root-pre-gpu.json'), layered_training_receipt=ref('layered200.json'),
        strict_verifier={'path': c.STRICT_PATH, 'sha256': c.STRICT},
        strict_common={'path': str(Path(c.STRICT_PATH).parent / 'common.py'), 'sha256': c.STRICT_COMMON},
        source_inventory={'path': c.ROOT + '/tools/final200-eval-source-publication-v1/source-inventory.actual.json',
            'sha256': c.INVENTORY},
        native_import_closure={'path': c.ROOT + '/runs/formal-3304/evidence/c5bf-native-cpu-preparation-once-v1/native-cpu-preparation.json.closure.json',
            'sha256': c.CLOSURE}, native_module_count=94,
        input_manifest=ref('ordered-inputs.json'), common_manifest_sha256=c.COMMON,
        final_parquet_sha256=c.FINAL, ordered_question_ids_sha256=c.IDS,
        source_counts=copy.deepcopy(c.COUNTS),
        source_files={name: hashlib.sha256((EVAL_REPO / name).read_bytes()).hexdigest()
                      for name in c.REQUIRED_SOURCES}, roles={})
    for role in c.ROLES:
        files = {'config.json': pin(role + '-config'), 'tokenizer.json': pin(role + '-tokenizer'),
                 'model-00001-of-00001.safetensors': pin(role + '-weights')}
        if role == 'official_base':
            files['VERIFIED-MODEL-MANIFEST.json'] = c.MODEL_MANIFEST
        value['roles'][role] = dict(model_directory=c.MODELS[role], model_files=files,
            strict_proof=dict(kind='STRICT_ACTUAL_TRAINING3304_HF200_PREREQUISITES', training_commit=c.TRAIN,
                training_job_id='3304', actual_outer_updates=200, hf_step=200, final_questions=51713,
                receipt_sha256=value['layered_training_receipt']['sha256'], model_role=role,
                model_directory=c.MODELS[role]),
            strict_stdout_original=ref(role + '-strict-stdout'),
            strict_wait_original=ref(role + '-strict-wait.json'),
            strict_waited=True, strict_returncode=0, resolved_config=resolved(role))
    return value


class ContractTests(unittest.TestCase):
    def test_blocked_template_has_no_future_job_stage_or_pass(self):
        value = c.blocked_template()
        self.assertIs(value['execute_allowed'], False)
        for key in ('actual_layered_training_receipt', 'actual_independent200_qualification',
            'actual_eval_job_id', 'actual_allocation', 'actual_model200', 'actual_input_manifest',
            'actual_root_pre_gpu_review', 'actual_terminal', 'actual_resource_release',
            'actual_training_checkpoint31_before', 'actual_training_checkpoint31_after'):
            self.assertIsNone(value[key])
        self.assertEqual(value['actual_stages'], dict.fromkeys(c.ROLES))
        self.assertEqual(value['actual_role_outputs'], dict.fromkeys(c.ROLES))
        self.assertIn('OWN_RNG_CONSUMPTION_ALLOWED', value['independent_eval_rng_scope'])
        self.assertIn('ENV_DONE_PERMISSIVE', value['metric_scope'])
        with self.assertRaises(ValueError):
            c.proposed_role_stage(value, 'official_base')

    def test_proposed_stages_use_identical_native_budget_encoding(self):
        value = fixture()
        stages = [c.proposed_role_stage(value, role) for role in c.ROLES]
        for stage, role in zip(stages, c.ROLES):
            actual = legacy.budget_snapshot(OmegaConf.create(value['roles'][role]['resolved_config']))
            self.assertEqual(c.canonical(stage['generation_budget']), legacy.encoded(actual))
            legacy.verify_budget(OmegaConf.create(value['roles'][role]['resolved_config']), stage)
            self.assertEqual(stage['stage_construction_scope'], 'PROPOSED_CONTENT_NOT_EXTERNALLY_REVIEWED_OR_PUBLISHED')
        self.assertEqual(stages[0]['generation_budget_sha256'], stages[1]['generation_budget_sha256'])
        self.assertEqual(stages[0]['input_manifest_sha256'], stages[1]['input_manifest_sha256'])
        self.assertNotEqual(stages[0]['model_directory'], stages[1]['model_directory'])

    def test_budget_matches_actual_cpu_authored_default_except_real_service_url(self):
        raw = NATIVE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '2b42362994a7f51f11044cd305b01d11e143a45b57beae9cafc9ef6d49250d14')
        authored = json.loads(raw)['configuration']['authored_default_budget']
        self.assertEqual(authored['retriever.url'], 'http://127.0.0.1:8000/retrieve')
        authored['retriever.url'] = c.BUDGET['retriever.url']
        self.assertEqual(c.canonical(authored), c.canonical(c.BUDGET))
        self.assertEqual(sum(c.COUNTS.values()), 51713)

    def test_all_seven_inactive_fields_must_be_resolved_null(self):
        self.assertEqual(len(c.inactive_overrides()), 7)
        for name in c.INACTIVE:
            with self.subTest(field=name):
                config = resolved('official_base')
                assign(config, 'baselines.opsd_e1.' + name, '${oc.env:OLD3304_IDENTITY}')
                with self.assertRaisesRegex(ValueError, 'resolved field'):
                    c.validate_resolved_config(config)

    def test_stage_contains_only_driver_manifest_pin_never_inline_targets(self):
        value = fixture()
        # The real c5bf input reader owns target aliases; no model input is built here.
        value['driver_only_targets'] = ['CPU_SYNTHETIC_GOLD_SENTINEL']
        stage = c.proposed_role_stage(value, 'official_base')
        self.assertNotIn('CPU_SYNTHETIC_GOLD_SENTINEL', c.canonical(stage).decode())
        self.assertNotIn('driver_only_targets', stage)
        self.assertEqual(stage['input_manifest_sha256'], value['input_manifest']['sha256'])

    def test_old_job_and_cross_path_cgroup_cannot_supply_new_allocation(self):
        for mutate in (lambda v: v['allocation'].update(job_id='3304'),
            lambda v: v['allocation'].update(cgroup='1:freezer:/job_990001/step_00\n'),
            lambda v: v['allocation'].update(cgroup='1:freezer:/job_990001/step_1\n2:pids:/job_4/step_0\n')):
            value = fixture(); mutate(value)
            with self.assertRaises(ValueError): c.validate_control_inputs(value)
        value = fixture(); value['allocation']['cgroup'] = '1:freezer:/job_990001.scope/step_0.scope/task_0\n'
        c.validate_control_inputs(value)

    def test_bool_uid_pid_clock_and_cardinality_are_rejected(self):
        for key in ('uid', 'pid', 'start_ticks'):
            value = fixture(); value['allocation'][key] = False
            with self.assertRaises(ValueError): c.validate_control_inputs(value)
        for count in (True, 211):
            value = fixture(); value['native_module_count'] = count
            with self.assertRaises(ValueError): c.validate_control_inputs(value)

    def test_exact_strict_nine_fields_and_true_waited_zero(self):
        for mutation in ('extra', 'old500', 'unwaited', 'failed', 'bool-exit'):
            value = fixture(); entry = value['roles']['actual_hf200']
            if mutation == 'extra': entry['strict_proof']['status'] = 'PASS'
            if mutation == 'old500': entry['strict_proof']['hf_step'] = 500
            if mutation == 'unwaited': entry['strict_waited'] = False
            if mutation == 'failed': entry['strict_returncode'] = 1
            if mutation == 'bool-exit': entry['strict_returncode'] = False
            with self.assertRaises(ValueError): c.validate_control_inputs(value)

    def test_wrong_role_model_source_cohort_and_training_receipt_rejected(self):
        changes = (lambda v: v['roles']['actual_hf200'].update(model_directory=c.HF200.replace('200', '5')),
            lambda v: v.update(source_commit=c.TRAIN),
            lambda v: v.update(ordered_question_ids_sha256=pin('wrong-ordered-ids')),
            lambda v: v['source_counts'].update(nq=3609),
            lambda v: v['native_import_closure'].update(sha256=pin('old211')),
            lambda v: v['layered_training_receipt'].update(sha256='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'))
        for change in changes:
            value = fixture(); change(value)
            with self.assertRaises(ValueError): c.validate_control_inputs(value)
        with self.assertRaises(ValueError): c.proposed_role_stage(fixture(), 'hf500')

    def test_actual_budget_changes_no_pg_and_unknown_sources_fail_closed(self):
        for key, wrong in (('data.max_obs_length', 501), ('retriever.topk', 4),
            ('actor_rollout_ref.actor.use_kl_loss', True), ('actor_rollout_ref.rollout.temperature', 1.0),
            ('trainer.total_training_steps', 1)):
            value = fixture(); assign(value['roles']['actual_hf200']['resolved_config'], key, wrong)
            with self.assertRaises(ValueError): c.validate_control_inputs(value)
        for name in ('../other.py', '/other.py', 'unknown.json'):
            value = fixture(); value['source_files'][name] = pin('invalid-source')
            with self.assertRaises(ValueError): c.validate_control_inputs(value)

    def test_complete_membership_remains_external_but_manifest_omission_is_rejected(self):
        value = fixture()
        del value['roles']['official_base']['model_files']['VERIFIED-MODEL-MANIFEST.json']
        with self.assertRaises(ValueError): c.validate_control_inputs(value)
        value = fixture(); value['roles']['actual_hf200']['model_files']['../model.safetensors'] = pin('outside')
        with self.assertRaises(ValueError): c.validate_control_inputs(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
