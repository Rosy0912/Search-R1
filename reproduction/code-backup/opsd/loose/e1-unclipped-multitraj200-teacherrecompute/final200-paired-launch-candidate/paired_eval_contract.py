"""Pure schema/template preparation, not a publisher, verifier or GPU launcher.

These functions require supplied evidence identities but do not authenticate
their files or signatures. Only future independently qualified original inputs
may be used to publish a stage. There is deliberately no production main.
"""
import copy
import hashlib
import json
from pathlib import PurePosixPath
import re

TRAIN = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
EVAL = 'c5bf4c8db33562a259fa227a940dc69543901b98'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
REPO = ROOT + '/repo-final200-eval-c5bf4c8db335'
BASE = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
HF200 = ROOT + '/runs/formal-3304/checkpoints/actor/global_step_200'
MODELS = {'official_base': BASE, 'actual_hf200': HF200}
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MODEL_MANIFEST = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COMMON = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
FINAL = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
IDS = '4e9df91a75beba40c3316576f6da049851e736df4a1e5c873f4f0cdb883fd4b1'
COUNTS = {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267, 'hotpotqa': 7405,
          '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}
INVENTORY = 'fd8fa25d8321a4cd11b9faa5d8fdbd5f1692172a9e51ea5eba447c3565b8cc0d'
CLOSURE = '378277254a4b7d92236dfc4a8618a9de5fc49cf3ed16664ac41f21b918d1b20e'
STRICT = '7dff811b01b302f45c729ee12ae9ee22bb561b359f24fcd0d37352192db2d075'
STRICT_COMMON = 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
STRICT_PATH = ROOT + '/tools/final200-prerequisites-85f014e167c8/verify_actual200_prerequisites.py'
ROLES = ('official_base', 'actual_hf200')
INACTIVE = ('run_root', 'train_log_path', 'receipt_path', 'public_key_path',
            'public_key_sha256', 'binding_manifest_path', 'binding_manifest_sha256')
REQUIRED_SOURCES = {
    'search_r1/baselines/final200_evaluation.py', 'search_r1/baselines/sharedholdout.py',
    'search_r1/baselines/corrected_evaluation.py', 'search_r1/baselines/evaluation.py',
    'search_r1/baselines/bapo.py', 'search_r1/qa_protocol.py', 'verl/trainer/main_ppo.py',
    'verl/utils/reward_score/qa_em.py', 'search_r1/baselines/main.py',
    'verl/trainer/ppo/ray_trainer.py', 'search_r1/llm_agent/generation.py',
    'search_r1/baselines/actor.py', 'search_r1/baselines/distillation.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'verl/utils/dataset/rl_dataset.py',
    'search_r1/baselines/corrected_runtime_capture.py', 'verl/workers/fsdp_workers.py',
    'verl/single_controller/base/worker.py', 'verl/single_controller/base/decorator.py',
    'verl/single_controller/ray/base.py'}
# Exact authored scalar budget independently read in actual c5bf CPU step2.
# The only runtime substitution is the actual same-node service URL18084.
BUDGET = {'actor_rollout_ref.rollout.dtype': 'bfloat16',
          'actor_rollout_ref.rollout.enforce_eager': True,
          'actor_rollout_ref.rollout.free_cache_engine': True,
          'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
          'actor_rollout_ref.rollout.ignore_eos': False,
          'actor_rollout_ref.rollout.max_num_batched_tokens': 8192,
          'actor_rollout_ref.rollout.max_num_seqs': 1024,
          'actor_rollout_ref.rollout.n': 1,
          'actor_rollout_ref.rollout.temperature': 1,
          'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
          'actor_rollout_ref.rollout.top_k': -1, 'actor_rollout_ref.rollout.top_p': 1,
          'algorithm.no_think_rl': False, 'baselines.seed': 42,
          'data.max_obs_length': 500, 'data.max_prompt_length': 4096,
          'data.max_response_length': 500, 'data.max_start_length': 2048,
          'data.val_batch_size': 64, 'do_search': True, 'force_final_answer': False,
          'max_turns': 4, 'retriever.topk': 3, 'retriever.url': 'http://127.0.0.1:18084/retrieve',
          'trainer.n_gpus_per_node': 8, 'trainer.nnodes': 1,
          'native_validation_sampling': {'best_of': 1, 'do_sample': False, 'min_p': 0.0,
              'n': 1, 'temperature': 0.0, 'top_k': -1, 'top_p': 1.0}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def same(value, expected, message):
    require(canonical(value) == canonical(expected), message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid typed integer: ' + label)


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'Actual full SHA required')


def reference(value, expected=None):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Actual original reference required')
    require(type(value['path']) is str and PurePosixPath(value['path']).is_absolute() and
            '..' not in PurePosixPath(value['path']).parts, 'Absolute non-traversing original path required')
    sha(value['sha256'])
    if expected is not None:
        same(value, expected, 'Bound reference differs')


def selected(config, dotted):
    value = config
    for key in dotted.split('.'):
        require(type(value) is dict and key in value, 'Missing resolved field: ' + dotted)
        value = value[key]
    return value


def validate_resolved_config(config):
    expected = {'trainer.val_only': True, 'trainer.val_before_train': True,
        'trainer.total_training_steps': 0, 'trainer.smoke_only': False,
        'baselines.method': 'opsd', 'baselines.holdout_eval_only': False,
        'baselines.opsd_e1.enabled': False, 'baselines.opsd_e1.gate_required': False,
        'baselines.opsd_e1.profile': None, 'baselines.corrected_evaluation.enabled': False,
        'baselines.final200_evaluation.enabled': True, 'baselines.local_runtime_only': True,
        'baselines.local_runtime_host': 'h100-10', 'data.val_data_num': None,
        'data.return_raw_chat': True, 'actor_rollout_ref.rollout.n_agent': 1,
        'actor_rollout_ref.actor.use_kl_loss': False, 'actor_rollout_ref.actor.kl_loss_coef': 0,
        'actor_rollout_ref.actor.entropy_coeff': 0, 'algorithm.kl_ctrl.kl_coef': 0,
        'baselines.distillation.teacher_path': BASE,
        'baselines.distillation.teacher_device': 'cuda'}
    expected.update({k: v for k, v in BUDGET.items() if k != 'native_validation_sampling'})
    expected.update({'baselines.opsd_e1.' + name: None for name in INACTIVE})
    for key, target in expected.items():
        require(type(selected(config, key)) is type(target) and selected(config, key) == target,
                'Wrong actual eval-only resolved field: ' + key)
    return copy.deepcopy(BUDGET)


def validate_control_inputs(value):
    """Validate supplied schema; caller still must authenticate every original."""
    require(type(value) is dict and value.get('kind') == 'ROOT_QUALIFIED_PAIRED_FINAL200_STAGE_CONSTRUCTION_INPUTS',
            'Future complete original inputs remain BLOCKED')
    same(value['source_commit'], EVAL, 'Wrong eval source')
    same(value['training_source_commit'], TRAIN, 'Wrong stored training source')
    execution = value['allocation']
    require(type(execution) is dict, 'Actual new allocation missing')
    job, step = execution['job_id'], execution['step_id']
    require(type(job) is str and re.fullmatch('[1-9][0-9]*', job) is not None and job != '3304',
            'A separate actual scheduler allocation is required')
    require(type(step) is str and re.fullmatch('(?:[0-9]+|batch|extern)', step) is not None, 'Actual step missing')
    require(execution['node'] == 'h100-10' and type(execution['uid']) is int and execution['uid'] == 0,
            'Actual owned same-node allocation required')
    integer(execution['pid'], 'allocation PID', 1)
    integer(execution['start_ticks'], 'allocation start', 1)
    pattern = r'(?:^|/)job[_-]' + re.escape(job) + r'(?:\.scope)?/step[_-]' + re.escape(step) + r'(?:\.scope)?(?:/|$)'
    require(type(execution['cgroup']) is str and re.search(pattern, execution['cgroup'], re.M),
            'Actual job and step must join the same cgroup path')
    uuids = execution['gpu_uuids']
    require(type(uuids) is list and len(uuids) == len(set(uuids)) == 8 and
            all(type(u) is str and re.fullmatch('GPU-[0-9a-fA-F-]+', u) for u in uuids), 'Eight actual UUIDs required')
    reference(value['root_pre_gpu_review'])
    reference(value['layered_training_receipt'])
    require(value['layered_training_receipt']['sha256'] != 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc',
            'Signed first5 is not an actual200 layered receipt')
    reference(value['strict_verifier'], {'path': STRICT_PATH, 'sha256': STRICT})
    reference(value['strict_common'], {'path': str(PurePosixPath(STRICT_PATH).parent / 'common.py'), 'sha256': STRICT_COMMON})
    reference(value['source_inventory'])
    same(value['source_inventory']['sha256'], INVENTORY, 'Actual c5bf857 source inventory required')
    reference(value['native_import_closure'])
    same(value['native_import_closure']['sha256'], CLOSURE, 'Actual c5bf94 closure required, not old211')
    require(type(value['native_module_count']) is int and value['native_module_count'] == 94, 'Actual module cardinality differs')
    reference(value['input_manifest'])
    same(value['common_manifest_sha256'], COMMON, 'Common dataset differs')
    same(value['final_parquet_sha256'], FINAL, 'Actual final parquet differs')
    same(value['ordered_question_ids_sha256'], IDS, 'Ordered final cohort differs')
    same(value['source_counts'], COUNTS, 'Actual final source counts differ')
    source_files = value['source_files']
    require(type(source_files) is dict and REQUIRED_SOURCES <= set(source_files), 'Actual generation source pins missing')
    for name, pin in source_files.items():
        require(type(name) is str and name.endswith('.py') and not PurePosixPath(name).is_absolute() and
                '..' not in PurePosixPath(name).parts, 'Unsafe Python source path')
        sha(pin)
    require(type(value['roles']) is dict and set(value['roles']) == set(ROLES), 'Both actual roles are required')
    budgets = []
    for role in ROLES:
        entry = value['roles'][role]
        same(entry['model_directory'], MODELS[role], 'Actual qualified model role/path differs')
        expected = dict(kind='STRICT_ACTUAL_TRAINING3304_HF200_PREREQUISITES', training_commit=TRAIN,
            training_job_id='3304', actual_outer_updates=200, hf_step=200, final_questions=51713,
            receipt_sha256=value['layered_training_receipt']['sha256'], model_role=role, model_directory=MODELS[role])
        same(entry['strict_proof'], expected, 'Actual strict proof must contain exact nine fields')
        reference(entry['strict_stdout_original'])
        reference(entry['strict_wait_original'])
        require(entry['strict_waited'] is True and type(entry['strict_returncode']) is int and entry['strict_returncode'] == 0,
                'Actual strict CLI must truly finish0')
        files = entry['model_files']
        require(type(files) is dict and {'config.json', 'tokenizer.json'} <= set(files) and
                any(n.endswith('.safetensors') for n in files), 'Fresh complete model member map required')
        for name, pin in files.items():
            require(type(name) is str and PurePosixPath(name).name == name and name not in ('.', '..'), 'Unsafe model member')
            sha(pin)
        # Complete membership/fullSHA still comes from qualified originals, not this count.
        if role == 'official_base':
            same(files.get('VERIFIED-MODEL-MANIFEST.json'), MODEL_MANIFEST, 'Official manifest member omitted')
        budget = validate_resolved_config(entry['resolved_config'])
        same(selected(entry['resolved_config'], 'actor_rollout_ref.model.path'), MODELS[role], 'Resolved loaded model role differs')
        budgets.append(budget)
    same(budgets[0], budgets[1], 'Paired runtime budgets differ')
    return {'scope': 'SOFTWARE_SCHEMA_ONLY_NOT_ORIGINAL_OR_SIGNATURE_OR_GPU_QUALIFICATION',
            'generation_budget': budgets[0], 'allocation_job_id': job}


def proposed_role_stage(value, role):
    """Return source-compatible proposed content, not a reviewed/published stage."""
    require(role in ROLES, 'Unsupported final200 model role')
    checked = validate_control_inputs(value)
    budget = checked['generation_budget']
    return dict(kind='REVIEWED_FINAL200_PAIRED_EVAL_STAGE', model_role=role, training_commit=TRAIN,
        training_job_id='3304', actual_outer_updates=200, eval_commit=EVAL,
        eval_job_id=value['allocation']['job_id'], common_manifest_sha256=COMMON,
        official_base_revision=REVISION, official_base_manifest_sha256=MODEL_MANIFEST,
        model_directory=MODELS[role], model_files=copy.deepcopy(value['roles'][role]['model_files']),
        source_files=copy.deepcopy(value['source_files']),
        input_manifest_sha256=value['input_manifest']['sha256'],
        generation_budget=budget, generation_budget_sha256=hashlib.sha256(canonical(budget)).hexdigest(),
        training_hf_receipt=copy.deepcopy(value['layered_training_receipt']),
        strict_training_hf_verifier=copy.deepcopy(value['strict_verifier']),
        stage_construction_scope='PROPOSED_CONTENT_NOT_EXTERNALLY_REVIEWED_OR_PUBLISHED',
        actual_allocation=copy.deepcopy(value['allocation']), root_pre_gpu_review=copy.deepcopy(value['root_pre_gpu_review']),
        actual_c5bf_source_inventory=copy.deepcopy(value['source_inventory']),
        actual_c5bf_native_import_closure=copy.deepcopy(value['native_import_closure']))


def inactive_overrides():
    return ['baselines.opsd_e1.' + name + '=null' for name in INACTIVE]


def blocked_template():
    return dict(version=1, kind='BLOCKED_PAIRED_FINAL200_EVAL_LAUNCH_SOFTWARE_TEMPLATE',
        status='BLOCKED_ACTUAL200_SIGNED_LAYERED_RECEIPT_NEW_ALLOCATION_AND_STAGES_REQUIRED',
        training_source_commit=TRAIN, stored_training_job_id='3304', eval_source_commit=EVAL,
        expected_eval_repository=REPO, expected_roles=list(ROLES), expected_generation_budget=copy.deepcopy(BUDGET),
        mandatory_inactive_hydra_overrides=inactive_overrides(),
        actual_layered_training_receipt=None, actual_independent200_qualification=None,
        actual_eval_job_id=None, actual_allocation=None, actual_model200=None, actual_input_manifest=None,
        actual_root_pre_gpu_review=None, actual_stages={role: None for role in ROLES},
        actual_role_outputs={role: None for role in ROLES}, actual_terminal=None, actual_resource_release=None,
        actual_training_checkpoint31_before=None, actual_training_checkpoint31_after=None,
        training_checkpoint_guard_scope='ORIGINAL_MODEL_ADAM_SCHEDULER_RNG_DRIVER_FILES_FULLSHA_UNCHANGED',
        independent_eval_rng_scope='FIXED_SEED_ACTUAL_SAMPLING_RECORDED_OWN_RNG_CONSUMPTION_ALLOWED',
        metric_scope='ENV_DONE_PERMISSIVE_COMPLETION_SEPARATE_FROM_REPAIRED_STRICT_TERMINAL_EM_F1_FORMAT',
        existing_native_cpu_closure_sha256=CLOSURE, existing_native_module_count=94,
        existing_strict200_manifest_sha256='85f014e167c8ca0fbb0ea4cf74bf2065e012d8efa1ec89c38b1f7240b09a62eb',
        original_strict_publisher_outcome='UNKNOWN_WAIT_FALSE_RETURNCODE_NULL_DO_NOT_RETRY',
        execute_allowed=False)
