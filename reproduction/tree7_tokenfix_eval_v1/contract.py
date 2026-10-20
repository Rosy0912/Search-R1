"""Fail closed on unaccepted training, wrong protocol, or reused repeat slots."""
import json
from pathlib import Path
import re
from reproduction.official_eval_v2.protocol import PROFILE, SCIENCE, digest, sha, validate_rows

BASE = Path('/plm-dhw/zhangjunming/Workspace/Recursive-Tree-GRPO-Tree7-TokenFix-20261020-v1')
COMMON = Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1')
ROOT = BASE / 'evaluations/tree179-tokenfix-commonv2-v1'
MODEL = BASE / 'exports/tree7-tokenfix-final179-v1/huggingface'
UID = '009919da-8569-49dd-8772-a61062c779d7'
TRAIN_SOURCE = '14a6afa983460e1a76f7c17b3a77b11be6fc4088e6ab212e8c27774c627eecb5'
TRAIN_PLAN = '902214efde6ebda1012fde72106d33df91c50a1ae63ffe954066892773746a0a'
DATA_MANIFEST = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
FINAL_JSONL = 'a1b95766beecb66753b5ee4c3dd2552d6a47e375c4c117200248dae6b6fc08d5'
ID_GOLD = '4953085d4fdaf823fe355972d8dceae9e167e27d264f0f6b7f8bcf64473c6845'
ALLOCATION = dict(gpus=8, cpus=64, memory_gib=600, machine_type='n2ls.ie.i60b',
                  gpu_model='A100', gpu_memory_gib=80)
RUNTIME = '/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/native-env/bin/python'
FROZEN = {
    'reproduction/official_eval_v2/driver.py': 'f3b81d83ba60bca1b3d5bd5415b34b8ee80ffc7df2c418c7797096014a8a3f93',
    'reproduction/official_eval_v2/protocol.py': '4da23b50ad4e80b4da17c2a17c677ebbd74b6b8eacd5e9740e76d4bc46ed92ed',
    'reproduction/official_eval_v2/port_recovery.py': 'ec586e53ca75b9f2e21cd7f68001445234768e81acd967fb7dbe4238d63776d5',
    'search_r1/baselines/preflight.py': '2c918d0ef82281d0cfc8a16a198d9338348197f33886300b1098ba34d8b7d7d4',
}


def job_name(repeat):
    assert type(repeat) is int and repeat in (1, 2, 3)
    return 'zhangjunming-tg7-tf1-r' + str(repeat)


def model_identity(files):
    return digest(dict(method='official_treegrpo_sampled_token_prefix_repair_v1', training_uid=UID, actual_iterations=179,
                       directory_label=180, files=files, evaluation_profile=PROFILE))


def validate_lineage(native, hf, native_sha):
    assert native['uid'] == hf['uid'] == UID
    assert native['status'] == 'CPU_ACTOR_NATIVE_FILE_TENSOR_ADAM_RNG_SCHEDULER_AUDIT_PASSED'
    assert native['source_manifest_sha256'] == TRAIN_SOURCE and native['plan_sha256'] == hf['plan_sha256'] == TRAIN_PLAN
    assert native['not_old_tree179_model'] and native['condition']=='common_initialization_tree7_sampled_token_prefix_repair_v1'
    assert native['logical_parameters']==339 and native['complete_iterations_history']==179
    assert native['actual_iterations'] == 179 and native['directory_label'] == 180
    assert native['actual_adam_per_rank'] == 1432 and native['world_size'] == 8
    assert native['local_small_backup_verified'] and len(native['files']) == 24
    assert len(native['workers']) == 8 and {w['rank'] for w in native['workers']} == set(range(8))
    assert all(w['adam_min'] == w['adam_max'] == 1432 and w['scheduler']['last_epoch'] == 179
               for w in native['workers'])
    assert hf['status'] == 'FINAL_HF_OFFICIAL_CPU_MERGE_ALL_TENSORS_AND_METADATA_ACCEPTED'
    assert hf['actual_training_iterations'] == 179 and hf['final_directory_label'] == 180
    assert hf['local_small_backup_verified'] and hf['native_acceptance_sha256'] == native_sha
    assert hf['target'] == str(MODEL) and hf['cuda_initialized'] is False
    assert hf['route_sha256']=='15ce94973d0b3cdc8dcba2c4d3c879422e0449ba7a9f3fb350c748cf5e946470'
    assert hf['native_route_sha256']=='ae83cf3655613242deabd47a36613f4e5acdbc8a3b1bc9df5e44e5a743b61ad0'
    assert hf['full_hf_cpu_reload']['logical_parameters']==339 and hf['full_hf_cpu_reload']['elements']==7615616512
    assert hf['full_hf_cpu_reload']['all_logical_parameters_bit_exact_to_exported_files'] and hf['full_hf_cpu_reload']['tied_storage'] is False
    comparison = hf['parameter_comparison']
    assert comparison['exact_native_bf16_projection_all_parameters']
    assert comparison['logical_parameters'] == 339 and comparison['rank_parameter_comparisons'] == 2712
    assert comparison['parameters_different_from_initialization']
    files = {n: r['sha256'] for n, r in hf['files'].items()}
    assert {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(files)
    assert any(n.endswith('.safetensors') for n in files)
    # The export acceptance uses the original sorted-json file identity convention.
    import hashlib
    assert hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == hf['hf_file_identity_sha256']
    return files


def validate_plan(plan, native, hf, reload):
    assert plan['profile'] == PROFILE and plan['science'] == SCIENCE
    assert plan['kind'] == 'trained-treegrpo' and plan['repeat_count'] == 3
    assert plan['job'] == job_name(plan['repeat'])
    assert plan['root'] == str(ROOT / ('r' + str(plan['repeat'])))
    assert plan['training_uid'] == UID and plan['training_source'] == TRAIN_SOURCE
    assert plan['actual_training_iterations'] == 179 and plan['final_directory_label'] == 180
    assert plan['model'] == str(MODEL) and plan['model_revision'] is None
    files = validate_lineage(native, hf, plan['evidence']['native']['sha256'])
    assert plan['model_files'] == files and plan['model_identity'] == model_identity(files)
    assert plan['hf_file_identity_sha256'] == hf['hf_file_identity_sha256']
    assert reload['status'] == 'ACTUAL_HF_CPU_RELOAD_ALL339_PARAMETERS_AND_COMMONV2_51713_PROMPTS_VERIFIED'
    assert reload['uid'] == UID and reload['hf_file_identity_sha256'] == hf['hf_file_identity_sha256']
    assert reload['hf_acceptance_sha256'] == plan['evidence']['hf']['sha256']
    assert reload['all_loaded_parameters_exact'] and reload['logical_parameters_compared'] == 339
    assert reload['parameter_elements'] == 7615616512 and reload['cuda_initialized'] is False
    assert reload['all51713_prompt_tokenization_verified'] and reload['max_prompt_tokens'] <= 2048
    assert reload['data_sha256'] == plan['data_sha256'] and reload['id_gold_identity'] == plan['id_gold_identity']
    assert reload['profile'] == PROFILE and reload['science'] == SCIENCE
    assert plan['allocation'] == ALLOCATION and plan['no_auto_retry'] is True
    assert (plan['gpus'], plan['cpus'], plan['memory_gib']) == (8, 64, 600)
    assert re.fullmatch('[a-f0-9]{40}', plan['source_commit'])
    assert plan['source'] == str(BASE / 'sources' / plan['source_commit'])
    assert plan['data'] == str(ROOT / 'final.jsonl')
    assert plan['data_sha256'] == FINAL_JSONL and plan['id_gold_identity'] == ID_GOLD
    assert plan['data_manifest_sha256'] == DATA_MANIFEST
    assert plan['retriever_receipt'] == str(COMMON / 'receipts/pj-retriever-ready.json')
    for name, value in FROZEN.items():
        assert plan['source_files'][name] == value
    return Path(plan['root'])


def verify(plan):
    evidence = {}
    for name in ('native', 'hf', 'reload'):
        record = plan['evidence'][name]
        assert record['path'] == str(ROOT / (name + '-acceptance.json'))
        assert sha(record['path']) == record['sha256']
        evidence[name] = json.loads(Path(record['path']).read_text())
    root = validate_plan(plan, **evidence)
    for name, value in plan['source_files'].items():
        assert sha(Path(plan['source']) / name) == value, name
    for name, value in plan['model_files'].items():
        assert sha(MODEL / name) == value, name
    assert sha(plan['data']) == plan['data_sha256']
    rows = [json.loads(v) for v in Path(plan['data']).open()]
    validate_rows(rows)
    assert digest([{k: r[k] for k in ('final_index', 'question_id', 'source', 'gold_answers')}
                   for r in rows]) == plan['id_gold_identity']
    assert sha(plan['retriever_receipt']) == plan['retriever_receipt_sha256']
    ready = json.loads(Path(plan['retriever_receipt']).read_text())
    from urllib.parse import urlsplit
    u = urlsplit(ready['url'])
    assert u.scheme == 'http' and u.port == 29147 and u.path in ('', '/', '/retrieve')
    assert not u.query and not u.fragment
    assert plan['retriever_url'] == ready['url'].rstrip('/') + ('' if u.path == '/retrieve' else '/retrieve')
    return rows, root


def validate_gate(plan, submission, gate):
    assert submission['job'] == gate['actual']['name'] == plan['job']
    assert submission['uid'] == gate['actual']['uid']
    assert gate['actual']['state'] == 'RUNNING' and gate['actual']['resources'] == ALLOCATION
    assert submission['plan_sha256'] == gate['plan_sha256']
    assert gate['source'] == plan['source_commit']
    return gate['actual']

INITIAL_MODEL=Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-20261019-v1/models/base/a09a35458c702b33eeacc393d103063234e8bc28')
TRAINING_JOB='zhangjunming-tree7-tokenfix-v1'
TRAINING_ROOT=BASE/'run'
PROTECTED_FREE_BYTES=7500000000000

def cpu_headroom():
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 for limit,used in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
  if Path(limit).exists() and Path(used).exists():
   t=Path(limit).read_text().strip()
   if t!='max':mem=min(mem,max(0,int(t)-int(Path(used).read_text())))
 return mem
