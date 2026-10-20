"""After final export acceptance: exclusive CPU reload and fixed-three plan preparation.

This command has no scheduler submission and never initializes CUDA. An existing
prepare directory is an incident to inspect, not an instruction to retry.
"""
import argparse
import json
import os
from pathlib import Path
import time
from reproduction.official_eval_v2.driver import once, search
from reproduction.official_eval_v2.protocol import prompt, question_from_old_prompt, TOTAL
from .contract import *


def compare_loaded(model, target, expected_count=339, expected_elements=7615616512):
    import torch
    from safetensors import safe_open
    loaded = model.state_dict()
    assert len(loaded) == expected_count and sum(v.numel() for v in loaded.values()) == expected_elements
    observed = set()
    for path in sorted(Path(target).glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as stream:
            for name in stream.keys():
                assert name not in observed and name in loaded
                tensor = stream.get_tensor(name)
                assert tensor.dtype == loaded[name].dtype == torch.bfloat16
                assert torch.isfinite(loaded[name]).all().item()
                assert torch.equal(tensor, loaded[name]), name
                observed.add(name)
    assert observed == set(loaded), 'Missing, duplicated, or unexpected exported parameter'
    return dict(logical_parameters_compared=len(observed), parameter_elements=sum(v.numel() for v in loaded.values()),
                all_loaded_parameters_exact=True)


def build_plan(repeat, source, source_files, native, hf, reload, evidence, data_sha, identity, original_sha, receipt, endpoint):
    files = validate_lineage(native, hf, evidence['native']['sha256'])
    plan = dict(profile=PROFILE, science=SCIENCE, kind='trained-treegrpo', repeat=repeat, repeat_count=3,
                root=str(ROOT / ('r' + str(repeat))), job=job_name(repeat),
                source=str(BASE / 'sources' / source), source_commit=source, source_files=source_files,
                model=str(MODEL), model_revision=None, model_files=files, model_identity=model_identity(files),
                hf_file_identity_sha256=hf['hf_file_identity_sha256'], training_uid=UID, training_source=TRAIN_SOURCE,
                actual_training_iterations=179, final_directory_label=180, evidence=evidence,
                data=str(ROOT / 'final.jsonl'), data_sha256=data_sha, id_gold_identity=identity,
                original_data_sha256=original_sha, data_manifest_sha256=DATA_MANIFEST,
                retriever_receipt=str(receipt), retriever_receipt_sha256=sha(receipt), retriever_url=endpoint,
                gpus=8, cpus=64, memory_gib=600, allocation=ALLOCATION, no_auto_retry=True)
    validate_plan(plan, native, hf, reload)
    return plan


def run(bundle):
    import importlib.metadata as metadata
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from vllm.engine.arg_utils import EngineArgs
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    assert metadata.version('vllm') == '0.6.3' and metadata.version('torch').split('+')[0] == '2.4.0'
    bundle = json.loads(Path(bundle).read_text())
    assert bundle['uid'] == UID and bundle['actual_scheduler_state'] == 'COMPLETED'
    assert bundle['worker_exit_code'] == '0' and bundle['completed_iterations'] == 179
    source = bundle['source_commit']
    src = BASE / 'sources' / source
    assert src == Path(__file__).resolve().parents[2]
    for name, value in bundle['source_files'].items():
        assert sha(src / name) == value, name
    for name, value in FROZEN.items():
        assert bundle['source_files'][name] == value
    paths = {name: Path(bundle[name]['path']) for name in ('native', 'hf')}
    for name, path in paths.items():
        assert sha(path) == bundle[name]['sha256']
    native, hf = [json.loads(paths[name].read_text()) for name in ('native', 'hf')]
    files = validate_lineage(native, hf, bundle['native']['sha256'])
    for name, value in files.items():
        assert sha(MODEL / name) == value, name
    # Read the immutable original training receipt/worker return, independently
    # of the caller's SDK terminal claim. No live source is changed.
    training_root = TRAINING_ROOT
    submission = json.loads((training_root / 'submission.json').read_text())
    assert submission['uid'] == UID and submission['condition'] == 'common_initialization_tree7_sampled_token_prefix_repair_v1' and submission['plan_sha256'] == TRAIN_PLAN
    assert sha(submission['plan']) == TRAIN_PLAN and (training_root / 'exit.code').read_text().strip() == '0'
    assert json.loads((training_root / 'audit/progress.json').read_text())['completed_upstream_iteration'] == 179
    assert cpu_headroom()>20*1024**3, 'Insufficient actual CPU headroom; no namespace created'
    assert __import__('os').statvfs(BASE).f_bavail*__import__('os').statvfs(BASE).f_frsize>PROTECTED_FREE_BYTES
    assert not ROOT.exists(), 'Inspect prior prepare claim/partial, never replay'
    ROOT.mkdir(parents=True, exist_ok=False)
    once(ROOT / 'prepare-once.json', dict(at=time.time(), bundle=bundle, no_gpu_action=True))
    for name, path in paths.items():
        (ROOT / (name + '-acceptance.json')).open('xb').write(path.read_bytes())
    original = COMMON / 'common-data/final.parquet'
    manifest = COMMON / 'common-data/manifest.json'
    assert sha(manifest) == DATA_MANIFEST
    dm = json.loads(manifest.read_text())
    assert sha(original) == dm['splits']['final']['sha256']
    rows = []
    for i, row in enumerate(pq.read_table(original).to_pylist()):
        q = question_from_old_prompt(row['prompt']); gold = row['reward_model']['ground_truth']['target']
        rows.append(dict(final_index=i, question_id=row['extra_info']['index'], source=row['data_source'],
                         gold_answers=[gold] if isinstance(gold, str) else list(gold), question=q, prompt=prompt(q)))
    validate_rows(rows)
    data = ROOT / 'final.jsonl'
    with data.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    identity = digest([{k: r[k] for k in ('final_index', 'question_id', 'source', 'gold_answers')} for r in rows])
    assert sha(data) == FINAL_JSONL and identity == ID_GOLD, 'Common correctedv2 inputs differ'
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    assert (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.vocab_size, cfg.tie_word_embeddings) == ('qwen2', 3584, 28, 28, 4, 152064, False)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    initialized = INITIAL_MODEL
    base_tok = AutoTokenizer.from_pretrained(initialized, local_files_only=True)
    assert tok.get_vocab() == base_tok.get_vocab() and tok.chat_template == base_tok.chat_template
    assert tok.special_tokens_map == base_tok.special_tokens_map
    assert tok.eos_token_id == 151645 and tok.pad_token_id == 151643
    max_len = max(len(tok.apply_chat_template([dict(role='user', content=r['prompt'])],
                      add_generation_prompt=True, tokenize=True)) for r in rows)
    assert max_len <= 2048
    args = EngineArgs(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16', tensor_parallel_size=1,
                      enforce_eager=True, gpu_memory_utilization=.65, max_model_len=4596,
                      max_num_batched_tokens=8192, max_num_seqs=128, seed=0, disable_log_stats=True)
    assert (args.seed, args.tensor_parallel_size, args.max_model_len) == (0, 1, 4596)
    started = time.monotonic()
    assert cpu_headroom()>20*1024**3
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                local_files_only=True, low_cpu_mem_usage=True, device_map='cpu')
    comparison = compare_loaded(model, MODEL)
    del model
    for name, value in files.items():
        assert sha(MODEL / name) == value, name
    assert not torch.cuda.is_initialized()
    reload = dict(at=time.time(), status='ACTUAL_HF_CPU_RELOAD_ALL339_PARAMETERS_AND_COMMONV2_51713_PROMPTS_VERIFIED',
                  uid=UID, hf_file_identity_sha256=hf['hf_file_identity_sha256'], hf_acceptance_sha256=bundle['hf']['sha256'],
                  **comparison, all51713_prompt_tokenization_verified=True, max_prompt_tokens=max_len,
                  data_sha256=sha(data), id_gold_identity=identity, profile=PROFILE, science=SCIENCE,
                  cuda_initialized=False, elapsed_reload_seconds=time.monotonic()-started, no_gpu_action=True)
    once(ROOT / 'reload-acceptance.json', reload)
    evidence = {name: dict(path=str(ROOT / (name + '-acceptance.json')),
                          sha256=sha(ROOT / (name + '-acceptance.json'))) for name in ('native', 'hf', 'reload')}
    receipt = COMMON / 'receipts/pj-retriever-ready.json'
    ready = json.loads(receipt.read_text())
    from urllib.parse import urlsplit
    u = urlsplit(ready['url']); assert u.scheme == 'http' and u.port == 29147 and u.path in ('', '/', '/retrieve')
    assert not u.query and not u.fragment
    endpoint = ready['url'].rstrip('/') + ('' if u.path == '/retrieve' else '/retrieve')
    result, response_sha = search(endpoint, ['Beijing capital'])
    assert len(result) == 1 and len(result[0]) == 3
    once(ROOT / 'retriever-preflight.json', dict(at=time.time(), endpoint=endpoint, response_sha256=response_sha, documents=3))
    (ROOT / 'plans').mkdir(); plans = []
    for repeat in (1, 2, 3):
        plan = build_plan(repeat, source, bundle['source_files'], native, hf, reload, evidence,
                          sha(data), identity, sha(original), receipt, endpoint)
        path = ROOT / 'plans' / ('r' + str(repeat) + '.json')
        once(path, plan); (ROOT / ('r' + str(repeat))).mkdir()
        verify(plan)
        plans.append(dict(plan=str(path), plan_sha256=sha(path), job=plan['job']))
    once(ROOT / 'prepared.json', dict(at=time.time(), status='TRAINED7B_TREE_COMMONV2_THREE_PLANS_CPU_PREPARED_NO_SUBMISSION',
              uid=UID, source=source, plans=plans, evidence=evidence, no_gpu_action=True, no_auto_retry=True))
    print('PREPARED=' + (ROOT / 'prepared.json').read_text(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True); a = p.parse_args(); run(a.bundle)
