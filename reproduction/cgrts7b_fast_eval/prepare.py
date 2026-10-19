"""Prepare exactly one already-authorized milestone50 evaluation on the CPU."""
import json
import time
from pathlib import Path
from .entry import JOB, TRAIN_JOB, TRAIN_SOURCE, validate_proof, verify
from reproduction.official_eval_v2.driver import once, search
from reproduction.official_eval_v2.protocol import digest, sha

BASE = Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1')
ROOT = BASE / 'cgrts-fast-milestone50-v1-20261019'
TRAIN_ROOT = BASE / 'runs/original-hostmem-v1'
REFERENCE = BASE / 'reevaluation-official-v2d-20261019/plans/base-r1.json'


def prepare():
    import torch
    import importlib.metadata as metadata
    from transformers import AutoConfig, AutoTokenizer
    from vllm.engine.arg_utils import EngineArgs
    assert not torch.cuda.is_initialized()
    assert metadata.version('vllm') == '0.6.3'
    assert metadata.version('torch').split('+')[0] == '2.4.0'
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    source = Path(deployment['source'])
    assert source == Path(__file__).resolve().parents[2]
    for name, expected in deployment['files'].items():
        assert sha(source / name) == expected, name
    assert not (ROOT / 'prepared.json').exists() and not (ROOT / 'plans').exists()
    proofpath = ROOT / 'native50-acceptance.json'
    proof = json.loads(proofpath.read_text()); validate_proof(proof)
    manifest = TRAIN_ROOT / 'checkpoints/training_state/global_step_50/manifest.json'
    assert sha(manifest) == proof['manifest_sha256']
    native = json.loads(manifest.read_text())
    assert native['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
    assert native['hf_identity'] == proof['hf_identity']
    model = TRAIN_ROOT / 'checkpoints/actor/global_step_50'
    for name, expected in proof['hf_identity']['files'].items():
        assert sha(model / name) == expected, name
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    assert (config.hidden_size, config.num_hidden_layers,
            config.num_attention_heads, config.num_key_value_heads) == (3584, 28, 28, 4)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert (tokenizer.eos_token_id, tokenizer.pad_token_id) == (151645, 151643)
    args = EngineArgs(model=str(model), tokenizer=str(model), dtype='bfloat16',
        tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=.65,
        max_model_len=4596, max_num_batched_tokens=8192, max_num_seqs=128,
        seed=0, disable_log_stats=True)
    assert args.seed == 0 and args.max_model_len == 4596
    reference = json.loads(REFERENCE.read_text())
    root = ROOT / 'cgrts50-r1'
    assert not root.exists()
    plan = dict(reference, root=str(root), kind='cgrts50', job=JOB, repeat_count=1,
        source=str(source), source_commit=deployment['source_commit'], source_files=deployment['files'],
        model=str(model), model_revision='139134-step50', model_files=proof['hf_identity']['files'],
        model_identity=digest(dict(training_job=TRAIN_JOB, training_source=TRAIN_SOURCE,
                                  completed_step=50, hf_identity=proof['hf_identity'])),
        milestone_checkpoint=dict(step=50, training_job=TRAIN_JOB,
            training_root=str(TRAIN_ROOT), native_acceptance=str(proofpath),
            native_acceptance_sha256=sha(proofpath), reference_plan=str(REFERENCE),
            reference_plan_sha256=sha(REFERENCE), not_final200=True,
            reference_model_label_only=False, accepted_scientific_repeat_count_for_milestone=1))
    rows = verify(plan)
    maximum = max(len(tokenizer.apply_chat_template([dict(role='user', content=r['prompt'])],
        add_generation_prompt=True, tokenize=True)) for r in rows)
    assert maximum <= 2048
    result, response = search(plan['retriever_url'], ['Beijing capital'])
    assert len(result) == 1 and len(result[0]) == 3
    assert not torch.cuda.is_initialized()
    (ROOT / 'plans').mkdir(); root.mkdir()
    path = ROOT / 'plans/cgrts50-r1.json'; once(path, plan)
    receipt = dict(at=time.time(), plan=str(path), plan_sha256=sha(path),
        source_commit=deployment['source_commit'], actual_full_model_source_data_sha=True,
        native50_independent_full_cpu_audit_local_sha=True, maximum_prompt_tokens=maximum,
        prompt_count=len(rows), retriever_health_response_sha256=response,
        no_cuda_initialization=True, no_training_mutation=True, no_gpu_submission=True,
        not_final200=True)
    once(root / 'preflight.json', receipt); once(ROOT / 'prepared.json', receipt)
    print('PREPARED=' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    prepare()
