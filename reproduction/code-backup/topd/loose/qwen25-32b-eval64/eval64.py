"""Standalone original Search-R1 greedy64 evaluation; never trains or submits jobs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

REVISION = '1818d35814b8319459f4bd55ed1ac8709630f003'
OFFICIAL_METADATA_SHA = 'e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
DATA_SHA = 'af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756'
GREEDY = dict(n=1, best_of=1, temperature=0., top_p=1., top_k=-1,
              min_p=0., max_tokens=500, detokenize=False, ignore_eos=False)
REUSED_SOURCES = {
    'search_r1/llm_agent/generation.py', 'search_r1/llm_agent/tensor_helper.py',
    'search_r1/baselines/evaluation.py', 'search_r1/baselines/bapo.py',
    'search_r1/baselines/preflight.py', 'search_r1/baselines/topd_local_gpu.py',
    'search_r1/search/retrieval_server.py', 'verl/trainer/main_ppo.py',
    'verl/utils/reward_score/qa_em.py', 'verl/utils/dataset/rl_dataset.py',
    'verl/utils/tokenizer.py', 'verl/utils/torch_functional.py', 'verl/protocol.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if hasattr(value, 'item'):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(jsonable(value), f, indent=2, allow_nan=False)


def asset_gate(binding_path, root):
    """CPU-only fresh full-file SHA, index completeness and local-source binding."""
    root = Path(root).resolve()
    binding = json.loads(Path(binding_path).read_text())
    require(binding['status'] == 'INVENTORY_VERIFIED' and binding['revision'] == REVISION,
            'Need complete official pinned inventory, never partial download')
    model = Path(binding['model_directory']).resolve()
    data = Path(binding['data_path']).resolve()
    require(root in model.parents and root in data.parents, 'Assets must be on own local root')
    require(sha(data) == DATA_SHA == binding['data_sha256'], 'Frozen original64 drift')
    files = binding['model_files']
    metadata_path = Path(binding['official_metadata_path']).resolve()
    require(root in metadata_path.parents and sha(metadata_path) == OFFICIAL_METADATA_SHA
            == binding['official_metadata_sha256'], 'Official captured HF inventory drift')
    metadata = json.loads(metadata_path.read_text())
    require(metadata['id'] == 'Qwen/Qwen2.5-32B' and metadata['sha'] == REVISION,
            'Official model/revision mismatch')
    official = {r['rfilename']: r for r in metadata['siblings']}
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    require(len(shards) == 17 and shards == {n for n in files if n.endswith('.safetensors')},
            'Require all17 indexed shards, no substitute model')
    required = {'config.json', 'generation_config.json', 'merges.txt', 'vocab.json',
                'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    require(required <= set(files), 'Missing configuration/tokenizer identity')
    for name, expected in files.items():
        path = (model / name).resolve()
        require(model in path.parents and path.is_file() and sha(path) == expected,
                'Model file missing/drift/nonlocal: ' + name)
        require(name in official and path.stat().st_size == official[name]['size'], 'Official file size/name mismatch')
        if 'lfs' in official[name]:
            require(expected == official[name]['lfs']['sha256'], 'Official LFS SHA mismatch')
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            require(blob == official[name]['blobId'], 'Official Git blob identity mismatch')
    from safetensors import safe_open
    actual_keys = {}
    for name in sorted(shards):
        with safe_open(str(model / name), framework='pt', device='cpu') as shard:
            for key in shard.keys():
                require(key not in actual_keys and shard.get_slice(key).get_dtype() == 'BF16',
                        'Duplicate/non-BF16 official weight: ' + key)
                actual_keys[key] = name
    require(actual_keys == index['weight_map'], 'Index/tensor inventory mismatch')
    config = json.loads((model / 'config.json').read_text())
    require(config['model_type'] == 'qwen2' and config['architectures'] == ['Qwen2ForCausalLM']
            and config['num_hidden_layers'] == 64 and config['hidden_size'] == 5120
            and config['num_attention_heads'] == 40 and config['num_key_value_heads'] == 8
            and config['eos_token_id'] == 151643 and config['torch_dtype'] == 'bfloat16'
            and not config.get('quantization_config'), 'Not official nonquantized Qwen2.5-32B Base')
    import subprocess
    head = subprocess.check_output(['git', '-C', str(root / 'repo'), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == binding['code_commit'], 'Production source HEAD drift')
    for name, expected in binding['source_files'].items():
        path = (root / 'repo' / name).resolve()
        require(root / 'repo' in path.parents and sha(path) == expected, 'Source byte drift: ' + name)
    require(REUSED_SOURCES <= set(binding['source_files']), 'Missing reused-source identity')
    for name, expected in binding['tool_files'].items():
        path = (Path(__file__).parent / name).resolve()
        require(Path(__file__).parent.resolve() in path.parents and sha(path) == expected,
                'Evaluation tool drift: ' + name)
    require({'eval64.py', 'supervisor.py', 'runtime_safety.py', 'eval64.sbatch'} <= set(binding['tool_files']),
            'Incomplete reviewed evaluation tools')
    require(sha(root / 'tools/topd-short-run.py') == binding['owned_launcher_sha256'],
            'Owned-process safety implementation drift')
    runtime_files = binding['runtime_files']
    require({'assets-root/assets/wiki/e5_Flat.index', 'assets-root/assets/wiki/wiki-18.jsonl'} <= set(runtime_files)
            and any(n.startswith('assets-root/assets/e5-base-v2/') and n.endswith(('.safetensors', '.bin')) for n in runtime_files)
            and any(n.startswith('retriever-gpu-lib/') and n.endswith('.so') for n in runtime_files),
            'Missing GPU retrieval asset/overlay frozen inventory')
    required_runtime = set()
    for directory in ('assets-root/assets/e5-base-v2', 'retriever-gpu-lib'):
        path = root / directory
        require(path.is_dir(), 'Missing local retrieval dependency directory')
        for file in path.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and '.cache' not in file.parts:
                required_runtime.add(str(file.relative_to(root)))
    require(required_runtime <= set(runtime_files), 'Incomplete actual retrieval dependency freeze')
    for name, expected in runtime_files.items():
        path = (root / name).resolve()
        require(root in path.parents and path.is_file() and sha(path) == expected,
                'Local retrieval asset/overlay drift: ' + name)
    return binding, model, data


def quality_gate(rows):
    require(len(rows) == 64 and len({r['uid'] for r in rows}) == 64,
            'Need exactly64 unique complete QA rows')
    require([r['sample_position'] for r in rows] == list(range(64)), 'Question order drift')
    require(all(r['em'] in (0., 1.) and math.isfinite(r['f1']) and 0 <= r['f1'] <= 1
                for r in rows), 'Invalid actual QA metrics')
    correct = sum(int(r['em']) for r in rows)
    return dict(correct=correct, denominator=64, greedy_pass_at_1=correct / 64,
                required_correct=26, threshold=.4, candidate_quality_accepted=correct >= 26,
                next_path='TOPD_TEACHER_BACKEND_VALIDATION' if correct >= 26 else 'SEARCH_G1_GRPO100',
                scope='CANDIDATE_QUALITY_ONLY_NOT_TOPD_TRAINING_APPROVAL')


class StandaloneAdapter:
    """Official vLLM RequestOutput -> original CPU DataProto rollout contract."""
    def __init__(self, engine, tokenizer, sampling_params, output):
        self.engine, self.tokenizer, self.sampling = engine, tokenizer, sampling_params
        self.output, self.calls = Path(output), 0

    def generate_sequences(self, prompts):
        import torch
        from verl import DataProto
        from verl.utils.torch_functional import get_eos_mask
        require(prompts.meta_info.get('do_sample') is False, 'Only original greedy validation allowed')
        idx, mask, positions = [prompts.batch[k] for k in ('input_ids', 'attention_mask', 'position_ids')]
        require(idx.device.type == 'cpu', 'Evaluation inputs must remain CPU')
        pad = self.tokenizer.pad_token_id
        # Same left-padding removal as original vllm_rollout._pre_process_inputs.
        token_lists = []
        for row in idx:
            valid = torch.nonzero(row != pad, as_tuple=False)
            require(len(valid) > 0, 'Empty prompt')
            token_lists.append(row[int(valid[0]):].tolist())
        started = time.time()
        outputs = self.engine.generate(prompts=None, prompt_token_ids=token_lists,
                                       sampling_params=self.sampling, use_tqdm=False)
        require(len(outputs) == len(token_lists), 'vLLM request count changed')
        raw, finish = [], []
        for item, source in zip(outputs, token_lists):
            require(list(item.prompt_token_ids) == source and len(item.outputs) == 1,
                    'vLLM request order or n changed')
            ids = list(item.outputs[0].token_ids)
            require(len(ids) <= 500, 'Response exceeds unchanged500 budget')
            raw.append(ids)
            finish.append(dict(finish_reason=item.outputs[0].finish_reason,
                               stop_reason=item.outputs[0].stop_reason, generated_tokens=len(ids)))
        response = torch.full((len(raw), 500), pad, dtype=torch.long)
        for i, ids in enumerate(raw):
            response[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        response_mask = get_eos_mask(response, prompts.meta_info['eos_token_id'], dtype=mask.dtype)
        tensors = dict(prompts=idx, responses=response, input_ids=torch.cat((idx, response), -1),
                       attention_mask=torch.cat((mask, response_mask), -1),
                       position_ids=torch.cat((positions, positions[:, -1:] + torch.arange(1, 501)[None]), -1))
        self.calls += 1
        from safetensors.torch import save_file
        path = self.output / ('generation-%02d.safetensors' % self.calls)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
        write_json(path.with_suffix('.json'), dict(call=self.calls, started_epoch=started,
                   completed_epoch=time.time(), prompt_token_ids=token_lists, raw_response_token_ids=raw,
                   results=finish, tensors_sha256=sha(path), padding_scope='manager DP8 duplicate rows retained here and stripped by original manager'))
        return DataProto.from_dict(tensors, meta_info=prompts.meta_info.copy())


def recording_manager_class():
    from search_r1.llm_agent.generation import LLMGenerationManager
    class RecordingManager(LLMGenerationManager):
        def __init__(self, *args, output, **kwargs):
            super().__init__(*args, **kwargs)
            import requests
            self.session = requests.Session()
            self.session.trust_env = False
            self.output, self.turn = Path(output), 0

        def _batch_search(self, queries):
            require(self.config.search_url == 'http://127.0.0.1:18085/retrieve', 'No remote retrieval')
            payload = dict(queries=queries, topk=self.config.topk, return_scores=True)
            response = self.session.post(self.config.search_url, json=payload, timeout=(5, 900))
            response.raise_for_status()
            result = response.json()
            require(len(result['result']) == len(queries) and all(len(r) == 3 for r in result['result']),
                    'Retriever top3/count mismatch')
            write_json(self.output / ('search-%02d.json' % self.turn),
                       dict(request=payload, response=result, response_bytes_sha256=hashlib.sha256(response.content).hexdigest()))
            return result

        def execute_predictions(self, predictions, pad_token, active_mask=None, do_search=True):
            self.turn += 1
            result = super().execute_predictions(predictions, pad_token, active_mask, do_search)
            write_json(self.output / ('turn-%02d.json' % self.turn),
                       dict(turn=self.turn, predictions=predictions, active_mask=active_mask,
                            observations=result[0], dones=result[1], valid_action=result[2], is_search=result[3]))
            return result
    return RecordingManager


def load_questions(data, tokenizer, output):
    from verl.utils.dataset.rl_dataset import RLHFDataset
    dataset = RLHFDataset(str(data), tokenizer, max_prompt_length=4096, truncation='error',
                         cache_dir=str(Path(output) / 'dataset-cache'))
    require(len(dataset.dataframe) == 64, 'Original dataset must contain64 rows')
    dataset.dataframe = dataset.dataframe.sample(64, random_state=42)
    identities = []
    for i, (_, row) in enumerate(dataset.dataframe.iterrows()):
        content = json.dumps(jsonable(row.to_dict()), sort_keys=True, ensure_ascii=False, allow_nan=False)
        identities.append(dict(sample_position=i, uid=hashlib.sha256(content.encode()).hexdigest(),
                               source=row['data_source'], original_row=jsonable(row.to_dict())))
    require(len({r['uid'] for r in identities}) == 64, 'Duplicate question identities')
    write_json(Path(output) / 'questions.json', identities)
    return dataset, identities


def child_execution_gate(root):
    """Direct script invocation cannot bypass own supervisor/cgroup/leased8GPU boundary."""
    import socket
    require(Path(root).resolve() == Path('/dataset/shaoshitong/search-r1-rl/topd'), 'Require own root')
    require(socket.gethostname().split('.')[0] == 'h100-11'
            and os.environ.get('SLURM_JOB_NODELIST') == 'h100-11'
            and os.environ.get('SLURM_JOB_ID'), 'Require own local Slurm allocation')
    require(int(os.environ.get('SLURM_GPUS_ON_NODE', '0')) == 8
            and int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) >= 128
            and int(os.environ.get('SLURM_MEM_PER_NODE', '0')) >= 384 * 1024, 'Require8GPU128CPU384GiB')
    require(os.environ.get('SEARCH_R1_EVAL64_SUPERVISOR_PID') == str(os.getppid()), 'Require own live supervisor parent')
    relative = os.environ.get('SEARCH_R1_EVAL64_CGROUP', '')
    require('/job_' + os.environ['SLURM_JOB_ID'] + '/' in relative + '/', 'Require own job cgroup identity')
    for pid in (os.getpid(), os.getppid()):
        rows = Path('/proc/%d/cgroup' % pid).read_text().splitlines()
        require(any(line.split(':', 2)[-1] == relative for line in rows), 'Child or parent escaped own cgroup')
    runtime_path = Path(os.environ.get('SEARCH_R1_EVAL64_RUNTIME', '')).resolve()
    require(Path(root) / 'logs' in runtime_path.parents and sha(runtime_path) == os.environ.get('SEARCH_R1_EVAL64_RUNTIME_SHA256'), 'Own active runtime binding drift')
    runtime = json.loads(runtime_path.read_text())
    require(runtime['supervisor_pid'] == os.getppid() and runtime['cgroup'] == relative, 'Runtime supervisor identity mismatch')
    from runtime_safety import active_gate
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    return active_gate(runtime, pre_model=True)


def run(args):
    root = Path(args.root).resolve()
    binding, model, data = asset_gate(args.binding, root)
    output = Path(args.output).resolve()
    require(root / 'logs' in output.parents, 'Output must be own logs')
    output.mkdir(exist_ok=False)
    import torch
    torch.set_num_threads(1)
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl import DataProto
    from search_r1.llm_agent.generation import GenerationConfig
    from search_r1.baselines.evaluation import batch_answer_metrics, aggregate_answer_metrics
    from verl.trainer.main_ppo import decode_model_response_spans
    from search_r1.baselines.bapo import answer_from_spans
    tokenizer = hf_tokenizer(str(model), local_files_only=True, trust_remote_code=False)
    require(tokenizer.eos_token_id == 151643, 'Unexpected EOS')
    dataset, identities = load_questions(data, tokenizer, output)
    write_json(output / 'effective-config.json', dict(scope='EVAL64_NO_TRAINING', binding_sha256=sha(args.binding),
               code_commit=binding['code_commit'], data_sha256=sha(data), model_revision=REVISION,
               sampling=GREEDY, seed=42, tp=8, dtype='bfloat16', eager=True, quantization=None,
               max_turns=4, start=2048, rolling=4096, response=500, observation=500, topk=3,
               force_final_answer=False, max_model_len=4596, gpu_memory_utilization=.4, max_num_seqs=64,
               tokenizer_vocab_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
               chat_template=tokenizer.chat_template, bos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id))
    write_json(output / 'pre-model-active-gate.json', child_execution_gate(root))
    from vllm import LLM, SamplingParams
    engine = LLM(model=str(model), tokenizer=str(model), tensor_parallel_size=8,
                 distributed_executor_backend='mp', dtype='bfloat16', quantization=None,
                 trust_remote_code=False, enforce_eager=True, gpu_memory_utilization=.4,
                 max_model_len=4596, max_num_seqs=64, seed=42)
    # Separate readiness, never included in QA or quality denominator.
    readiness = engine.generate(prompt_token_ids=[tokenizer.encode('Hello', add_special_tokens=False)],
                                sampling_params=SamplingParams(temperature=0., max_tokens=1), use_tqdm=False)
    require(len(readiness) == 1 and len(readiness[0].outputs) == 1, 'Readiness failed')
    write_json(output / 'readiness.json', dict(scope='EXCLUDED_FROM64', output_tokens=list(readiness[0].outputs[0].token_ids)))
    adapter = StandaloneAdapter(engine, tokenizer, SamplingParams(**GREEDY), output)
    config = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
                              max_response_length=500, max_obs_length=500, num_gpus=8,
                              no_think_rl=False, search_url='http://127.0.0.1:18085/retrieve',
                              topk=3, force_final_answer=False)
    manager = recording_manager_class()(tokenizer, adapter, config, is_validation=True, output=output)
    batch = DataProto.from_single_dict(collate_fn([dataset[i] for i in range(64)]))
    gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    gen.meta_info = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                         recompute_log_prob=False, do_sample=False, validate=True)
    start = time.time()
    final = manager.run_llm_loop(gen, gen.batch['input_ids'][:, -2048:].clone())
    batch = batch.union(final)
    from safetensors.torch import save_file
    save_file({k: v.long().contiguous() for k, v in batch.batch.items()}, str(output / 'final.safetensors'))
    write_json(output / 'final-meta.json', final.meta_info)
    metrics = batch_answer_metrics(batch, tokenizer)
    rows = []
    for i, metric in enumerate(metrics):
        width = batch.batch['responses'].shape[-1]
        mask = batch.batch['attention_mask'][i, -width:].bool() & batch.batch['info_mask'][i, -width:].bool()
        spans = decode_model_response_spans(tokenizer, batch.batch['responses'][i], mask)
        rows.append(dict(metric, sample_position=i, uid=identities[i]['uid'], answer=answer_from_spans(spans),
                         model_spans=spans, turns=final.meta_info['turns_stats'][i],
                         unfinished=final.meta_info['active_mask'][i], actual_search_calls=final.meta_info['valid_search_stats'][i]))
    with (output / 'qa.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
    write_json(output / 'result.json', dict(status='COMPLETE64_UNREVIEWED_NO_TRAINING',
               quality=quality_gate(rows), metrics=aggregate_answer_metrics(rows),
               wall_seconds=time.time()-start, final_sha256=sha(output / 'final.safetensors'),
               qa_sha256=sha(output / 'qa.jsonl')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/dataset/shaoshitong/search-r1-rl/topd')
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output')
    parser.add_argument('--check-assets-only', action='store_true')
    args = parser.parse_args()
    if args.check_assets_only:
        asset_gate(args.binding, args.root)
        print('OFFLINE32B_COMPLETE_INVENTORY_PASS_NO_GPU')
        return
    require(args.output, 'Need exclusive own output')
    run(args)


if __name__ == '__main__':
    main()
