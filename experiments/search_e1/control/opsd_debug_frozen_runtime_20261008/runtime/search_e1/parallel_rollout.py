"""One vLLM process per allocated GPU; verify and merge disjoint question shards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import write_jsonl, write_json


def run_parallel(args, rows):
    devices = [s.strip() for s in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if s.strip()]
    if args.backend != 'vllm' or args.workers > len(devices):
        raise ValueError('Parallel vLLM requires enough explicitly allocated CUDA_VISIBLE_DEVICES')
    if args.workers > len(rows) or not rows:
        raise ValueError('Need at least one question per worker')
    expected_ids = {row['question_id'] for row in rows}
    if len(expected_ids) != len(rows):
        raise ValueError('Question IDs must be unique before sharding')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the final name before starting children. Preserve partial artifacts on failure.
    with output.open('x') as merged:
        parts = Path(str(output) + '.parts')
        parts.mkdir()
        processes, logs = [], []
        shards = [rows[i::args.workers] for i in range(args.workers)]
        try:
            for rank, shard in enumerate(shards):
                source = parts / f'questions-{rank}.jsonl'
                target = parts / f'rollouts-{rank}.jsonl'
                write_jsonl(source, shard)
                log = (parts / f'worker-{rank}.log').open('x')
                logs.append(log)
                command = [sys.executable, '-m', 'search_e1.rollout', '--model', args.model,
                           '--data', str(source), '--output', str(target), '--backend', 'vllm',
                           '--workers', '1', '--k', str(args.k), '--temperature', str(args.temperature),
                           '--batch-size', str(args.batch_size), '--seed', str(args.seed + rank),
                           '--retriever-url', args.retriever_url]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank],
                                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                processes.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT))
            last_completed = -1
            while True:
                states = [process.poll() for process in processes]
                if any(state is not None and state != 0 for state in states):
                    raise RuntimeError(f'Rollout worker failed: exit codes {states}; inspect {parts}')
                completed = sum(state == 0 for state in states)
                if completed != last_completed:
                    print({'completed_rollout_workers': completed, 'workers': args.workers}, flush=True)
                    last_completed = completed
                if completed == args.workers:
                    break
                time.sleep(5)
            seen, scores = set(), {}
            for rank, shard in enumerate(shards):
                allowed = {row['question_id'] for row in shard}
                count = 0
                with (parts / f'rollouts-{rank}.jsonl').open() as part:
                    for line in part:
                        row = json.loads(line)
                        key = (row['question_id'], row['sibling'])
                        if key[0] not in allowed or not 0 <= key[1] < args.k or key in seen:
                            raise RuntimeError('Unexpected or duplicate question/sibling in rollout shard')
                        seen.add(key)
                        count += 1
                        dataset = row.get('data_source', 'unknown')
                        total, n = scores.get(dataset, (0., 0))
                        scores[dataset] = (total + row['reward'], n+1)
                        merged.write(line)
                if count != len(shard) * args.k:
                    raise RuntimeError('Rollout shard did not cover all assigned question/sibling pairs')
            if len(seen) != len(rows) * args.k:
                raise RuntimeError('Merged rollout coverage incomplete')
            metrics = {dataset: total/n for dataset, (total, n) in scores.items()}
            write_json(str(output) + '.metrics.json',
                       {'em': metrics, 'macro_em': sum(metrics.values())/len(metrics),
                        'count': len(seen), 'workers': args.workers, 'coverage_verified': True})
            write_json(parts/'manifest.json', {'workers': args.workers, 'questions': len(rows),
                       'k': args.k, 'worker_seeds': [args.seed+i for i in range(args.workers)],
                       'shard_questions': [len(shard) for shard in shards], 'coverage_verified': True})
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
