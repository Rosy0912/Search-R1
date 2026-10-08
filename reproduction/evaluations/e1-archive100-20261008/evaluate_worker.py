"""Task-private fixed512 inference using unchanged frozen Search-E1 code."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

BASE = Path(__file__).resolve().parent
PYTHON = '/home/shaoshitong/search-e1-config-study-20261004/env/bin/python'


def save(phase, **fields):
    value = dict(phase=phase, observed_epoch=time.time(), pid=os.getpid(), **fields)
    temporary = BASE / 'status.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(BASE / 'status.json')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def query(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU admission query failed')
    return result.stdout.strip()


def score_all(rows, predictions):
    from search_e1.core import reward, normalize
    from search_r1.qa_protocol import extract_final_answer
    expected = {row['question_id']: row for row in rows}
    assert len(rows) == len(expected) == len(predictions) == 512
    seen, scored = set(), []
    for row in predictions:
        key = row['question_id']
        assert key in expected and key not in seen
        seen.add(key)
        gold = expected[key]
        assert all(row[field] == gold[field] for field in ('question', 'answers', 'data_source'))
        spans, previous = [], 0
        for start, end in row['policy_spans']:
            assert previous <= start <= end <= len(row['response'])
            spans.append(row['response'][start:end])
            previous = end
        answer = extract_final_answer(spans)
        em = reward(row['response'], gold['answers'], row['policy_spans'])
        assert row['reward'] == em
        aliases = gold['answers'] if isinstance(gold['answers'], list) else [gold['answers']]
        values = []
        for target in aliases:
            a = normalize(answer).split() if answer is not None else []
            b = normalize(target).split()
            shared = sum((collections.Counter(a) & collections.Counter(b)).values())
            values.append(2 * shared / (len(a) + len(b)) if a and b else 0.)
        scored.append(dict(row, extracted_answer=answer, strict_em=em,
                           answer_format_valid=answer is not None, token_f1=max(values, default=0.)))
    assert seen == set(expected)
    groups = {'overall': scored}
    for source in sorted({row['data_source'] for row in scored}):
        groups[source] = [row for row in scored if row['data_source'] == source]
    metrics = {}
    for source, subset in groups.items():
        n = len(subset)
        metrics[source] = dict(count=n, correct=sum(row['strict_em'] for row in subset),
            em=sum(row['strict_em'] for row in subset) / n,
            token_f1=sum(row['token_f1'] for row in subset) / n,
            answer_format_valid=sum(row['answer_format_valid'] for row in subset) / n,
            search_calls_mean=sum(row['search_calls'] for row in subset) / n,
            terminations=dict(collections.Counter(row['termination'] for row in subset)))
    return scored, metrics


def main():
    os.umask(0o077)
    lock = (BASE / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / 'completion.json').exists()
    spec = json.loads((BASE / 'model-manifest.json').read_text())
    plan = json.loads((BASE / 'plan.json').read_text())
    code = BASE / 'code'
    sys.path.insert(0, str(code))
    for relative, expected in plan['strict_reward_source_manifest'].items():
        assert sha(code / relative) == expected, 'Frozen code/data checksum mismatch'
    partial = BASE / 'model.partial'
    save('verifying_checkpoint', checked_files=0, checkpoint_bytes=spec['checkpoint_bytes'])
    verified = []
    for row in spec['files']:
        path = partial / row['name']
        assert path.is_file() and not path.is_symlink()
        before = path.stat()
        assert before.st_size == row['bytes']
        digest = sha(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert digest == row['sha256'], 'Checkpoint SHA256 mismatch'
        verified.append(dict(name=row['name'], bytes=before.st_size, sha256=digest))
        save('verifying_checkpoint', checked_files=len(verified), checkpoint_bytes=spec['checkpoint_bytes'])
    assert not (BASE / 'model').exists()
    partial.rename(BASE / 'model')
    (BASE / 'checkpoint-verification.json').write_text(json.dumps(dict(files=verified,
        verified_epoch=time.time(), pinned_hf_revision=plan['model_revision']), indent=2) + '\n')
    gpu = query(['nvidia-smi', '--id=0', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
    apps = query(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
    parts = [part.strip() for part in gpu.split(',')]
    assert parts[1] == 'GPU-15061c70-49d6-3014-da87-915b370c1e89'
    assert parts[2] == '0' and int(parts[3]) < 512 and int(parts[4]) > 47000 and not apps, 'GPU0 is no longer free'
    endpoint = plan['inference']['retriever_same_HKUST_cluster']
    health_url = endpoint.rsplit('/', 1)[0] + '/health'
    with urllib.request.urlopen(health_url, timeout=15) as response:
        health = json.load(response)
    assert health.get('ready') is True and health.get('ntotal') == 21015324
    (BASE / 'admission.json').write_text(json.dumps(dict(gpu=gpu, compute_apps=apps,
        retrieval=health, admitted_epoch=time.time()), indent=2) + '\n')
    output = BASE / 'predictions.jsonl'
    assert not output.exists()
    command = [PYTHON, '-B', '-m', 'search_e1.rollout', '--model', str(BASE / 'model'),
        '--data', str(code / 'data/validation.jsonl'), '--output', str(output), '--k', '1',
        '--temperature', '0', '--seed', '42', '--backend', 'vllm', '--batch-size', '64',
        '--workers', '1', '--retriever-url', endpoint]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=parts[1], PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', VLLM_ATTENTION_BACKEND='XFORMERS', OMP_NUM_THREADS='8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    (BASE / 'resolved-inference.json').write_text(json.dumps(dict(command=command,
        environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','VLLM_ATTENTION_BACKEND','OMP_NUM_THREADS')},
        historical_equivalence=False, scope='Fresh greedy fixed512; no training or model updates'), indent=2) + '\n')
    started = time.time()
    with (BASE / 'inference.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=str(code), env=env, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            count = sum(1 for _ in output.open()) if output.exists() else 0
            save('inference', child_pid=child.pid, completed_predictions=count, total=512, elapsed_seconds=time.time()-started)
            time.sleep(10)
        assert child.returncode == 0, 'Inference failed; retain task log and any partial predictions'
    rows = [json.loads(line) for line in (code / 'data/validation.jsonl').read_text().splitlines()]
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    scored, metrics = score_all(rows, predictions)
    with (BASE / 'scored-predictions.jsonl').open('x') as file:
        for row in scored:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    (BASE / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    checks = {name:sha(BASE / name) for name in ['predictions.jsonl','scored-predictions.jsonl',
        'metrics.json','inference.log','admission.json','resolved-inference.json','checkpoint-verification.json',
        'plan.json','model-manifest.json','evaluate_worker.py']}
    result = dict(completed_epoch=time.time(), elapsed_seconds=time.time()-started,
        metrics=metrics, sha256=checks, complete_question_coverage=True, model_updated=False)
    (BASE / 'completion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    save('complete', metrics=metrics, completed_predictions=512, elapsed_seconds=time.time()-started)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failed', error_type=type(error).__name__, error=str(error)[:1200])
        raise
