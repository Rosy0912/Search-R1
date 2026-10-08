"""Read-only, same-node classification of retained action/raw-token prefixes.

Counts are physical retained leaf occurrences, including inherited tree prefixes.
This compares recorded IDs, without claiming a tokenizer or engine re-execution.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN = ROOT / 'runs/tree-safety300-3268'
MANIFEST_SHA = '84e2cabc65d9390980a1145e83eeb090d48a529c30161ff6d77bab94d42576eb'
IDENTITY = dict(outer_step=111, job_id='3268',
                source='a632ba91840fb116aacfc0e770681a860647b295',
                config='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b',
                data='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24')


def load_bytes(path, expected):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    fields = ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    assert all(getattr(before, k) == getattr(after, k) for k in fields), 'File changed while reading'
    assert hashlib.sha256(payload).hexdigest() == expected, str(path)
    return payload


def ids(value):
    assert isinstance(value, list) and all(type(t) is int and t >= 0 for t in value), 'Invalid token IDs'
    return value


def prefix_length(action, raw):
    for i, (a, b) in enumerate(zip(action, raw)):
        if a != b:
            return i
    return min(len(action), len(raw))


def analyze_rows(rows, expected_identity, expected_rows, expected_tokens=None):
    counts = Counter()
    examples = []
    for row_id, row in enumerate(rows):
        assert type(row['row']) is int and row['row'] == row_id and row['identity'] == expected_identity
        leaf = ids(row['token_ids'])
        actions = row['metadata']['action_records']
        assert actions, 'Empty retained action record'
        timeline = []
        for action in actions:
            a, obs = ids(action['action_token_ids']), ids(action['observation_token_ids'])
            assert type(action['action_tokens']) is int and type(action['observation_tokens']) is int
            assert action['action_tokens'] == len(a) and action['observation_tokens'] == len(obs)
            timeline.extend(a + obs)
        assert len(leaf) >= len(timeline) and leaf[len(leaf) - len(timeline):] == timeline
        prompt = leaf[:len(leaf) - len(timeline)]
        prior = []
        effective = sum(action['action_tokens'] for action in actions)
        assert effective > 0, 'Empty supervised leaf'
        for step, action in enumerate(actions):
            a, obs, raw = action['action_token_ids'], action['observation_token_ids'], ids(action['raw_generation_token_ids'])
            context = ids(action['generation_context_token_ids'])
            context_equal = context == prompt + prior
            assert type(action['scoring_context_matches_generation']) is bool
            assert context_equal == action['scoring_context_matches_generation'], 'Recorded context flag disagrees with IDs'
            equal = prefix_length(a, raw)
            matched_prefix = equal if context_equal else 0
            counts['records'] += 1
            counts['retained_action_token_occurrences'] += len(a)
            counts['original_raw_recorded_token_slots'] += len(raw)
            counts['own_record_exact_causal_prefix_token_occurrences'] += matched_prefix
            counts['own_record_nonmatching_token_occurrences'] += len(a) - matched_prefix
            counts['generation_vs_final_prefix_equal_records'] += int(context_equal)
            counts['generation_vs_final_prefix_different_records'] += int(not context_equal)
            counts['action_is_exact_raw_prefix_records'] += int(equal == len(a))
            counts['action_differs_within_raw_prefix_records'] += int(equal < min(len(a), len(raw)))
            counts['action_extends_past_raw_records'] += int(len(a) > len(raw) and equal == len(raw))
            counts['first_difference_is_final_action_token_records'] += int(equal < min(len(a), len(raw)) and equal == len(a) - 1)
            counts['raw_tail_discarded_after_exact_prefix_records'] += int(equal == len(a) and len(raw) > len(a))
            # Original equal-leaf objective: each leaf contributes total weight 1.
            counts['own_record_exact_context_weight_mass'] += matched_prefix / effective
            counts['leaf_weight_mass'] += len(a) / effective
            if matched_prefix < len(a) and len(examples) < 16:
                examples.append(dict(row=row_id, step=step, action_tokens=len(a), raw_slots=len(raw),
                    generation_prefix_equal=context_equal, first_action_raw_difference=equal,
                    retained_token_at_difference=a[equal] if equal < len(a) else None,
                    raw_token_at_difference=raw[equal] if equal < len(raw) else None,
                    action_ids_sha256=hashlib.sha256(json.dumps(a, separators=(',', ':')).encode()).hexdigest(),
                    raw_ids_sha256=hashlib.sha256(json.dumps(raw, separators=(',', ':')).encode()).hexdigest()))
            prior.extend(a + obs)
        counts['leaf_rows'] += 1
    assert counts['leaf_rows'] == expected_rows
    if expected_tokens is not None:
        assert counts['retained_action_token_occurrences'] == expected_tokens
    total = counts['retained_action_token_occurrences']
    return dict(counts=dict(counts), own_record_exact_token_coverage=counts['own_record_exact_causal_prefix_token_occurrences'] / total,
                own_record_equal_leaf_weight_coverage=counts['own_record_exact_context_weight_mass'] / expected_rows,
                examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'h100-9'
    assert not os.environ.get('CUDA_VISIBLE_DEVICES', ''), 'Run under a proper0GPU reservation'
    assert os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Proper CPU Slurm allocation required'
    started, cpu = time.perf_counter(), time.process_time()
    directory = RUN / 'output/trajectories/outer-000111'
    manifest = json.loads(load_bytes(directory / 'manifest.json', MANIFEST_SHA))
    assert manifest['status'] == 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED'
    assert manifest['identity'] == IDENTITY and manifest['rows'] == 3840
    name = 'trajectories.jsonl.gz'
    payload = load_bytes(directory / name, manifest['payloads'][name])
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode='rb') as stream:
        rows = (json.loads(line) for line in stream)
        result = analyze_rows(rows, IDENTITY, 3840, 4237098)
    result.update(status='ACTUAL_RETAINED_ACTION_RAW_PREFIX_CLASSIFICATION_ONLY', identity=IDENTITY,
        manifest_sha256=MANIFEST_SHA, trajectories_sha256=manifest['payloads'][name],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_slurm_job_id=os.environ['SLURM_JOB_ID'], wall_seconds=time.perf_counter() - started,
        process_cpu_seconds=time.process_time() - cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        scope='Recorded own-action raw prefixes only. Inherited physical leaf occurrences count repeatedly. No tokenizer, model, GPU, logits or optimizer execution. Other raw calls may cover tokens not matched to the recorded own action. Differences cannot alone assign decode/retokenization/EOS/engine cause or prove any sampled-policy correction.')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
