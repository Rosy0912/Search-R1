"""Read exact shared products; never partition or rewrite their bytes."""
import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

MANIFEST_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
COUNTS = dict(train=169028, validation=512, stage=64, final=51713)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def probe_question_id(chat):
    # Same normalized question identity used by the G1 probe, separate from
    # shared canonical identity. Exact file SHA binds every original field.
    text = chat[0]['content']
    matches = list(re.finditer(r'Question:\s*', text))
    if not matches:
        raise ValueError('Unknown common prompt question schema')
    question = re.sub(r'\s+', ' ', text[matches[-1].end():]).strip().casefold()
    return hashlib.sha256(question.encode()).hexdigest()


def verify(root):
    import pyarrow.parquet as pq
    root = Path(root)
    manifest_path = root / 'manifest.json'
    assert digest(manifest_path) == MANIFEST_SHA, 'Wrong shared manifest bytes'
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest['splits']) == set(COUNTS)
    products, identities = {}, {}
    for split, count in COUNTS.items():
        record = manifest['splits'][split]
        assert record['path'] == split + '.parquet'
        path = root / record['path']
        assert digest(path) == record['sha256'], 'Wrong shared product: ' + split
        table = pq.read_table(path)
        assert table.num_rows == record['rows'] == count
        sources = dict(Counter(table['data_source'].to_pylist()))
        assert sources == record['source_counts']
        ids = [probe_question_id(chat) for chat in table['prompt'].to_pylist()]
        identities[split] = set(ids)
        products[split] = dict(sha256=record['sha256'], rows=count,
            source_counts=sources, unique_probe_questions=len(set(ids)),
            ordered_probe_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
    intersections = {}
    for a, b in itertools.combinations(COUNTS, 2):
        overlap = len(identities[a] & identities[b])
        intersections[a + ':' + b] = overlap
        assert overlap == 0, 'Probe-normalized question overlap: ' + a + ':' + b
    exclusion = manifest['exclusion_audit']
    assert exclusion['path'] == 'exclusions-audit.json'
    assert digest(root / exclusion['path']) == exclusion['sha256']
    # No misleading claim that the policy-training calibration subset is held out.
    return dict(manifest_sha256=MANIFEST_SHA, products=products,
        probe_normalized_cross_split_intersections=intersections,
        calibration_policy='derivative of train; retain all canonical train rows',
        accepted=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--proof', required=True)
    args = parser.parse_args()
    result = verify(args.data)
    with Path(args.proof).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
