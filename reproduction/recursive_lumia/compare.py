"""Compare immutable native evaluation files by question identity and outer round."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load_evaluation(path):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['rows_sha256']:
        raise ValueError('Evaluation row bytes differ from their native receipt')
    rows = [json.loads(line) for line in raw.splitlines()]
    identifiers = [row['question_id'] for row in rows]
    if (len(rows) != metadata['count'] or len(set(identifiers)) != len(rows)
            or any(not isinstance(q, str) or not q for q in identifiers)
            or any(row['em'] not in (0., 1.) for row in rows)):
        raise ValueError('Missing/repeated question identities or invalid native EM')
    count = {'dev512':512, 'final51713':51713}.get(metadata['split'])
    if count != len(rows):
        raise ValueError('Evaluation does not cover its declared complete split')
    identity_hash = hashlib.sha256(json.dumps(identifiers, ensure_ascii=False).encode()).hexdigest()
    if identity_hash != metadata['ordered_ids_sha256']:
        raise ValueError('Evaluation question order differs from native receipt')
    if not math.isclose(sum(row['em'] for row in rows)/len(rows), metadata['em'], abs_tol=1e-12, rel_tol=0):
        raise ValueError('Native aggregate disagrees with stored per-question EM')
    return metadata, {row['question_id']:row for row in rows}, dict(
        rows_path=str(path.resolve()), metadata_path=str(metadata_path.resolve()),
        rows_sha256=metadata['rows_sha256'], metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())


def counts(pairs):
    result = dict(questions=len(pairs), both_correct=0, baseline_only_correct=0,
                  candidate_only_correct=0, both_incorrect=0)
    labels = {(1.,1.):'both_correct', (1.,0.):'baseline_only_correct',
              (0.,1.):'candidate_only_correct', (0.,0.):'both_incorrect'}
    for baseline, candidate in pairs:
        result[labels[(baseline, candidate)]] += 1
    result['baseline_em'] = (result['both_correct']+result['baseline_only_correct'])/len(pairs)
    result['candidate_em'] = (result['both_correct']+result['candidate_only_correct'])/len(pairs)
    result['candidate_minus_baseline_percentage_points'] = 100*(
        result['candidate_only_correct']-result['baseline_only_correct'])/len(pairs)
    return result


def compare(baseline_path, candidate_path):
    baseline, left, left_files = load_evaluation(baseline_path)
    candidate, right, right_files = load_evaluation(candidate_path)
    for key in ('split','completed_outer','data_identity'):
        if baseline[key] != candidate[key]:
            raise ValueError('Cannot pair different '+key)
    if left.keys() != right.keys():
        raise ValueError('Native evaluations do not cover exactly the same questions')
    if any(left[q]['source'] != right[q]['source'] for q in left):
        raise ValueError('Per-question dataset identity differs')
    grouped = {}
    for q in sorted(left):
        grouped.setdefault(left[q]['source'], []).append((left[q]['em'], right[q]['em']))
    return dict(status='VERIFIED_SAME_ROUND_PAIRED_EM', split=baseline['split'],
        completed_outer=baseline['completed_outer'], data_identity=baseline['data_identity'],
        baseline=dict(metadata=baseline, files=left_files), candidate=dict(metadata=candidate, files=right_files),
        overall=counts([(left[q]['em'], right[q]['em']) for q in sorted(left)]),
        by_source={source:counts(pairs) for source,pairs in sorted(grouped.items())},
        limits=['Descriptive same-question, same-round comparison; no multi-seed uncertainty claim.',
                'Equal training rounds do not match compute; screening, retries and GPU time must be reported separately.',
                'Hardware and numerical backend differences must be considered when interpreting method differences.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(output=str(path), **result['overall']), ensure_ascii=False))


if __name__ == '__main__': main()
