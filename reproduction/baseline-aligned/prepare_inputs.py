"""Import shared prepared files and observed baseline order; never resplit data."""
import argparse
import hashlib
import json
from pathlib import Path
from search_r1.baselines.baseline_alignment_data import prepared_manifest, order_manifest


def main():
    p = argparse.ArgumentParser()
    for name in ('train', 'validation', 'stage', 'final', 'baseline_order', 'output'):
        p.add_argument('--' + name.replace('_', '-'), required=True)
    p.add_argument('--stage-train-overlap-policy', choices=['disjoint', 'preserve-common-baseline-train'], default='disjoint')
    a = p.parse_args()
    manifest = prepared_manifest(a.train, a.validation, a.final, a.stage, a.stage_train_overlap_policy)
    observed = json.loads(Path(a.baseline_order).read_text())
    # The exported order must originate from the actual baseline DataLoader;
    # seed equality or a newly generated shuffle is not evidence of parity.
    if observed.get('source') != 'baseline-observed-indices':
        raise ValueError('Require the exported actual baseline epoch indices')
    order = order_manifest(manifest['splits']['train']['question_ids'], 512, observed['epochs'])
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, doc in [('shared-data.json', manifest), ('shared-order.json', order)]:
        raw = (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        (root / name).write_bytes(raw)
        print(name + ' SHA256=' + hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
