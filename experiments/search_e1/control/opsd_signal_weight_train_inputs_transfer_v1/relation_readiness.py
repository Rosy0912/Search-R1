"""Consume pinned G TRAIN records with the unchanged scientific extractor, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from train_inputs import CONFIGS, S, digest, require


def main():
    p = argparse.ArgumentParser(); p.add_argument('--helper', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); helper = Path(a.helper); output = Path(a.output)
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only owner required')
    require(output.is_absolute() and output.parent == S / 'analysis' and not output.exists(), 'Fresh independent remote directory required')
    manifest = helper / 'source-manifest.json'
    require(digest(manifest) == 'af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314', 'Original scientific source manifest differs')
    source = json.loads(manifest.read_text())
    for name, ref in source['files'].items():
        require(digest(helper / name) == ref['sha256'], 'Original scientific file differs: ' + name)
    sys.path.insert(0, str(helper))
    from result_analysis import extract_committed_relations, relationships
    configs = {}; updates = []; ledgers = []
    for group, sha in CONFIGS.items():
        root = S / ('group-' + group)
        cp = root / 'receipts/config.json'
        require(digest(cp) == sha, 'Frozen G configuration differs')
        cfg = json.loads(cp.read_text()); cfg['_config_sha256'] = sha; configs[group] = cfg
        lp = root / 'runs/training-em-25-complete.json'; ledger = json.loads(lp.read_text())
        require(ledger['accepted'] and len(ledger['rows']) == 25 and ledger['run_id'] == cfg['run_id'], 'G25 not complete')
        updates.extend(ref for row in ledger['rows'] for ref in row['source_update_refs'])
        ledgers.append(dict(group=group, path=str(lp), sha256=digest(lp), selected_pairs=ledger['rows'][-1]['cumulative_pair_exposure']))
    rows = extract_committed_relations(updates, configs)
    counts = Counter(row['variant'] for row in rows)
    require(all(counts[g['group']] == g['selected_pairs'] for g in ledgers), 'Selected trajectory coverage differs')
    features = ('gap', 'entropy', 'weight', 'raw_loss', 'post_loss', 'weighted_post_loss', 'student_to_teacher_raw_kl')
    coverage = {key: dict(available=sum(row.get(key) is not None for row in rows), missing=sum(row.get(key) is None for row in rows)) for key in features}
    require(all(math.isfinite(row[key]) for row in rows for key in features if row.get(key) is not None), 'Nonfinite feature')
    require(all(coverage[key]['missing'] == 0 for key in features if key != 'gap'), 'Required feature missing')
    result = relationships(rows)
    require('torch' not in sys.modules, 'Unexpected Torch import in CPU analysis')
    output.mkdir(parents=False, exist_ok=False)
    rp = output / 'selected-train-rows.jsonl'
    with rp.open('x') as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True) + '\n')
    with (output / 'conditional-train-relations.json').open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2); f.write('\n')
    value = dict(accepted=True, checked_unix=time.time(), original_scientific_files_verified=len(source['files']),
                 groups=dict(counts), selected_trajectories=len(rows), committed_updates=len(updates), ledgers=ledgers,
                 feature_coverage=coverage, rows=dict(path=str(rp), sha256=digest(rp)), torch_imported=False,
                 gpu_started=False, heldout_results_available=False,
                 scope='Actual selected G training trajectories with successful reference only; conditional associations are not heldout EM, current-policy pass probability or causal improvement')
    with (output / 'readiness.json').open('x') as f: json.dump(value, f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__': main()
