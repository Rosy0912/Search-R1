"""Lossless schema conversion of the fixed train/dev rows for official rLLM."""
import json
from pathlib import Path
from .contract import DATA_SHA, digest, sha, task_from_common


def prepare(common, target):
    import pyarrow.parquet as pq
    from rllm.data import DatasetRegistry
    common = Path(common)
    target = Path(target)
    assert sha(common / 'manifest.json') == DATA_SHA
    manifest = json.loads((common / 'manifest.json').read_text())
    assert not target.exists(), 'Existing conversion is preserved; inspect before recovery'
    target.mkdir(parents=True)
    DatasetRegistry._REGISTRY_DIR = str(target / 'registry')
    DatasetRegistry._REGISTRY_FILE = str(target / 'registry/dataset_registry.json')
    DatasetRegistry._DATASET_DIR = str(target / 'datasets')
    proof = {'common_manifest_sha256': DATA_SHA, 'splits': {}}
    datasets = {}
    for split, expected in [('train', 169028), ('validation', 512)]:
        entry = manifest['splits'][split]
        source = common / entry['path']
        assert entry['rows'] == expected and sha(source) == entry['sha256']
        rows = pq.read_table(source).to_pylist()
        tasks = [task_from_common(row) for row in rows]
        ids = [task['index'] for task in tasks]
        assert len(tasks) == len(set(ids)) == expected
        import hashlib
        ids_sha = hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
        assert ids_sha == entry['question_ids_sha256']
        dataset = DatasetRegistry.register_dataset('branpo_common', tasks, split)
        actual = DatasetRegistry.load_dataset('branpo_common', split)
        assert actual.get_data() == tasks, 'Question/gold/ID round trip changed'
        verl_rows = pq.read_table(dataset.get_verl_data_path()).to_pylist()
        assert [row['extra_info'] for row in verl_rows] == tasks
        proof['splits'][split] = dict(rows=expected, source_sha256=entry['sha256'],
            ordered_ids_sha256=ids_sha, exact_question_gold_id_digest=digest(tasks),
            tasks_sha256=sha(dataset.get_data_path()),
            verl_sha256=sha(dataset.get_verl_data_path()),
            tasks=dataset.get_data_path(), verl=dataset.get_verl_data_path(),
            full_round_trip_exact=True)
        datasets[split] = dataset
    with (target / 'receipt.json').open('x') as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)
    return datasets, proof
