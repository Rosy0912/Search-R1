"""CPU preparation using pinned, existing raw assets; never reuse trained weights."""
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    base = Path(sys.argv[1]).resolve()
    revision = 'bcafb8dd07d453be3cbeeeb3f78be1841bddf92c'
    cache = Path('/home/jmzhang/.cache/huggingface/hub/datasets--RUC-NLPIR--FlashRAG_datasets/snapshots') / revision
    raw = base / 'assets/raw'
    from search_r1.baselines.holdout import E1_SOURCES
    pairs = [('nq', 'train'), ('hotpotqa', 'train')] + [(s, 'test' if s in ('nq','triviaqa','popqa','bamboogle') else 'dev') for s in E1_SOURCES]
    for source, split in pairs:
        src = (cache / source / (split + '.jsonl')).resolve(strict=True)
        dest = raw / source / (split + '.jsonl')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src, dest)
        assert sha(src) == sha(dest)
    data = base / 'common-data'
    if not data.exists():
        from release.recursive_bran_a.prepare_data import prepare
        prepare(raw, data, revision)
    manifest = json.loads((data / 'manifest.json').read_text())
    assert {k: v['rows'] for k,v in manifest['splits'].items()} == dict(train=169028, validation=512, stage=64, final=51713)
    prior = Path('/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/preparation/official-model-manifest.json')
    official = json.loads(prior.read_text())
    assert official['revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    model = Path('/home/jmzhang/Workspace/Search-E1-20261001/models/Qwen2.5-3B-Instruct')
    for row in official['files']:
        assert sha(model / row['file']) == row['sha256'], row['file']
    receipt = dict(model=str(model), model_revision=official['revision'], official_manifest_sha256=sha(prior),
                   data=str(data), data_manifest_sha256=sha(data/'manifest.json'),
                   dataset_revision=revision, counts={k:v['rows'] for k,v in manifest['splits'].items()},
                   index='/home/jmzhang/self-evolution/Search-R1/data/retriever_data/e5_Flat.index',
                   corpus='/home/jmzhang/self-evolution/Search-R1/data/retriever_data/wiki-18.jsonl',
                   encoder='/home/jmzhang/models/e5-base-v2')
    (base/'receipts').mkdir(exist_ok=True)
    dest = base/'receipts/assets.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == receipt
    else:
        dest.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
