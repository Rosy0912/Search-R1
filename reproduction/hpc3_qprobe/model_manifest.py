"""The same pinned official model, bound to existing HPC3 read-only files."""
import json
from pathlib import Path

from search_g1.qprobe_features import OFFICIAL_REVISION, sha_file

FORMAT = 'hpc3-official-qwen-aa8e725-nine-files-v1'
FILES = {
    'config.json': 'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b',
    'generation_config.json': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'model-00001-of-00002.safetensors': '67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2',
    'model-00002-of-00002.safetensors': 'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d',
    'model.safetensors.index.json': 'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}


def verify(model, manifest):
    model = Path(model)
    if manifest.get('format') != FORMAT or manifest.get('repo') != 'Qwen/Qwen2.5-3B-Instruct' or manifest.get('revision') != OFFICIAL_REVISION:
        raise ValueError('HPC3_OFFICIAL_MODEL_IDENTITY_DIFFERS')
    records = {r['file']: r for r in manifest['files']}
    if set(records) != set(FILES) or len(manifest['files']) != len(FILES):
        raise ValueError('HPC3_OFFICIAL_MODEL_FILE_SET_DIFFERS')
    for name, expected in FILES.items():
        path = model/name
        record = records[name]
        if record['sha256'] != expected or path.stat().st_size != record['bytes'] or sha_file(path) != expected:
            raise ValueError('HPC3_OFFICIAL_MODEL_FULL_SHA_DIFFERS: ' + name)
    config = json.loads((model/'config.json').read_text())
    if config['model_type'] != 'qwen2' or config['hidden_size'] != 2048:
        raise ValueError('HPC3_OFFICIAL_MODEL_ARCHITECTURE_DIFFERS')
    return dict(FILES)


def create(model, path):
    model = Path(model)
    manifest = dict(format=FORMAT, repo='Qwen/Qwen2.5-3B-Instruct', revision=OFFICIAL_REVISION,
        files=[dict(file=name, bytes=(model/name).stat().st_size, sha256=expected)
               for name, expected in sorted(FILES.items())])
    verify(model, manifest)
    from search_g1.qprobe_warmup import write_once
    write_once(path, manifest)
    return manifest
