"""Holdout-only checkpoint selection and immutable HF teacher export."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from .holdout import file_hash

CANDIDATES = (5, 100, 200, 300, 400, 500)


def model_manifest(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not (path/'config.json').is_file() or not (path/'tokenizer_config.json').is_file():
        raise ValueError('Teacher must be a complete HF model/tokenizer directory')
    if not any(p.suffix == '.safetensors' or (p.name.startswith('pytorch_model') and p.suffix == '.bin') for p in files):
        raise ValueError('Missing HF weights')
    for index in path.glob('*.index.json'):
        weight_map = json.loads(index.read_text()).get('weight_map')
        if not weight_map:
            raise ValueError('Invalid HF shard index')
        for name in set(weight_map.values()):
            shard = (path/name).resolve()
            if not shard.is_relative_to(path.resolve()) or not shard.is_file():
                raise ValueError('Missing/outside HF weight shard: ' + name)
    hashes = {str(p.relative_to(path)): file_hash(p) for p in files}
    return dict(files=hashes, sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def select(records, split_manifest):
    content_id = split_manifest['content_identity']['sha256']
    holdout_rows = split_manifest['holdout_rows']
    if len(records) != len(CANDIDATES) or sorted(r['outer_update'] for r in records) != list(CANDIDATES):
        raise ValueError('Require exactly the six specified checkpoint candidates')
    for r in records:
        if r['split_content_identity'] != content_id or r['evaluation_split'] != 'common-holdout':
            raise ValueError('Teacher selection requires identical common holdout')
        if r['seed'] != 42 or r['evaluated_count'] != holdout_rows or r['holdout_rows'] != holdout_rows or holdout_rows <= 0:
            raise ValueError('Incomplete/nonstandard holdout evaluation')
        if not math.isfinite(r['mean_answer_f1']) or not 0 <= r['mean_answer_f1'] <= 1:
            raise ValueError('Invalid holdout F1')
    return min(records, key=lambda r: (-r['mean_answer_f1'], r['outer_update']))


def export(records, split_manifest, student, destination):
    from transformers import AutoTokenizer
    winner = select(records, split_manifest)
    for record in records:
        if model_manifest(record['hf_directory'])['sha256'] != record['checkpoint_sha256']:
            raise ValueError('Candidate changed after evaluation')
    source = Path(winner['hf_directory'])
    source_hash = model_manifest(source)
    if source_hash['sha256'] != winner['checkpoint_sha256']:
        raise ValueError('Candidate HF directory changed after evaluation')
    base_hash = model_manifest(student)
    st = AutoTokenizer.from_pretrained(student, local_files_only=True)
    tt = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if st.get_vocab() != tt.get_vocab() or (st.bos_token_id, st.eos_token_id) != (tt.bos_token_id, tt.eos_token_id):
        raise ValueError('Teacher/student tokenizer incompatibility')
    weights = lambda m: {k:v for k,v in m['files'].items() if k.endswith('.safetensors') or k.startswith('pytorch_model')}
    if weights(base_hash) == weights(source_hash):
        raise ValueError('Teacher is still untrained initialization')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Immutable teacher destination already exists')
    shutil.copytree(source, destination)
    if model_manifest(destination) != source_hash:
        raise ValueError('Export hash mismatch')
    report = dict(selected=winner, candidates=records, split_content_identity=split_manifest['content_identity']['sha256'],
                  selection_rule='maximum holdout F1; ties earliest update', teacher=source_hash,
                  student=base_hash, tokenizer_compatible=True, evaluation_seed=42)
    (destination/'teacher-manifest.json').write_text(json.dumps(report, indent=2))
    for p in destination.rglob('*'):
        p.chmod(0o555 if p.is_dir() else 0o444)
    destination.chmod(0o555)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--student', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    manifest_path = Path(args.split_manifest)
    split_manifest = json.loads(manifest_path.read_text())
    for name, expected in split_manifest['sha256'].items():
        if file_hash(manifest_path.parent/name) != expected:
            raise ValueError('Split data hash mismatch')
    report = export(json.loads(Path(args.records).read_text()), split_manifest, args.student, args.destination)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
