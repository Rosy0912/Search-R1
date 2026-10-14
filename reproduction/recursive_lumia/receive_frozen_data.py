"""Receive a file-listed tar stream without storing a dataset on the relay host."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time


def receive(stream, spec, output, source_commit):
    output = Path(output).absolute()
    stage = output.with_name(output.name + '.receiving')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Dataset already exists; inspect it rather than overwrite')
    files = spec['files']
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError('Invalid file name in frozen specification')
    stage.mkdir(parents=True, exist_ok=False)
    intent = dict(started_at=time.time(), source_commit=source_commit,
                  manifest_sha256=spec['manifest_sha256'], expected_files=len(files))
    (stage/'transfer-intent.json').write_text(json.dumps(intent))
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            if member.name not in files or member.name in seen or not member.isfile():
                raise ValueError('Unexpected, repeated or non-regular archive member')
            expected = files[member.name]
            if member.size != expected['bytes']:
                raise ValueError('File size differs from frozen dataset')
            target = stage/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as src, target.open('xb') as dst:
                while True:
                    block = src.read(8*1024**2)
                    if not block:
                        break
                    digest.update(block)
                    dst.write(block)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError('SHA mismatch: ' + member.name)
            seen.add(member.name)
            print(json.dumps(dict(verified=member.name, bytes=member.size)), flush=True)
    if seen != set(files):
        raise ValueError('Dataset stream is incomplete')
    manifest = json.loads((stage/'manifest.json').read_text())
    if (files['manifest.json']['sha256'] != spec['manifest_sha256']
            or {k:v['rows'] for k,v in manifest['splits'].items()} != spec['counts']):
        raise ValueError('Frozen manifest identity or split counts differ')
    receipt = dict(**intent, completed_at=time.time(), status='ALL_BYTES_VERIFIED',
                   files=len(seen), bytes=sum(v['bytes'] for v in files.values()),
                   scope='Byte-identical frozen data only; no GPU or training executed')
    (stage/'transfer-complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
    stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(receive(sys.stdin.buffer, json.loads(Path(args.spec).read_text()),
                             args.output, args.source_commit)), flush=True)


if __name__ == '__main__':
    main()
