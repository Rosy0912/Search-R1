"""Atomic streaming concatenation, checking each pinned source SHA256."""
import hashlib
from pathlib import Path

def assemble(specs, destination):
    destination=Path(destination);temp=destination.with_suffix(destination.suffix+'.assembling')
    combined=hashlib.sha256();total=0
    with temp.open('wb') as out:
        for spec in specs:
            paths=([Path(spec['prefix'])] if spec.get('prefix') else [])+[Path(row[2]) for row in spec['ranges']]
            source_hash=hashlib.sha256();size=0
            for path in paths:
                with path.open('rb') as src:
                    for block in iter(lambda:src.read(8*1024**2),b''):
                        out.write(block);source_hash.update(block);combined.update(block);size+=len(block)
            if size!=spec['size'] or source_hash.hexdigest()!=spec['sha']:
                raise RuntimeError('Full asset SHA256 mismatch: '+spec['name'])
            total+=size
            print('FULL_ASSET_HASH_VERIFIED',spec['name'],spec['sha'],flush=True)
    temp.replace(destination)
    return dict(path=destination.name,bytes=total,sha256=combined.hexdigest())
