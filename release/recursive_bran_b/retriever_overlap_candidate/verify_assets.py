"""Once-only transfer checksums; no full-file hashing on the retrieval hot path."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True)
    p.add_argument('--corpus-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from flat_memmap import file_metadata
    assets=Path(a.assets);manifest=json.loads((assets/'manifest.json').read_text())
    out={}
    for key,name,wanted,size in [('index','index/e5_Flat.index',manifest['index']['files'][0]['sha256'],64559075373),
            ('corpus','corpus/wiki-18.jsonl',a.corpus_sha256,14393579520)]:
        f=assets/name;before=file_metadata(f);assert before['bytes']==size
        actual=digest(f);assert actual==wanted and file_metadata(f)==before
        out[key]=dict(sha256=actual,bytes=size,metadata=before)
        print('ASSET_VERIFIED='+json.dumps(dict(asset=key,bytes=size,sha256=actual)),flush=True)
    for row in manifest['encoder']['files']:
        f=assets/'encoder'/row['path'];assert f.stat().st_size==row['bytes'] and digest(f)==row['sha256']
    out['encoder_manifest_sha256']=hashlib.sha256(json.dumps(manifest['encoder'],sort_keys=True).encode()).hexdigest()
    Path(a.output).write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
