"""Materialize only verified immutable assets, never overwrite a finished output."""
import gzip, hashlib, json, os, subprocess, sys
from pathlib import Path
root=Path(os.environ['SEARCH_R1_BASE']); assets=Path(os.environ['SEARCH_R1_ASSETS']); repo=root/'repo'
subprocess.run([sys.executable,str(repo/'reproduction/h20/verify_assets.py')],check=True)
wiki=assets/'assets/wiki'
manifest=json.loads((assets/'assets.json').read_text())
print('WIKI_MANIFEST',[(x['path'],x['size']) for x in manifest if x['path'].startswith('wiki/')],flush=True)
# Fixed manifest names, not unverified wildcard ordering.
parts=[wiki/x['path'].split('/',1)[1] for x in manifest if x['path'].startswith('wiki/') and 'part_' in x['path']]
assert len(parts)==2 and 'part_aa' in parts[0].name and 'part_ab' in parts[1].name
outputs={}
for name, sources in [('e5_Flat.index',parts),('wiki-18.jsonl',[wiki/'wiki-18.jsonl.gz'])]:
    dest=wiki/name; h=hashlib.sha256(); size=0
    expected=hashlib.sha256(); expected_size=0
    staged=dest.with_name(name+'.preparing')
    writer=None if dest.exists() else staged.open('wb')
    try:
        for src in sources:
            opener=gzip.open if src.suffix=='.gz' else open
            with opener(src,'rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    expected.update(block); expected_size+=len(block)
                    if writer is not None: writer.write(block)
    finally:
        if writer is not None: writer.close()
    candidate=dest if dest.exists() else staged
    with candidate.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block);size+=len(block)
    assert size==expected_size and h.hexdigest()==expected.hexdigest(), 'Materialized asset differs from immutable sources'
    if candidate==staged: staged.replace(dest)
    outputs[name]=dict(bytes=size,sha256=h.hexdigest())
subprocess.run([sys.executable,str(repo/'reproduction/h20/prepare_paper_data.py')],check=True)
(root/'logs/materialized-assets.json').write_text(json.dumps(outputs,indent=2))
print('ASSET_MATERIALIZATION_READY',json.dumps(outputs),flush=True)
(root/'logs/data.ready').touch()
