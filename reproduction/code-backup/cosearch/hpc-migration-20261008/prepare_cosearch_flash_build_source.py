"""Freeze original cached FlashAttention sources, excluding generated build outputs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path('/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch/author-native/cache/uv/sdists-v9/pypi/flash-attn/2.8.1/gq4V0ZAYZdcQdbBq/src')
ROOT = Path('/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch/.migration-flash-build-20261009')

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):
            h.update(b)
    return h.hexdigest()

def main():
    lock = (ROOT / 'selection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert 'Version: 2.8.1\n' in (SOURCE / 'PKG-INFO').read_text()
    (ROOT / 'selection-controller.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'source':str(SOURCE)}))
    manifest = ROOT / 'manifest.jsonl'
    if not manifest.exists():
        rows = []
        for directory, dirs, files in os.walk(str(SOURCE)):
            dirs[:] = sorted(d for d in dirs if d not in ('build','__pycache__') and not d.endswith('.egg-info'))
            for name in sorted(files):
                if name.endswith(('.so','.o','.pyc')):
                    continue
                p = Path(directory) / name
                rel = str(p.relative_to(SOURCE))
                assert not p.is_symlink(), rel
                rows.append({'kind':'file','path':rel,'bytes':p.stat().st_size,'sha256':digest(p)})
        assert any(x['path']=='setup.py' for x in rows)
        assert sum(x['bytes'] for x in rows) < 500000000
        rows.append({'kind':'complete','files':len(rows),'bytes':sum(x['bytes'] for x in rows),'source':str(SOURCE),'version':'2.8.1'})
        tmp = manifest.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows));os.replace(tmp,manifest)
    subprocess.run([sys.executable,str(ROOT/'prepare_cosearch_hf_archive.py'),str(SOURCE),str(ROOT),'flash-attn-281-build-source'],check=True)

if __name__ == '__main__':
    main()
