"""Start only the private CPU parent; GPU activation is deferred to real probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--asset-proof',required=True);p.add_argument('--base',required=True);a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
    source=Path(a.source);base=Path(a.base);source.relative_to(root);base.relative_to(root)
    assert not base.exists(), 'Reuse the existing service rather than duplicate it'
    base.mkdir();(base/'logs').mkdir();(base/'receipts').mkdir()
    token=base/'receipts/private-probe-token';token.write_text(secrets.token_hex(32));token.chmod(0o600)
    helper=source/'release/recursive_bran_b/retriever_overlap_candidate'
    env=dict(CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(source)+':'+str(helper),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',
        OPENBLAS_NUM_THREADS='16',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        NVIDIA_TF32_OVERRIDE='0',
        HF_DATASETS_OFFLINE='1',HF_HOME=str(root/'cache/hf'),
        HF_DATASETS_CACHE=str(root/'cache/phase-datasets'),XDG_CACHE_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'))
    python='/home/shaoshitong/search-e1-config-study-20261004/env/bin/python'
    cmd=['env']+[k+'='+v for k,v in env.items()]+[python,'-B','-u',str(helper/'phase_http_server.py'),
        '--assets',a.assets,'--asset-proof',a.asset_proof,'--base',a.base,'--token-file',str(token),'--gpu-count','4']
    command='exec '+shlex.join(cmd)+' > '+shlex.quote(str(base/'logs/server.log'))+' 2>&1'
    script=base/'receipts/cpu-launch.sh';script.write_text(command+'\n');script.chmod(0o700)
    physical_source=Path('/home/shaoshitong/search-e1-migration-20261011')/source.relative_to('/')
    wrapper=physical_source/'release/recursive_bran_b/view_hkust5.sh'
    # The already-running tmux server is outside this bwrap filesystem view.
    start=['bash',str(wrapper),'bash',str(script)]
    subprocess.run(['tmux','new-session','-d','-s',base.name,shlex.join(start)],check=True)
    record=dict(source=str(source),commit=(source/'SOURCE_COMMIT').read_text().strip(),base=str(base),
        session=base.name,gpu_activation=False,gpu_devices=[0,1,2,3],
        launch_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    (base/'receipts/cpu-start.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))


if __name__=='__main__':main()
