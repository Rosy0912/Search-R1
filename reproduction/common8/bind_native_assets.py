"""Bind an isolated code deployment to existing remote assets without local tensors."""
import argparse,hashlib,json,os,time
from pathlib import Path

def bind(base):
    base=Path(base).resolve();assert str(base).startswith('/mnt/data/shaoshitong/search-g1-')
    old=Path('/mnt/data/shaoshitong/g1f-ef498ac3')
    assert (base/'repo/reproduction/common8/launch_native.sh').is_file()
    assert not (base/'logs').exists()
    links={n:old/n for n in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib')}
    for name,target in links.items():
        assert target.exists();(base/name).symlink_to(target.resolve(),target_is_directory=True)
    for name in ('data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/name).mkdir(exist_ok=False)
    for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
        target=old/'data'/name;assert target.is_file();(base/'data'/name).symlink_to(target.resolve())
    target=old/'preparation/official-model-manifest-original.json';assert target.is_file()
    (base/'preparation/official-model-manifest-original.json').write_bytes(target.read_bytes())
    proof=dict(base=str(base),outputs_remote=str((base/'checkpoints').resolve()),
        links={n:str((base/n).resolve()) for n in links},training_started=False,created_unix=time.time())
    with (base/'preparation/asset-binding.json').open('x') as f:json.dump(proof,f);f.flush();os.fsync(f.fileno())
    return proof
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);print(json.dumps(bind(p.parse_args().base)))
