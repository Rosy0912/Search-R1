"""Independent CPU mirror of only the ten evaluation-boundary checkpoints.

Regular files stay on the same HKUST node; this is a second-inode persistence
copy, not off-node disaster recovery. No Torch, GPU, locks or source mutations.
"""
import argparse
from pathlib import Path
import shutil
try:
    from .utils import *
except ImportError:
    from utils import *

def run(request_ref):
    value,rp=checked_json(request_ref);source=value['checkpoint'];checkpoint,cp=checked_json(source)
    require(value['schema']=='opsd-pr100-boundary-mirror-request-v1' and value['completed_outers'] in range(10,101,10),'Wrong boundary mirror')
    require(checkpoint['accepted'] is True and checkpoint['run_id']==value['training_run_id'] and checkpoint['config_sha256']==value['training_config_sha256']
        and checkpoint['cursor']['outer_index']==value['completed_outers']-1,'Wrong source checkpoint identity')
    root=Path(cp['path']).parent;destination=Path(value['destination']).resolve()
    require(str(root).startswith('/data/') and str(destination).startswith('/data/') and not destination.exists()
        and not destination.is_relative_to(root) and not root.is_relative_to(destination),'Mirror must be fresh remote independent path')
    paths=sorted(p for p in root.rglob('*') if p.is_file())
    require(not any(p.is_symlink() for p in root.rglob('*')) and not any('pending' in p.name for p in paths),'Checkpoint contains alias/pending file')
    originals=[fileproof(p) for p in paths];destination.mkdir(parents=True)
    copies=[]
    for original in originals:
        src=Path(original['path']);target=destination/src.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target);proof=fileproof(target,original['sha256'])
        require(proof['bytes']==original['bytes'] and (proof['device'],proof['inode'])!=(original['device'],original['inode']),'Not independent exact copy')
        check_metadata([original]);copies.append(dict(source=original,destination=proof))
    check_metadata(originals)
    return atomic_json(destination.parent/(destination.name+'-complete.json'),dict(schema='opsd-pr100-boundary-mirror-complete-v1',accepted=True,
        request=rp,checkpoint=cp,completed_outers=value['completed_outers'],files=copies,member_bytes=sum(p['bytes'] for p in originals),
        file_count=len(originals),scope='Ten selected boundary checkpoints only; same-node separate inode copies, not disaster recovery.'),exclusive=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True);a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'Mirror must remain CPU-only')
    result=run(dict(path=a.request,sha256=a.request_sha256));print(json.dumps(dict(accepted=True,complete=result)))
if __name__=='__main__':main()
