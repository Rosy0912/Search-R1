"""Run-private node /tmp compilation cache, separate from shared durable assets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from reproduction.common8.native_ray import verify as verify_ray, identity
from reproduction.hpc3_qprobe.control import publish

NAMES={'TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor',
       'TORCH_EXTENSIONS_DIR':'extensions','TMPDIR':'tmp'}


def verify(base,run):
    assert run['execution_site']=='hpc3' and run['training_world_size']==4
    ray=Path(verify_ray(run,run['ray_tmp_dir']))
    proof=json.loads((base/'logs/node-jit-admission.json').read_text())
    root=ray/'jit'
    assert proof['run_id']==run['run_id'] and proof['root']==str(root)
    assert identity(root)==proof['root_identity']
    for name in NAMES.values():assert identity(root/name)==proof['directory_identities'][name]
    return root


def prepare(base,run):
    base=Path(base)
    assert run['execution_site']=='hpc3' and run['training_world_size']==4
    ray=Path(verify_ray(run,run['ray_tmp_dir']));root=ray/'jit'
    root.mkdir(mode=0o700,exist_ok=False)
    for name in NAMES.values():(root/name).mkdir(mode=0o700)
    value=dict(run_id=run['run_id'],root=str(root),root_identity=identity(root),
        directory_identities={name:identity(root/name) for name in NAMES.values()},
        shared_output_filesystem_device=base.stat().st_dev,node_tmp_filesystem_device=root.stat().st_dev,
        node_tmp_free_bytes=shutil.disk_usage(root).free,checked_unix=time.time(),
        scope='private node compilation/temp cache only; models/checkpoints/teacher data remain on shared remote storage',
        actual_compilation_pending=True)
    publish(base/'logs/node-jit-admission.json',value)
    assert verify(base,run)==root
    return {**{key:str(root/name) for key,name in NAMES.items()},'SEARCH_G1_NODE_JIT_ROOT':str(root)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    run=json.loads((args.base/'logs/run-manifest.json').read_text())
    print(verify(args.base,run))
