"""Owner-run CPU loading launcher. GPU activation requires a separate receipt."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text())
    base=Path(config['candidate_base'])
    if base!=Path('/data/shaoshitong/search-e1-retriever-resident-20261008'):raise RuntimeError('Invalid independent base')
    if config['physical_gpus']!=[0,1,2,3] or config['port']!=18005:raise RuntimeError('Candidate GPU/port scope differs')
    if (base/'receipts/enable-gpu.json').exists() or (base/'receipts/admit-requests.json').exists():raise RuntimeError('Fresh CPU startup requires absent activation/admission controls')
    if len(os.environ.get('E1_RESIDENT_PROBE_TOKEN',''))<32:raise RuntimeError('Set private probe token externally; never commit it')
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',E1_BASE=str(base),E1_ENCODER_DEVICE='cpu',NVIDIA_TF32_OVERRIDE='0',PYTHONPATH=str(base/'source')+':'+str(base/'runtime/upstream/Search-R1'))
    python=str(base/'env/bin/python')
    argv=[python,str(base/'source/server.py'),'--base',str(base),'--assets',str(base/'assets'),'--asset-manifest',str(base/'assets/manifest.json'),'--asset-bindings',str(base/'receipts/asset-bindings.json'),'--host',config['host'],'--port','18005','--gpu-uuids',','.join(config['gpu_uuids'])]
    os.execve(python,argv,env)


if __name__=='__main__':main()
