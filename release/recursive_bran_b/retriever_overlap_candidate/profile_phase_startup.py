"""Actual full-index cold/warm/release timing on authorized devices only.

This does not admit a retriever or change training configuration. Assets remain
remote; a small JSON receipt and stage timestamps explain phase overhead.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from gpu_phase_proxy import PhaseGpuScanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3'
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(80)
    q=rng.normal(size=(65,768)).astype(np.float32)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    original=SimpleNamespace(d=768,ntotal=21015324,vector_rows=32768,query_micro=64)
    scanner=PhaseGpuScanner(original,a.index,gpu_count=4)
    receipt=dict(scope='full_index_phase_cost_only',gpu_devices=[0,1,2,3],started_at=time.time(),scans=[])
    try:
        for count in (1,65):
            t=time.perf_counter();scanner.search_many([q[:count]],[3])
            row=dict(queries=count,seconds=time.perf_counter()-t,
                worker_pid=scanner.worker_pid,tiles=scanner.last_scan_profile['tiles_merged'])
            receipt['scans'].append(row)
            print('PHASE_COST='+json.dumps(row),flush=True)
        receipt['release']=scanner.release_gpu()
        receipt['parent_cuda_initialized']=torch.cuda.is_initialized()
    except BaseException as error:
        receipt['error']=repr(error)
        raise
    finally:
        if scanner.worker_pid is not None:receipt['final_release']=scanner.release_gpu()
        receipt['finished_at']=time.time()
        Path(a.receipt).write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
