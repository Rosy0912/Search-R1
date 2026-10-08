"""CPU-only exact tile/shard-boundary fixture; never live GPU acceptance."""
import hashlib
import json
from pathlib import Path
import argparse
import faiss
import numpy as np
import torch
import torch_streaming_flat
from torch_streaming_flat import TorchStreamingFlatIP
from resident_scan import ResidentTiledFlatIP
import resident_scan
from operator_trace import RecordMM


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(16); faiss.omp_set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rng=np.random.default_rng(42)
    checks=[]
    for fixture in ['random','ties']:
        vectors=rng.standard_normal((37,9)).astype(np.float32)
        if fixture=='ties':vectors[:]=0
        index=faiss.IndexFlatIP(9);index.add(vectors)
        original=TorchStreamingFlatIP(index,'cpu',vector_rows=8,query_micro=64)
        resident=ResidentTiledFlatIP(index,['cpu']*4,vector_rows=8,query_micro=64)
        lengths=[0,1,63,64,65,128,129,320]
        queries=[rng.standard_normal((n,9)).astype(np.float32) for n in lengths]
        for parity in [0,1]:
            ks=[3+(i+parity)%2 for i in range(len(lengths))]
            trace=[]
            with RecordMM() as actual:
                arrays=resident.search_many(queries,ks,trace.append)
            declared=[[e['left'],e['right']] for e in trace if e['event']=='gemm']
            assert actual.shapes==declared
            for i,(q,k,(scores,ids)) in enumerate(zip(queries,ks,arrays)):
                with RecordMM() as oracle:
                    reference_s,reference_i=original.search(q,k)
                per_request=[[e['left'],e['right']] for e in trace if e['event']=='gemm' and e['request']==i]
                check=dict(fixture=fixture,queries=len(q),topk=k,scores_exact=bool(np.array_equal(scores,reference_s)),ids_exact=bool(np.array_equal(ids,reference_i)),shape_exact=per_request==oracle.shapes)
                assert all(check[key] for key in ['scores_exact','ids_exact','shape_exact']),check
                checks.append(check)
    receipt=dict(accepted_cpu_fixture=True,full_gpu_accepted=False,scope='CPU fixture only: four logical shard boundaries, tail tiles, empty/query tails and tied ranking. Does not test CUDA copies or cross-GPU arithmetic.',checks=checks,original_sha256=hashlib.sha256(Path(torch_streaming_flat.__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(resident_scan.__file__).read_bytes()).hexdigest(),cuda_initialized=torch.cuda.is_initialized())
    assert receipt['cuda_initialized'] is False
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(accepted_cpu_fixture=True,checks=len(checks),output=str(args.output))))


if __name__=='__main__':main()
