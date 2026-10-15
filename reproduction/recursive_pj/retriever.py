"""PJ resident FP32 retrieval using the same encoder operations and GPU scan as Lumia."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ASSETS = Path('/plm-shared/zhangjunming/Workspace/Search-Nested-PG-Public-Assets-20261009-v2')
HASHES = {'e5_Flat.index':'69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166',
          'wiki-18.jsonl':'85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9',
          'encoder/model.safetensors':'d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693'}


def main():
    import sys
    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from reproduction.recursive_lumia.gpu_index import FourGPUFlatIP
    from release.recursive_bran_a.ablation150_v1.contracts import sha
    base = Path(sys.argv[1]); job = os.environ['SEARCH_R1_JOB_ID']
    assert os.environ['SEARCH_R1_EXECUTION_BACKEND'] == 'pj_pt'
    assert torch.cuda.device_count() == 4 and not os.environ.get('SLURM_JOB_ID')
    directory = base/'receipts'/('retriever-'+job); directory.mkdir(exist_ok=False)
    for name, expected in HASHES.items():
        assert sha(ASSETS/name) == expected, name
        print('ASSET_VERIFIED='+name, flush=True)
    torch.set_num_threads(48); faiss.omp_set_num_threads(48)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    index = faiss.read_index(str(ASSETS/'e5_Flat.index'))
    scan = FourGPUFlatIP(index)
    assert scan.ntotal == 21015324 and scan.d == 768
    corpus = ASSETS/'wiki-18.jsonl'; offsets = []
    with tarfile.open(corpus, 'r:') as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('.jsonl')]
        assert len(members) == 1
        member = members[0]; position = member.offset_data; end = position+member.size
        with archive.extractfile(member) as f:
            for line in f:
                if line.strip(): offsets.append(position)
                position += len(line)
        assert position == end and len(offsets) == index.ntotal
    offsets = np.asarray(offsets, dtype=np.int64)
    handle = corpus.open('rb'); mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    def doc(i):
        value = json.loads(mapping[int(offsets[i]):int(offsets[i+1]) if i+1<len(offsets) else end])
        assert int(value['id']) == i
        return value
    model = AutoModel.from_pretrained(ASSETS/'encoder', local_files_only=True).eval().cpu().float()
    tokenizer = AutoTokenizer.from_pretrained(ASSETS/'encoder', use_fast=True, local_files_only=True)
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
    @torch.no_grad()
    def encode(queries):
        # Same prefix, padding, pooling, normalization, precision and batch512 as native Encoder.
        parts = []
        for start in range(0, len(queries), 512):
            inputs = tokenizer(['query: '+q for q in queries[start:start+512]], max_length=256,
                               padding=True, truncation=True, return_tensors='pt')
            output = model(**inputs, return_dict=True)
            hidden = output.last_hidden_state.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.)
            pooled = hidden.sum(dim=1)/inputs['attention_mask'].sum(dim=1)[..., None]
            parts.append(torch.nn.functional.normalize(pooled, dim=-1).numpy().astype(np.float32, order='C'))
        return np.concatenate(parts) if parts else np.empty((0,768),np.float32)
    with (base/'common-data/raw/nq/train.jsonl').open() as f:
        queries = [json.loads(next(f))['question'] for _ in range(64)]
    vectors = encode(queries)
    expected_s, expected_i = index.search(vectors, 3); actual_s, actual_i = scan.search(vectors, 3)
    assert np.array_equal(expected_i, actual_i)
    error = float(np.max(np.abs(expected_s-actual_s))); assert error <= 2e-5
    assert [[doc(int(i)) for i in row] for row in expected_i[:4]] == [[doc(int(i)) for i in row] for row in actual_i[:4]]
    guard = threading.Lock()
    record = dict(at=time.time(), root=str(base), job=job, execution_backend='pj_pt',
        pid=os.getpid(), host=socket.gethostname(), url='http://'+socket.gethostbyname(socket.gethostname())+':29147',
        source=os.environ['RECURSIVE_SOURCE_COMMIT'], encoder_dtype='float32', index_dtype='float32',
        versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(4)], assets=HASHES,
        validation=dict(queries=64,top3_identical=True,max_score_error=error,documents_identical=True,
                        universal_equivalence_claim=False))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def respond(self, value):
            body=json.dumps(value).encode();self.send_response(200)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path == '/health': self.respond(record)
            else: self.send_error(404)
        def do_POST(self):
            if self.path != '/retrieve': return self.send_error(404)
            started=time.time()
            try:
                size=int(self.headers['Content-Length']);assert 0<size<=32*1024**2
                payload=json.loads(self.rfile.read(size));queries=payload['queries']
                assert isinstance(queries,list) and len(queries)<=2560 and all(isinstance(q,str) for q in queries)
                assert payload.get('topk',3) == 3
                with guard:
                    scores,ids=scan.search(encode(queries),3)
                    result=[[dict(document=doc(int(i)),score=float(s)) for i,s in zip(row,sc)] for row,sc in zip(ids,scores)]
                    if not payload.get('return_scores',False): result=[[x['document'] for x in row] for row in result]
                self.respond(dict(result=result))
                with (directory/'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(at=time.time(),seconds=time.time()-started,queries=len(queries)))+'\n')
            except Exception as exc:
                self.send_error(500,type(exc).__name__)
                raise
    server=ThreadingHTTPServer(('0.0.0.0',29147),Handler)
    with (base/'receipts/pj-retriever-ready.json').open('x') as f: json.dump(record,f,indent=2)
    print('PJ_RETRIEVER_READY='+json.dumps(record),flush=True)
    server.serve_forever()

if __name__ == '__main__': main()
