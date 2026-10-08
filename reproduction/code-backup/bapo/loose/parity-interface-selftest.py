import importlib.util, json, sys, types
from pathlib import Path
import numpy as np
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3')
w=root/'work/parity-interface-selftest';w.mkdir(exist_ok=True)
model=w/'model';model.mkdir(exist_ok=True);(model/'config.json').write_text('{}')
(w/'index').write_bytes(b'FAKE_INDEX_SELFTEST');(w/'corpus').write_text('{}\n')
torch=types.ModuleType('torch');torch.__version__='MOCK_CPU_ONLY';torch.set_num_threads=lambda n:None
class Cuda:
 def __getattr__(self,k):raise AssertionError('CPU mode attempted CUDA')
torch.cuda=Cuda();sys.modules['torch']=torch
faiss=types.ModuleType('faiss');faiss.__version__='MOCK_CPU_ONLY';faiss.__file__='SELFTEST';faiss.omp_set_num_threads=lambda n:None
sys.modules['faiss']=faiss
server=types.ModuleType('search_r1.search.retrieval_server')
class Config:
 def __init__(self,**kw):self.__dict__.update(kw)
class Index:
 d=2;ntotal=100
 def search(self,emb,k):return np.tile(np.linspace(1,.5,k,dtype=np.float32),(len(emb),1)),np.tile(np.arange(k,dtype=np.int64),(len(emb),1))
class Retriever:
 def __init__(self,c):
  assert not c.faiss_gpu and c.encoder_device=='cpu' and c.retrieval_use_fp16
  self.index=Index();self.corpus=[{'id':str(i),'contents':'doc '+str(i)} for i in range(100)]
  self.encoder=types.SimpleNamespace(encode=lambda qs:np.ones((len(qs),2),dtype=np.float32))
server.Config=Config;server.DenseRetriever=Retriever;server.load_docs=lambda c,ids:[c[i] for i in ids]
sys.modules['search_r1']=types.ModuleType('search_r1');sys.modules['search_r1.search']=types.ModuleType('search_r1.search');sys.modules['search_r1.search.retrieval_server']=server
spec=importlib.util.spec_from_file_location('parity',root/'outputs/bapo-retrieval-parity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for name in ('cpu.json','compare.json'):
 p=w/name
 if p.exists():p.unlink()
sys.argv=['parity','--mode','cpu','--repo',str(root/'work/bapo'),'--work-root',str(w),'--index',str(w/'index'),'--corpus',str(w/'corpus'),'--model',str(model),'--hash-large-assets','--output',str(w/'cpu.json')]
m.main();a=json.loads((w/'cpu.json').read_text());assert len(a['results'])==14
assert all(len(row['50']['ids'])==50 and len(row['3']['ids'])==3 for row in a['results'])
b=json.loads(json.dumps(a));b['mode']='SYNTHETIC_SECOND_RESULT'
for key in ('ids','scores','contents_sha256','corpus_ids'):
 b['results'][0]['3'][key][0],b['results'][0]['3'][key][1]=b['results'][0]['3'][key][1],b['results'][0]['3'][key][0]
b['results'][0]['50']['scores'][0]+=.01
report=m.compare(a,b)
assert report['summary']['3']['exact_rank_queries']==13
assert report['summary']['3']['mean_overlap_fraction']==1
assert report['summary']['50']['max_matched_id_score_abs_delta']>.009
(w/'compare.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('CPU MOCK INTERFACE ONLY: 14 fixed queries, top3/top50 IDs/scores/doc hashes; zero CUDA use; rank swap and same-ID score delta detected. Not real retrieval evidence.')
