import ast,sys,unittest,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch,faiss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from joined_scan import install_joined_scan
from torch_streaming_flat import TorchStreamingFlatIP
UP=Path(__file__).resolve().parents[1]/'archive/upstream/Search-R1'
tree=ast.parse((UP/'search_r1/search/retrieval_server.py').read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DenseRetriever')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_batch_search')
for a in fn.args.args:a.annotation=None
fn.returns=None
load_docs=lambda corpus,ids:[corpus[int(i)] for i in ids]
ns=dict(torch=torch,tqdm=lambda x,**kw:x,load_docs=load_docs)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-dense-retriever>', 'exec'),ns)
class JoinedTests(unittest.TestCase):
 def test_original_encode_calls_scores_docs_and_single_scan(self):
  rng=np.random.default_rng(9);vectors=rng.normal(size=(137,31)).astype('float32');queries=rng.normal(size=(513,31)).astype('float32')
  idx=faiss.IndexFlatIP(31);idx.add(vectors)
  def retriever():
   calls=[];scans=[];backend=TorchStreamingFlatIP(idx,'cpu',vector_rows=17,query_micro=64)
   def encode(ids):calls.append(list(ids));return queries[ids].copy()
   def search(q,k):scans.append(len(q));return backend.search(q,k)
   return SimpleNamespace(batch_size=128,topk=3,corpus=[dict(id=i,text=str(i)) for i in range(137)],encoder=SimpleNamespace(encode=encode),index=SimpleNamespace(search=search),calls=calls,scans=scans)
  for n in [0,1,127,128,129,257,512,513]:
   a,b=retriever(),retriever();install_joined_scan(b,load_docs)
   for scores in [True,False]:
    a.calls.clear();b.calls.clear();a.scans.clear();b.scans.clear()
    old=ns['_batch_search'](a,list(range(n)),return_score=scores);new=b._batch_search(list(range(n)),return_score=scores)
    self.assertEqual(old,new);self.assertEqual(a.calls,b.calls);self.assertEqual(len(b.scans),int(n>0))
if __name__=='__main__':unittest.main()
