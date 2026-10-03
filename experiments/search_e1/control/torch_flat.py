"""Full FP32 Flat-IP search, with all vectors resident across selected devices.

No quantization or ANN: per-shard top-k candidates contain the global top-k.
GPU/CPU rounding and exact-score ties can change ordering; acceptance compares
scores and IDs, and does not claim bitwise equivalence.
"""
import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor

class TorchShardedFlatIP:
 def __init__(self,index,devices,query_micro=64):
  if index.metric_type!=faiss.METRIC_INNER_PRODUCT or not hasattr(index,'get_xb'):
   raise ValueError('Only the verified flat inner-product index is supported')
  if not 0<query_micro<=64:raise ValueError('Query microbatch must be in 1..64')
  self.ntotal=int(index.ntotal);self.d=int(index.d);self.metric_type=faiss.METRIC_INNER_PRODUCT
  self.cpu_index=index;self.query_micro=query_micro
  self.host=faiss.rev_swig_ptr(index.get_xb(),self.ntotal*self.d).reshape(self.ntotal,self.d)
  if not devices:raise ValueError('Nonempty devices required')
  self.bounds=[(self.ntotal*i//len(devices),self.ntotal*(i+1)//len(devices)) for i in range(len(devices))]
  if not devices or any(end<=start for start,end in self.bounds):raise ValueError('Nonempty shards required')
  self.shards=[]
  for (start,end),device in zip(self.bounds,devices):
   tensor=torch.from_numpy(self.host[start:end]).to(torch.device(device))
   self.shards.append((start,tensor))
  self.pool=ThreadPoolExecutor(max_workers=len(devices))
 @torch.no_grad()
 def _search_shard(self,shard,queries,k):
  start,vectors=shard;q=torch.from_numpy(queries).to(vectors.device)
  distances=q@vectors.T
  scores,ids=torch.topk(distances,min(k,len(vectors)),dim=1,sorted=True)
  return scores.cpu().numpy(),ids.cpu().numpy()+start
 def search(self,queries,k):
  q=np.ascontiguousarray(queries,dtype=np.float32)
  if q.ndim!=2 or q.shape[1]!=self.d or not np.isfinite(q).all():raise ValueError('Invalid query embeddings')
  if not 0<int(k)<=self.ntotal:raise ValueError('Invalid top-k')
  all_scores=[];all_ids=[]
  for offset in range(0,len(q),self.query_micro):
   batch=q[offset:offset+self.query_micro]
   futures=[self.pool.submit(self._search_shard,shard,batch,int(k)) for shard in self.shards]
   candidates=[f.result() for f in futures]
   scores=np.concatenate([x[0] for x in candidates],axis=1)
   ids=np.concatenate([x[1] for x in candidates],axis=1)
   # Stable order for equal scores across shards; shard-local tied IDs are
   # backend-defined, so numerical-tie differences are reported separately.
   order=np.argsort(-scores,axis=1,kind='stable')[:,:int(k)]
   all_scores.append(np.take_along_axis(scores,order,axis=1))
   all_ids.append(np.take_along_axis(ids,order,axis=1))
  if not len(q):return np.empty((0,k),np.float32),np.empty((0,k),np.int64)
  return np.concatenate(all_scores),np.concatenate(all_ids)
