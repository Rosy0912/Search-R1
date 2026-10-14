"""Resident four-device FP32 FlatIP scan with the published tile dimensions."""
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import torch

class FourGPUFlatIP:
    def __init__(self,index):
        import faiss
        assert torch.cuda.device_count()==4 and index.metric_type==faiss.METRIC_INNER_PRODUCT
        self.ntotal,self.d,self.metric_type=index.ntotal,index.d,index.metric_type
        self.oracle=index
        vectors=faiss.rev_swig_ptr(index.get_xb(),self.ntotal*self.d).reshape(self.ntotal,self.d)
        tiles=[(i,min(i+32768,self.ntotal)) for i in range(0,self.ntotal,32768)]
        self.groups=[tiles[len(tiles)*i//4:len(tiles)*(i+1)//4] for i in range(4)]
        self.vectors=[]
        self.pool=ThreadPoolExecutor(max_workers=4)
        self.lock=threading.Lock()
        for gpu,group in enumerate(self.groups):
            first,last=group[0][0],group[-1][1]
            self.vectors.append(torch.from_numpy(vectors[first:last]).to('cuda:'+str(gpu)))
            torch.cuda.synchronize(gpu)

    def search(self,queries,k):
        queries=np.ascontiguousarray(queries,dtype=np.float32)
        assert queries.ndim==2 and queries.shape[1]==self.d and np.isfinite(queries).all()
        assert 1<=k<=100 and len(queries)<=2560
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        if len(queries)==0:return np.empty((0,k),np.float32),np.empty((0,k),np.int64)
        def worker(gpu):
            with torch.cuda.device(gpu),torch.no_grad():
                q=torch.from_numpy(queries).to('cuda:'+str(gpu));group=self.groups[gpu];origin=group[0][0]
                best_s=torch.full((len(q),k),float('-inf'),device=q.device)
                best_i=torch.full((len(q),k),-1,dtype=torch.long,device=q.device)
                for first,last in group:
                    matrix=self.vectors[gpu][first-origin:last-origin]
                    for offset in range(0,len(q),64):
                        stop=min(offset+64,len(q))
                        scores,ids=torch.topk(q[offset:stop]@matrix.T,min(k,last-first),dim=1)
                        joined_s=torch.cat((best_s[offset:stop],scores),dim=1)
                        joined_i=torch.cat((best_i[offset:stop],ids+first),dim=1)
                        values,choice=torch.topk(joined_s,k,dim=1)
                        best_s[offset:stop]=values;best_i[offset:stop]=torch.gather(joined_i,1,choice)
                torch.cuda.synchronize(gpu)
                return best_s.cpu(),best_i.cpu()
        with self.lock:
            parts=list(self.pool.map(worker,range(4)))
            scores=torch.cat([x[0] for x in parts],dim=1);ids=torch.cat([x[1] for x in parts],dim=1)
            values,choice=torch.topk(scores,k,dim=1)
            selected=torch.gather(ids,1,choice)
            assert torch.isfinite(values).all() and (selected>=0).all()
            return values.numpy(),selected.numpy()
