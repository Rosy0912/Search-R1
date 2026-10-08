import os,time,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import torch,numpy as np
from search_r1.search.retrieval_server import Encoder
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo')
out=root/'runs/bapo-fp32-cpu-parity-20261003/encoder-fp32-512-long.json'
assert not out.exists()
torch.set_num_threads(8)
encoder=Encoder('e5',str(root/'assets-root/assets/e5-base-v2'),'mean',256,False,'cpu')
queries=['Which scientific observations describe the relationship between planets, their moons, and the stars in the sky? '*32]*512
c=time.process_time();t=time.perf_counter();x=encoder.encode(queries)
result={'status':'SYNTHETIC_CPU_THROUGHPUT_ONLY','device':'cpu','dtype':'float32','query_count':len(queries),'max_length':256,'threads':8,'queries_sha256':hashlib.sha256(json.dumps(queries).encode()).hexdigest(),'shape':list(x.shape),'finite':bool(np.isfinite(x).all()),'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-c,'actual_outer_updates':0}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
