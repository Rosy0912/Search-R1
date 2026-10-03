"""Short-lived GPU encoder comparator; never leaves CUDA in the CPU service."""
import os,json,numpy as np,torch
from pathlib import Path
os.environ['E1_ENCODER_DEVICE']='cuda'
from search_r1.search import retrieval_server as server
base=Path(os.environ['E1_BASE']);queries=json.loads((base/'receipts/retriever-probe-queries.json').read_text())
config=server.Config()
encoder=server.Encoder('e5',str(base/'assets/encoder'),config.retrieval_pooling_method,config.retrieval_query_max_length,True)
embeddings=encoder.encode(queries)
if not np.isfinite(embeddings).all():raise FloatingPointError('Nonfinite FP16 probe embeddings')
np.save(base/'receipts/retriever-gpu-fp16-embeddings.npy',embeddings)
print('FP16_ENCODER_PROBE_COMPLETE',len(queries),flush=True)
