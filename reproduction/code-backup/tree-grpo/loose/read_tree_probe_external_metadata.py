import hashlib,json,socket
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools')
wanted={'resource_guard_identity.py':'284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88','tree_shared_owned_runtime_fixed.py':'4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0','expected-gpu-uuids.json':'afefc6b20298a0cfb9594b70988369ed529e4c3088e40d26c3486ce0c2c50127'}
records=[]
for name,digest in wanted.items():
 p=root/name;b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==digest
 records.append(dict(path=str(p),sha256=actual,bytes=len(b)))
print(json.dumps(dict(status='EXACT_SAME_NODE_EXTERNAL_SOURCE_METADATA_VERIFIED',records=records,actual_gpu_calls=0,tensor_assets_loaded=False,optimizer_writes=0),sort_keys=True))
