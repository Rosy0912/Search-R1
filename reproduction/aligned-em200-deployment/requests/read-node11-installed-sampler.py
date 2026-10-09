import json,os,hashlib,subprocess
from pathlib import Path
assert os.uname().nodename=='h100-11'
p=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/env/lib/python3.10/site-packages/vllm/model_executor/layers/sampler.py');raw=p.read_bytes();lines=raw.decode().splitlines()
out={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'sampling_kernel_source':'\n'.join(lines[380:530]),'sampling_entry_source':'\n'.join(lines[730:880]),'dmesg':subprocess.run(['dmesg','--level=err,warn'],capture_output=True,text=True).stdout[-10000:]}
print(json.dumps(out))
