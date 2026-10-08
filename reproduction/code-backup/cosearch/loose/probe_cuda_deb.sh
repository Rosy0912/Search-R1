set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
curl -fsSL --connect-timeout 15 --max-time 60 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz -o "$A/cache/cuda-packages.gz"
python3 - "$A" <<'PY'
import gzip,json,sys
from pathlib import Path
r=Path(sys.argv[1]);pkgs=[]
for block in gzip.decompress((r/'cache/cuda-packages.gz').read_bytes()).decode().split('\n\n'):
 d={}
 for line in block.splitlines():
  if ': ' in line and not line.startswith(' '):k,v=line.split(': ',1);d[k]=v
 if d.get('Package') in ['cuda-nvcc-12-8','cuda-cudart-dev-12-8','cuda-cccl-12-8','cuda-crt-12-8','cuda-cudart-12-8','cuda-nvvm-12-8','cuda-driver-dev-12-8']:
  pkgs.append(d)
(r/'tools/cuda-debs.json').write_text(json.dumps(pkgs,indent=2))
for x in pkgs:print(x.get('Package'),x.get('Version'),x.get('Filename'),x.get('Depends'))
PY
