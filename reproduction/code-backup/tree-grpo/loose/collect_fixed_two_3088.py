import json,hashlib,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');out={}
for directory in (root/'runs/tree-fixed-two-3088',root/'runs/tree-fixed-two-cpu-3088-3089'):
 for path in directory.rglob('*'):
  if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log') and path.stat().st_size<3000000:
   data=path.read_bytes();out[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
for name in ('tree-fixed-two-3088.out','tree-fixed-two-hf-3089.out','tree-fixed-two-cpu-terminal-3090.out'):
 path=root/'logs'/name
 if path.exists():
  data=path.read_bytes();out[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
print(json.dumps({'status':'EVIDENCE_ONLY_NOT_ACCEPTANCE','files':out,'gpu_now':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}))
