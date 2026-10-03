import pathlib,subprocess,json,sys
for p in pathlib.Path('/home/shaoshitong/miniconda3/envs').glob('*/bin/python'):
 try:
  code="import sys,importlib.metadata as m,json;names=['torch','vllm','transformers','flash-attn','datasets'];d={x.metadata.get('Name','').lower():x.version for x in m.distributions()};print(json.dumps(dict(python=sys.version.split()[0],packages={k:d.get(k) for k in names})))"
  out=subprocess.check_output([str(p),'-c',code],stderr=subprocess.DEVNULL,timeout=20).decode().strip()
  print(str(p),out,flush=True)
 except Exception as e:print(str(p),type(e).__name__,flush=True)
