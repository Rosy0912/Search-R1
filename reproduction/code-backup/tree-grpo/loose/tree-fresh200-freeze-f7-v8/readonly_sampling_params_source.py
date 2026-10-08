"""Read installed vLLM source metadata on h1009 without importing vLLM/CUDA."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket

assert socket.gethostname().split('.')[0]=='h100-9'
dist=importlib.metadata.distribution('vllm')
path=Path(dist.locate_file('vllm/sampling_params.py'))
raw=path.read_bytes();text=raw.decode();tree=ast.parse(text)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingParams')
lines=text.splitlines();selected=[]
for node in cls.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('__post_init__','_verify_args','_verify_non_beam_search','_verify_greedy_sampling'):
        selected.append({'name':node.name,'line':node.lineno,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])})
    elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ('n','best_of','_real_n'):
        selected.append({'name':node.target.id,'line':node.lineno,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])})
print(json.dumps({'status':'READ_ONLY_INSTALLED_VLLM_SOURCE_NOT_NATIVE_RUNTIME_ACCEPTANCE','node':socket.gethostname(),'version':dist.version,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'selected':selected,'model_or_PT_loads':0,'generation_calls':0,'optimizer_writes':0}))
