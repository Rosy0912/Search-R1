import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
base=Path('/data/search-r1/assets/qwen2.5-3b')
ckpt=Path('/data/search-r1/checkpoints/search-r1-reviewed-smoke/actor/global_step_2')
name='model.layers.0.self_attn.q_proj.weight'
def tensor(path):
    index=json.loads((path/'model.safetensors.index.json').read_text())
    files=[path/f for f in set(index['weight_map'].values())]+[path/'model.safetensors.index.json']
    assert all(f.is_file() and f.stat().st_size>0 for f in files)
    if path==ckpt and os.environ.get('SEARCH_R1_RUN_STARTED'):
        assert all(f.stat().st_mtime >= float(os.environ['SEARCH_R1_RUN_STARTED']) for f in files), 'Stale checkpoint from an earlier run'
    with safe_open(str(path/index['weight_map'][name]),framework='pt',device='cpu') as f:
        return f.get_tensor(name).float()
a,b=tensor(base),tensor(ckpt)
assert a.shape==b.shape and torch.isfinite(b).all()
delta=(b-a).abs()
result={'parameter':name,'shape':list(a.shape),'finite':True,'changed_elements':int((delta>0).sum()),'max_abs_delta':float(delta.max())}
assert result['changed_elements']>0, 'No update found'
print(json.dumps(result,indent=2))
