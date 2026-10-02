"""Gate success on final weights and complete seven-dataset outcome metrics."""
import json
import os
from pathlib import Path
import re
import torch
from safetensors import safe_open
BASE=Path('/data/search-r1')
ckpt=BASE/'checkpoints/paper-grpo-qwen2.5-3b/actor/global_step_500'
def validate_saved_checkpoint(directory, canonical, run_started):
    """Match canonical serialized keys/shapes; allow reshards and stream finite checks."""
    index_path=directory/'model.safetensors.index.json'
    saved=json.loads(index_path.read_text())['weight_map']
    expected=json.loads((canonical/'model.safetensors.index.json').read_text())['weight_map']
    assert expected and set(saved)==set(expected), 'Checkpoint keys differ from canonical model'
    paths={directory/f for f in saved.values()} | {index_path}
    assert all(p.is_file() and p.stat().st_size>0 and p.stat().st_mtime>=run_started for p in paths), 'Missing or stale checkpoint file'
    shapes={}
    for filename in set(expected.values()):
        with safe_open(str(canonical/filename),framework='pt',device='cpu') as reader:
            for key in (k for k,v in expected.items() if v==filename):
                shapes[key]=tuple(reader.get_slice(key).get_shape())
    for filename in set(saved.values()):
        with safe_open(str(directory/filename),framework='pt',device='cpu') as reader:
            keys={k for k,v in saved.items() if v==filename}
            assert set(reader.keys())==keys, 'Incomplete checkpoint shard'
            for key in keys:
                assert tuple(reader.get_slice(key).get_shape())==shapes[key], 'Checkpoint shape differs: '+key
                tensor=reader.get_tensor(key)
                assert torch.isfinite(tensor).all(), 'Nonfinite checkpoint tensor: '+key
                del tensor
    return len(saved)
validated_count=validate_saved_checkpoint(ckpt,BASE/'assets/qwen2.5-3b',float(os.environ['SEARCH_R1_RUN_STARTED']))
text=(BASE/'logs/paper-grpo-qwen2.5-3b.log').read_text(errors='replace')
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
metrics=None
for line in text.splitlines():
    if 'SEARCH_R1_FINAL_METRICS_JSON=' in line:
        metrics=json.loads(line.split('SEARCH_R1_FINAL_METRICS_JSON=',1)[1].strip())
assert isinstance(metrics,dict), 'Missing final validation'
expected=json.loads((BASE/'data/paper-grpo/dataset-audit.json').read_text())['eval_counts']
for source,count in expected.items():
    assert metrics['val/evaluated_count/'+source]==count, 'Incomplete evaluation '+source
    assert 0<=metrics['val/test_score/'+source]<=1, 'Invalid EM '+source
print(json.dumps({'final_checkpoint':str(ckpt),'completed_updates':500,
                  'validated_tensor_count':validated_count,'finite_all_saved_tensors':True,'metrics':metrics},indent=2))
