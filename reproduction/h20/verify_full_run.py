"""Gate success on final weights and complete seven-dataset outcome metrics."""
import json
import os
from pathlib import Path
import re
from safetensors import safe_open
BASE=Path('/data/search-r1')
ckpt=BASE/'checkpoints/paper-grpo-qwen2.5-3b/actor/global_step_500'
index=json.loads((ckpt/'model.safetensors.index.json').read_text())
files={ckpt/f for f in index['weight_map'].values()}
assert files and all(f.is_file() and f.stat().st_size>0 for f in files)
assert all(f.stat().st_mtime>=float(os.environ['SEARCH_R1_RUN_STARTED']) for f in files), 'Stale final checkpoint'
for f in files:
    with safe_open(str(f),framework='pt',device='cpu') as reader:
        assert set(reader.keys())=={k for k,v in index['weight_map'].items() if ckpt/v==f}, 'Incomplete checkpoint shard'
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
print(json.dumps({'final_checkpoint':str(ckpt),'completed_updates':500,'metrics':metrics},indent=2))
