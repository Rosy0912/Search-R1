import json,hashlib,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-formal-2971/output/checkpoints/tree_grpo-qwen2.5-3b');h=p/'actor/global_step_5';index=json.loads((h/'model.safetensors.index.json').read_text());shards=sorted(set(index['weight_map'].values()));assert len(shards)==3 and all((h/s).is_file() and (h/s).stat().st_size>0 for s in shards)
a=p/'algorithm/global_step_5/state.json';state=json.loads(a.read_text());assert state['method']=='tree_grpo' and state['global_step']==5
print(json.dumps(dict(status='CHECKPOINT5_FILES_PRESENT_NOT_FULL_RELOAD_ACCEPTANCE',epoch=time.time(),job=2971,checkpoint=str(h),files={f.name:f.stat().st_size for f in h.iterdir() if f.is_file()},tensor_keys=len(index['weight_map']),index_sha256=hashlib.sha256((h/'model.safetensors.index.json').read_bytes()).hexdigest(),algorithm_state_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),algorithm_state=state),ensure_ascii=False))
