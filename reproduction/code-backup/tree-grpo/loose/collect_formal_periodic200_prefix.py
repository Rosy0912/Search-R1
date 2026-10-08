import json,hashlib,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-formal-2971');files={};cp=p/'output/checkpoints/tree_grpo-qwen2.5-3b';h=cp/'actor/global_step_200'
paths=[p/'resolved.yaml',p/'launch-manifest.json',p/'output/entry-audit/entry-config.yaml',p/'output/entry-audit/entry.jsonl',*sorted((p/'output/validation-cost').glob('*')),*sorted((p/'output/tree-rank-audit').glob('step-000200-rank-*.json')),cp/'algorithm/global_step_200/state.json',h/'model.safetensors.index.json']
for f in paths:
 if f.is_file():
  raw=f.read_bytes();assert len(raw)<2_000_000;files[str(f.relative_to(p))]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
index=json.loads((h/'model.safetensors.index.json').read_text());assert len(set(index['weight_map'].values()))==3 and all((h/s).stat().st_size>0 for s in set(index['weight_map'].values()))
print(json.dumps(dict(epoch=time.time(),run=str(p),files=files,checkpoint200_files={f.name:f.stat().st_size for f in h.iterdir() if f.is_file()},selected_stage_log=[line for line in (p/'stage.log').read_text().splitlines() if 'step:50 - ' in line or 'step:200 - ' in line or 'TREE_VALIDATION_COST_INSTALLED_JSON=' in line]),ensure_ascii=False))
