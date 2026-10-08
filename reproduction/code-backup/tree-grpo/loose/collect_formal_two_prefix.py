import json,hashlib,re,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-formal-2971')
files={}
for f in [p/'input-config.yaml',p/'launch-manifest.json',p/'resolved.yaml',p/'output/entry-audit/entry-config.yaml',p/'output/entry-audit/entry.jsonl',*sorted((p/'output/tree-audit').glob('step_[12]_rollout_*.json')),*sorted((p/'output/validation-cost').glob('*')), *sorted((p/'output/tree-rank-audit').glob('*.json'))]:
 if f.is_file() and f.stat().st_size<20_000_000:
  raw=f.read_bytes(); files[str(f.relative_to(p))]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
log=(p/'stage.log').read_text()
lines=[x for x in log.splitlines() if any(s in x for s in ['TREE_VALIDATION_COST_INSTALLED_JSON=','BASELINE_EVALUATION_METRICS_JSON=','training/actual_outer_updates'])]
print(json.dumps({'run':str(p),'collected_epoch':time.time(),'files':files,'selected_stage_log':lines},ensure_ascii=False))
