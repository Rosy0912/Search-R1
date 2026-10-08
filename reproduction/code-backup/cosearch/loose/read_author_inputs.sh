set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
"$R/env/bin/python" - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
import pyarrow.parquet as pq,json
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch');c=OmegaConf.load(r/'runs/cosearch-local-gpu-five-20261003-n/resolved.yaml')
print(json.dumps({'train':str(c.data.train_files),'val':str(c.data.val_files),'actor':c.actor_rollout_ref.model.path}))
t=pq.ParquetFile(str(c.data.train_files)).read_row_group(0);rows=t.slice(0,1).to_pylist();print(json.dumps({'schema':str(t.schema),'row':rows[0]},ensure_ascii=False)[:6500])
PY
