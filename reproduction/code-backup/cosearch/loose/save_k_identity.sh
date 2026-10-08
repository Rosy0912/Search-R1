set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
"$R/env/bin/python" - <<'PY'
import json,hashlib
from pathlib import Path
from omegaconf import OmegaConf
import torch,transformers,ray
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-k')
c=OmegaConf.load(r/'resolved.yaml');launch=json.loads((r/'launch.json').read_text())
assert launch['job_id']=='2961' and launch['commit'].startswith('409c0e5')
checks={'actor_param_offload':c.actor_rollout_ref.actor.fsdp_config.param_offload,'actor_grad_offload':c.actor_rollout_ref.actor.fsdp_config.grad_offload,'actor_optimizer_offload':c.actor_rollout_ref.actor.fsdp_config.optimizer_offload,'reference_param_offload':c.actor_rollout_ref.ref.fsdp_config.param_offload,'free_cache_engine':c.actor_rollout_ref.rollout.free_cache_engine}
assert all(checks.values())
record={'run':r.name,'job':2961,'source':launch['commit'],'launch':launch,'offload_checks':checks,'ranker':OmegaConf.to_container(c.baselines.ranker,resolve=True),'torch':torch.__version__,'transformers':transformers.__version__,'ray':ray.__version__,'artifact_sha256':{name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ('native-ranker-probe.json','native-ranker-probe.log','resolved.yaml','launch.json','slurm-allocation.txt')}}
(r/'native-probe-identity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
