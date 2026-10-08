set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ROOT/tools:$ROOT/repo:$ROOT/repo/verl" COSEARCH_NATIVE_RUN_ID=author-native-phase-aware-warm-five-m "$ROOT/env/bin/python" - <<'GATE'
import json,os,subprocess
from pathlib import Path
from launch_observed import compose_launch_config
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
for name in ['cosearch-native-warm','cosearch-author-five','cosearch-native-hf-probe']:
 assert not subprocess.check_output(['squeue','-h','--name='+name,'-o','%i'],text=True).strip(),name
assert json.loads((root/'runs/author-native-grammar-actor-continue-v4/driver-exit.json').read_text())['returncode']==0
assert json.loads((root/'runs/author-native-format-t1-probe-n/driver-exit.json').read_text())['returncode']==0
assert json.loads((root/'runs/author-native-warm-five-l/driver-exit.json').read_text())['returncode']==1
c,reload=compose_launch_config();assert not reload and c.trainer.resume_mode=='disable'
for label,key in [('actor','actor_rollout_ref'),('ranker','reranker_actor_rollout_ref')]:
 role=c[key];expected=root/('runs/author-native-grammar-actor-continue-v4/warm-start/actor' if label=='actor' else 'runs/author-native-grammar-warm-start-v2/warm-start/ranker');assert role.model.path==str(expected)
 ref_model=role.ref.get('model')
 assert ref_model is None or ref_model.get('path',role.model.path)==role.model.path
assert not os.environ.get('COSEARCH_NATIVE_GPU_PROBE_ONLY')
assert json.loads((root/'tools/phase-cpu-gate.json').read_text())['passed']
assert json.loads((root/'tools/warm-initialization.json').read_text())['phase_sampling_correction']['pure_original_default_reproduction'] is False
print('PHASE_AWARE_M_OWN_RELEASE_FRESH_ROLE_INITIALIZATION_GATE_PASS_0RL',flush=True)
GATE
unset COSEARCH_NATIVE_GPU_PROBE_ONLY
COSEARCH_NATIVE_NEXT_RUN_ID=author-native-phase-aware-warm-five-m "$ROOT/env/bin/python" "$ROOT/tools/start_native_if_ready.py"
squeue -h --name=cosearch-author-five -o '%i %T %N'
