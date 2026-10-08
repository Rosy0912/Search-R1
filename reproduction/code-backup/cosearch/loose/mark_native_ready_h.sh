set -euo pipefail
export COSEARCH_NATIVE_GPU_PROBE_ONLY=1
export COSEARCH_NATIVE_NEXT_RUN_ID=author-native-gpu-routing-probe-h
python3 - <<'READY'
import json,hashlib,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
assert json.loads((root/'tools/cpu-launch-gate.json').read_text())['passed']
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True).strip()
files=['observe_worker.py','launch_observed.py','checkpoint_observer.py','native_job.py','check_native_five.py','start_native_entry.sh','start_native_if_ready.py','cpu_launch_gate.py','author-cli.json','tool.yaml','source-manifest.json','probe_gpu_routing.py']
value=dict(cpu_preparation_passed=True,actual_gpu_resources_tested=False,created_at=time.time(),tools_sha256={name:hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest() for name in files})
(root/'tools/preparation-ready.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
READY
/dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python /dataset/shaoshitong/search-r1-rl/cosearch/author-native/tools/start_native_if_ready.py
