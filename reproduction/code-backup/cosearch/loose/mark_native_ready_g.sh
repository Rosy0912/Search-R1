set -euo pipefail
unset COSEARCH_NATIVE_GPU_PROBE_ONLY
export COSEARCH_NATIVE_NEXT_RUN_ID=author-native-raw-five-g
/dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python - <<'GATE'
from pathlib import Path
import json,subprocess
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
h=root/'runs/author-native-gpu-routing-probe-h'
e=root/'runs/author-native-raw-five-f'
assert json.loads((e/'driver-exit.json').read_text())['returncode']==1
gate=json.loads((root/'tools/vllm-registry-subprocess-cpu-gate.json').read_text())
import hashlib
assert gate['passed'] and gate['real_rl_updates']==0
for n,expected in gate['tools_sha256'].items():
    assert hashlib.sha256((root/'tools'/n).read_bytes()).hexdigest()==expected
assert json.loads((h/'driver-exit.json').read_text())['returncode']==0
assert not subprocess.check_output(['squeue','-h','--name=cosearch-author-five','-o','%i'],text=True).strip()
a=json.loads((h/'evidence/routing-deferred.json').read_text())
assert a['actual_hook_all8'] and a['distinct_uuids']==8 and a['eight_gpu_nccl_allreduce']==[8.]*8
cpu=json.loads((h/'evidence/routing-cpu-reuse.json').read_text())
assert len({x['pid'] for x in cpu})<len(cpu) and all(x['device_count']==8 and not x['assigned'].get('GPU') for x in cpu)
assert len({x['mask'] for x in cpu})==1
print('H_RESOURCE_DIAGNOSTIC_ONLY_PASS_NOT_RL')
GATE
python3 - <<'READY'
import json,hashlib,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
assert json.loads((root/'tools/cpu-launch-gate.json').read_text())['passed']
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True).strip()
files=['observe_worker.py','launch_observed.py','checkpoint_observer.py','native_job.py','check_native_five.py','start_native_entry.sh','start_native_if_ready.py','cpu_launch_gate.py','author-cli.json','tool.yaml','source-manifest.json','probe_gpu_routing.py','sitecustomize.py']
value=dict(cpu_preparation_passed=True,actual_gpu_resources_tested=False,created_at=time.time(),tools_sha256={name:hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest() for name in files})
(root/'tools/preparation-ready.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
READY
/dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python /dataset/shaoshitong/search-r1-rl/cosearch/author-native/tools/start_native_if_ready.py
