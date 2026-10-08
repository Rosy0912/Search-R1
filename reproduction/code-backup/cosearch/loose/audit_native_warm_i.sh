set -euo pipefail
python3 - <<'PY'
import json,hashlib,re,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');r=root/'runs/author-native-warm-five-i';launch=json.loads((r/'launch.json').read_text());a=[]
for p in (r/'evidence/workers').glob('*.jsonl'):
 for line in p.read_text().splitlines():
  v=json.loads(line)
  if v.get('kind')=='ack':a.append(v)
assert len(a)==8 and {(v['role'],v['rank']) for v in a}=={(role,i) for role in ['main','reranker'] for i in range(4)}
assert {v['uuid'] for v in a}==set(launch['leased_uuids']) and len(set(v['uuid'] for v in a))==8
manifest=hashlib.sha256((r/'tools/source-manifest.json').read_bytes()).hexdigest();hook=hashlib.sha256((r/'tools/observe_worker.py').read_bytes()).hexdigest()
for v in a:
 p=Path('/proc')/str(v['pid']);assert int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==v['process_start_ticks'];assert (p/'cgroup').read_text()==v['process_cgroup'] and '/job_3072/' in v['process_cgroup'];assert v['adam_state_initial']==0;assert v['source_manifest_sha256']==manifest and v['hook_sha256']==hook and v['job']=='3072';assert v['hostname']=='h100-16'
controllers=[]
for p in (r/'evidence').glob('controller-*.jsonl'):
 for line in p.read_text().splitlines():
  v=json.loads(line)
  if v.get('kind')!='launch_configuration':controllers.append(v)
value=dict(run=r.name,job=3072,checked_at=time.time(),live_all8_init_acks_verified=True,fresh_all8_adam_zero=True,acks=a,controllers=controllers,training_log_identity=dict(inode=(r/'training.log').stat().st_ino,device=(r/'training.log').stat().st_dev),real_joint_outer_count=sum(v.get('kind')=='outer_end' for v in controllers),accepted=False)
print(json.dumps(value))
PY
