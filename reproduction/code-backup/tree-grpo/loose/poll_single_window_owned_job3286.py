"""Read only own3286 bounded JSON/log metadata. No PT, model or mutation."""
from pathlib import Path
import base64
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import route

REMOTE = r'''
from pathlib import Path
import hashlib,json,re,socket,subprocess,time
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');job='3286';run=root/'runs/tree-probability-probe-3286'
result=dict(host=socket.gethostname(),job_id=job,epoch=time.time(),mutation=False,tensor_assets_read=False)
q=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True,timeout=15)
result['scheduler']=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
def meta(path):
 b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for name in ['initial-allocation.json','termination-cost.json','launch.json','torchrun-command.json','result.json']:
 p=run/name
 if p.exists():
  v=json.loads(p.read_bytes());v.pop('rank_records',None)
  result[name]=dict(identity=meta(p),record=v)
for name,p in [('probe.log',run/'probe.log'),('launcher.log',root/'logs/tree-probability-probe-3286.out')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));b=f.read()
  result[name]=dict(path=str(p),bytes=p.stat().st_size,tail=b.decode(errors='replace'))
p=root/'instruct300/probe-scheduler-observation-3286.jsonl'
if p.exists():
 rows=p.read_text().splitlines();result['scheduler_observations']=dict(identity=meta(p),samples=len(rows),last=json.loads(rows[-1]),terminal=[json.loads(x) for x in rows if json.loads(x).get('terminal_state_observed')])
p=run/'resources.jsonl'
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines() if x]
 if rows:result['resources']=dict(identity=meta(p),samples=len(rows),last=rows[-1],max_foreign_memory_fraction=max(g['foreign_memory_fraction'] for r in rows for g in r['gpus']),max_foreign_sm_upper=max(g['foreign_sm_upper'] for r in rows for g in r['gpus']),whole_card_5s_peak_used_bytes=max(g['total_used_memory_bytes'] for r in rows for g in r['gpus']))
ready=[]
for rank in range(8):
 p=run/('gpu-records/rank-%02d/ready.json'%rank)
 if not p.exists():continue
 v=json.loads(p.read_bytes())
 keys=['status','rank','world_size','window_profile','bundle_sha256','window_inputs_sha256','generation_invocations','original_generation_call','original_prompt_rows','maximum_new_emitted_tokens','actual_new_emitted_tokens','actor_native_forwards','actor_scoring_native_forwards','warmup_native_forwards','optimizer_instantiated','optimizer_committed','scheduler_instantiated','scheduler_advanced','backward_called','original_master_state_unchanged','original_modes_restored','actual_loaded_binary_identity_proven','probability_accepted','training_accepted','GPU_capacity_proven','own_wall_seconds','tensor_evidence','actual_runtime','raw_probability_observations','anchor_observations','cpu_preparation','services','wrappers']
 record={k:v[k] for k in keys if k in v}
 w=v['live_window_weights'];record['live_window_weights']=w
 record['hf_summary']={k:v['hf'][k] for k in ['status','serialized_count','unique_parameter_count']}
 observer=v['observer'];record['observer_summary']={k:observer[k] for k in ['status','observer_evidence_complete','rank','counts','hooks_restored','errors','training_accepted','probability_accepted','observer_replaced_native_outputs','observer_changed_sampling_parameters']}
 record['observer_summary'].update(generated_rows=len(observer['generated']),emitted_tokens=sum(x['true_response_length'] for x in observer['generated']),anchors=observer['anchors'])
 ready.append(dict(identity=meta(p),record=record))
result['ready']=ready
print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
'''

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
destination = Path(__file__).resolve().parents[1]/'outputs/tree-instruct-300'/('PPO_WINDOW_GPU3286_METADATA_'+stamp+'.json')
answer = subprocess.run(route('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -c '+shlex.quote(REMOTE)), capture_output=True)
if answer.returncode:
    print(answer.stderr.decode(errors='replace'), file=sys.stderr)
    destination.with_suffix('.failed.json').write_text(json.dumps(dict(status='CONNECTION_OR_READ_FAILED_CURRENT_UNKNOWN',returncode=answer.returncode,stderr=answer.stderr.decode(errors='replace')),indent=2)+'\n')
    sys.exit(answer.returncode)
value=json.loads(answer.stdout)
destination.write_bytes(answer.stdout)
match=__import__('re').search(r'(?:^|\s)JobState=(\S+)',value['scheduler']['stdout'])
print(json.dumps(dict(metadata_file=str(destination),sha256=hashlib.sha256(answer.stdout).hexdigest(),scheduler=match.group(1) if value['scheduler']['returncode']==0 and match else 'UNKNOWN',ready_ranks=[r['record']['rank'] for r in value['ready']],result_status=value.get('result.json',{}).get('record',{}).get('status')),sort_keys=True))
