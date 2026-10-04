"""Admit only verified archive training on GPU7; inspect mode allocates nothing."""
import json,os,subprocess,time,sys
from pathlib import Path
b=Path(os.environ['E1_BASE']);archive=Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
uuid='GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3'
inspect='--inspect-live' in sys.argv
receipts=[]
for path in (archive/'receipts').glob('archive-*.json'):
 try:
  r=json.loads(path.read_text())
  if r.get('profile')=='archive' and r.get('status')=='RUNNING' and time.time()-r.get('updated_unix',0)<90:receipts.append(r)
 except (OSError,ValueError):pass
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
selected=[x.split(',') for x in apps.splitlines() if x.startswith(uuid)]
record=json.loads((archive/'receipts/shared-gpu-oracle.json').read_text()) if inspect else None
training=[];services=[]
for row in selected:
 pid=int(row[1]);p=Path('/proc')/str(pid)
 assert p.stat().st_uid==os.getuid(),('foreign UID',pid)
 env=p.joinpath('environ').read_bytes().split(b'\0');cmd=p.joinpath('cmdline').read_bytes()
 if inspect and pid==record['pid']:
  assert record['accepted'] and record['base']==str(b) and record['uuid']==uuid
  assert ('E1_BASE='+str(b)).encode() in env and str(b/'source/retriever_stream.py').encode() in cmd
  assert int(row[2])<=1024;services.append(pid);continue
 owned=[r for r in receipts if ('E1_RUN_ID='+r['run_id']).encode() in env]
 assert owned and (b'ray::WorkerDict' in cmd or b'search_e1.opsd_main' in cmd),('unexpected GPU7 occupant',pid)
 training.append(pid)
assert len(training)<=1 and (not inspect or len(services)==1)
gpu=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert gpu[0]==uuid;total,used=map(int,gpu[1:])
assert used<total-(6144 if inspect else 7168),(used,total)
historical=max([39574]+[int(r.get('peaks_mib',{}).get('7',0)) for r in receipts])
assert historical+1024<total-6144,(historical,total)
proof=dict(uuid=uuid,training=training,services=services,used_mib=used,total_mib=total,historical_peak_mib=historical,time=time.time(),inspect_only=inspect)
if not inspect:(b/'receipts/restart-shared-gpu-admission.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof),flush=True)
