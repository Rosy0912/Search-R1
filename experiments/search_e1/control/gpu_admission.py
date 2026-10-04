"""Check selected GPU UUIDs, not utilization alone; never terminate outsiders."""
import subprocess,csv,os,json,sys
from pathlib import Path
selected=[int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
def query(args):return list(csv.reader(subprocess.check_output(['nvidia-smi',*args],text=True,timeout=20).splitlines(),skipinitialspace=True))
gpus=query(['--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
rows=[g for g in gpus if int(g[0]) in selected]
if len(rows)!=len(selected):raise SystemExit('GPU inventory incomplete')
apps=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
shared_path=Path(os.environ['E1_BASE'])/'receipts/shared-gpu-oracle.json'
shared=None
if shared_path.exists():
 shared=json.loads(shared_path.read_text())
 assert shared['accepted'] and shared['max_process_mib']<=1024
 pid=int(shared['pid']);proc=Path('/proc')/str(pid)
 assert proc.stat().st_uid==os.getuid()
 assert ('E1_BASE='+shared['base']).encode() in proc.joinpath('environ').read_bytes().split(b'\0')
 assert (shared['base']+'/source/retriever_stream.py').encode() in proc.joinpath('cmdline').read_bytes()
 import urllib.request
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(shared['health_url'],timeout=10) as response:health=json.load(response)
 assert health['ready'] and health['acceptance_sha256']==shared['acceptance_sha256']
for g in rows:
 if shared and g[1]==shared['uuid']:
  occupants=[p for p in apps if p[0]==g[1]]
  if len(occupants)!=1 or int(occupants[0][1])!=int(shared['pid']) or int(occupants[0][3])>shared['max_process_mib']:
   raise SystemExit('Shared GPU has unexpected occupant: '+str(g))
  if not g[3].isdigit() or int(g[3])>shared['max_process_mib']+128:
   raise SystemExit('Shared GPU memory exceeds verified oracle bound: '+str(g))
  continue
 if g[4]=='[N/A]' or int(g[3])>128 or int(g[4])>1:raise SystemExit('Selected GPU busy/unknown: '+str(g))
 if any(p[0]==g[1] for p in apps):raise SystemExit('Selected GPU has compute processes: '+str(g))
print(json.dumps(dict(devices=rows,compute_apps=apps,owned_shared_oracle=shared)),flush=True)
