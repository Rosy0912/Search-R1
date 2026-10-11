"""K5 owns GPUs0-3 only; the approved retrieval process stays on K4."""
import json,os,socket,subprocess,urllib.request
UUIDS=['GPU-15061c70-49d6-3014-da87-915b370c1e89','GPU-0c0aeed1-c7b9-c9ad-4b1f-7396b6c19f4c','GPU-d2609c94-e30a-a106-e1cc-6d4ae3e044c2','GPU-d6d81990-8737-f64c-aaef-d0ceabb5123a']
EXPECTED_HEALTH={'ready':True,'ntotal':21015324,'d':768,'tf32':False,'query_micro':64,'source_manifest_sha256':'b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508','acceptance_sha256':'f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443'}
def exact_gpu_gate(selected,inventory):
 if socket.gethostname().split('.')[0]!='xie-lab-4' or os.getuid()!=1001 or list(selected)!=['0','1','2','3']:
  raise RuntimeError('HKUST5 hardware/UID boundary changed')
 if [inventory[x]['uuid'] for x in selected]!=UUIDS:raise RuntimeError('HKUST5 selected UUID changed')
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 for line in rows.splitlines():
  gpu,pid,memory=[x.strip() for x in line.split(',')]
  if gpu in UUIDS:raise RuntimeError('Selected K5 GPU is occupied; no takeover or oracle exemption')
 health=json.load(urllib.request.urlopen('http://10.123.4.11:18003/health',timeout=30))
 if any(health.get(k)!=v for k,v in EXPECTED_HEALTH.items()):raise RuntimeError('K4 retriever numeric/source identity changed')
 return health
