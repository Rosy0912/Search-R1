"""Exact selected-card admission for the user-authorized HKUST4 experiment B."""
import os,subprocess,json,urllib.request
from pathlib import Path
UUIDS=['GPU-ed24344f-9cd8-6d29-e8b9-d7582a8cfd65','GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2','GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f','GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3']

def exact_gpu_gate(selected,inventory):
 if os.getuid()!=1001 or list(selected)!=['4','5','6','7']:
  raise RuntimeError('B selected hardware/UID changed')
 if [inventory[x]['uuid'] for x in selected]!=UUIDS:
  raise RuntimeError('B physical GPU UUIDs changed')
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 oracle_seen=False
 for line in rows.splitlines():
  gpu,pid,memory=[x.strip() for x in line.split(',')]
  if gpu not in UUIDS:continue
  if gpu!=UUIDS[-1] or int(pid)!=1749337 or float(memory)>1024:
   raise RuntimeError('Foreign process on selected GPU; never take over')
  p=Path('/proc',pid)
  if p.stat().st_uid!=1001 or int((p/'stat').read_text().split(') ',1)[1].split()[19])!=641133475:
   raise RuntimeError('Retriever kernel identity changed')
  oracle_seen=True
 if not oracle_seen:raise RuntimeError('Pinned oracle not present')
 health=json.load(urllib.request.urlopen('http://10.123.4.11:18003/health',timeout=15))
 expected={'ready':True,'ntotal':21015324,'d':768,'tf32':False,'query_micro':64,
  'source_manifest_sha256':'b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508',
  'acceptance_sha256':'f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443'}
 if any(health.get(k)!=v for k,v in expected.items()):raise RuntimeError('Retriever actual numeric/source gate changed')
 return health
