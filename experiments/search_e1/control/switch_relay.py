import os,json,time,signal,subprocess,requests
from pathlib import Path
base=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004');old=Path('/data/shaoshitong/search-e1-retriever-gpu-20261004')
s=requests.Session();s.trust_env=False
h=s.get('http://10.123.4.11:18002/health',timeout=10);h.raise_for_status();health=h.json()
assert health['acceptance_sha256']=='75c4a3b2e60bdfb0d1c450e01f9cd99bec2d5245d753d28704755b1d08fdb30d' and health['source_manifest_sha256']=='b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508'
assert (base/'receipts/stream-overlap.accepted').exists()
proc=Path('/proc/2440107');assert proc.stat().st_uid==os.getuid()
assert ('E1_BASE='+str(old)).encode() in proc.joinpath('environ').read_bytes().split(b'\0')
assert str(old/'source/experiments/search_e1/control/retriever_gpu.py').encode() in proc.joinpath('cmdline').read_bytes()
for _ in range(360):
 sockets=subprocess.check_output(['ss','-tnp'],text=True)
 active=[line for line in sockets.splitlines() if ':18001' in line and '2440107' in line and line.startswith('ESTAB')]
 if not active:break
 time.sleep(2)
else:raise RuntimeError('Original oracle has active request; leave it running')
proof=dict(original_pid=2440107,original_base=str(old),new_backend=health,time=time.time(),active_old_requests=[])
(base/'receipts/oracle-relocation.json').write_text(json.dumps(proof,indent=2));(old/'receipts/relocation.json').write_text(json.dumps(proof,indent=2))
os.kill(2440107,signal.SIGTERM)
for _ in range(60):
 if not proc.exists():break
 time.sleep(1)
else:raise RuntimeError('Owned oracle did not exit; no second port listener started')
command='env E1_BASE='+str(base)+' CUDA_VISIBLE_DEVICES= E1_RETRIEVER_TARGET=http://10.123.4.11:18002 E1_RETRIEVER_BIND=10.120.17.101 E1_RETRIEVER_PORT=18001 '+str(base/'env/bin/python')+' '+str(base/'source/experiments/search_e1/control/retriever_proxy.py')+' > '+str(base/'logs/oracle-relay.log')+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s','e1-oracle-relay-20261004',command],check=True)
for _ in range(60):
 try:
  r=s.get('http://10.120.17.101:18001/health',timeout=15);r.raise_for_status();assert r.json()==health;break
 except requests.RequestException:time.sleep(1)
else:raise RuntimeError('Relay not healthy; restore owned service or fix relay')
r=s.post('http://10.120.17.101:18001/retrieve',json=dict(queries=['capital of France'],topk=3,return_scores=True),timeout=180);r.raise_for_status();assert len(r.json()['result'][0])==3
(base/'receipts/migration-backend.accepted').write_text(json.dumps(dict(health=health,probe=r.json(),time=time.time()),indent=2))
print('STABLE_URL_RELAY_ACCEPTED_OWNED_ORACLE_GPUS_FREED',flush=True)
