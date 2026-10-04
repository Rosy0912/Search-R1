import os,json,time,signal,subprocess,requests
from pathlib import Path
b=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004')
s=requests.Session();s.trust_env=False
health=s.get('http://10.123.4.11:18003/health',timeout=15).json()
assert health['acceptance_sha256']=='f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443'
assert health['source_manifest_sha256']=='b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508'
proof=json.loads((b/'receipts/joined-live-comparison.json').read_text());assert proof['129']['documents_identical'] and proof['2440']['queries']==2440
pid=3421872;p=Path('/proc/'+str(pid));assert p.stat().st_uid==os.getuid()
env=p.joinpath('environ').read_bytes().split(b'\0');cmd=p.joinpath('cmdline').read_bytes()
assert b'E1_RETRIEVER_TARGET=http://10.123.4.11:18002' in env
assert str(b/'source/experiments/search_e1/control/retriever_proxy.py').encode() in cmd
# Drain owned relay before a brief restart. Original archive relay retries transport.
for _ in range(900):
 rows=subprocess.check_output(['ss','-tnp'],text=True)
 active=[r for r in rows.splitlines() if r.startswith('ESTAB') and ':18001' in r and 'pid='+str(pid)+',' in r]
 if not active:break
 time.sleep(2)
else:raise RuntimeError('Owned relay busy; leave old relay unchanged')
(b/'receipts/relay-before-joined-transition.json').write_text(json.dumps(dict(pid=pid,env_target='http://10.123.4.11:18002',cmd=cmd.decode().replace('\0',' '),health=health,time=time.time()),indent=2))
os.kill(pid,signal.SIGTERM)
for _ in range(60):
 if not p.exists():break
 time.sleep(1)
else:raise RuntimeError('Owned relay did not drain; do not bind a second listener')
command='env E1_BASE='+str(b)+' CUDA_VISIBLE_DEVICES= E1_RETRIEVER_TARGET=http://10.123.4.11:18003 E1_RETRIEVER_BIND=10.120.17.101 E1_RETRIEVER_PORT=18001 '+str(b/'env/bin/python')+' '+str(b/'source/experiments/search_e1/control/retriever_proxy.py')+' > '+str(b/'logs/oracle-relay-joined.log')+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s','e1-oracle-joined-relay-20261004',command],check=True)
for _ in range(60):
 try:
  r=s.get('http://10.120.17.101:18001/health',timeout=15);r.raise_for_status();assert r.json()==health;break
 except requests.RequestException:time.sleep(1)
else:raise RuntimeError('Joined relay not ready; recover owned relay')
r=s.post('http://10.120.17.101:18001/retrieve',json=dict(queries=['capital of France'],topk=3,return_scores=True),timeout=180);r.raise_for_status();assert len(r.json()['result'][0])==3
(b/'receipts/joined-relay.accepted').write_text(json.dumps(dict(health=health,probe=r.json(),time=time.time()),indent=2));print('JOINED_RELAY_ACCEPTED',flush=True)
