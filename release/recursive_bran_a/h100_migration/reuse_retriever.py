"""Reuse exactly the same live admitted CPU service; no new numeric gate claim."""
import json,pathlib,sys,os,requests
try:
 b=pathlib.Path(sys.argv[1]);identity=json.loads((b/'receipts/source-identity.json').read_text());gate=json.loads((b/'receipts/full-scan-http-gate.json').read_text())
 h=requests.get('http://127.0.0.1:18086/health',timeout=2).json();pid=h['pid'];p=pathlib.Path('/proc',str(pid))
 assert h['ready'] and gate['accepted'] and pid==identity['pid']==gate['candidate_pid']
 assert h['source_identity_sha256']==identity['source_identity_sha256']==gate['source_identity_sha256']
 assert p.stat().st_uid==os.getuid() and int((p/'stat').read_text().rsplit(') ',1)[1].split()[19])==gate['candidate_starttick']
 assert not h['gpu_worker_active']
 print(json.dumps({'reuse_same_live_admitted_service':True,'pid':pid,'numeric_gate_repeated':False}),flush=True)
except (OSError,ValueError,KeyError,AssertionError,requests.RequestException):raise SystemExit(1)
