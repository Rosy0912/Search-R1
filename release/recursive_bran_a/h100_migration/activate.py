import pathlib,json,time,sys,os,requests
b=pathlib.Path(sys.argv[1]);pid=int(sys.argv[2]);deadline=time.time()+900
while time.time()<deadline:
 if not pathlib.Path('/proc',str(pid)).exists():raise RuntimeError('Own retriever exited during CPU load')
 try:
  h=requests.get('http://127.0.0.1:18086/health',timeout=2).json()
  identity=json.loads((b/'receipts/source-identity.json').read_text())
  if h['pid']==pid and identity['pid']==pid and h['source_identity_sha256']==identity['source_identity_sha256']:
   (b/'receipts/enable-gpu.json').write_text(json.dumps({'pid':pid,'source_identity_sha256':h['source_identity_sha256']}))
   break
 except (requests.RequestException,OSError,ValueError,KeyError):pass
 time.sleep(1)
else:raise RuntimeError('Own retriever CPU load timed out')
for _ in range(120):
 h=requests.get('http://127.0.0.1:18086/health',timeout=2).json()
 if h.get('activation_error'):raise RuntimeError(h['activation_error'])
 if h.get('gpu_ready'):print(json.dumps(h),flush=True);sys.exit(0)
 time.sleep(1)
raise RuntimeError('Own retriever activation timed out')
