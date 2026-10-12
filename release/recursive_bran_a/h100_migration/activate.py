import pathlib,json,time,sys,os
b=pathlib.Path(sys.argv[1]);pid=int(sys.argv[2]);deadline=time.time()+900
while time.time()<deadline:
 p=b/'receipts/source-identity.json'
 if p.exists():
  identity=json.loads(p.read_text());assert identity['pid']==pid
  (b/'receipts/enable-gpu.json').write_text(json.dumps({'pid':pid,'source_identity_sha256':identity['source_identity_sha256']}))
  import requests
  for _ in range(120):
   try:
    h=requests.get('http://127.0.0.1:18086/health',timeout=2).json()
    if h.get('activation_error'):raise RuntimeError(h['activation_error'])
    if h.get('gpu_ready'):print(json.dumps(h),flush=True);sys.exit(0)
   except requests.RequestException:pass
   time.sleep(1)
  raise RuntimeError('Own retriever activation timed out')
 if not pathlib.Path('/proc',str(pid)).exists():raise RuntimeError('Own retriever exited during CPU load')
 time.sleep(1)
raise RuntimeError('Own retriever CPU load timed out')
