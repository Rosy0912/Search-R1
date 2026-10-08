import pathlib,tarfile,io,hashlib,json,base64
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/runs/formal200-8c29851b54174f2083023a98cde9c19b-3312')
b=io.BytesIO();records=[]
with tarfile.open(fileobj=b,mode='w:gz')as t:
 for p in sorted(r.rglob('*')):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb')as f:
   while True:
    x=f.read(8388608)
    if not x:break
    h.update(x)
  n=p.relative_to(r).as_posix();z=p.stat().st_size;records.append({'path':n,'bytes':z,'sha256':h.hexdigest()})
  if z<20000000:t.add(str(p),arcname=n,recursive=False)
 raw=(json.dumps({'job_id':3312,'run_root':str(r),'files':records},indent=2)+'\n').encode();info=tarfile.TarInfo('terminal-full-file-hash-manifest.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
raw=b.getvalue();print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data_base64':base64.b64encode(raw).decode()}))
