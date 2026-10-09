import subprocess,shlex,sys
secret=sys.stdin.readline().strip()
code=r'''import requests,sys,time,json,hashlib,pathlib
token=sys.stdin.readline().strip();s=requests.Session();s.trust_env=False;t=time.time();n=0;status=None;err=None;h=hashlib.sha256()
try:
 with s.get('https://hf-mirror.com/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=0-8388607'},stream=True,timeout=(5,5)) as r:
  status=r.status_code
  if status==206:
   for b in r.iter_content(65536):
    n+=len(b);h.update(b)
    if n>=8388608 or time.time()-t>15:break
except Exception as e:err=type(e).__name__
p=next(pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009').glob('teacher_model-00028*.part'));f=p.open('rb');expected=hashlib.sha256(f.read(n)).hexdigest();v={'epoch':time.time(),'route':'mirror_direct_no_proxy','bytes':n,'seconds':time.time()-t,'HTTP':status,'error':err,'sample_SHA_matches_existing_partial':bool(n and h.hexdigest()==expected),'mac_used':False};p.parent.joinpath('mirror-probe-latest.json').write_text(json.dumps(v));print(json.dumps(v))
'''
p=subprocess.Popen(['ssh','hkust2','python3 -c '+shlex.quote(code)],stdin=subprocess.PIPE);p.stdin.write((secret+chr(10)).encode());p.stdin.close();raise SystemExit(p.wait(timeout=35))
