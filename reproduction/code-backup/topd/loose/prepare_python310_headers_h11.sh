python3 - <<'PY'
import pathlib,subprocess,json,hashlib,os,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/assets/compiler-headers');root.mkdir(parents=True,exist_ok=True)
dest=root/'python3.10-3.10.12-22.04.18-amd64';dest.mkdir(exist_ok=False);(dest/'packages').mkdir()
packages=[('libpython3.10-dev',4765042,'c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9'),('python3.10-dev',507980,'7caa163f4d83ae871f985ec4339fb4c96ae7ef8cb3568b8aec9138f86aacbeb7')]
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
for name,size,sha in packages:
 fn=name+'_3.10.12-1~22.04.18_amd64.deb';url='https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/'+fn;p=dest/'packages'/fn
 result=subprocess.run(['curl','--fail','--location','--connect-timeout','10','--max-time','25','--output',str(p)+'.partial',url],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 record={'epoch':time.time(),'node':'h100-11','url':url,'returncode':result.returncode,'stderr':result.stderr,'expected_sha256':sha,'expected_size':size}
 with (dest/'download-attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(record),flush=True)
 if result.returncode:raise SystemExit(1)
 b=pathlib.Path(str(p)+'.partial').read_bytes();assert len(b)==size and hashlib.sha256(b).hexdigest()==sha
 pathlib.Path(str(p)+'.partial').rename(p)
PY
