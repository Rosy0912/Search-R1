import pathlib,json,hashlib,base64,subprocess,sys,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');d=json.load(sys.stdin);raw=base64.b64decode(d['sealer_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()=='5745ab029a8167a2f17667d57a80f9961e164ae42d8d27b050c64b5af1d9ff50'
p=r/'logs/seal-hardware3287-evidence.py'
with p.open('xb') as f:f.write(raw)
h=r/'logs/hardware3287-head-terminal-ledger.json'
with h.open('x') as f:f.write(json.dumps(d['head'],sort_keys=True)+'\n')
terminal=base64.b64decode(d['terminal_base64'],validate=True)
subprocess.run([str(r/'env/bin/python'),'-B',str(p),'--job-id','3287','--head-ledger',str(h)],input=terminal,cwd=r/'repo',env=dict(os.environ,PYTHONPATH=str(r/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),check=True)
