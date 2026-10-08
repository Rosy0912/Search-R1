import sys,pathlib,hashlib,subprocess
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/logs/capture-hardware3287-allocation.py');raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()=='f0701eda22ffe0fcd2881dbaa6aefe65e6378f234d21c1baa6afbc6c2b6fba82'
with p.open('xb') as f:f.write(raw)
subprocess.run(['python3',str(p),'--job-id','3287'],check=True)
