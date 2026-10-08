import json,time,hashlib,gzip,base64,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');out={'epoch':time.time(),'job_id':3234,'files':{},'commands':{}}
files=[p for p in r.iterdir() if p.is_file()]+list((r/'output').glob('*result*'))+list((r/'output').glob('*failure*'))+list((r/'output').glob('*guard*'))
for name in ['resource-samples.jsonl','resource-guard.json','run-result.json']:
 p=r/name
 if p.exists():files.append(p)
for p in sorted(set(files)):
 raw=p.read_bytes();out['files'][str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw)).decode()}
for name,cmd in [('sacct',['sacct','-j','3234,3236','--format=JobID,JobName,State,ExitCode,Elapsed,NodeList','-P']),('scontrol3236',['scontrol','show','job','3236']),('disk',['df','-h','/dataset']),('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),('background',['ps','-p','1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446','-o','pid,lstart,stat,args'])]:
 try:
  x=subprocess.run(cmd,capture_output=True,text=True,timeout=20);out['commands'][name]={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
 except Exception as e:out['commands'][name]={'error':str(e)}
print(json.dumps(out))
