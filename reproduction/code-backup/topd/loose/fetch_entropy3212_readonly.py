import pathlib,sys,json,base64,hashlib,time
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
SRC=r'''
import pathlib,json,hashlib,base64,time,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');folder=R/'logs/entropy-device-proof-3212'
r=subprocess.run(['scontrol','show','job','-dd','3212'],capture_output=True,text=True,timeout=5)
out={'job_id':3212,'epoch':time.time(),'controller':{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},'files':[],'errors':[]}
paths=list(folder.glob('*.json'))+list(folder.glob('*.log'))+list((folder/'generated-candidate-kernels').glob('*.py'))
paths += [R/'logs/r1-entropy-source-v5-cpu-3212/result.json',R/'logs/entropy-device-proof-3212.log',R/'logs/entropy-device-proof-v1-r1-final-submission.json',R/'logs/entropy-device-proof-v1-r1-final-3212-actual-final.json']
terminal=r.returncode==0 and any('JobState='+v in r.stdout for v in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'))
if terminal:paths+=list(folder.glob('*.jsonl'))
for p in sorted(set(paths)):
 if not p.exists():continue
 try:
  a=p.stat();assert a.st_size<16*1024*1024,'Large original requires chunked collector';b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns) and len(b)==a.st_size,'File changed while reading'
  out['files'].append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime_ns':z.st_mtime_ns,'base64':base64.b64encode(b).decode()})
 except Exception as e:out['errors'].append({'path':str(p),'error':repr(e)})
print(json.dumps(out))
'''
r=send(SRC,attempts=3,retry_safe=True,timeout=90);r.check_returncode();d=json.loads(r.stdout);folder=pathlib.Path('outputs/remote-evidence/entropy-device3212');folder.mkdir(parents=True,exist_ok=True)
for f in d['files']:
 b=base64.b64decode(f.pop('base64'));assert len(b)==f['size'] and hashlib.sha256(b).hexdigest()==f['sha256'];p=folder/(pathlib.Path(f['path']).name+'.'+f['sha256'][:12]);
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as out:out.write(b)
 f['local_path']=str(p.resolve())
p=folder/('snapshot-%d.json'%int(d['epoch']*1000000));p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'snapshot':str(p),'controller':d['controller'],'files':[{k:f[k] for k in ('path','size','sha256')} for f in d['files']],'errors':d['errors']}))
