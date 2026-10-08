"""Readonly actual source-contract evidence, never teacher approval."""
import pathlib,sys,json,hashlib,base64
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).parent/'g1-native-cpu-deploy-v6'))
from stream_transport import send
R='/dataset/shaoshitong/search-r1-rl/topd/logs';N='standard-search-r1-teacher-identity-v5-native-r1';RUN='standard-r1-teacher-identity-v5-r1-cpu-3179'
files=[R+'/'+RUN+'/'+n for n in ['chain.log','score.log','owner.log','ipc.log','supervisor.log','hf.log','result.json']]+[R+'/'+RUN+'.log',R+'/'+N+'-native-cpu-submission.json',R+'/'+N+'-native-cpu-3179-control.jsonl',R+'/'+N+'-native-cpu-3179-actual-final.json']
code='import pathlib,hashlib,json,base64,time\nrows=[]\n'
code+='for n in '+repr(files)+':\n p=pathlib.Path(n);a=p.stat();assert p.is_file() and not p.is_symlink() and a.st_size<4*1024*1024\n b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)\n rows.append(dict(path=n,size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))\n'
code+='print(json.dumps(dict(scope="READONLY_RAW_SOURCE_CPU_NOT_TEACHER",epoch=time.time(),files=rows)))\n'
r=send(code,attempts=3,retry_safe=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-identity-native3179');out.mkdir(exist_ok=False,parents=True)
for row in data['files']:
 b=base64.b64decode(row.pop('data'),validate=True);assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256']
 name=pathlib.Path(row['path']).name;(out/name).write_bytes(b);row['local_file']=name
(out/'fetch-refs.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
