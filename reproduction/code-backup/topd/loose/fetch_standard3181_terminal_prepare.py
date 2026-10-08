"""Authorized captured-terminal install and readonly small evidence fetch."""
import pathlib,json,hashlib,base64,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).parent/'g1-native-cpu-deploy-v6'))
from stream_transport import send
P=pathlib.Path('outputs/remote-evidence/r1-standard32b-profile3181-actual-final.json');b=P.read_bytes();parent_sha=hashlib.sha256(b).hexdigest();t=json.loads(b)
assert t['job_id']==3181 and t['scontrol']['returncode']==0 and 'JobState=FAILED' in t['scontrol']['stdout'] and 'ExitCode=1:0' in t['scontrol']['stdout']
flat=dict(t['scontrol'],job_id=3181,epoch=t['epoch'],parent_raw={'local_path':str(P.resolve()),'sha256':parent_sha,'field':'scontrol; stdout original unchanged'})
flatbytes=(json.dumps(flat,indent=2)+'\n').encode();local=pathlib.Path('outputs/standard3181-captured-controller-terminal.json');local.write_bytes(flatbytes)
root='/dataset/shaoshitong/search-r1-rl/topd';dest=root+'/logs/standard3181-captured-controller-terminal.json';ev=root+'/logs/r1-standard32b-profile2-v7/evidence'
code='import pathlib,json,hashlib,base64,os,time\n'
code+='b=base64.b64decode('+repr(base64.b64encode(flatbytes).decode())+');p=pathlib.Path('+repr(dest)+')\n'
code+='if p.exists():assert p.read_bytes()==b\nelse:\n tmp=p.with_name(p.name+".private");assert not tmp.exists()\n with tmp.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())\n os.link(tmp,p);tmp.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)\n'
files=[dest,ev+'/train-result.json',ev+'/owned-cleanup.json',ev+'/effective-runtime-config.json']
code+='rows=[]\nfor n in '+repr(files)+':\n p=pathlib.Path(n);a=p.stat();assert a.st_size<4*1024*1024 and not p.is_symlink();b=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_mtime_ns,a.st_size)==(z.st_ino,z.st_mtime_ns,z.st_size)\n rows.append(dict(path=n,size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))\nprint(json.dumps(dict(files=rows,epoch=time.time())))\n'
r=send(code,attempts=3,retry_safe=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-r1-profile3181-small');out.mkdir(exist_ok=False,parents=True)
for row in rows['files']:
 raw=base64.b64decode(row.pop('data'),validate=True);assert len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['sha256'];(out/pathlib.Path(row['path']).name).write_bytes(raw)
(out/'fetch-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(parent_raw_sha256=parent_sha,files=rows['files']),indent=2))
