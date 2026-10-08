"""Install immutable independent review/gate evidence together to own remote logs."""
import base64,hashlib,json,pathlib,subprocess,sys
records=[]
for name in sys.argv[1:]:
 p=pathlib.Path(name);b=p.read_bytes();records.append({'name':p.name,'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
code=r'''import base64,hashlib,json,os,pathlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs')
records=json.loads(base64.b64decode('PAYLOAD'))
for record in records:
 name=record['name'];assert pathlib.PurePath(name).name==name
 b=base64.b64decode(record['data']);assert hashlib.sha256(b).hexdigest()==record['sha256'];p=r/name
 assert not p.is_symlink()
 if p.exists():assert p.read_bytes()==b
 else:
  temp=r/(name+'.install-private');assert not temp.exists()
  with temp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  os.link(temp,p);temp.unlink()
 fd=os.open(r,os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps({'status':'EVIDENCE_INSTALL_SHA_EXACT','path':str(p),'sha256':record['sha256']}))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(records).encode()).decode())
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'INSTALL_REVIEWS'\n"+code+'\nINSTALL_REVIEWS\n',text=True,capture_output=True)
pathlib.Path('outputs/g1-initial-independent-gate-install.log').write_text(p.stdout+p.stderr);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr);raise SystemExit(p.returncode)
