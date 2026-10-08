"""Read-only compressed fetch of two complete A traces, preserving native SHA."""
import base64,gzip,hashlib,json,pathlib,subprocess,sys
job=int(sys.argv[1]);root=sys.argv[2];assert root.startswith('/dataset/shaoshitong/search-r1-rl/topd/logs/') and '..' not in pathlib.PurePosixPath(root).parts and "'" not in root
code="""import base64,gzip,hashlib,json,pathlib
root=pathlib.Path('ROOT');out=[]
for name in ('generation-token-trace.json','hidden-input-token-trace.json'):
 p=root/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<80*1024**2
 b=p.read_bytes();out.append(dict(relative_path='initial/outer-0/'+name,native_path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(gzip.compress(b)).decode()))
print(json.dumps(out))
""".replace('ROOT',root)
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'G1_TRACES'\n"+code+'\nG1_TRACES\n',text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
target=pathlib.Path('outputs/remote-evidence/g1-initial'+str(job));target.mkdir(exist_ok=True);refs=[]
for r in json.loads(p.stdout):
 b=gzip.decompress(base64.b64decode(r.pop('data')));assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256'];f=target/r['relative_path'];f.parent.mkdir(parents=True,exist_ok=True)
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
 refs.append(r)
(target/'fetch-large-trace-refs.json').write_text(json.dumps(refs,indent=2));print(json.dumps(refs))
