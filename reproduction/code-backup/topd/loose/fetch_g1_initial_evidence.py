import base64,hashlib,json,pathlib,subprocess,sys
root='/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial-static-cpu-3102/recipe/initial-allocation-evidence'
code="""import base64,hashlib,json,pathlib
root=pathlib.Path('ROOT');out=[]
if root.exists():
 for p in sorted(root.rglob('*')):
  if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log') and p.stat().st_size<16*1024**2:
   b=p.read_bytes();out.append({'relative_path':str(p.relative_to(root)),'native_path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
print(json.dumps(out))
""".replace('ROOT',root)
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'G1_EVIDENCE'\n"+code+'\nG1_EVIDENCE\n',text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
target=pathlib.Path('outputs/remote-evidence/g1-initial3108');target.mkdir(exist_ok=True);refs=[]
for r in json.loads(p.stdout):
 b=base64.b64decode(r.pop('data'));assert hashlib.sha256(b).hexdigest()==r['sha256'];f=target/r['relative_path'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);refs.append(r)
(target/'fetch-refs.json').write_text(json.dumps(refs,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='native_path'} for r in refs]))
