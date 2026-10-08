import base64,hashlib,json,pathlib,subprocess,sys
jid=int(sys.argv[1]);assert jid>0
remote="python3 - <<'INNER'\nimport base64,hashlib,json,pathlib\nr=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');out=r/'logs/qwen25-32b-window-dual-burst-JID'\npaths=[p for p in out.rglob('*') if p.is_file()]+[r/'logs/topd-window-dual-telemetry-burst-JID.log',r/'logs/qwen25-32b-readonly-watch-JID.jsonl']\nprint(json.dumps({(str(p.relative_to(out)) if out in p.parents else p.name):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'base64':base64.b64encode(p.read_bytes()).decode()} for p in paths if p.is_file()}))\nINNER".replace('JID',str(jid))
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
d=json.loads(p.stdout);out=pathlib.Path('outputs/remote-evidence/window-dual-burst-'+str(jid));out.mkdir(exist_ok=True)
for n,x in d.items():
 assert not pathlib.PurePath(n).is_absolute() and '..' not in pathlib.PurePath(n).parts;(out/n).parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(x['base64']);assert hashlib.sha256(b).hexdigest()==x['sha256'];(out/n).write_bytes(b)
print(json.dumps({n:{'sha256':x['sha256'],'bytes':(out/n).stat().st_size} for n,x in d.items()},indent=2))
