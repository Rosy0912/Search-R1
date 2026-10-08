import base64,hashlib,json,pathlib,subprocess
remote="""python3 - <<'INNER'
import base64,hashlib,json,pathlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');out=r/'logs/qwen25-32b-terminal-fixed413-eval64-3036'
paths=[out/n for n in ('supervisor-result.json','evaluation.log','final-gpu.json','baseline.json','baseline-nvml.json','active-runtime.json','active-occupancy.jsonl','retriever-readiness.json')]
paths += [r/'logs/qwen25-32b-startup-guard-3036.jsonl',r/'logs/qwen25-32b-readonly-watch-3036.jsonl',r/'logs/qwen25-32b-startup-guard-3036.log']
print(json.dumps({p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'base64':base64.b64encode(p.read_bytes()).decode()} for p in paths if p.exists()}))
INNER"""
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
d=json.loads(p.stdout);out=pathlib.Path('outputs/remote-evidence/eval64-3036-failed');out.mkdir(exist_ok=True)
for n,x in d.items():
 b=base64.b64decode(x['base64']);assert hashlib.sha256(b).hexdigest()==x['sha256'];(out/n).write_bytes(b)
print(json.dumps({n:{'sha256':x['sha256'],'bytes':(out/n).stat().st_size} for n,x in d.items()},indent=2))
