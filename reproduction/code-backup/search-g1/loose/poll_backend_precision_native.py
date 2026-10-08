import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
p=Path('/Users/shaoshitong/Documents/Search-G1/preparation');h=json.loads((p/'backend-precision-native-launch.json').read_text())
code='from pathlib import Path\nimport json,subprocess,time\nb=Path('+repr(h['base'])+')\nrecords=[]\nfor p in (b/"logs").glob("precision-rank-*.jsonl"):\n for l in p.read_text().splitlines():\n  try:records.append(json.loads(l))\n  except ValueError:pass\nl=b/"logs/slurm-3250.log"\nprint(json.dumps(dict(checked_unix=time.time(),job=subprocess.run(["scontrol","show","job","3250","-o"],capture_output=True,text=True).stdout,records=records,log_tail=l.read_text(errors="replace")[-3500:] if l.exists() else "")))'
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(p/'backend-precision-native-observation.json').write_text(json.dumps(x,indent=2));print(json.dumps(dict(checked_unix=x["checked_unix"],job=x["job"],records=[{k:v for k,v in r.items() if "buffers" not in k} for r in x["records"]],log_tail=x["log_tail"]),indent=2))
