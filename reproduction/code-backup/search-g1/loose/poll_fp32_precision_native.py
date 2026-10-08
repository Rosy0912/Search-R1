import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');h=json.loads((prep/'fp32-precision-native-launch.json').read_text())
code='import json,subprocess,time;from pathlib import Path\nb=Path('+repr(h['base'])+');job='+repr(h['job_id'])+'\nrecords=[]\nfor p in (b/"logs").glob("precision-rank-*.jsonl"):\n for line in p.read_text().splitlines():\n  try:records.append(json.loads(line))\n  except ValueError:pass\nl=b/"logs"/("slurm-"+job+".log")\nprint(json.dumps(dict(checked_unix=time.time(),job=subprocess.check_output(["scontrol","show","job",job,"-o"],text=True),records=records,log_tail=l.read_text(errors="replace")[-4000:] if l.exists() else "")))'
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'fp32-precision-native-observation.json').write_text(json.dumps(x,indent=2));print(json.dumps(dict(job=x['job'],record_count=len(x['records']),log_tail=x['log_tail'][-1700:]),indent=2))
