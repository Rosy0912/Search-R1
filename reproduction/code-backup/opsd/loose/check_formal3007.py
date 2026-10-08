"""Read-only current formal evidence check using the deployed immutable validator."""
import json
import shlex
import subprocess
from pathlib import Path

code = r'''
from pathlib import Path
import sys,hashlib,json,re,datetime,subprocess
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
sys.path.insert(0,str(root/'tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6'))
from verify_formal_prerequisites import verify_outers
from strict_metric_parser import parse_metric_summary
run=root/'runs/opsd-formal-3007'
raw=(run/'training.log').read_bytes()
text=raw.decode()
physical=text.splitlines(keepends=True)
complete=''.join(physical if not physical or physical[-1].endswith('\n') else physical[:-1])
accepted_eof=False
# Ray can expose a whole logger record before its newline. Accept only its full
# three-decimal terminal field or one exact complete typed warning frame.
if physical and not physical[-1].endswith('\n') and 'actor/optimizer_steps:' in physical[-1]:
 try:
  _,_,events=parse_metric_summary(physical[-1])
  if events or re.search(r'timing_per_token_ms/gen:[+-]?[0-9]+\.[0-9]{3}\Z',physical[-1]):
   complete+=physical[-1];accepted_eof=True
 except AssertionError:
  pass
steps=[int(m[1]) for line in complete.splitlines() if 'actor/optimizer_steps:' in line for m in [re.search(r'(?:^|\s)step:(\d+) - ',line)] if m]
proof=verify_outers(complete,max(steps))
proof.update(raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=3007,excluded_unterminated_tail_bytes=len(raw)-len(complete.encode()),complete_eof_metric_accepted=accepted_eof)
proof['slurm']=subprocess.check_output(['scontrol','show','job','3007'],text=True)
proof['source']=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
assert proof['source']=='8e60976d364ab7d7ae51714415813c1359dafe9a'
proof['tracked_source_status']=subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain','--untracked-files=no'],text=True)
assert not proof['tracked_source_status']
assert 'JobId=3007 ' in proof['slurm'] and 'NodeList=h100-10' in proof['slurm']
proof['training_finished']=(run/'training-finished.txt').exists()
proof['exit_marker']=(run/'exit.txt').read_text() if (run/'exit.txt').exists() else None
print(json.dumps(proof))
'''

work = Path(__file__).resolve().parent
result = subprocess.run(
    ['python3', str(work / 'remote.py'), 'python3 -c ' + shlex.quote(code)],
    capture_output=True, text=True, check=True,
)
proof = json.loads(result.stdout)
temporary = work / 'job3007-root-latest-numeric-check.json.partial'
temporary.write_text(json.dumps(proof, indent=2) + '\n')
temporary.replace(work / 'job3007-root-latest-numeric-check.json')
print(json.dumps({key: value for key, value in proof.items() if key != 'slurm'}))
