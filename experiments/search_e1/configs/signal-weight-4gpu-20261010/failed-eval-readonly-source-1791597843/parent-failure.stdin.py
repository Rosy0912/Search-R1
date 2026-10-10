from pathlib import Path
import json,hashlib,time,os,sys
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010');R=S/'group-G_POS/runs';E=R/'evaluations/outer-15';D=E/'execution'
def ref(p):
 b=p.read_bytes();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def j(p):return json.loads(p.read_bytes())
assert os.getuid()==1001
status=j(R/'status.json')
matches=[]
for p in sorted((R/'phase-exits').glob('*.json')):
 d=j(p)
 if any(x.get('identity',{}).get('pid')==1715934 for x in d.get('child_exit_evidence',[])):
  matches.append(dict(reference=ref(p),evidence=d))
history=[]
hp=R/'child-history.jsonl'
for line in hp.read_bytes().splitlines():
 d=json.loads(line)
 if d.get('phase')=='evaluation-running' or d.get('identity',{}).get('pid')==1715934:history.append(d)
lp=E/'controller.log';b=lp.read_bytes()
lines=b.decode('utf-8',errors='replace').splitlines()
stack=[];active=False
for line in lines:
 if line.startswith('Traceback (most recent call last):'):stack=[];active=True
 if active:
  if line.startswith(('  File ','Traceback ')) or line.lstrip().startswith(('gates.Hold:','Hold:','ValueError:','RuntimeError:','FileNotFoundError:','ProcessLookupError:','AssertionError:','TypeError:','KeyError:','AttributeError:','NameError:','OSError:','ImportError:')):stack.append(line[:2000])
  elif line.startswith('    ') and len(line)<1000 and not any(w in line for w in ('response','question','messages','prompt','content')):stack.append(line)
out=dict(schema='opsd-signal-native15-parent-failure-readonly-v1',checked_unix=time.time(),uid=os.getuid(),status=status,actual_parent_phase_terminal=matches,matching_evaluation_child_history=history,controller_log_reference=ref(lp),controller_log_filtered_last_traceback=stack,execution_files=[dict(name=p.relative_to(D).as_posix(),bytes=p.stat().st_size) for p in sorted(D.glob('*.json'))],no_signals_or_launches=True,no_raw_question_or_trajectory_in_stdout=True,no_torch='torch' not in sys.modules)
print(json.dumps(out,sort_keys=True))
