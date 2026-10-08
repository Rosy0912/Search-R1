from pathlib import Path
import json,subprocess
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/answer-reward-fix')
p= root.parent/'reward-fix-branches.json';rows=json.loads(p.read_text())
def run(*a):
 x=subprocess.run(a,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if x.returncode:raise RuntimeError(x.stdout)
 return x.stdout.strip()
run('git','fetch','origin')
for row in rows:
 if row['pushed']:continue
 remote=run('git','rev-parse','origin/'+row['branch'])
 if remote!=row['old']:
  run('git','switch',row['local']);run('git','rebase','origin/'+row['branch'])
  row['head']=run('git','rev-parse','HEAD')
  row['protocol_tests']=run('/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-m','unittest','discover','-s','tests/reproduction','-p','test_terminal_answer_protocol.py','-v')
 run('git','push','origin',row['local']+':refs/heads/'+row['branch'])
 row['pushed']=True;p.write_text(json.dumps(rows,indent=2));print('PUSHED',row['branch'],row['head'][:12],flush=True)
run('git','push','origin','codex/fix-final-answer-reward','codex/fix-baseline-answer-reward','codex/fix-g1-answer-reward')
refs=run('git','ls-remote','--heads','origin');refmap={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in refs.splitlines()}
for row in rows:
 row['remote_verified']=refmap.get(row['branch'])==row['head']
 if not row['remote_verified']:raise RuntimeError('Remote changed: '+row['branch'])
p.write_text(json.dumps(rows,indent=2));print('ALL REMOTES VERIFIED',len(rows),flush=True)
