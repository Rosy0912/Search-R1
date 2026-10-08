from pathlib import Path
import subprocess,json,os
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/answer-reward-fix');OUT=ROOT.parent/'reward-fix-branches.json';rows=json.loads(OUT.read_text());PYTHON='/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
def run(*args,check=True,env=None):
 p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
 if check and p.returncode:raise RuntimeError(' '.join(args)+'\n'+p.stdout)
 return p
run('git','fetch','origin','--prune')
for row in rows:
 run('git','switch',row['local'])
 remote=run('git','rev-parse','origin/'+row['branch']).stdout.strip()
 if remote!=row['old']:
  run('git','rebase','origin/'+row['branch']);row['old_before_rebase']=row['old'];row['old']=remote
 row['head']=run('git','rev-parse','HEAD').stdout.strip()
 run('git','diff','origin/'+row['branch'],'--check')
 row['protocol_tests']=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_terminal_answer_protocol.py','-v').stdout
 if (ROOT/'tests/reproduction/test_reward_and_validation.py').exists():
  row['provenance_tests']=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_reward_and_validation.py','-v').stdout
 if (ROOT/'verl/trainer/main_ppo_format.py').exists():
  row['format_manager_tests']=run(PYTHON,str(ROOT.parent/'check_format_reward_manager.py'),env=dict(os.environ,PYTHONPATH=str(ROOT))).stdout
 else:row['format_manager_tests']='Entry absent on this branch'
 OUT.write_text(json.dumps(rows,indent=2));print('READY '+row['branch'],flush=True)
refs=[r['local']+':refs/heads/'+r['branch'] for r in rows]
refs+=['codex/fix-final-answer-reward:refs/heads/codex/fix-final-answer-reward','codex/fix-baseline-answer-reward:refs/heads/codex/fix-baseline-answer-reward','codex/fix-g1-answer-reward:refs/heads/codex/fix-g1-answer-reward']
print(run('git','push','--atomic','origin',*refs).stdout,flush=True)
remote_heads=dict(line.split()[::-1] for line in run('git','ls-remote','--heads','origin').stdout.splitlines())
for row in rows:
 assert remote_heads['refs/heads/'+row['branch']]==row['head'],row['branch']
 row['pushed']=True
OUT.write_text(json.dumps(rows,indent=2));print('VERIFIED REMOTE HEADS '+str(len(rows)),flush=True)
