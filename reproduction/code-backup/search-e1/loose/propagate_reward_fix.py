from pathlib import Path
import subprocess, json, sys
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/answer-reward-fix')
OUT=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/reward-fix-branches.json')
PYTHON='/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
def run(*args,check=True):
 p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if check and p.returncode: raise RuntimeError(' '.join(args)+'\n'+p.stdout)
 return p

def baseline():
 p=ROOT/'search_r1/baselines/bapo.py';s=p.read_text().replace('import torch','import torch\nfrom search_r1.qa_protocol import extract_final_answer, parse_model_action',1)
 a=s.index('def answer_from_spans(');b=s.index('\ndef ',a+5)
 s=s[:a]+'def answer_from_spans(spans):\n    return extract_final_answer(spans)\n\n'+s[b:]
 a=s.index('def legal_action_records(');b=s.index('\ndef ',a+5)
 s=s[:a]+'''def legal_action_records(records):
    if not records:
        return False
    final_kind = None
    for i, record in enumerate(records):
        action = parse_model_action(record['text'], strict_prefix=True)
        if action is None:
            return False
        final_kind = action[0]
        if final_kind == 'answer' and i != len(records) - 1:
            return False
    return final_kind == 'answer'

'''+s[b:];p.write_text(s)

def resolve_core(branch):
 for name in ['qa_em.py','qa_em_format.py']:
  path='verl/utils/reward_score/'+name
  s=run('git','show','origin/'+branch+':'+path).stdout
  s=s.replace('import random','import random\nfrom search_r1.qa_protocol import extract_final_answer, extract_legacy_final_answer',1)
  a=s.index('def extract_solution(');b=s.index('\ndef ',a+5)
  s=s[:a]+'def extract_solution(solution_str):\n    return extract_legacy_final_answer(solution_str)\n\n'+s[b:]
  if name=='qa_em.py':
   a=s.index('def compute_score_generated_em(');b=s.find('\ndef ',a+5);b=len(s) if b<0 else b
   s=s[:a]+'''def compute_score_generated_em(model_spans, ground_truth, format_score=0., score=1.):
    """Score only a legal terminal answer in the final model-authored span."""
    answer = extract_final_answer(model_spans)
    if answer is None:
        return 0.
    return score if em_check(answer, ground_truth['target']) else format_score

'''+s[b:]
  (ROOT/path).write_text(s.rstrip()+'\n')
 run('git','add','verl/utils/reward_score/qa_em.py','verl/utils/reward_score/qa_em_format.py')
 run('git','-c','core.editor=true','cherry-pick','--continue')

branches=[s.removeprefix('origin/') for s in run('git','for-each-ref','--format=%(refname:short)','refs/remotes/origin').stdout.splitlines() if s!='origin/HEAD']
rows=[]
for idx,branch in enumerate(branches):
 old=run('git','rev-parse','origin/'+branch).stdout.strip()
 local='codex/reward-fix-'+str(idx)
 run('git','switch','-c',local,'origin/'+branch)
 p=run('git','cherry-pick','eaa74b8',check=False)
 if p.returncode: resolve_core(branch)
 core=run('git','rev-parse','HEAD').stdout.strip();adapters=[]
 if (ROOT/'search_r1/baselines/bapo.py').exists():
  baseline();run('git','add','search_r1/baselines/bapo.py')
  run('git','commit','-m','fix: align baseline F1 and action legality with terminal QA protocol')
  adapters.append(run('git','rev-parse','HEAD').stdout.strip())
 if branch=='search-g1':
  run('git','cherry-pick','codex/fix-g1-answer-reward')
  adapters.append(run('git','rev-parse','HEAD').stdout.strip())
 run('git','diff','origin/'+branch,'--check')
 test=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_terminal_answer_protocol.py','-v').stdout
 provenance='not present'
 if (ROOT/'tests/reproduction/test_reward_and_validation.py').exists():
  provenance=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_reward_and_validation.py','-v').stdout
 row=dict(branch=branch,old=old,local=local,core=core,adapters=adapters,head=run('git','rev-parse','HEAD').stdout.strip(),protocol_tests=test,provenance_tests=provenance,pushed=False)
 rows.append(row);OUT.write_text(json.dumps(rows,indent=2))
 print('VERIFIED '+branch+' '+row['head'][:12],flush=True)
print('ALL VERIFIED '+str(len(rows)),flush=True)
