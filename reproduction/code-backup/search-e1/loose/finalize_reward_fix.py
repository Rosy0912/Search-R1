from pathlib import Path
import subprocess,json,ast
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/answer-reward-fix');OUT=ROOT.parent/'reward-fix-branches.json';rows=json.loads(OUT.read_text())
PYTHON='/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
def run(*args,check=True):
 p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if check and p.returncode: raise RuntimeError(' '.join(args)+'\n'+p.stdout)
 return p
for row in rows:
 if row.get('format_adapter'):continue
 if 'format_adapter' in row: continue
 run('git','switch',row['local']);p=run('git','cherry-pick','codex/fix-final-answer-reward',check=False)
 if p.returncode:
  unmerged=run('git','diff','--name-only','--diff-filter=U').stdout.splitlines()
  if unmerged!=['tests/reproduction/test_terminal_answer_protocol.py']:raise RuntimeError(p.stdout)
  path=unmerged[0];old=run('git','show','HEAD:'+path).stdout;new=run('git','show','codex/fix-final-answer-reward:'+path).stdout
  tree=ast.parse(old)
  for node in ast.walk(tree):
   if isinstance(node,ast.FunctionDef) and node.name=='test_g1_contexts_reject_nonterminal_answers':
    method='\n'.join(old.splitlines()[node.lineno-1:node.end_lineno])+'\n\n'
    new=new.replace('    def test_legacy_api_and_nested_actions(self):',method+'    def test_legacy_api_and_nested_actions(self):')
  (ROOT/path).write_text(new);run('git','add',path);run('git','-c','core.editor=true','cherry-pick','--continue')
 row['format_adapter']=run('git','rev-parse','HEAD').stdout.strip();row['head']=row['format_adapter']
 run('git','diff','origin/'+row['branch'],'--check')
 row['protocol_tests']=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_terminal_answer_protocol.py','-v').stdout
 if (ROOT/'tests/reproduction/test_reward_and_validation.py').exists():
  row['provenance_tests']=run(PYTHON,'-m','unittest','discover','-s','tests/reproduction','-p','test_reward_and_validation.py','-v').stdout
 OUT.write_text(json.dumps(rows,indent=2));print('FINAL VERIFIED '+row['branch'],flush=True)
