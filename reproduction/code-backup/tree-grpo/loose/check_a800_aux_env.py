import sys,os,pathlib,importlib.metadata as md,json,time
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval')
if '--paths-cleaned' not in sys.argv:
 changes={}
 for p in (root/'env/lib/python3.10/site-packages').glob('*.pth'):
  old=p.read_text();lines=old.splitlines(keepends=True)
  kept=[]
  for line in lines:
   if '/data/search-r1/' in line:
    assert line.strip()=='/data/search-r1/repo', (str(p),line)
   else:kept.append(line)
  new=''.join(kept)
  if new!=old:p.write_text(new);changes[str(p)]={'before':old,'after':new}
 (root/'logs/own-copied-pth-repair.json').write_text(json.dumps(changes,indent=2)+'\n')
 os.execv(str(root/'env/bin/python'),[str(root/'env/bin/python'),str(root/'tools/check_a800_aux_env.py'),'--paths-cleaned'])
assert sys.prefix==str(root/'env'),sys.prefix
paths=sys.path;assert not any('/data/search-r1/' in p for p in paths),paths
pth={str(p):p.read_text() for p in (root/'env/lib/python3.10/site-packages').glob('*.pth')}
assert not any('/data/search-r1/' in s for s in pth.values()),pth
packages={d.metadata['Name']:d.version for d in md.distributions()}
(root/'logs/environment-freeze.txt').write_text('\n'.join(k+'=='+v for k,v in sorted(packages.items(),key=lambda kv:kv[0].lower()))+'\n')
import torch,transformers,numpy
for mod in [torch,transformers,numpy]:assert str(root/'env') in mod.__file__,mod.__file__
record={'sys_prefix':sys.prefix,'sys_path':paths,'pth_files':pth,'module_paths':{m.__name__:m.__file__ for m in [torch,transformers,numpy]},'source_environment':'/data/search-r1/env copied read only; no inherited source purelib','aux_environment_scope':'retriever only; no Ray CLI invoked','pip_unavailable':'source venv has no pip; freeze through importlib.metadata','epoch':time.time()}
(root/'logs/environment-isolation.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'logs/preparation-completed.json').write_text(json.dumps({'epoch':time.time(),'scope':'own env copy, independent immutable SHA verification; no GPU operation'})+'\n')
print(json.dumps(record))
