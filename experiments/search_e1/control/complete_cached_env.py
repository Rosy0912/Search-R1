"""Resolve from verified completed wheels; fetch only missing public dependencies."""
import os,subprocess,re
from pathlib import Path
base=Path(os.environ['E1_BASE']);py=str(base/'env/bin/python');control=base/'source/experiments/search_e1/control';req=str(control/'requirements-hkust.txt');wheels=str(base/'wheels')
env=dict(os.environ,PIP_INDEX_URL='https://pypi.org/simple',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_TIMEOUT='15',PIP_RETRIES='1',PIP_CACHE_DIR=str(base/'pip-cache'),TMPDIR=str(base/'tmp'))
for attempt in range(30):
 p=subprocess.run([py,'-m','pip','install','--dry-run','--no-index','--find-links',wheels,'-r',req],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (base/'logs/local-resolution.log').write_text(p.stdout)
 if p.returncode==0:
  print('OFFLINE_ENV_RESOLUTION_ACCEPTED',flush=True)
  subprocess.run(['bash',str(base/'finish_env.sh')],env=env,check=True);break
 m=re.search(r'Could not find a version that satisfies the requirement (.+?) \(from',p.stdout)
 if not m:raise RuntimeError('Offline resolution failed without a missing dependency; inspect log')
 missing=m.group(1).split(';',1)[0].strip();print('FETCH_MISSING',missing,flush=True)
 subprocess.run([py,str(control/'pip_ipv4.py'),'download','--only-binary=:all:','--find-links',wheels,'-d',wheels,'-c',req,missing],env=env,check=True)
else:raise RuntimeError('Thirty dependency fetch rounds exhausted')
