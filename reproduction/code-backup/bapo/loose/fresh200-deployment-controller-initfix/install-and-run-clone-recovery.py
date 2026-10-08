"""Parent-operated additive recovery transport; never a launch or scheduler action."""
from pathlib import Path
import json,shlex,subprocess
B=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
SCRIPT_SHA='8dc544900245626b74179379f5ae45ea637561b9edf02d79ba9bf27fe4e3da3d'
code="""
import hashlib,os,pathlib,subprocess,sys
if sys.flags.optimize:raise RuntimeError('optimized transport forbidden')
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
assert sys.platform=='linux' and os.uname().nodename=='h100-5'
assert root.is_dir() and root.resolve()==root and not root.is_symlink()
assert (root/'logs').is_dir() and not (root/'logs').is_symlink()
assert (root/'source-expected.txt').read_text().strip()=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert not (root/'repo').exists() and not (root/'repo').is_symlink()
raw=sys.stdin.buffer.read()
assert hashlib.sha256(raw).hexdigest()=='8dc544900245626b74179379f5ae45ea637561b9edf02d79ba9bf27fe4e3da3d'
path=root/'logs/recover-source-clone-reviewed.py'
with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
fd=os.open(root/'logs',os.O_RDONLY);os.fsync(fd);os.close(fd)
subprocess.run([str(root/'env/bin/python'),'-B',str(path),'--execute-worker'],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
"""
if __name__=='__main__':
 import hashlib
 assert hashlib.sha256((D/'recover-source-clone.py').read_bytes()).hexdigest()==SCRIPT_SHA
 log=B/'evidence/fresh200-controller-initfix-clone-recovery.actual.log'
 with log.open('x')as out:
  result=subprocess.run(['python3',str(B/'h100_exec.py'),'--socks','--timeout','120','--stdin-file',str(D/'recover-source-clone.py'),'python3 -B -c '+shlex.quote(code)],stdout=out,stderr=subprocess.STDOUT)
 print(json.dumps(dict(exit_code=result.returncode,raw_log=str(log),scope='ADDITIVE_SOURCE_CLONE_FINISH_ONLY_NO_GPU_NO_LAUNCH')))
 raise SystemExit(result.returncode)
