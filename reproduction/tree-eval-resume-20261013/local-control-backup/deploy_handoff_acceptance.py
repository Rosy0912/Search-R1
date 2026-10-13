"""Publish only reviewed CPU verification code; never models or scores."""
import base64, hashlib, importlib.util, json, shlex, subprocess
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('ssh', HERE/'ssh_retained18.py')
ssh = importlib.util.module_from_spec(spec); spec.loader.exec_module(ssh)
repo = HERE/'branpo-source'
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/tree-branpo-tail-20261011'],cwd=repo,text=True).split()[0]
assert remote == commit, 'Publish and verify source first'
source = repo/'reproduction/tree-eval-resume-20261013'
review = json.loads((source/'HANDOFF_HF_CODE_REVIEW.json').read_text())
wrapper = (source/'verify_handoff_hf.py').read_bytes()
assert hashlib.sha256(wrapper).hexdigest() == '8cc378a52009a8a41740831aa2259f380004da94760b3e7e98c4c8243a9c79ff'
dest = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/handoff-hf-'+commit
files = {name:base64.b64encode((source/name).read_bytes()).decode() for name in ['verify_handoff_hf.py','HANDOFF_HF_CODE_REVIEW.json','HANDOFF_HF_CODE_REVIEW.md','README.md']}
code = 'import base64,hashlib,json,os\nfrom pathlib import Path\np=Path('+repr(dest)+');p.mkdir()\nfiles='+repr(files)+'\nresult={}\nfor name,b64 in files.items():\n data=base64.b64decode(b64)\n with (p/name).open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())\n assert (p/name).read_bytes()==data\n result[name]=hashlib.sha256(data).hexdigest()\nprint(json.dumps(result))'
r = ssh.run('python3 -c '+shlex.quote(code),timeout=120)
assert r.returncode == 0, r.stdout+r.stderr
receipt = dict(source_commit=commit,remote_sha=remote,tools=dest,files=json.loads(r.stdout))
Path('outputs/tree-eval-resume-20261013/HANDOFF_HF_DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
