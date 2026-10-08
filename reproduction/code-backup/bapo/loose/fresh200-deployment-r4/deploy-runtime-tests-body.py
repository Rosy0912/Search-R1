M={'source_commit': '5121a8d9349eadb5a81119cc4573165338e041bd', 'tar_sha256': 'c255144afcbeab38ba41cc6d63c87be97d7701edbfbd5e26c8867635d40178d8', 'files': {'test_freeze_scalability.py': '8d71d683b17d2242f600193ba5332a0e9f4d19b04db5b7f037ef5574a7fe9e40', 'test_launch_transaction.py': '8b9385ce8c510f5af5a6cc60ff6cec8ade6de63406aeb43ee2864e04ba4a9fb7', 'test_resource_watch_recovery.py': '5acb3fb153e4d8a96b7ef0af7433d501a1786b4989abfd639d55deb79ea51ef0', 'test_runtime_anchor.py': 'bfa31421fa7289a608f6663d580e65375c48c04fcb6bb1e8d197d42d44e4ef30'}}
import sys,os,json,hashlib,tarfile,io,subprocess,time
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized forbidden')
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r4-20261006')
assert sys.platform=='linux' and os.uname().nodename=='h100-5' and sys.prefix==str(R/'env')
assert (R/'source-expected.txt').read_text().strip()==M['source_commit']
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==M['tar_sha256']
assert not (R/'tests').exists();(R/'tests').mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
 assert {x.name for x in tar.getmembers()}=={'tests/'+k for k in M['files']}
 for x in tar.getmembers():
  assert x.isfile() and x.name.startswith('tests/') and '..' not in x.name
  data=tar.extractfile(x).read();assert hashlib.sha256(data).hexdigest()==M['files'][Path(x.name).name]
  with (R/x.name).open('xb') as f:f.write(data)
started=time.monotonic()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'repo'))
with (R/'logs/runtime-tests.actual.log').open('x') as out:
 result=subprocess.run([str(R/'env/bin/python'),'-B','-m','unittest','discover','-s',str(R/'tests'),'-v'],cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
record=dict(source_commit=M['source_commit'],host=os.uname().nodename,exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,files=M['files'],raw_log_sha256=hashlib.sha256((R/'logs/runtime-tests.actual.log').read_bytes()).hexdigest(),scope='ACTUAL_LINUX_OWN_CHILD_RESOURCE_TESTS_NOT_GPU')
with (R/'logs/runtime-tests.actual.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record));raise SystemExit(result.returncode)
