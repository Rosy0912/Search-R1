import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
src=r'''
import pathlib,json,hashlib,tarfile,socket
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
packages={'topd-instruct300-source-v1':'d50beaeaa97d9db3f2948e3fd0d17d90ed6227b2939f3b9e42e1f554d6951bb3','topd-vllm-teacher-v1':'ead268797f25e2e50418f17a6343b7b44a1c7274f77bb2e5cb2cd37866f2f774','topd-instruct300-launch-v1':'0145fc86fb2608d32ecffa60c5e76743763b148585894a1b52e6b6ed2057cfd5'}
for name,digest in packages.items():
 archive=R/'tools'/(name+'.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest
 target=R/'tools'/name
 if not target.exists():
  with tarfile.open(archive) as tf:
   for m in tf.getmembers():
    p=pathlib.PurePosixPath(m.name)
    assert p.parts[0]==name and '..' not in p.parts and not p.is_absolute() and m.isfile()
   tf.extractall(R/'tools')
 print(json.dumps({'installed':str(target),'archive_sha256':digest,'scope':'CODE_ONLY_NO_GPU_NO_TRAINING_NO_SUBMISSION'}),flush=True)
'''
r=send(src,attempts=1,timeout=120)
pathlib.Path('outputs/topd-instruct300-code-install-result.log').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
