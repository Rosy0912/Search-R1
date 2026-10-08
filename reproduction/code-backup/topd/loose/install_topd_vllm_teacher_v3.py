import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
src=r'''
import pathlib,json,hashlib,tarfile,socket
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
packages={'topd-vllm-teacher-v3':'bd39d31936fdaab410bdad0a11f1b561389ba72dcdb5244dbea11c7962e60efd'}
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
pathlib.Path('outputs/topd-vllm-teacher-v3-install-result.log').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
