import sys,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
r=send(pathlib.Path('outputs/standard-r1-v14-own-directory-prepare.py').read_text(),attempts=1,timeout=120)
pathlib.Path('outputs/standard-r1-v14-own-directory-prepare-result.log').write_bytes(r.stdout+r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
