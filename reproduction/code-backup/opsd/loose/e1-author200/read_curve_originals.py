"""Read only the two bound original logs and saved resolved config for plots."""
import base64, datetime, gzip, hashlib, importlib.util, json, shlex, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
transport_path = HERE / 'transport.py'
assert hashlib.sha256(transport_path.read_bytes()).hexdigest() == 'a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
spec = importlib.util.spec_from_file_location('curve_transport', transport_path)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
output = HERE / ('curve-originals-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
output.mkdir(exist_ok=False)
code = r'''
import base64,datetime,gzip,hashlib,json,os,socket,stat
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
paths={
 'driver.out':Path('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs/worker-8350026ae4bc8387c3d3b6880792734dd21e46ad5a9c09e08669a24e-02000000-1533312.out'),
 'train.log':root/'train.log',
 'resolved-config.json':root/'full_state/global_step_5/resolved-config.json',
 'driver200.json':root/'full_state/global_step_200/driver.json'
}
result={'scope':'BOUND_SMALL_ORIGINALS_FOR_CURVE_NOT_FINAL_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'files':{}}
assert result['node']=='h100-10' and os.getuid()==0
def ident(s):return [s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
for name,path in paths.items():
 assert path.resolve(strict=True)==path
 fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_uid==0 and before.st_size<32*1024*1024
  raw=f.read();after=os.fstat(f.fileno())
 assert len(raw)==before.st_size and ident(before)==ident(after)==ident(path.stat())
 packed=gzip.compress(raw,compresslevel=6,mtime=0)
 result['files'][name]={'path':str(path),'identity':ident(before),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':hashlib.sha256(packed).hexdigest(),'gzip_base64':base64.b64encode(packed).decode()}
print(json.dumps(result,sort_keys=True))
'''
(output / 'reader-source.py').write_text(code)
try:
    r = transport.remote('python3 -c ' + shlex.quote(code), timeout=90)
except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
    (output / 'transport.stdout.original').write_bytes(error.stdout or b'')
    (output / 'transport.stderr.original').write_bytes(error.stderr or b'')
    (output / 'failure.json').write_text(json.dumps({'error': type(error).__name__, 'returncode': getattr(error, 'returncode', None)}))
    print(json.dumps({'output': str(output), 'error': type(error).__name__}))
    raise SystemExit(1)
(output / 'transport.stdout.original').write_bytes(r.stdout)
(output / 'transport.stderr.original').write_bytes(r.stderr)
x = json.loads(r.stdout)
for name, z in x['files'].items():
    packed = base64.b64decode(z.pop('gzip_base64'), validate=True)
    assert hashlib.sha256(packed).hexdigest() == z['gzip_sha256']
    raw = gzip.decompress(packed)
    assert len(raw) == z['bytes'] and hashlib.sha256(raw).hexdigest() == z['sha256']
    (output / name).write_bytes(raw)
x['transport_sha256'] = hashlib.sha256(r.stdout).hexdigest()
x['actual_transport_returncode'] = r.returncode
(output / 'summary.json').write_text(json.dumps(x, indent=2) + '\n')
print(json.dumps({'output': str(output), 'files': {k: v['bytes'] for k,v in x['files'].items()}, 'transport_sha256': x['transport_sha256']}))
