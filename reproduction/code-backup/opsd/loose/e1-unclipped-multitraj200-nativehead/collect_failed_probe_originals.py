"""Read-only original collector for this actual failed allocation, not acceptance."""
from pathlib import Path
import argparse,base64,hashlib,json,shlex,subprocess,datetime
from transport import remote
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert a.binding.resolve(strict=True).parent==H and a.output.parent.resolve(strict=True)==a.output.parent and not a.output.exists()
spec=json.loads(a.binding.read_bytes());job=(H/'actual-job-id.txt').read_text().strip();terminal=H/f'job{job}-scheduler-terminal-original.txt';assert job=='3302' and 'JobState=FAILED ' in terminal.read_text() and 'ExitCode=1:0 ' in terminal.read_text()
a.output.mkdir(mode=0o700);run=spec['remote_root']+'/runs/formal-'+job
code=f'''import pathlib,os,json,base64,hashlib,datetime
r=pathlib.Path({run!r});out={{}}
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.suffix not in ('.json','.log','.txt','.ndjson'):continue
 assert not p.is_symlink() and p.resolve(strict=True)==p and p.stat().st_uid==0
 assert p.stat().st_size<20*1024*1024
 raw=p.read_bytes();out[str(p.relative_to(r))]={{'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'uid':p.stat().st_uid,'physical':str(p)}}
print(json.dumps({{'job_id':{job!r},'source_commit':{spec['source_commit']!r},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':out}},sort_keys=True))
'''
try:q=remote('python3 -c '+shlex.quote(code),timeout=90)
except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
 (a.output/'transport.stdout.original').write_bytes(e.stdout or b'');(a.output/'transport.stderr.original').write_bytes(e.stderr or b'');raise
(a.output/'transport.stdout.original').write_bytes(q.stdout);(a.output/'transport.stderr.original').write_bytes(q.stderr);v=json.loads(q.stdout);records={}
assert v['job_id']==job and v['source_commit']==spec['source_commit']
for name,row in v['files'].items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 raw=base64.b64decode(row['base64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(raw)
 records[name]={k:row[k] for k in ['sha256','bytes','uid','physical']}
summary={'job_id':job,'source_commit':spec['source_commit'],'observed_utc':v['utc'],'terminal_sha256':hashlib.sha256(terminal.read_bytes()).hexdigest(),'files':records,'scope':'Original failed0 probe collection; no RL/Adam/whole-run acceptance inferred'}
(a.output/'inventory.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'job_id':job,'count':len(records),'inventory_sha256':hashlib.sha256((a.output/'inventory.json').read_bytes()).hexdigest()}))
