"""Read actual submitted allocation originals; no authorization or GPU action."""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from transport import remote

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    job=(HERE/'actual-job-id.txt').read_text().strip()
    assert job.isdigit() and int(job)>0
    spec=json.loads((HERE/'runtime-tools-bound-667c68e0f91b-v2.json').read_bytes())
    run=spec['remote_root']+'/runs/formal-'+job
    assert args.output.parent.resolve(strict=True)==args.output.parent and not args.output.exists()
    args.output.mkdir(mode=0o700)
    code='''import base64,hashlib,json,pathlib,subprocess,datetime,os,socket
r=pathlib.Path(RUN);ready=r/'approval-ready.json'
names=['approval-ready.json','bindings/manifest.json','bindings/public.json','evidence/pre-gpu-import-check.json','evidence/pre-gpu-profile.json','evidence/pre-gpu-python-environment.json']
binding=json.loads((r/'bindings/manifest.json').read_bytes())
names += ['bindings/'+name+'.json' for name in binding['files']]
files={}
for name in names:
 p=r/name;raw=p.read_bytes();files[name]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'uid':p.stat().st_uid,'physical':str(p.resolve(strict=True))}
live={}
for name,argv in [('scontrol',['scontrol','show','job','-o',JOB]),('gpu_inventory',['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),('gpu_applications',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])]:
 q=subprocess.run(argv,capture_output=True,text=True,timeout=30);live[name]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
v=json.loads(ready.read_bytes());p=pathlib.Path('/proc')/str(v['pid'])
live['wrapper_proc']={n:(p/n).read_text() for n in ['status','stat','cgroup','cmdline']}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':JOB,'uid':os.getuid(),'hostname':socket.gethostname(),'run_root':str(r),'files':files,'live':live},sort_keys=True))
'''.replace('RUN',repr(run)).replace('JOB',repr(job))
    try:
        result=remote('python3 -c '+shlex.quote(code),timeout=90)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
        (args.output/'transport.stdout.original').write_bytes(exc.stdout or b'')
        (args.output/'transport.stderr.original').write_bytes(exc.stderr or b'')
        (args.output/'transport-failure.original.json').write_text(json.dumps({'error_type':type(exc).__name__,'returncode':getattr(exc,'returncode',None),'scope':'Read-only collection failed; original output preserved'},indent=2)+'\n')
        raise
    (args.output/'transport.stdout.original').write_bytes(result.stdout)
    (args.output/'transport.stderr.original').write_bytes(result.stderr)
    record=json.loads(result.stdout)
    for name,row in record['files'].items():
        assert Path(name).is_absolute() is False and '..' not in Path(name).parts
        raw=base64.b64decode(row['base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
    (args.output/'actual-observation.json').write_text(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2)+'\n')
    print(json.dumps({'job_id':job,'output':str(args.output),'ready_sha256':record['files']['approval-ready.json']['sha256'],'source':spec['source_commit'],'scope':'Actual read-only pre-GPU originals, no initialization approval'}))

if __name__=='__main__':main()
