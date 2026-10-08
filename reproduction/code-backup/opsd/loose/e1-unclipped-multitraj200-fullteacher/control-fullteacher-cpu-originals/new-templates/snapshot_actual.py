"""Read own current job/log/GPU originals. Counts are observations, not acceptance."""
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
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);a=p.parse_args()
    job=(HERE/'actual-job-id.txt').read_text().strip();assert job.isdigit() and int(job)>0
    assert a.binding.resolve(strict=True).parent==HERE
    spec=json.loads(a.binding.read_bytes())
    run=spec['remote_root']+'/runs/formal-'+job
    assert a.output.parent.resolve(strict=True)==a.output.parent and not a.output.exists()
    a.output.mkdir(mode=0o700)
    code='''import pathlib,subprocess,json,base64,datetime,hashlib
r=pathlib.Path(RUN);files={}
for name in ['formal-wrapper.log','train.log','retriever.log','job-exit.txt','evidence/hardware-exit.json','evidence/capacity-micro4/exit.json','evidence/capacity-micro4/micro-4.json','evidence/capacity-micro4/torchrun.log','evidence/train-process-exit.json','process-finished-review-required.json','checkpoint-5-manifest.json','checkpoint-5-ready.json']:
 p=r/name
 if p.is_file():
  raw=p.read_bytes();files[name]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
live={}
for name,argv in [('scontrol',['scontrol','show','job','-o',JOB]),('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),('gpu_apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
 q=subprocess.run(argv,capture_output=True,text=True,timeout=25);live[name]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':JOB,'run_root':str(r),'files':files,'live':live},sort_keys=True))
'''.replace('RUN',repr(run)).replace('JOB',repr(job))
    try:q=remote('python3 -c '+shlex.quote(code),timeout=60)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        (a.output/'transport.stdout.original').write_bytes(e.stdout or b'');(a.output/'transport.stderr.original').write_bytes(e.stderr or b'');raise
    (a.output/'transport.stdout.original').write_bytes(q.stdout);(a.output/'transport.stderr.original').write_bytes(q.stderr)
    d=json.loads(q.stdout)
    for name,row in d['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['sha256']
        t=a.output/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(raw)
    rows=[]
    for line in (a.output/'train.log').read_bytes().splitlines(keepends=True) if (a.output/'train.log').exists() else []:
        if not line.endswith(b'\n') or b'OPSD_E1_OUTER_JSON=' not in line:continue
        rows.append(json.loads(line.split(b'OPSD_E1_OUTER_JSON=',1)[1]))
    record={k:v for k,v in d.items() if k!='files'}
    record.update(observed_complete_driver_outer_records=len(rows),last_driver_record=rows[-1] if rows else None,scope='Original read-only observation; independent all8/count/checkpoint/effect acceptance separate')
    (a.output/'actual-observation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'job_id':job,'observed_complete_outer_records':len(rows),'last_driver':rows[-1] if rows else None,'live':d['live'],'files':list(d['files'])}))

if __name__=='__main__':main()
