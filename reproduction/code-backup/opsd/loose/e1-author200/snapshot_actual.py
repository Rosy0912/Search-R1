"""Copy read-only evidence for an observed author200 job; never submit or sign.

Only the new run root on h100-10 is queried. Source/model/native tensor files
are not copied. All local outputs are fresh, and remote evidence is untouched.
"""
import argparse
import base64
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def checked_job(job):
    if type(job) is not str or not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Observed actual positive decimal job ID required')
    return job


def remote_code(job, include_resources=False, resource_limit=256*1024*1024):
    """Render standard-library read-only observation code with no GPU imports."""
    checked_job(job)
    if type(resource_limit) is not int or resource_limit < 1:
        raise ValueError('Positive resource-prefix byte limit required')
    values = dict(root=ROOT, job=job, source=SOURCE,
                  include_resources=include_resources, resource_limit=resource_limit)
    return 'INPUT = ' + repr(values) + '\n' + REMOTE_BODY


REMOTE_BODY = r'''
import base64,csv,datetime,gzip,hashlib,io,json,os,re,socket,stat,subprocess
from pathlib import Path
root=Path(INPUT['root']);job=INPUT['job'];run=root/'runs'/('formal-'+job)
ORIGINAL_LOGS=('train.log','formal-wrapper.log','retriever.log',
              'evidence/hardware-torchrun.log','evidence/ray-start.log')
observed={'scope':'READ_ONLY_ACTUAL_AUTHOR200_SNAPSHOT_NOT_ACCEPTANCE',
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':job,'run_root':str(run),'node':socket.gethostname(),'reader_uid':os.getuid(),
 'expected_source':INPUT['source'],'files':{},'commands':{},'errors':[],
 'processes':{},'resource_history':None}
def command(name,args):
 try:
  p=subprocess.run(args,capture_output=True,timeout=20)
  result={'args':args,'returncode':p.returncode,'stdout':p.stdout.decode(errors='replace'),
          'stderr':p.stderr.decode(errors='replace')}
 except (OSError,subprocess.TimeoutExpired) as e:
  result={'args':args,'error':type(e).__name__,'detail':str(e)}
 observed['commands'][name]=result
 return result
def safe_under(path):
 path=Path(path)
 if not path.is_absolute() or path.resolve(strict=True)!=path:
  raise ValueError('Physical nonsymlink evidence path required')
 path.relative_to(run)
 if not path.is_file() or path.stat().st_uid!=os.getuid():
  raise ValueError('Own regular evidence file required')
 return path
def fingerprint(s):
 return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
         'ctime_ns':s.st_ctime_ns,'uid':s.st_uid,'mode':stat.S_IMODE(s.st_mode)}
def capture(path,key=None,limit=32*1024*1024,tail=False):
 try:
  path=safe_under(path);before=path.stat()
  if before.st_size>limit and not tail:
   observed['errors'].append({'path':str(path),'reason':'exceeds_small_original_limit','bytes':before.st_size});return
  start=max(0,before.st_size-limit) if tail else 0
  with path.open('rb') as f:
   f.seek(start);raw=f.read(before.st_size-start)
  after=path.stat();name=key or str(path.relative_to(run))
  observed['files'][name]={'physical_path':str(path),'base64':base64.b64encode(raw).decode(),
    'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'source_offset':start,
    'before':fingerprint(before),'after':fingerprint(after),
    'complete_original_at_start':start==0 and len(raw)==before.st_size,
    'same_inode':before.st_dev==after.st_dev and before.st_ino==after.st_ino,
    'unchanged_during_read':fingerprint(before)==fingerprint(after)}
 except (OSError,ValueError) as e:
  observed['errors'].append({'path':str(path),'error':type(e).__name__,'detail':str(e)})
def process(pid):
 key=str(pid)
 if key in observed['processes']:return observed['processes'][key]
 try:
  p=Path('/proc')/key;raw_stat=(p/'stat').read_text();fields=raw_stat.rsplit(')',1)[1].split()
  status=(p/'status').read_text();cgroup=(p/'cgroup').read_text()
  uids=[int(x) for x in re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()]
  own_job=bool(re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cgroup,re.M))
  value={'pid':pid,'start_ticks':fields[19],'state':fields[0],'uids':uids,'cgroup':cgroup,
         'comm':raw_stat.split('(',1)[1].rsplit(')',1)[0],'in_observed_job':own_job}
  if own_job:
   # Environment is a deliberate allowlist; never copy authentication material.
   allowed={'SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','PYTHONPATH',
            'SEARCH_R1_REPO','SEARCH_R1_BASELINE_RUN_ROOT','RAY_ADDRESS','RAY_TMPDIR',
            'OPSD_E1_OFFICIAL_MODEL','OPSD_E1_RETRIEVER_URL'}
   env=(p/'environ').read_bytes().split(b'\0');value['environment_allowlist']={}
   for entry in env:
    k,sep,v=entry.partition(b'=')
    if sep and k.decode(errors='replace') in allowed:
     value['environment_allowlist'][k.decode()]=v.decode(errors='replace')
   value['stdout_fd_target']=os.readlink(str(p/'fd/1'))
  observed['processes'][key]=value;return value
 except (OSError,ValueError,AttributeError,IndexError) as e:
  value={'pid':pid,'unavailable':True,'error':type(e).__name__,'detail':str(e)}
  observed['processes'][key]=value;return value
command('scontrol',['scontrol','show','job',job,'-o'])
command('sacct',['sacct','-j',job,'-n','-P','--format=JobID,JobName,State,ExitCode,Elapsed,ElapsedRaw,NodeList,AllocTRES'])
gpu=command('nvml_inventory',['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
apps=command('nvml_unfiltered_apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
observed['gpu_inventory_csv_rows']=list(csv.reader(io.StringIO(gpu.get('stdout',''))))
observed['unfiltered_application_csv_rows']=list(csv.reader(io.StringIO(apps.get('stdout',''))))
for row in observed['unfiltered_application_csv_rows']:
 if len(row)==3:
  try:process(int(row[1].strip()))
  except ValueError:observed['errors'].append({'reason':'invalid_unfiltered_nvml_pid','row':row})
try:
 if root.resolve(strict=True)!=root or root.stat().st_uid!=os.getuid():raise ValueError('Own physical new root required')
 if not run.exists():
  observed['run_directory_exists']=False
 else:
  if run.resolve(strict=True)!=run or run.stat().st_uid!=os.getuid():raise ValueError('Own physical actual run required')
  observed['run_directory_exists']=True
  for path in sorted(run.rglob('*')):
   if path.is_symlink():
    observed['errors'].append({'path':str(path),'reason':'symlink_not_followed'});continue
   if not path.is_file():continue
   name=path.name
   if path.suffix in ('.json','.txt') and name!='resources.ndjson':capture(path)
   elif str(path.relative_to(run)) in ORIGINAL_LOGS:
    capture(path)
    if path.parent==run and name in ('train.log','formal-wrapper.log'):
     capture(path,key=name+'.tail-original',limit=256*1024,tail=True)
  # Full node-local original logs referenced by actual owned stdout FDs only.
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    cg=(p/'cgroup').read_text()
    if re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cg,re.M):process(int(p.name))
   except OSError:pass
  for value in list(observed['processes'].values()):
   if not value.get('in_observed_job') or set(value.get('uids',()))!={os.getuid()}:continue
   fd=value.get('stdout_fd_target','');path=Path(fd)
   # Ray drivers can have forwarding delay. Keep their own original separately.
   if path.is_absolute() and path.name.endswith('.out') and re.fullmatch(r'worker-[0-9a-f]+-[0-9a-f]+-'+str(value['pid'])+r'\.out',path.name):
    try:
     if path.resolve(strict=True)!=path or not path.is_file() or path.stat().st_uid!=os.getuid():raise ValueError('Own physical Ray stdout required')
     before=path.stat();raw=path.read_bytes();after=path.stat()
     if len(raw)>32*1024*1024:raise ValueError('Ray original log exceeds small-copy budget')
     observed['files']['ray-stdout-pid-'+str(value['pid'])+'.out']={'physical_path':str(path),
       'fd_source':'/proc/'+str(value['pid'])+'/fd/1','base64':base64.b64encode(raw).decode(),
       'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'source_offset':0,
       'before':fingerprint(before),'after':fingerprint(after),
       'complete_original_at_start':len(raw)==before.st_size,'same_inode':before.st_ino==after.st_ino,
       'unchanged_during_read':fingerprint(before)==fingerprint(after)}
    except (OSError,ValueError) as e:
     observed['errors'].append({'path':str(path),'error':type(e).__name__,'detail':str(e)})
  history=run/'resources.ndjson'
  if history.is_file():
   history=safe_under(history);before=history.stat();record={'physical_path':str(history),
     'before':fingerprint(before),'retained_remote':True,'included':False}
   if INPUT['include_resources']:
    n=min(before.st_size,INPUT['resource_limit'])
    with history.open('rb') as f:raw=f.read(n)
    packed=gzip.compress(raw,compresslevel=6,mtime=0)
    record.update(included=True,prefix_bytes=len(raw),complete_prefix_at_start=len(raw)==before.st_size,
      sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=hashlib.sha256(packed).hexdigest(),
      gzip_bytes=len(packed),gzip_base64=base64.b64encode(packed).decode())
   record['after']=fingerprint(history.stat());observed['resource_history']=record
except (OSError,ValueError) as e:
 observed['errors'].append({'error':type(e).__name__,'detail':str(e)})
repo=root/('repo-'+INPUT['source'][:12])
if repo.exists():
 command('source_head',['git','--no-optional-locks','-C',str(repo),'rev-parse','HEAD'])
 command('source_porcelain',['git','--no-optional-locks','-C',str(repo),'status','--porcelain'])
observed['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(observed,sort_keys=True,allow_nan=False))
'''


def save_observation(output, raw):
    """Preserve transport original and independently verify every copied byte."""
    observation = strict_json(raw)
    if observation.get('job_id') != output.name.split('-')[0]:
        # Callers choose a timestamped directory prefixed with the actual job ID.
        raise ValueError('Observation/local directory actual job ID differs')
    (output/'observed-remote-original.json').write_bytes(raw)
    copied = {}
    for name, record in observation['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact-relative output path')
        value = base64.b64decode(record['base64'], validate=True)
        if len(value) != record['bytes'] or hashlib.sha256(value).hexdigest() != record['sha256']:
            raise ValueError('Copied remote artifact bytes differ')
        destination = output/'original-files'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        copied[name] = {k:v for k,v in record.items() if k!='base64'}
    history = observation.get('resource_history')
    if history and history.get('included'):
        packed = base64.b64decode(history['gzip_base64'], validate=True)
        if len(packed)!=history['gzip_bytes'] or hashlib.sha256(packed).hexdigest()!=history['gzip_sha256']:
            raise ValueError('Compressed resource prefix differs')
        plain = gzip.decompress(packed)
        if len(plain)!=history['prefix_bytes'] or hashlib.sha256(plain).hexdigest()!=history['sha256']:
            raise ValueError('Lossless resource prefix differs')
        (output/'resources-prefix-original.ndjson.gz').write_bytes(packed)
    summary = {k:v for k,v in observation.items() if k!='files'}
    if history: summary['resource_history']={k:v for k,v in history.items() if k!='gzip_base64'}
    summary['files'] = copied
    summary['transport_original_sha256'] = hashlib.sha256(raw).hexdigest()
    (output/'observation-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-id',required=True,type=checked_job)
    parser.add_argument('--resources-prefix',action='store_true')
    parser.add_argument('--resource-prefix-limit',type=int,default=256*1024*1024)
    args=parser.parse_args()
    # Transport is the reviewed own-node route, never submit_reviewed.py.
    import transport
    when=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=HERE/'actual-snapshots'/(args.job_id+'-'+when)
    output.mkdir(parents=True,exist_ok=False)
    code=remote_code(args.job_id,args.resources_prefix,args.resource_prefix_limit)
    try:
        result=transport.remote('python3 -c '+shlex.quote(code),timeout=90)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        (output/'transport-failed.stdout').write_bytes(error.stdout or b'')
        (output/'transport-failed.stderr').write_bytes(error.stderr or b'')
        (output/'transport-failure.json').write_text(json.dumps({'error':type(error).__name__,
          'detail':str(error),'job_id':args.job_id,'scope':'READ_FAILURE_NOT_JOB_FAILURE'})+'\n')
        raise
    (output/'transport.stderr').write_bytes(result.stderr)
    summary=save_observation(output,result.stdout)
    print(json.dumps({'snapshot':str(output),'job_id':args.job_id,
      'gpu_rows':len(summary['gpu_inventory_csv_rows']),
      'unfiltered_application_rows':len(summary['unfiltered_application_csv_rows']),
      'copied_originals':len(summary['files']),'errors':summary['errors'],
      'scope':'READ_ONLY_SNAPSHOT_NOT_ACCEPTANCE'},sort_keys=True))


if __name__=='__main__':
    main()
