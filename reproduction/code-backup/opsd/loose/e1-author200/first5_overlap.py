"""Source-bound, once-only CPU overlap inspection. No signature, release or GPU work.

Root deploys this reviewed external file and invokes start after an observed
actual PAUSED5. A detached supervisor waits on the controller; that controller
waits on srun. Every command, original stream and actual waited exit is retained.
"""
import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
FIRST5=ROOT/'tools/first5-12424d01025e'
FIRST5_SHA='12424d01025e9759d495500eef87de37712a29e1dfe989d3772d05cfa3bfe330'
NATIVE_SHA='1f209cab7e025439584a938b4aab7d71358db66ee3f7d63ab348285247ebbd65'
NATIVE_PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
REPO=ROOT/('repo-'+SOURCE[:12])
HERE=Path(__file__).resolve().parent
SCOPE='CPU_FIRST5_INSPECTION_ONLY_NO_ACCEPTANCE_SIGNATURE_RELEASE_OR_GPU'


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'


def unique(raw):
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value,'Duplicate JSON key');value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def hashfile(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def physical(path,directory=False):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path and path.stat().st_uid==0,
         'Physical UID 0 path required: '+str(path))
    need(path.is_dir() if directory else path.is_file(),'Wrong path kind')
    return path


def pin(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'Actual SHA pin required')
    return value


def write_original(path,value):
    with path.open('xb') as handle:
        handle.write(canonical(value));handle.flush();os.fsync(handle.fileno())


def host():
    need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Wrong host/physical UID')


def process(pid,job=None,required=True):
    need(type(pid) is int and pid>0,'Actual process PID required')
    path=Path('/proc')/str(pid)
    try:
        status=(path/'status').read_text();cgroup=(path/'cgroup').read_text()
        start=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
        argv=(path/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        if required:raise
        return {'pid':pid,'present':False}
    need(set(map(int,re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()))=={0},'Four process UIDs differ')
    state=re.search(r'^State:\s+(\S+)',status,re.M).group(1)
    if required:need(state in ('R','S','D','I'),'Process not live')
    if job is not None:
        need(re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',cgroup,re.M) is not None,'Foreign process cgroup')
    return {'pid':pid,'present':True,'live':state in ('R','S','D','I'),'uid':0,'start_ticks':start,'state':state,
            'cgroup_original':cgroup,'status_original':status,
            'argv':[item.decode(errors='replace') for item in argv]}


def verify_tools(overlap_pin):
    host();path=physical(HERE/'manifest.json');need(hashfile(path)==pin(overlap_pin),'Overlap tool manifest changed')
    manifest=unique(path.read_bytes())
    need(manifest['source_commit']==SOURCE and manifest['first5_manifest_sha256']==FIRST5_SHA
         and manifest['native_manifest_sha256']==NATIVE_SHA and manifest['scope']==SCOPE
         and set(manifest['tools'])=={'first5_overlap.py'}
         and manifest['contract']=={'global_mini_batch':32,'adam_steps_per_rank_per_outer':16,
              'saved_adam_steps':80,'horizon':200,'warmup_steps':0,'learning_rate':5e-6},'Wrong reviewed overlap contract')
    need(hashfile(physical(HERE/'first5_overlap.py'))==manifest['tools']['first5_overlap.py'],'Overlap tool bytes changed')
    tools=physical(FIRST5/'manifest.json');need(hashfile(tools)==FIRST5_SHA,'Existing first5 manifest changed')
    first=unique(tools.read_bytes());need(first['source_commit']==SOURCE
         and first['native_payload_tools_manifest_sha256']==NATIVE_SHA,'Existing first5 native binding changed')
    for name,expected in first['tools'].items():need(hashfile(physical(FIRST5/name))==expected,'Existing first5 tool changed')
    return first


def run_root(job):
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) is not None,'Observed actual job ID required')
    return ROOT/'runs'/('formal-'+job)


def pause_context(config):
    verify_tools(config['overlap_tools_manifest_sha256'])
    script=physical(FIRST5/'inspect_first5_cpu.py')
    spec=importlib.util.spec_from_file_location('strict_author200_cpu_inspector',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.JOB=config['job_id'];module.RUN=run_root(config['job_id']);module.TOOLS_MANIFEST_SHA=FIRST5_SHA
    manifest,manifest_path,manifest_pin,before=module.context()
    need(manifest_pin==config['checkpoint_manifest_sha256']
         and manifest['pause_nonce']==config['pause_nonce'],'Actual PAUSED checkpoint/nonce differs')
    driver=process(manifest['driver_pid'],config['job_id'])
    need(driver['start_ticks']==config['driver_start_ticks'],'Actual paused driver start changed')
    control=subprocess.check_output(['scontrol','show','job',config['job_id'],'-o'],text=True,timeout=20)
    fields=dict(token.split('=',1) for token in control.split() if '=' in token)
    need(fields.get('JobId')==config['job_id'] and fields.get('JobState')=='RUNNING'
         and fields.get('NodeList')=='h100-10' and fields.get('TresPerNode')=='gres:gpu:h100:8'
         and fields.get('OverSubscribe')=='NO','Actual parent allocation changed')
    return {'manifest_sha256':manifest_pin,'manifest_path':str(manifest_path),
            'driver':driver,'inspector_context':before,'scontrol_original':control}


def command(job,output):
    return ['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1',
            '--cpus-per-task=1','--gres=none','--nodelist=h100-10','env',
            'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(REPO),
            str(NATIVE_PYTHON),str(FIRST5/'inspect_first5_cpu.py'),'--job-id',job,
            '--tools-manifest-sha256',FIRST5_SHA,'--output',str(output)]


def load_launch(path,expected):
    host();path=physical(path);need(hashfile(path)==pin(expected),'Actual launch bytes changed')
    config=unique(path.read_bytes());verify_tools(config['overlap_tools_manifest_sha256'])
    need(config['source_commit']==SOURCE and config['scope']==SCOPE,'Wrong launch scope/source')
    run=physical(run_root(config['job_id']),True);need(path==run/'evidence/first5-cpu-overlap-launch.json','Foreign launch path')
    output=physical(config['output'],True);output.relative_to(run/'evidence')
    need(output.parent==run/'evidence' and config['command']==command(config['job_id'],output),'Wrong output or CPU-only command')
    pin(config['checkpoint_manifest_sha256']);pin(config['pause_nonce'])
    need(type(config['driver_start_ticks']) is str and re.fullmatch('[1-9][0-9]*',config['driver_start_ticks']) is not None,'Actual driver start required')
    return config,output


def child_command(role,launch,launch_pin):
    return [sys.executable,str(HERE/'first5_overlap.py'),role,'--launch',str(launch),'--launch-sha256',launch_pin]


def waited(code):
    return code if code>=0 else 128-code


def controller(config,output):
    before=pause_context(config)
    write_original(output/'controller-start-original.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'process':process(os.getpid()),'command':config['command'],'scope':SCOPE,'before':before})
    with (output/'srun-stdout-original.txt').open('xb') as stdout,(output/'srun-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(config['command'],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        try:
            write_original(output/'srun-start-original.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'process':process(child.pid,required=False),'controller_pid':os.getpid(),'command':config['command']})
        finally:returncode=child.wait()
    write_original(output/'srun-exit-original.json',{'returncode':returncode,'waited_by_controller_pid':os.getpid(),
                   'srun_pid':child.pid,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
    return waited(returncode)


def supervise(config,output,launch,launch_pin):
    pause_context(config)
    write_original(output/'supervisor-start-original.json',{'process':process(os.getpid()),'scope':SCOPE,
                   'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    argv=child_command('controller',launch,launch_pin)
    with (output/'controller-stdout-original.txt').open('xb') as stdout,(output/'controller-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        try:
            write_original(output/'controller-launch-original.json',{'process':process(child.pid,required=False),'command':argv,
                           'supervisor_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        finally:returncode=child.wait()
    write_original(output/'controller-exit-original.json',{'returncode':returncode,'controller_pid':child.pid,
                   'waited_by_supervisor_pid':os.getpid(),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
    return waited(returncode)


def start(args):
    run=physical(run_root(args.job_id),True);evidence=physical(run/'evidence',True)
    output=Path(args.output)
    need(output.is_absolute() and output.parent==evidence and not output.exists() and not output.is_symlink(),
         'Fresh physical output directory directly under own evidence required')
    config={'version':1,'source_commit':SOURCE,'scope':SCOPE,'job_id':args.job_id,
            'overlap_tools_manifest_sha256':pin(args.overlap_tools_manifest_sha256),
            'checkpoint_manifest_sha256':pin(args.checkpoint_manifest_sha256),
            'pause_nonce':pin(args.pause_nonce),'driver_start_ticks':args.driver_start_ticks,
            'output':str(output),'command':command(args.job_id,output)}
    before=pause_context(config)
    launch=evidence/'first5-cpu-overlap-launch.json'
    # The exclusive launch marker also covers an interrupted/unknown transport.
    # It is never erased to retry an inspection. Originals survive every failure.
    need(not launch.exists() and not launch.is_symlink(),'Existing launch; inspect originals, never automatically repeat')
    output.mkdir(mode=0o700)
    write_original(output/'root-start-context-original.json',before)
    write_original(launch,config);launch_pin=hashfile(launch)
    argv=child_command('supervise',launch,launch_pin)
    with (output/'supervisor-stdout-original.txt').open('xb') as stdout,(output/'supervisor-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        write_original(output/'durable-launch-original.json',{'process':process(child.pid,required=False),'command':argv,
                       'launcher_process':process(os.getpid()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'started':True,'launch':str(launch),'launch_sha256':launch_pin,
          'output':str(output),'supervisor_pid':child.pid,'scope':SCOPE},sort_keys=True))


def completion(config,output):
    parent=unique(physical(output/'controller-exit-original.json').read_bytes())
    child=unique(physical(output/'srun-exit-original.json').read_bytes())
    need(type(parent['returncode']) is int and type(child['returncode']) is int,'Noninteger actual waited exit')
    if parent['returncode']!=0 or child['returncode']!=0:return False
    completed=unique(physical(output/'inspection-completed-review-required.json').read_bytes())
    need(completed['scope']=='CPU_STORED_STATE_AND_HF_INSPECTION_ONLY_NOT_SIGNATURE_OR_RELEASE'
         and completed['source_commit']==SOURCE and completed['job_id']==config['job_id']
         and completed['manifest_sha256']==config['checkpoint_manifest_sha256']
         and completed['cuda_is_initialized'] is False,'Wrong original completion identity')
    step=unique(physical(output/'cpu-step-original.json').read_bytes())
    need(step['job_id']==config['job_id'] and type(step['uid']) is int and step['uid']==0
         and step['manifest_sha256']==config['checkpoint_manifest_sha256']
         and step['cuda_visible_devices']=='' and step['cuda_is_initialized'] is False
         and re.search(r'(?:^|/)job_'+re.escape(config['job_id'])+r'(?:/|$)',step['cgroup'],re.M) is not None,
         'Actual CPU step environment/identity differs')
    for name in ('native','hf'):
        code=unique(physical(output/(name+'-exit-original.json')).read_bytes())
        report=unique(physical(output/(name+'.json')).read_bytes())
        need(type(code['returncode']) is int and code['returncode']==0
             and report['passed'] is True and report['source_commit']==SOURCE
             and report['job_id']==config['job_id']
             and report['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256'],
             'Actual '+name+' inspection result differs')
    return True


def observe(config,output,launch,launch_pin):
    originals=[]
    for path in sorted([launch]+[p for p in output.iterdir() if p.is_file()]):
        physical(path);size=path.stat().st_size
        record={'path':str(path),'bytes':size,'sha256':hashfile(path)}
        if size<=8*1024**2:record['base64']=base64.b64encode(path.read_bytes()).decode()
        else:record['retained_remote_only']=True
        originals.append(record)
    result={'scope':SCOPE,'job_id':config['job_id'],'source_commit':SOURCE,'output':str(output),
            'launch_sha256':launch_pin,'checkpoint_manifest_sha256':config['checkpoint_manifest_sha256'],
            'records':originals,'state':'INSPECTION_RUNNING_OR_OUTCOME_PENDING'}
    for name in ('durable-launch-original.json','supervisor-start-original.json','controller-launch-original.json','controller-start-original.json','srun-start-original.json'):
        path=output/name
        if path.exists():
            original=unique(physical(path).read_bytes());record=original['process']
            live=process(record['pid'],required=False)
            if live['present']:need(record['present'] and live['start_ticks']==record['start_ticks'],'Observed controller PID reused/unwitnessed')
            result[name]={'recorded':record,'live_now':live}
    controller_exit=output/'controller-exit-original.json';srun_exit=output/'srun-exit-original.json'
    if controller_exit.exists():
        result['controller_exit']=unique(physical(controller_exit).read_bytes())
        if srun_exit.exists():result['srun_exit']=unique(physical(srun_exit).read_bytes())
        result['state']='CPU_INSPECTION_FAILED_OR_INCOMPLETE_ORIGINALS_RETAINED'
        try:
            if completion(config,output):result['state']='CPU_INSPECTION_COMPLETED_INDEPENDENT_REVIEW_REQUIRED'
        except (FileNotFoundError,ValueError,KeyError) as error:
            result['completion_check_error']=str(error)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='role',required=True)
    start_parser=sub.add_parser('start')
    for field in ('job-id','checkpoint-manifest-sha256','pause-nonce','driver-start-ticks','overlap-tools-manifest-sha256','output'):
        start_parser.add_argument('--'+field,required=True)
    for role in ('controller','supervise','observe'):
        child=sub.add_parser(role);child.add_argument('--launch',required=True,type=Path);child.add_argument('--launch-sha256',required=True)
    args=parser.parse_args()
    if args.role=='start':start(args);return
    config,output=load_launch(args.launch,args.launch_sha256)
    if args.role=='controller':sys.exit(controller(config,output))
    if args.role=='supervise':sys.exit(supervise(config,output,args.launch,args.launch_sha256))
    observe(config,output,args.launch,args.launch_sha256)


if __name__=='__main__':main()
