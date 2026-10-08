"""Once-only durable unsigned receipt preparation after actual external PASS records.

No review records, signature, training release, SSH or GPU operation are created.
The existing source-bound receipt_payload.py performs the full actual validation.
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
REPO=ROOT/('repo-'+SOURCE[:12])
FIRST5=ROOT/'tools/first5-12424d01025e'
FIRST5_SHA='12424d01025e9759d495500eef87de37712a29e1dfe989d3772d05cfa3bfe330'
OVERLAP=ROOT/'tools/first5-overlap-1cfb93790632'
OVERLAP_SHA='1cfb9379063254ae242c7fdcfa4a57fb9927cc9d5fe00b97a49455aa4f652fea'
OVERLAP_SCRIPT_SHA='7ec32fbc071bcdb10de8824efa26d845588eb0431e14eaed834f4f5aaa4f98b2'
PRODUCER_SHA='79f29673f4dee991f295e9915ff163ada5073776638a330529571dabf007a64d'
PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
HERE=Path(__file__).resolve().parent
SCOPE='UNSIGNED_FIRST5_PAYLOAD_PREPARATION_ONLY_NO_REVIEW_SIGNATURE_RELEASE_OR_GPU'
_BASE=None


def need(ok,message):
    if not ok:raise ValueError(message)


def hashfile(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def base():
    global _BASE
    if _BASE is None:
        path=OVERLAP/'first5_overlap.py'
        need(path.resolve(strict=True)==path and path.stat().st_uid==0
             and hashfile(path)==OVERLAP_SCRIPT_SHA,'Reviewed overlap helper changed')
        spec=importlib.util.spec_from_file_location('pinned_first5_overlap_common',path)
        _BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_BASE)
    return _BASE


def verify_tools(expected):
    b=base();b.host();path=b.physical(HERE/'manifest.json')
    need(hashfile(path)==b.pin(expected),'Unsigned runner manifest changed')
    manifest=b.unique(path.read_bytes())
    need(manifest['source_commit']==SOURCE and manifest['scope']==SCOPE
         and manifest['first5_manifest_sha256']==FIRST5_SHA
         and manifest['overlap_manifest_sha256']==OVERLAP_SHA
         and manifest['overlap_script_sha256']==OVERLAP_SCRIPT_SHA
         and manifest['producer_sha256']==PRODUCER_SHA
         and set(manifest['tools'])=={'unsigned_receipt_overlap.py'},'Wrong unsigned runner binding')
    need(hashfile(b.physical(HERE/'unsigned_receipt_overlap.py'))==manifest['tools']['unsigned_receipt_overlap.py'],'Unsigned runner bytes changed')
    b.verify_tools(OVERLAP_SHA)
    need(hashfile(b.physical(FIRST5/'receipt_payload.py'))==PRODUCER_SHA,'Fixed producer changed')
    return b


def producer():
    path=FIRST5/'receipt_payload.py'
    need(hashfile(path)==PRODUCER_SHA,'Fixed receipt producer changed')
    spec=importlib.util.spec_from_file_location('pinned_unsigned_receipt_api',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def context(config):
    b=verify_tools(config['runner_manifest_sha256'])
    need(config['source_commit']==SOURCE and config['first5_tools_manifest_sha256']==FIRST5_SHA
         and config['scope']==SCOPE,'Wrong source/first5/scope')
    before=b.pause_context(dict(config,overlap_tools_manifest_sha256=OVERLAP_SHA))
    run=b.run_root(config['job_id']);mp=b.physical(run/'full_state/global_step_5/manifest.json')
    need(hashfile(mp)==config['checkpoint_manifest_sha256'],'Actual checkpoint changed')
    module=producer();manifest=module.read(mp);module.identity(manifest,run)
    specpath=module.file(config['evidence_spec'],run);evidence=module.read(specpath)
    need(set(evidence)=={'resources','retrieval','hf_reload','rank_audit','service'},'Actual five-role spec required')
    for rec in evidence.values():module.file(rec,run)
    need(Path(evidence['resources']['path'])!=run/'resources.ndjson','Immutable resource prefix required')
    for name,role in [('root_review','ROOT_AUDITOR'),('independent_review','INDEPENDENT_REVIEWER')]:
        rec=config[name]
        module.review_record(Path(rec['path']),rec['sha256'],run,manifest,
                             config['checkpoint_manifest_sha256'],role,evidence)
    for role in ('hf_reload','rank_audit'):
        report=module.read(Path(evidence[role]['path']))
        need(report['passed'] is True and report['source_commit']==SOURCE
             and report['job_id']==config['job_id']
             and report['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256']
             and report['cuda_is_initialized'] is False,'Actual '+role+' report is not ready/passed')
    need(not (run/'acceptance-5-receipt.json').exists()
         and not (run/'acceptance-5-receipt.json').is_symlink(),'Training already signed/released')
    return before


def producer_command(config,output):
    run=base().run_root(config['job_id'])
    argv=[str(PYTHON),str(FIRST5/'receipt_payload.py'),'--run-root',str(run),'--repo',str(REPO),
          '--manifest',str(run/'full_state/global_step_5/manifest.json'),
          '--manifest-sha256',config['checkpoint_manifest_sha256'],
          '--tools-manifest',str(FIRST5/'manifest.json'),'--tools-manifest-sha256',FIRST5_SHA]
    for name,arg in [('evidence_spec','evidence-spec'),('root_review','root-review'),('independent_review','independent-review')]:
        argv.extend(['--'+arg,config[name]['path'],'--'+arg+'-sha256',config[name]['sha256']])
    return argv+['--output',str(output/'unsigned-step5-payload.json')]


def child_command(role,launch,launch_pin):
    return [sys.executable,str(HERE/'unsigned_receipt_overlap.py'),role,'--launch',str(launch),'--launch-sha256',launch_pin]


def srun_command(config,launch,launch_pin):
    return ['srun','--jobid='+config['job_id'],'--overlap','--nodes=1','--ntasks=1',
            '--cpus-per-task=1','--gres=none','--nodelist=h100-10','env',
            'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(REPO),
            str(PYTHON),str(HERE/'unsigned_receipt_overlap.py'),'produce',
            '--launch',str(launch),'--launch-sha256',launch_pin]


def load_launch(path,expected):
    b=base();path=b.physical(path);need(hashfile(path)==b.pin(expected),'Actual launch bytes changed')
    config=b.unique(path.read_bytes());verify_tools(config['runner_manifest_sha256'])
    need(config['source_commit']==SOURCE and config['scope']==SCOPE,'Wrong launch source/scope')
    run=b.physical(b.run_root(config['job_id']),True)
    need(path==run/'evidence/first5-unsigned-receipt-overlap-launch.json','Foreign launch record')
    output=b.physical(config['output'],True);need(output.parent==run/'evidence','Foreign output directory')
    for key in ('checkpoint_manifest_sha256','pause_nonce'):b.pin(config[key])
    need(type(config['driver_start_ticks']) is str and re.fullmatch('[1-9][0-9]*',config['driver_start_ticks']) is not None,'Actual driver start required')
    need(config['producer_command']==producer_command(config,output),'Fixed producer command differs')
    return config,output


def produce(config,output):
    b=base();before=context(config)
    need(os.environ.get('SLURM_JOB_ID')==config['job_id']
         and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1'
         and os.environ.get('MKL_NUM_THREADS')=='1' and os.environ.get('OPENBLAS_NUM_THREADS')=='1','Wrong CPU-step environment')
    identity=b.process(os.getpid(),config['job_id'])
    import torch
    need(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'Native CPU-only Torch2.4 required')
    b.write_original(output/'cpu-step-original.json',{'process':identity,'job_id':config['job_id'],
          'source_commit':SOURCE,'checkpoint_manifest_sha256':config['checkpoint_manifest_sha256'],
          'cuda_visible_devices':'','cuda_is_initialized':False,'omp_num_threads':'1',
          'torch_version':torch.__version__,'before':before,'scope':SCOPE})
    b.write_original(output/'producer-command-original.json',{'command':config['producer_command'],
                     'scope':SCOPE,'cpu_step_pid':os.getpid()})
    with (output/'producer-stdout-original.txt').open('xb') as stdout,(output/'producer-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(config['producer_command'],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        try:b.write_original(output/'producer-start-original.json',{'process':b.process(child.pid,config['job_id'],required=False),'cpu_step_pid':os.getpid()})
        finally:returncode=child.wait()
    b.write_original(output/'producer-exit-original.json',{'returncode':returncode,'producer_pid':child.pid,
                     'waited_by_cpu_step_pid':os.getpid(),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
    if returncode!=0:return b.waited(returncode)
    after=context(config);need(not torch.cuda.is_initialized(),'CUDA initialized during unsigned preparation')
    payload=b.physical(output/'unsigned-step5-payload.json');value=b.unique(payload.read_bytes())
    need(value['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
         and value['source_commit']==SOURCE and value['job_id']==config['job_id']
         and value['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256']
         and value['pause_nonce']==config['pause_nonce'],'Unsigned payload actual identity differs')
    b.write_original(output/'unsigned-preparation-completed-review-required.json',{'scope':SCOPE,
          'job_id':config['job_id'],'source_commit':SOURCE,
          'checkpoint_manifest_sha256':config['checkpoint_manifest_sha256'],
          'payload':{'path':str(payload),'sha256':hashfile(payload)},'cuda_is_initialized':False,'after':after})
    print(json.dumps({'scope':SCOPE,'unsigned_payload':str(payload),'payload_sha256':hashfile(payload)},sort_keys=True))
    return 0


def controller(config,output,launch,launch_pin):
    b=base();before=context(config);argv=srun_command(config,launch,launch_pin)
    b.write_original(output/'controller-start-original.json',{'process':b.process(os.getpid()),'command':argv,'before':before,'scope':SCOPE})
    with (output/'srun-stdout-original.txt').open('xb') as stdout,(output/'srun-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        try:b.write_original(output/'srun-start-original.json',{'process':b.process(child.pid,required=False),'controller_pid':os.getpid(),'command':argv})
        finally:returncode=child.wait()
    b.write_original(output/'srun-exit-original.json',{'returncode':returncode,'srun_pid':child.pid,
                     'waited_by_controller_pid':os.getpid(),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
    return b.waited(returncode)


def supervise(config,output,launch,launch_pin):
    b=base();context(config);argv=child_command('controller',launch,launch_pin)
    b.write_original(output/'supervisor-start-original.json',{'process':b.process(os.getpid()),'scope':SCOPE})
    with (output/'controller-stdout-original.txt').open('xb') as stdout,(output/'controller-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        try:b.write_original(output/'controller-launch-original.json',{'process':b.process(child.pid,required=False),'supervisor_pid':os.getpid(),'command':argv})
        finally:returncode=child.wait()
    b.write_original(output/'controller-exit-original.json',{'returncode':returncode,'controller_pid':child.pid,
                     'waited_by_supervisor_pid':os.getpid(),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE})
    return b.waited(returncode)


def start(args):
    b=verify_tools(args.runner_manifest_sha256);need(args.source_commit==SOURCE,'Wrong explicitly supplied source')
    run=b.physical(b.run_root(args.job_id),True);evidence=b.physical(run/'evidence',True);output=Path(args.output)
    need(output.is_absolute() and output.parent==evidence and not output.exists() and not output.is_symlink(),'Fresh physical output directly under own evidence required')
    config={'version':1,'source_commit':SOURCE,'scope':SCOPE,'job_id':args.job_id,
            'runner_manifest_sha256':b.pin(args.runner_manifest_sha256),
            'first5_tools_manifest_sha256':args.first5_tools_manifest_sha256,
            'checkpoint_manifest_sha256':b.pin(args.checkpoint_manifest_sha256),
            'pause_nonce':b.pin(args.pause_nonce),'driver_start_ticks':args.driver_start_ticks,'output':str(output)}
    for name in ('evidence_spec','root_review','independent_review'):
        config[name]={'path':str(getattr(args,name)),'sha256':b.pin(getattr(args,name+'_sha256'))}
    config['producer_command']=producer_command(config,output)
    before=context(config);launch=evidence/'first5-unsigned-receipt-overlap-launch.json'
    need(not launch.exists() and not launch.is_symlink(),'Existing unsigned launch; preserve unknown outcome and never repeat automatically')
    output.mkdir(mode=0o700);b.write_original(output/'root-start-context-original.json',before)
    b.write_original(launch,config);launch_pin=hashfile(launch);argv=child_command('supervise',launch,launch_pin)
    with (output/'supervisor-stdout-original.txt').open('xb') as stdout,(output/'supervisor-stderr-original.txt').open('xb') as stderr:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        b.write_original(output/'durable-launch-original.json',{'process':b.process(child.pid,required=False),'launcher_process':b.process(os.getpid()),'command':argv})
    print(json.dumps({'started':True,'launch':str(launch),'launch_sha256':launch_pin,'output':str(output),
                      'supervisor_pid':child.pid,'scope':SCOPE},sort_keys=True))


def completion(config,output):
    b=base()
    for name in ('controller','srun','producer'):
        record=b.unique(b.physical(output/(name+'-exit-original.json')).read_bytes())
        need(type(record['returncode']) is int,'Noninteger actual waited exit')
        if record['returncode']!=0:return False
    completed=b.unique(b.physical(output/'unsigned-preparation-completed-review-required.json').read_bytes())
    need(completed['scope']==SCOPE and completed['source_commit']==SOURCE and completed['job_id']==config['job_id']
         and completed['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256']
         and completed['cuda_is_initialized'] is False,'Wrong original unsigned completion identity')
    cpu=b.unique(b.physical(output/'cpu-step-original.json').read_bytes())
    identity=cpu['process']
    need(cpu['source_commit']==SOURCE and cpu['job_id']==config['job_id']
         and cpu['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256']
         and cpu['cuda_visible_devices']=='' and cpu['cuda_is_initialized'] is False
         and cpu['omp_num_threads']=='1' and type(identity['uid']) is int and identity['uid']==0
         and re.search(r'(?:^|/)job_'+re.escape(config['job_id'])+r'(?:/|$)',identity['cgroup_original'],re.M) is not None,
         'Actual unsigned CPU step identity differs')
    payload=b.physical(completed['payload']['path']);need(payload==output/'unsigned-step5-payload.json'
         and hashfile(payload)==completed['payload']['sha256'],'Actual unsigned payload changed')
    value=b.unique(payload.read_bytes());need(value['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
         and value['source_commit']==SOURCE and value['job_id']==config['job_id']
         and value['pause_nonce']==config['pause_nonce']
         and value['checkpoint_manifest_sha256']==config['checkpoint_manifest_sha256'],'Actual unsigned payload identity differs')
    return True


def observe(config,output,launch,launch_pin):
    b=base();records=[]
    for path in sorted([launch]+[p for p in output.iterdir() if p.is_file()]):
        b.physical(path);size=path.stat().st_size;record={'path':str(path)}
        if size<=8*1024**2:
            with path.open('rb') as handle:raw=handle.read(size)
            need(len(raw)==size,'Observed original prefix truncated')
            record.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode(),scope='ORIGINAL_OBSERVED_PREFIX')
        else:record.update(bytes=size,retained_remote_only=True)
        records.append(record)
    result={'scope':SCOPE,'job_id':config['job_id'],'source_commit':SOURCE,'output':str(output),
            'launch_sha256':launch_pin,'records':records,'state':'UNSIGNED_PREPARATION_RUNNING_OR_OUTCOME_PENDING'}
    for name in ('durable-launch-original.json','supervisor-start-original.json','controller-launch-original.json',
                 'controller-start-original.json','srun-start-original.json','cpu-step-original.json','producer-start-original.json'):
        path=output/name
        if path.exists():
            original=b.unique(b.physical(path).read_bytes());identity=original['process']
            live=b.process(identity['pid'],required=False)
            if live['present']:need(identity['present'] and live['start_ticks']==identity['start_ticks'],'Observed unsigned PID reused/unwitnessed')
            result[name]={'recorded':identity,'live_now':live}
    if (output/'controller-exit-original.json').exists():
        result['state']='UNSIGNED_PREPARATION_FAILED_OR_INCOMPLETE_ORIGINALS_RETAINED'
        try:
            if completion(config,output):result['state']='UNSIGNED_PAYLOAD_READY_EXTERNAL_ROOT_SIGNATURE_REQUIRED'
        except (FileNotFoundError,ValueError,KeyError) as error:result['completion_check_error']=str(error)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='role',required=True)
    start_parser=sub.add_parser('start')
    for name in ('job-id','source-commit','checkpoint-manifest-sha256','pause-nonce','driver-start-ticks',
                 'runner-manifest-sha256','first5-tools-manifest-sha256','evidence-spec','evidence-spec-sha256',
                 'root-review','root-review-sha256','independent-review','independent-review-sha256','output'):
        start_parser.add_argument('--'+name,required=True)
    for role in ('controller','supervise','produce','observe'):
        child=sub.add_parser(role);child.add_argument('--launch',required=True,type=Path);child.add_argument('--launch-sha256',required=True)
    args=parser.parse_args()
    if args.role=='start':start(args);return
    config,output=load_launch(args.launch,args.launch_sha256)
    if args.role=='controller':sys.exit(controller(config,output,args.launch,args.launch_sha256))
    if args.role=='supervise':sys.exit(supervise(config,output,args.launch,args.launch_sha256))
    if args.role=='produce':sys.exit(produce(config,output))
    observe(config,output,args.launch,args.launch_sha256)


if __name__=='__main__':main()
