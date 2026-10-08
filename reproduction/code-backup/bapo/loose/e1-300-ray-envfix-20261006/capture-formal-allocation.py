#!/usr/bin/env python3
"""Read-only allocated formal input capture; no permission, GPU, or process signals."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876'
REQUIRED=('recovery-manifest.json','recovery-input-preparation.json','source-commit.txt',
 'slurm-allocation.txt','runtime-import-origins.log','cpu-gate.json','cpu-gate-source.txt',
 'tools-manifest.json','launch-approval.json','gpus-before.csv','processes-before.csv',
 'WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION')
MAX_RAW_BYTES=8*1024*1024
HARDWARE_RECEIPT_SHA='e8625dabce8c9cf5a97498cb0989b399937fd9719acfa186357b1d29cebe1e2c' # Actual final independent3287 original; never interim hold receipt.
CPU_SHA='e6cbb1cb66cdea1d5ad165dbd45a375868bb6711cdcd839e25cef6c32575318c'

def require(ok,msg):
 if not ok:raise ValueError(msg)

def job_id(text):
 require(isinstance(text,str) and re.fullmatch(r'[1-9][0-9]*',text)is not None,'Invalid positive actual job id')
 return int(text)

def validate_slurm(raw,job):
 for name,value in [('JobId',str(job)),('JobName','bapo-instruct300-formal300'),
                    ('JobState','RUNNING'),('Restarts','0'),('NodeList','h100-5')]:
  fields=re.findall(r'(?:^|\s)'+name+r'=([^\s]+)',raw)
  require(fields==[value],f'Slurm field differs or duplicated: {name}')

def read_raw(path,ordinary=True):
 p=Path(path)
 require(p.is_file() and (not ordinary or not p.is_symlink()),f'Missing/nonordinary raw: {p}')
 require(p.stat().st_size<=MAX_RAW_BYTES,f'Raw file too large to capture: {p}')
 raw=p.read_bytes();require(len(raw)<=MAX_RAW_BYTES,'Raw grew beyond bound')
 d={'path':str(p),'resolved_path':str(p.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
    'bytes':len(raw),'base64':base64.b64encode(raw).decode('ascii')}
 try:d['text']=raw.decode('utf-8')
 except UnicodeDecodeError:pass
 return d

def verify_dependencies(files):
 manifest=json.loads(files['tools-manifest.json']['text']);launch=json.loads(files['launch-approval.json']['text']);cpu=json.loads(files['cpu-gate.json']['text'])
 require(launch['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'and manifest['source_commit']==SOURCE and launch['source_commit']==SOURCE and launch['root']==str(ROOT),'Launch/tool identity differs')
 require(type(launch['tools_file_count'])is int and launch['tools_file_count']==36 and len(manifest['files'])==36,'Expected exact independently reviewed 36 tools')
 require(launch['tools_manifest_sha256']==files['tools-manifest.json']['sha256'],'Allocated tool manifest differs from approval')
 actual={p.name:read_raw(p)['sha256']for p in (ROOT/'tools').iterdir()if p.is_file()}
 require(actual==manifest['files'],'Frozen actual tool files differ')
 require(files['cpu-gate.json']['sha256']==CPU_SHA==launch['cpu_gate_record_sha256'],'Actual CPU gate differs')
 require(cpu['source_commit']==SOURCE and cpu['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'and cpu['sys_prefix']==str(ROOT/'env'),'Actual CPU identity differs')
 for key,value in [('passed',395),('failed',0),('skipped',0),('exit_code',0)]:
  require(type(cpu[key])is int and cpu[key]==value,'Actual CPU results differ')
 require(cpu['test_launcher_sha256']==actual['cpu-tests.py'],'CPU launcher differs from tested source')
 for label in ('before','after'):
  require(cpu['import_origins'][label]['source_commit']==SOURCE and cpu['import_origins'][label]['repo']==str(ROOT/'repo'),'Actual CPU import origin differs')

def hardware_binding():
 receiptpath=ROOT/'logs/hardware-acceptance.json';receipt=read_raw(receiptpath)
 require(receipt['sha256']==HARDWARE_RECEIPT_SHA,'Hardware receipt differs from actual signed acceptance')
 r=json.loads(receipt['text'])
 require(r['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and r['source_commit']==SOURCE and r['root']==str(ROOT)and type(r['job_id'])is int and r['job_id']==3287 and r['run_root']==str(ROOT/'runs/bapo-instruct300-hardware-3287'),'Actual hardware receipt identity differs')
 require(r['actual_gpu_accepted']is True and r['production_training_permission_signed']is False,'Hardware receipt scope differs')
 bound=r['independent_review'];path=Path(bound['path'])
 require(path.is_absolute()and path.parent==ROOT/'logs'and path.resolve()==path,'Hardware report must be ordinary direct newROOT/logs file')
 report=read_raw(path)
 require(report['sha256']==bound['sha256']and type(bound['bytes'])is int and report['bytes']==bound['bytes'],'Actual hardware report hash/bytes differ')
 return {'hardware-acceptance.json':receipt,'hardware-independent-review':report}

def capture(job,live_slurm):
 require(type(job)is int and job>0,'Job id must be a positive integer')
 validate_slurm(live_slurm,job)
 run=ROOT/'runs'/f'bapo-instruct300-formal300-{job}'
 require(run.is_dir() and not run.is_symlink() and run.resolve()==run,'Unexpected run directory')
 files={name:read_raw(run/name)for name in REQUIRED}
 verify_dependencies(files)
 hardware=hardware_binding()
 for name in ('source-commit.txt','cpu-gate-source.txt'):
  require(files[name].get('text','').strip()==SOURCE,f'Source mismatch: {name}')
 require(not (run/'recovery-permission.json').exists(),'Permission already published; not pre-GPU wait capture')
 manifest=json.loads(files['recovery-manifest.json']['text'])
 require(type(manifest['target_job_id'])is int and manifest['target_job_id']==job and
         manifest['target_source_commit']==SOURCE and manifest['target_run_root']==str(run),
         'Allocated manifest identity differs')
 require(manifest['step']==25 and manifest['next_step']==26 and manifest['world_size']==8,
         'Manifest recovery boundary differs')
 allocation=files['slurm-allocation.txt'].get('text','')
 require(re.findall(r'^SLURM_JOB_ID=(\d+)$',allocation,re.M)==[str(job)],'Allocation environment job differs')
 # All small original input proofs are captured; no state tensor/model weight loading or rehashing.
 referenced={}
 for key in ('resolved_config','historical_first5_receipt','train_prefix'):
  binding=manifest[key];x=read_raw(Path(binding['path']),ordinary=False)
  require(x['sha256']==binding['sha256'] and x['bytes']==binding['bytes'],f'Input binding differs: {key}')
  referenced[key]=x
 small_shared=[x for x in manifest['shared_inputs']if Path(x['path']).name in
               ('manifest.json','VERIFIED-MODEL-MANIFEST.json','config.json')]
 require(small_shared,'Missing shared input manifests')
 for index,binding in enumerate(small_shared):
  x=read_raw(binding['path'],ordinary=False)
  require(x['sha256']==binding['sha256'] and x['bytes']==binding['bytes'],'Shared input binding differs')
  referenced[f'shared_input_{index}']=x
 optional={}
 for name in ('resource-watch.jsonl','own-process-cleanup.json','recovery-permission-wait.log'):
  p=run/name
  if p.is_file():optional[name]=read_raw(p)
 # Repeat immutable allocated input reads and waiting-state check to reject capture races.
 for name,first in files.items():require(read_raw(run/name)['sha256']==first['sha256'],'Allocated raw changed during capture')
 verify_dependencies(files)
 require(hardware_binding()==hardware,'Hardware acceptance changed during capture')
 require(not (run/'recovery-permission.json').exists(),'Permission appeared during capture')
 return {'status':'READ_ONLY_ACTUAL_ALLOCATED_INPUTS_NOT_PERMISSION_OR_GPU_ACCEPTANCE',
  'job_id':job,'source_commit':SOURCE,'root':str(ROOT),'run_root':str(run),
  'captured_at_unix':time.time(),'files':files,'referenced_inputs':referenced,'optional_raw':optional,
  'hardware_acceptance':hardware,
  'slurm_live':{'text':live_slurm,'sha256':hashlib.sha256(live_slurm.encode()).hexdigest()},
  'large_state_or_weights_revalidated':False,'actual_gpu':False,'permission_signed':False,
  'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
 global HARDWARE_RECEIPT_SHA
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-id',required=True);p.add_argument('--hardware-receipt-sha',required=True)
 a=p.parse_args();job=job_id(a.job_id)
 require(a.hardware_receipt_sha==HARDWARE_RECEIPT_SHA,'Exact final independent3287 receipt SHA required')
 live=subprocess.check_output(['scontrol','show','job','-dd',str(job)],text=True,timeout=15)
 print(json.dumps(capture(job,live),sort_keys=True))

if __name__=='__main__':main()
