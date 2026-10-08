#!/usr/bin/env python3
"""Read-only allocated hardware input capture; no permission, GPU, or process signals."""
import argparse,ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006')
SOURCE='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'
REQUIRED=('recovery-manifest.json','recovery-input-preparation.json','source-commit.txt',
 'slurm-allocation.txt','runtime-import-origins.log','cpu-gate.json','cpu-gate-source.txt',
 'tools-manifest.json','launch-approval.json','gpus-before.csv','processes-before.csv',
 'WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION')
MAX_RAW_BYTES=8*1024*1024

def require(ok,msg):
 if not ok:raise ValueError(msg)

def job_id(text):
 require(isinstance(text,str) and re.fullmatch(r'[1-9][0-9]*',text)is not None,'Invalid positive actual job id')
 return int(text)

def validate_slurm(raw,job):
 for name,value in [('JobId',str(job)),('JobName','bapo-instruct300-hardware'),
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

def capture(job,live_slurm):
 require(type(job)is int and job>0,'Job id must be a positive integer')
 validate_slurm(live_slurm,job)
 run=ROOT/'runs'/f'bapo-instruct300-hardware-{job}'
 require(run.is_dir() and not run.is_symlink() and run.resolve()==run,'Unexpected run directory')
 files={name:read_raw(run/name)for name in REQUIRED}
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
 root_inputs={name:read_raw(ROOT/'logs'/name)for name in ('runtime-executable.json','cpu-gate-durable.log','cpu-gate-durable-exit.json','final-launch-independent-review.md','title-dependency-preflight.json','title-dependency-preflight.log','title-dependency-preflight-exit.json','guardian-tests-durable.log','guardian-tests-durable-exit.json')}
 approval=json.loads(files['launch-approval.json']['text']);tools=json.loads(files['tools-manifest.json']['text'])
 require(tools['source_commit']==SOURCE and len(tools['files'])==36,'Exact final36 tools required')
 require({p.name for p in (ROOT/'tools').iterdir()if p.is_file()}==set(tools['files']),'Unmanifested actual tool prevents capture')
 actual_tools={name:read_raw(ROOT/'tools'/name)for name in tools['files']}
 for name,item in actual_tools.items():require(item['sha256']==tools['files'][name],'Actual installed tool differs: '+name)
 for field,name in [('runtime_executable','runtime-executable.json'),('cpu_gate_log','cpu-gate-durable.log'),('cpu_gate_exit_record','cpu-gate-durable-exit.json'),('title_dependency_preflight','title-dependency-preflight.json'),('guardian_actual_linux_log','guardian-tests-durable.log'),('guardian_actual_linux_record','guardian-tests-durable-exit.json')]:
  binding=approval[field];x=root_inputs[name];require(binding['path']==x['path']and binding['sha256']==x['sha256']and binding['bytes']==x['bytes'],'New mandatory original binding differs: '+field)
 review=approval['independent_review'];x=root_inputs['final-launch-independent-review.md'];require(review['path']==x['path']and review['sha256']==x['sha256'],'Final independent report binding differs')
 require('identity-linux-fixture.py'in actual_tools and 'identity_evidence_validator.py'in actual_tools,'Identity fixture/validator tools must be bound')
 actual_source={'recovery.py':read_raw(ROOT/'repo/search_r1/baselines/recovery.py')}
 pins={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(actual_source['recovery.py']['text']).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('GUARD_JOURNAL_SHA','RESOURCE_WATCH_SHA')}
 require(pins=={'GUARD_JOURNAL_SHA':tools['files']['guard_journal.py'],'RESOURCE_WATCH_SHA':tools['files']['resource_watch.py']},'Actual core/source guard pins do not match installed tools')
 optional={} 
 for name in ('resource-watch.jsonl','own-process-cleanup.json','recovery-permission-wait.log'):
  p=run/name
  if p.is_file():optional[name]=read_raw(p)
 # Repeat immutable allocated input reads and waiting-state check to reject capture races.
 for name,first in files.items():require(read_raw(run/name)['sha256']==first['sha256'],'Allocated raw changed during capture')
 for name,first in root_inputs.items():require(read_raw(ROOT/'logs'/name)['sha256']==first['sha256'],'Original root evidence changed during capture')
 for name,first in actual_tools.items():require(read_raw(ROOT/'tools'/name)['sha256']==first['sha256'],'Tool changed during capture')
 require(read_raw(ROOT/'repo/search_r1/baselines/recovery.py')['sha256']==actual_source['recovery.py']['sha256'],'Actual source changed during capture')
 require(not (run/'recovery-permission.json').exists(),'Permission appeared during capture')
 return {'status':'READ_ONLY_ACTUAL_ALLOCATED_INPUTS_NOT_PERMISSION_OR_GPU_ACCEPTANCE',
  'job_id':job,'source_commit':SOURCE,'root':str(ROOT),'run_root':str(run),
  'captured_at_unix':time.time(),'files':files,'referenced_inputs':referenced,'optional_raw':optional,'root_inputs':root_inputs,'actual_tools':actual_tools,'actual_source':actual_source,
  'slurm_live':{'text':live_slurm,'sha256':hashlib.sha256(live_slurm.encode()).hexdigest()},
  'large_state_or_weights_revalidated':False,'actual_gpu':False,'permission_signed':False,
  'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-id',required=True)
 a=p.parse_args();job=job_id(a.job_id)
 live=subprocess.check_output(['scontrol','show','job','-dd',str(job)],text=True,timeout=15)
 print(json.dumps(capture(job,live),sort_keys=True))

if __name__=='__main__':main()
