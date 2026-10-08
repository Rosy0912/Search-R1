"""Local original allocation checker; no signing, restoration, GPU or submission."""
from pathlib import Path
import argparse,base64,csv,hashlib,io,json,re,yaml
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006';SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876';B=Path(__file__).resolve().parent
CPU_SHA='e6cbb1cb66cdea1d5ad165dbd45a375868bb6711cdcd839e25cef6c32575318c';MANIFEST_SHA='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46'
def h(raw):return hashlib.sha256(raw).hexdigest()
def validate_waiting_environment(w,job,stage_sha,stage_pid):
 run=ROOT+'/runs/bapo-instruct300-formal300-'+str(job)
 assert w['status']=='ACTUAL_READ_ONLY_EXISTING_WAIT_SLEEP_CHILD_ENVIRONMENT_NOT_PERMISSION'and w['job_id']==job and type(w['job_id'])is int and w['source_commit']==SOURCE and w['root']==ROOT and w['run_root']==run
 assert w['signals_sent']==w['processes_spawned']==0 and type(w['signals_sent'])is int and type(w['processes_spawned'])is int and w['gpu_executed']is False
 assert w['stage_path']==ROOT+'/tools/stage.sh'and w['stage_sha256']==stage_sha and len(w['proofs'])==1
 p=w['proofs'][0];c=p['child_before'];a=p['parent_before'];assert c==p['child_after']and a==p['parent_after']
 for v,exe in ((c,'/usr/bin/sleep'),(a,'/usr/bin/bash')):
  for k in ('pid','ppid','start_ticks'):assert type(v[k])is int and v[k]>0
  assert type(v['uid'])is int and v['uid']==0
  assert v['executable']['resolved_path']==exe
  for k in ('device','inode'):assert type(v['executable'][k])is int and v['executable'][k]>0
  raw=base64.b64decode(v['command_base64'],validate=True);assert [x.decode()for x in raw.split(b'\0')if x]==v['command']
  raw=base64.b64decode(v['selected_environment_base64'],validate=True);rows=[x for x in raw.split(b'\0')if x];assert len(rows)==len({x.split(b'=',1)[0]for x in rows});assert {x.split(b'=',1)[0].decode():x.split(b'=',1)[1].decode()for x in rows}==v['environment']
 assert c['ppid']==a['pid']==stage_pid and c['command']==['sleep','5']
 assert a['command']in [[exe,ROOT+'/tools/stage.sh','formal300']for exe in ('bash','/bin/bash','/usr/bin/bash')]
 assert a['environment']['SLURM_JOB_ID']==str(job)and a['environment']['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
 assert c['environment']==dict(SPT_NOENV='1',BAPO_RUN_ID=run,BAPO_E1_RUN=run,SLURM_JOB_ID=str(job),SOURCE=SOURCE,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
 return True

def check(d,job,receipt_sha,hardware_manifest,waiting_environment=None):
 assert type(job)is int and job>0 and job!=3287 and receipt_sha=='e8625dabce8c9cf5a97498cb0989b399937fd9719acfa186357b1d29cebe1e2c'
 run=ROOT+'/runs/bapo-instruct300-formal300-'+str(job)
 assert d['status']=='READ_ONLY_ACTUAL_ALLOCATED_INPUTS_NOT_PERMISSION_OR_GPU_ACCEPTANCE'and d['root']==ROOT and d['source_commit']==SOURCE and d['run_root']==run and type(d['job_id'])is int and d['job_id']==job and d['actual_gpu']is False and d['permission_signed']is False
 assert d['collector_sha256']==h((B/'capture-formal-allocation.py').read_bytes())
 items={}
 for group in ('files','referenced_inputs','hardware_acceptance'):
  for name,v in d[group].items():
   raw=base64.b64decode(v['base64'],validate=True);assert type(v['bytes'])is int and len(raw)==v['bytes']and h(raw)==v['sha256']and raw.decode()==v['text']and v['path']==v['resolved_path'];items[(group,name)]=raw
   if group=='files':assert v['path']==run+'/'+name
 assert len(d['files'])==12 and len(d['hardware_acceptance'])==2 and len(d['referenced_inputs'])==6
 f=lambda n:json.loads(items[('files',n)])
 m=f('recovery-manifest.json');assert type(m['target_job_id'])is int and m['target_job_id']==job and m['target_run_root']==run and m['target_source_commit']==SOURCE and m['step']==m['actual_outer_updates']==25 and m['next_step']==26 and m['world_size']==8
 receipt_raw=items[('hardware_acceptance','hardware-acceptance.json')];assert h(receipt_raw)==receipt_sha;receipt=json.loads(receipt_raw)
 assert receipt['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and receipt['job_id']==3287 and receipt['root']==ROOT and receipt['source_commit']==SOURCE and receipt['actual_gpu_accepted']is True and receipt['production_training_permission_signed']is False
 report=items[('hardware_acceptance','hardware-independent-review')];assert h(report)==receipt['independent_review']['sha256']and len(report)==receipt['independent_review']['bytes']and d['hardware_acceptance']['hardware-independent-review']['path']==receipt['independent_review']['path']and Path(receipt['independent_review']['path']).parent==Path(ROOT)/'logs'
 assert len(receipt['gpu_witnesses'])==9 and len(receipt['gpu_validation']['native_v3_roundtrip'])==8 and len(receipt['gpu_validation']['raw_artifacts'])==14
 hm_raw=hardware_manifest.read_bytes();hm=json.loads(hm_raw);hr=ROOT+'/runs/bapo-instruct300-hardware-3287/recovery-manifest.json'
 assert any(v['path']==hr and v['sha256']==h(hm_raw)and v['bytes']==len(hm_raw)for v in receipt['evidence'])
 for k in ('version','old_source_commit','old_run_root','step','next_step','actual_outer_updates','world_size','ranks','driver','driver_report','algorithm','resolved_config','shared_inputs','hf_files','historical_first5_receipt','train_prefix','train_prefix_live'):assert m[k]==hm[k],k
 cpu=f('cpu-gate.json');assert h(items[('files','cpu-gate.json')])==CPU_SHA and cpu['source_commit']==SOURCE and cpu['sys_prefix']==ROOT+'/env'and cpu['passed']==395 and cpu['failed']==cpu['skipped']==cpu['exit_code']==0
 assert h(items[('files','tools-manifest.json')])==MANIFEST_SHA and len(f('tools-manifest.json')['files'])==36 and items[('files','tools-manifest.json')]==(B/'tools-manifest.json').read_bytes()
 assert items[('files','launch-approval.json')]==(B/'unsigned-actual-cpu-launch-r1/launch-approval.json').read_bytes()
 for n in ('cpu-gate-source.txt','source-commit.txt'):assert items[('files',n)]== (SOURCE+'\n').encode()
 slurm=d['slurm_live']['text'];assert h(slurm.encode())==d['slurm_live']['sha256']
 for k,v in [('JobId',str(job)),('JobName','bapo-instruct300-formal300'),('JobState','RUNNING'),('Restarts','0'),('NodeList','h100-5'),('NumNodes','1'),('JOB_GRES','gpu:h100:8')]:assert re.findall(r'(?:^|\s)'+k+'=([^\s]+)',slurm)==[v]
 allocation=items[('files','slurm-allocation.txt')].decode()
 for key,val in [('SLURM_JOB_ID',str(job)),('SLURM_JOB_UID','0'),('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')]:assert re.findall(r'^'+key+'=([^\r\n]+)$',allocation,re.M)==[val]
 assert waiting_environment is not None,'Fresh actual existing-wait-child SPT proof required; filtered originalallocation untouched'
 stage_pid=re.findall(r'^SLURM_TASK_PID=([0-9]+)$',allocation,re.M);assert len(stage_pid)==1
 validate_waiting_environment(waiting_environment,job,f('tools-manifest.json')['files']['stage.sh'],int(stage_pid[0]))
 uuids=sorted(row['uuid'].strip()for row in csv.DictReader(io.StringIO(items[('files','gpus-before.csv')].decode()),skipinitialspace=True));assert len(uuids)==len(set(uuids))==8 
 hg=(hardware_manifest.parent/'gpus-before.csv').read_bytes();assert any(v['path']==ROOT+'/runs/bapo-instruct300-hardware-3287/gpus-before.csv'and v['sha256']==h(hg)and v['bytes']==len(hg)for v in receipt['evidence'])
 assert uuids==sorted(row['uuid'].strip()for row in csv.DictReader(io.StringIO(hg.decode()),skipinitialspace=True))
 prep=f('recovery-input-preparation.json');assert prep['phase']=='formal300'and prep['job_id']==job and prep['run_root']==run and prep['source_commit']==SOURCE and prep['manifest']['sha256']==h(items[('files','recovery-manifest.json')])
 assert items[('files','WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION')]==b''
 for name,key in [('resolved_config','resolved_config'),('historical_first5_receipt','historical_first5_receipt'),('train_prefix','train_prefix')]:assert d['referenced_inputs'][name]['sha256']==m[key]['sha256']
 for name,v in d['referenced_inputs'].items():
  if name.startswith('shared_input_'):assert any(v['path']==a['path']and v['sha256']==a['sha256']and v['bytes']==a['bytes']for a in m['shared_inputs'])
 cfg=yaml.safe_load(items[('referenced_inputs','resolved_config')]);assert cfg['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfg['trainer']['total_training_steps']==301 and cfg['baselines']['bapo']['stage_validation_size']==64 and cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1
 origins=[]
 for line in items[('files','runtime-import-origins.log')].decode().splitlines():
  if line.startswith('CPU_IMPORT_ORIGINS_'):
   o=json.loads(line.split('=',1)[1]);origins.append(o)
   for v in o['functions'].values():assert v['path'].startswith(ROOT+'/repo/')and v['co_filename']==v['path']
 assert len(origins)==2 and origins[0]==origins[1]
 return dict(status='PASS_LOCAL_RAW_FORMAL_ALLOCATION_CHECK_NOT_INDEPENDENT_PERMISSION_OR_RESTORE',job_id=job,source_commit=SOURCE,manifest_sha256=h(items[('files','recovery-manifest.json')]),capture_sha256=h(json.dumps(d,sort_keys=True).encode()),hardware_receipt_sha256=receipt_sha,step=25,next_step=26,target_actual_outer_updates=300,actual_runtime_restore_verified=False,allocated_uuids=uuids)
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--hardware-receipt-sha',required=True);p.add_argument('--hardware-manifest',type=Path,required=True);p.add_argument('--waiting-environment',type=Path,required=True);a=p.parse_args();print(json.dumps(check(json.loads(a.capture.read_text()),a.job_id,a.hardware_receipt_sha,a.hardware_manifest,json.loads(a.waiting_environment.read_text()))))
if __name__=='__main__':main()
