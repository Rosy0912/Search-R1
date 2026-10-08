from pathlib import Path
import json,base64,hashlib,re,csv,io,yaml
B=Path(__file__).resolve().parent;E=B.parent/'evidence';D=E/'freeze-scalability-formal3269-allocation-files.json';d=json.loads(D.read_text());ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006';RUN=ROOT+'/runs/bapo-instruct300-formal300-3269';SOURCE='b7119791b86951727cea36a1881e8a525c932494';h=lambda b:hashlib.sha256(b).hexdigest()
assert d['job_id']==3269 and d['run_root']==RUN and d['source_commit']==SOURCE and d['root']==ROOT and d['actual_gpu']is False and d['permission_signed']is False
assert d['collector_sha256']=='73ed915a9ff391deba96631af002905e5013deae4d64b943a1856907a636b088'
items={}
for group in ('files','referenced_inputs','hardware_acceptance'):
 for name,v in d[group].items():
  b=base64.b64decode(v['base64'],validate=True);assert h(b)==v['sha256'] and len(b)==v['bytes'] and b.decode()==v['text'] and v['path']==v['resolved_path'];items[(group,name)]=b
assert len(d['files'])==12 and len(d['referenced_inputs'])==6 and len(d['hardware_acceptance'])==2
f=lambda n:json.loads(items[('files',n)])
m=f('recovery-manifest.json');assert h(items[('files','recovery-manifest.json')])=='0fc65696548aed593d75816fb941c25153283702fb4ba5ceda7ed2fea9cc1593'
assert m['target_job_id']==3269 and m['target_run_root']==RUN and m['target_source_commit']==SOURCE and m['step']==m['actual_outer_updates']==25 and m['next_step']==26 and m['world_size']==8
A=E/'freeze-scalability-hardware3263-raw-evidence';HR=ROOT+'/runs/bapo-instruct300-hardware-3263';hm=json.loads((A/'runs/bapo-instruct300-hardware-3263/recovery-manifest.json').read_text())
for k in ('version','old_source_commit','old_run_root','step','next_step','actual_outer_updates','world_size','ranks','driver','driver_report','algorithm','resolved_config','shared_inputs','hf_files','historical_first5_receipt','train_prefix','train_prefix_live'):assert m[k]==hm[k],k
receipt=json.loads(items[('hardware_acceptance','hardware-acceptance.json')]);assert items[('hardware_acceptance','hardware-acceptance.json')]==(B/'hardware3263-final-receipt.json').read_bytes();assert items[('hardware_acceptance','hardware-independent-review')]==(B/'HARDWARE3263-FINAL-INDEPENDENT-REVIEW.md').read_bytes()
assert receipt['job_id']==3263 and receipt['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW' and receipt['actual_gpu_accepted']is True and receipt['production_training_permission_signed']is False and len(receipt['gpu_witnesses'])==9 and len(receipt['gpu_validation']['native_v3_roundtrip'])==8 and len(receipt['gpu_validation']['raw_artifacts'])==14
cpu=f('cpu-gate.json');assert cpu['passed']==395 and cpu['failed']==cpu['skipped']==cpu['exit_code']==0 and cpu['source_commit']==SOURCE and cpu['sys_prefix']==ROOT+'/env' and d['files']['cpu-gate.json']['sha256']=='8a32af877fdc8393c8d13378756d606e2e3b89cb04b416c8025a7c011e300095'
assert f('tools-manifest.json')==json.loads((B/'final-launch-candidate-r2/tools-manifest.json').read_text()) and len(f('tools-manifest.json')['files'])==29
assert items[('files','launch-approval.json')]==(B/'final-launch-candidate-r2/launch-approval.json').read_bytes()
for n in ('cpu-gate-source.txt','source-commit.txt'):assert items[('files',n)].decode()==SOURCE+'\n'
slurm=d['slurm_live']['text'];assert h(slurm.encode())==d['slurm_live']['sha256']
for k,v in [('JobId','3269'),('JobState','RUNNING'),('Restarts','0'),('NodeList','h100-5'),('NumNodes','1'),('JOB_GRES','gpu:h100:8')]:assert re.search(r'(?:^|\s)'+k+'='+re.escape(v)+r'(?:\s|$)',slurm),(k,v)
a=items[('files','slurm-allocation.txt')].decode();assert '\nSLURM_JOB_ID=3269\n'in a and '\nSLURM_JOB_UID=0\n'in a and 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n'in a
# Exact device UUID rows agree with the hardware's physical allocation.
gpus=items[('files','gpus-before.csv')].decode();old=(A/'runs/bapo-instruct300-hardware-3263/gpus-before.csv').read_text();uuidset=lambda s:set(re.findall(r'GPU-[a-f0-9-]{36}',s));assert len(uuidset(gpus))==8 and uuidset(gpus)==uuidset(old)
prep=f('recovery-input-preparation.json');assert prep['phase']=='formal300' and prep['job_id']==3269 and prep['run_root']==RUN and prep['source_commit']==SOURCE and prep['manifest']['sha256']==d['files']['recovery-manifest.json']['sha256']
assert items[('files','WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION')]==b''
for name,key in [('resolved_config','resolved_config'),('historical_first5_receipt','historical_first5_receipt'),('train_prefix','train_prefix')]:assert d['referenced_inputs'][name]['sha256']==m[key]['sha256']
for name,v in d['referenced_inputs'].items():
 if name.startswith('shared_input_'):assert any(v['path']==b['path'] and v['sha256']==b['sha256'] and v['bytes']==b['bytes']for b in m['shared_inputs'])
cfg=yaml.safe_load(items[('referenced_inputs','resolved_config')]);assert cfg['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfg['trainer']['total_training_steps']==301 and cfg['baselines']['bapo']['stage_validation_size']==64 and cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1
orig=[]
for line in items[('files','runtime-import-origins.log')].decode().splitlines():
 if line.startswith('CPU_IMPORT_ORIGINS_'):
  o=json.loads(line.split('=',1)[1]);orig.append(o)
  for v in o['functions'].values():assert v['path'].startswith(ROOT+'/repo/')and v['co_filename']==v['path']
assert len(orig)==2 and orig[0]==orig[1]
print(json.dumps({'status':'PASS_INDEPENDENT_FORMAL3269_ACTUAL_ALLOCATION_REVIEW_NOT_RUNTIME_RESTORE','job_id':3269,'manifest_sha256':d['files']['recovery-manifest.json']['sha256'],'source_commit':SOURCE,'files_rehashed':20,'tools':29,'cpu_passed':395,'allocated_uuids':sorted(uuidset(gpus)),'hardware_job':3263,'gpu_witnesses':9,'native_v3':8,'step':25,'next_step':26,'target_actual_outer_updates':300,'actual_runtime_restore_verified':False}))
