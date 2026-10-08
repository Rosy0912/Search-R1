"""Independently validate small allocated originals. Never remote/sign/load CUDA/weights."""
import argparse,ast,base64,hashlib,json,pathlib,re,yaml
B=pathlib.Path(__file__).parent;ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006';SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876'
MANIFEST_SHA='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46'
CPU_SHA='e6cbb1cb66cdea1d5ad165dbd45a375868bb6711cdcd839e25cef6c32575318c'
RUNTIME_SHA='38666a2b8448e0ce8389eba49cfea57caf5fca4979b9000a61a667d3a1d2dc46'
def need(v,msg):
 if not v:raise ValueError(msg)
def h(raw):return hashlib.sha256(raw).hexdigest()
def sha(p):return h(pathlib.Path(p).read_bytes())
def json_strict(raw):
 def pairs(items):
  result={}
  for k,v in items:need(k not in result,'Duplicate original JSON key');result[k]=v
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite original JSON')))
def validate(capture,job,checkpoint,repo):
 need(type(job)is int and job>0,'Positive job required');run=ROOT+'/runs/bapo-instruct300-hardware-'+str(job);d=json_strict(pathlib.Path(capture).read_bytes())
 need(d['status']=='READ_ONLY_ACTUAL_ALLOCATED_INPUTS_NOT_PERMISSION_OR_GPU_ACCEPTANCE'and d['job_id']==job and d['source_commit']==SOURCE and d['root']==ROOT and d['run_root']==run and d['actual_gpu']is False and d['permission_signed']is False,'Actual allocated capture identity differs')
 need(d['collector_sha256']==sha(B/'capture-hardware-allocation.py'),'Independent collector bytes changed')
 decoded={}
 for group in ('files','referenced_inputs','root_inputs','actual_tools','actual_source'):
  decoded[group]={}
  for name,item in d[group].items():
   raw=base64.b64decode(item['base64'],validate=True);need(type(item['bytes'])is int and len(raw)==item['bytes']and h(raw)==item['sha256']and item['path']==item['resolved_path'],'Original raw binding mismatch '+name)
   if 'text'in item:need(raw.decode()==item['text'],'Original text differs')
   if group=='files':need(item['path']==run+'/'+name,'Run evidence path differs')
   if group=='root_inputs':need(item['path']==ROOT+'/logs/'+name,'Root original path differs')
   if group=='actual_tools':need(item['path']==ROOT+'/tools/'+name,'Actual tool path differs')
   if group=='actual_source':need(name=='recovery.py'and item['path']==ROOT+'/repo/search_r1/baselines/recovery.py','Actual source path differs')
   decoded[group][name]=raw
 files=decoded['files'];root=decoded['root_inputs'];get=lambda n:json_strict(files[n])
 need(files['source-commit.txt']==files['cpu-gate-source.txt']==(SOURCE+'\n').encode(),'Allocated source markers differ')
 need(files['WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION']==b'','Not prepermission waiting marker')
 need(h(files['tools-manifest.json'])==MANIFEST_SHA and h(files['cpu-gate.json'])==CPU_SHA and h(root['runtime-executable.json'])==RUNTIME_SHA,'New exact manifest/CPU/runtime mandatory bindings differ')
 manifest=get('recovery-manifest.json');need(manifest['target_job_id']==job and manifest['target_source_commit']==SOURCE and manifest['target_run_root']==run,'Recovery allocated identity differs')
 need(manifest['step']==manifest['actual_outer_updates']==25 and manifest['next_step']==26 and manifest['world_size']==8 and manifest['old_source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0','Recovery25 budget/old identity differs')
 c=json_strict(pathlib.Path(checkpoint).read_bytes());bindings=[x for rank in manifest['ranks']for x in (rank['state'],rank['report'])]+[manifest['driver'],manifest['driver_report'],manifest['algorithm']]+manifest['hf_files']
 need(len(bindings)==31 and {x['path']:(x['sha256'],x['bytes'])for x in bindings}=={x['path']:(x['sha256'],x['bytes'])for x in c['files']},'Complete immutable31 legacy checkpoint input bindings differ')
 need(c['generation_rng_in_old_checkpoint']is False and c['sampler_cursor']==12800,'Historical RNG gap/cursor differs')
 need(manifest['resolved_config']['sha256']==c['config_sha256'],'Historical actual config differs')
 for key in ('resolved_config','historical_first5_receipt','train_prefix'):
  item=d['referenced_inputs'][key];binding=manifest[key];need(item['path']==binding['path']and item['sha256']==binding['sha256']and item['bytes']==binding['bytes'],'Original referenced input differs '+key)
 cfg=yaml.safe_load(decoded['referenced_inputs']['resolved_config']);need(cfg['trainer']['total_training_steps']==301 and cfg['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfg['baselines']['bapo']['stage_validation_size']==64 and cfg['data']['train_batch_size']==512 and cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size']==128 and cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size']==16 and cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1,'Original algorithm/hardware budget differs')
 prefix=decoded['referenced_inputs']['train_prefix'];need(h(prefix)=='57b05f48a27ab5fd5c81e3e524f71569c6675f8c3ca234ac105b0c677e2c3dc0'and len(prefix)==959681,'Complete25 prefix identity differs');steps=[json.JSONDecoder().raw_decode(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1].lstrip())[0]['step']for line in prefix.decode().splitlines()if'BAPO_TRAIN_METRICS_JSON='in line];need(steps==list(range(1,26)),'Actual25 prefix steps differ')
 for name,item in d['referenced_inputs'].items():
  if name.startswith('shared_input_'):need(any(item['path']==x['path']and item['sha256']==x['sha256']and item['bytes']==x['bytes']for x in manifest['shared_inputs']),'Shared input small binding differs')
 cpu=get('cpu-gate.json');need(cpu['source_commit']==SOURCE and cpu['sys_prefix']==ROOT+'/env'and all(type(cpu[k])is int and cpu[k]==v for k,v in [('passed',395),('failed',0),('skipped',0),('exit_code',0)]),'Actual second CPU result differs')
 tools=get('tools-manifest.json');need(tools['source_commit']==SOURCE and len(tools['files'])==36 and set(decoded['actual_tools'])==set(tools['files']),'Actual new36 tools differ')
 for name,raw in decoded['actual_tools'].items():need(h(raw)==tools['files'][name]and raw==(B/'deployment-tools'/name).read_bytes(),'Actual new tool/code differs '+name)
 approval=get('launch-approval.json');need(approval==json_strict((B/'unsigned-actual-cpu-launch-r1/launch-approval.json').read_bytes())and approval['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'and approval['source_commit']==SOURCE and approval['root']==ROOT,'Independent new signed launch differs')
 need(approval['tools_manifest_sha256']==h(files['tools-manifest.json'])and approval['cpu_gate_record_sha256']==h(files['cpu-gate.json']),'Approved original CPU/tools differ')
 for key,name in [('runtime_executable','runtime-executable.json'),('cpu_gate_log','cpu-gate-durable.log'),('cpu_gate_exit_record','cpu-gate-durable-exit.json'),('title_dependency_preflight','title-dependency-preflight.json'),('guardian_actual_linux_log','guardian-tests-durable.log'),('guardian_actual_linux_record','guardian-tests-durable-exit.json')]:
  binding=approval[key];item=d['root_inputs'][name];need(binding['path']==item['path']and binding['bytes']==item['bytes']and binding['sha256']==item['sha256'],'Mandatory launch original binding differs '+key)
 review=approval['independent_review'];need(review['path']==ROOT+'/logs/final-launch-independent-review.md'and review['sha256']==h(root['final-launch-independent-review.md']),'Independent signed launch report differs')
 exit_record=json_strict(root['cpu-gate-durable-exit.json']);need(type(exit_record['exit_code'])is int and exit_record['exit_code']==0 and exit_record['log_sha256']==h(root['cpu-gate-durable.log'])==cpu['log_sha256']and h(root['cpu-gate-durable-exit.json'])==cpu['exit_record_sha256'],'Actual durable exit/log mismatch')
 title_exit=json_strict(root['title-dependency-preflight-exit.json']);need(type(title_exit['exit_code'])is int and title_exit['exit_code']==0 and title_exit['log_sha256']==h(root['title-dependency-preflight.log']),'Actual title original exit/log differs')
 guardian=json_strict(root['guardian-tests-durable-exit.json']);need(type(guardian['exit_code'])is int and guardian['exit_code']==0 and guardian['source_commit']==SOURCE and guardian['tools_manifest_sha256']==MANIFEST_SHA and guardian['training_submitted']is False and guardian['log_sha256']==h(root['guardian-tests-durable.log']),'Actual guardian original exit/log differs');gtext=root['guardian-tests-durable.log'].decode();need(len(re.findall(r'Ran 112 tests in ([0-9.]+)s',gtext))==1 and gtext.rstrip().endswith('OK'),'Actual guardian112 not passed')
 rt=json_strict(root['runtime-executable.json']);need(rt['source_commit']==SOURCE and rt['root']==ROOT and rt['sys_prefix']==ROOT+'/env','Runtime anchor identity differs')
 source_raw=decoded['actual_source']['recovery.py'];need(source_raw==(pathlib.Path(repo)/'search_r1/baselines/recovery.py').read_bytes(),'Actual source byte binding differs');pins={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(source_raw).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('GUARD_JOURNAL_SHA','RESOURCE_WATCH_SHA')};need(pins=={'GUARD_JOURNAL_SHA':tools['files']['guard_journal.py'],'RESOURCE_WATCH_SHA':tools['files']['resource_watch.py']},'Actual core/helper pins do not close')
 need(h(root['title-dependency-preflight.json'])=='d10fb8f5de5bbb282e5c7a5096bd563457de567e149b09eb71b7279f8a84fd61','Actual fresh source title record differs')
 import importlib.util
 spec=importlib.util.spec_from_file_location('r3_actual_allocation_builder',B/'build-unsigned-cpu-launch-candidate.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.validate_title(json_strict(root['title-dependency-preflight.json']),rt,B/'deployment-tools',tools)
 for name in ('actual-title-dependency-preflight.py','identity-linux-fixture.py','identity_evidence_validator.py','resource_watch.py','guard_journal.py','capture-runtime-executable.py'):need(name in tools['files'],'Missing actual mandatory identity tool '+name)
 origin={}
 for label in ('before','after'):
  marker='CPU_IMPORT_ORIGINS_'+label.upper()+'_JSON=';rows=[json_strict(line.split(marker,1)[1])for line in files['runtime-import-origins.log'].decode().splitlines()if marker in line];need(len(rows)==1,'Runtime origin marker count differs');origin[label]=rows[0];need(rows[0]['source_commit']==SOURCE and rows[0]['repo']==ROOT+'/repo','Runtime actualsource identity differs')
  for name,item in rows[0]['modules'].items():need(item['path'].startswith(ROOT+'/repo/')and sha(pathlib.Path(repo)/pathlib.Path(item['path']).relative_to(pathlib.Path(ROOT)/'repo'))==item['sha256'],'Runtime source module changed')
  need(rows[0]['functions']==cpu['import_origins']['before']['functions'],'Runtime production function origins changed')
 need(origin['before']['functions']==origin['after']['functions'],'Runtime functions changed')
 live=d['slurm_live']['text'];need(h(live.encode())==d['slurm_live']['sha256'],'Actual live Slurm raw differs')
 for name,value in [('JobId',str(job)),('JobName','bapo-instruct300-hardware'),('JobState','RUNNING'),('Restarts','0'),('NodeList','h100-5'),('NumNodes','1'),('JOB_GRES','gpu:h100:8')]:need(re.findall(r'(?:^|\s)'+name+r'=([^\s]+)',live)==[value],'Actual Slurm field differs '+name)
 allocation=files['slurm-allocation.txt'].decode();need(re.findall(r'^SLURM_JOB_ID=(\d+)$',allocation,re.M)==[str(job)],'Allocation job differs');need(re.findall(r'^CUDA_VISIBLE_DEVICES=([^\n]+)$',allocation,re.M)==['0,1,2,3,4,5,6,7'],'Eight actual visible devices differ')
 uuids=set(re.findall(r'GPU-[a-f0-9-]{36}',files['gpus-before.csv'].decode()));need(len(uuids)==8,'Eight physical GPU UUIDs required')
 prep=get('recovery-input-preparation.json');need(prep['phase']=='hardware'and prep['job_id']==job and prep['source_commit']==SOURCE and prep['run_root']==run and prep['manifest']['sha256']==h(files['recovery-manifest.json']),'Actual preparation not bound')
 return dict(status='PASS_INDEPENDENT_ALLOCATED_INPUT_CHECK_NOT_SIGNED_PERMISSION_OR_GPU',job_id=job,source_commit=SOURCE,run_root=run,allocation_capture_sha256=sha(capture),manifest_sha256=h(files['recovery-manifest.json']),legacy_checkpoint_bound_files=31,old_generation_rng_gap_acknowledged=True,cpu_passed=395,tool_count=36,title_dependency_preflight_cpu_accepted=True,guardian_cpu_passed=112,allocated_gpu_uuids=sorted(uuids),identity_actual_linux_fixture_accepted=False,actual_gpu_accepted=False,permission_signed=False,training_authorized=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--checkpoint-evidence',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);need(not out.exists(),'Fresh independent check output only');r=validate(a.capture,a.job_id,a.checkpoint_evidence,a.repo)
 with out.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r))
if __name__=='__main__':main()
