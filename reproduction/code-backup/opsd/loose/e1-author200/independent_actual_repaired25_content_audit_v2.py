"""Independent local joins of actual CPU25 originals, without remote actions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, re, types
B=Path(__file__).resolve().parent
SNAP=B/'actual-snapshots/3261-20261005T230513952741Z'
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
RUN=ROOT+'/runs/formal-3261'
OUT='evidence/checkpoint-25-cpu-inspection-repair-v2/'
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
NATIVE='eb1afffde0a47de45143af34951ae06bd4ce22bf299e9f5ad9d9f8ea6a1b0ced'
INSPECTOR='f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a'
TOOLS='f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd'
INVENTORY='32f2c38c121b0c59ab26036c358b4fd749218f8e3980d6700f54c78fabbe6c4e'
REQUEST='ffb7a19aceef85a7ae27756d238dde46f45dc2a13726141a26fae9bd625d94b7'
NUMEL=3085938688
def raw(p,pin=None):
 assert p.resolve(strict=True)==p and p.is_file(), str(p)
 a=p.stat()
 with p.open('rb') as f:
  c=os.fstat(f.fileno());r=f.read();d=os.fstat(f.fileno())
 z=p.stat();stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 assert stamp(a)==stamp(c)==stamp(d)==stamp(z)
 h=hashlib.sha256(r).hexdigest();assert pin is None or h==pin,(p,h)
 return r,h
def load(raw):
 def unique(pairs):
  d={}
  for k,v in pairs:assert k not in d,k;d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def digest(h):assert type(h) is str and re.fullmatch('[0-9a-f]{64}',h)
def exact_int(v,x):assert type(v) is int and v==x,(v,x)
summary_raw,summary_sha=raw(SNAP/'observation-summary.json');summary=load(summary_raw)
_,transport_sha=raw(SNAP/'observed-remote-original.json',summary['transport_original_sha256'])
assert summary['job_id']=='3261' and summary['node']=='h100-10' and summary['expected_source']==SOURCE
files={}
for name,rec in summary['files'].items():
 p=SNAP/'original-files'/name;assert p.is_relative_to(SNAP/'original-files')
 r,h=raw(p,rec['sha256']);assert len(r)==rec['bytes'];files[name]=(r,h)
assert len(files)==286
def get(name,pin=None):
 r,h=files[name];assert pin is None or h==pin,(name,h);return load(r)
n=get(OUT+'native.json',NATIVE)
inventory=get('evidence/checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json',INVENTORY)
prior=get('evidence/first5-cpu-inspection-once-v1/native.json','b50c6ed9995f6148ac250e75f20721bafc95d952f4d841c24542dce31f4adcb2')
tools_raw,_=raw(B/'final-acceptance-f5c72c942641/manifest.json',TOOLS);tools=load(tools_raw)
raw(B/'final-acceptance-f5c72c942641/inspect_author200_checkpoint_cpu.py',INSPECTOR)
assert n['scope']=='NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION' and n['passed'] is True
exact_int(n['version'],1);exact_int(n['checkpoint_step'],25)
assert n['source_commit']==SOURCE and n['config_sha256']==CONFIG and n['job_id']=='3261' and n['run_root']==RUN
assert n['inspector_source_sha256']==tools['tools']['inspect_author200_checkpoint_cpu.py']==INSPECTOR
assert n['immutable_math_helper_sha256']=='e949ebab98a49b12ac67fd6531089ef34513e276f169d9fd240c6aaccf623db4'
assert n['inventory_sha256']==INVENTORY
assert n['cuda_is_initialized'] is False and n['cpu_runtime']['torch_version']=='2.4.0+cu121'
exact_int(n['native_closure_module_count'],210);exact_int(n['actual_import_function_count'],89)
assert n['native_import_closure_sha256']=='3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
assert n['formal_manifest_sha256']=='bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c'
assert n['clock']==dict(adam_steps=400,cursor=12800,epoch=0,global_step=25,lr=5e-6,next_global_step=26,scheduler_last_epoch=25,scheduler_step_count=26,teacher_cuda_updates=25)
for k,v in n['clock'].items():assert type(v) is (float if k=='lr' else int)
assert n['original_raw_statuses']==inventory['raw_statuses']=={'ranks':['FULL_STATE_SERIALIZED_OK']*8,'driver':'DRIVER_STATE_SERIALIZED_OK'}
assert inventory['observation_scope']=='FILES_AND_IDENTITY_OBSERVED_NOT_LIVE_RESTORE_OR_EFFECT_ACCEPTANCE'
assert len(inventory['files'])==31 and len(n['all_files'])==271 and len(n['hf_files'])==12
assert len({(r['device'],r['inode']) for r in inventory['files'].values()})==31
for role,rec in inventory['files'].items():
 assert n['all_files'][rec['physical_path']]==rec['sha256'] and type(rec['bytes']) is int and rec['bytes']>0
 if role.startswith('actor/') and not role.startswith('actor/training_state/'):
  assert n['hf_files'][rec['physical_path']]==rec['sha256']
assert n['hf_all_rank_slices_exact'] is True
index=get('checkpoints/actor/global_step_25/model.safetensors.index.json')
keys=set(index['weight_map']);assert len(keys)==435 and len(set(index['weight_map'].values()))==3
cfg=get('full_state/global_step_5/resolved-config.json',CONFIG)
actor=cfg['actor_rollout_ref']['actor']
assert actor['baseline_method']==cfg['baselines']['method']=='opsd' and actor['use_kl_loss'] is False
assert actor['kl_loss_coef']==actor['entropy_coeff']==cfg['algorithm']['kl_ctrl']['kl_coef']==0
assert actor['optim']['lr']==5e-6 and actor['optim']['lr_warmup_steps']==0 and actor['grad_clip']==.1
assert actor['ppo_mini_batch_size']==32 and actor['ppo_micro_batch_size']==16 and actor['distillation']['component_clip']==.05
receipt=get('acceptance-5-receipt.json','812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b')
gpath=B.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
graw,_=raw(gpath,'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb')
gate=types.ModuleType('independent_actual25_gate');gate.__file__=str(gpath);exec(compile(graw,str(gpath),'exec',dont_inherit=True),gate.__dict__)
gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],get('bindings/public.json'))
assert n['accepted5_signed_receipt_sha256']==inventory['accepted5']['signed_receipt_sha256']==files['acceptance-5-receipt.json'][1]
assert n['accepted5_manifest_sha256']==inventory['accepted5']['manifest_sha256']=='8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
closure=get('bindings/import_closure.json')
for rec in closure['modules'].values():
 assert n['all_files'][rec['path']]==rec['sha256'],rec['path']
assert len(closure['modules'])==210
rank_summary=[]
for rank,r in enumerate(n['ranks']):
 old=prior['ranks'][rank];rj=get(f'checkpoints/actor/global_step_25/training_state/rank_{rank}.json')
 pt=inventory['files'][f'actor/training_state/rank_{rank}.pt']
 exact_int(r['rank'],rank);assert r['pid']==rj['pid']==old['pid'] and r['object_ids']==rj['object_ids']==old['object_ids']
 assert r['raw_saved_status']==rj['status']=='FULL_STATE_SERIALIZED_OK'
 assert r['state_sha256']==rj['state_sha256']==rj['sha256']==pt['sha256'] and r['state_digest']==rj['state_digest']
 assert r['actor_config_digest']==old['actor_config_digest']
 assert set(r['component_digests'])=={'model','optimizer','scheduler','rng','teacher'}
 for h in r['component_digests'].values():digest(h)
 exact_int(r['model_dtensor_entries'],435);assert r['model_hf_slice_exact'] is True and set(r['model_hf_slice_keys'])==keys
 assert set(r['model_layouts'])==set(r['changed_elements_vs_original_fp32'])==keys
 count=changes=0
 for key,layout in r['model_layouts'].items():
  oldlayout=old['model_layouts'][key]
  assert layout['mesh_coordinate']==[rank] and type(layout['mesh_coordinate'][0]) is int
  assert layout['mesh_device_type']=='cuda' and layout['mesh_shape']==[8] and layout['mesh_ranks']==list(range(8))
  assert layout['mesh_dim_names']==['fsdp'] and layout['placements']==['S(0)']
  assert layout['global_dtype']==layout['local_dtype']=='torch.float32'
  shape=layout['global_shape'];local=layout['local_shape']
  assert all(type(x) is int and x>0 for x in shape+local) and shape[0]%8==0 and local==[shape[0]//8]+shape[1:]
  assert shape==oldlayout['global_shape'] and local==oldlayout['local_shape']
  local_n=math.prod(local);changed=r['changed_elements_vs_original_fp32'][key]
  assert type(changed) is int and 0<=changed<=local_n;digest(layout['local_bytes_sha256'])
  if key!='lm_head.weight':count+=local_n;changes+=changed
 assert count==r['unique_local_model_numel']==NUMEL//8 and changes>0
 assert r['model_layouts']['lm_head.weight']['local_bytes_sha256']==r['model_layouts']['model.embed_tokens.weight']['local_bytes_sha256']
 assert r['changed_elements_vs_original_fp32']['lm_head.weight']==r['changed_elements_vs_original_fp32']['model.embed_tokens.weight']
 assert r['optimizer_state_entries']==37 and r['optimizer_step_values']==[400] and len(r['optimizer_moments'])==37
 assert [(a['parameter_id'],a['shape'],a['dtype']) for a in r['optimizer_moments']]==[(a['parameter_id'],a['shape'],a['dtype']) for a in old['optimizer_moments']]
 for opt in r['optimizer_moments']:
  exact_int(opt['step'],400);assert opt['dtype']=='torch.float32';digest(opt['exp_avg_sha256']);digest(opt['exp_avg_sq_sha256'])
 assert r['scheduler']=={'last_epoch':25,'step_count':26,'base_lrs':[5e-6],'last_lr':[5e-6]}
 rng=r['rng_content'];assert rng['python_state_valid'] is True and rng['numpy_algorithm']=='MT19937'
 digest(rng['numpy_state_sha256']);digest(rng['torch_cpu_state_sha256'])
 assert [x['device'] for x in rng['cuda']]==[x['device'] for x in old['rng_content']['cuda']]==[0]
 for x in rng['cuda']:exact_int(x['bytes'],16);digest(x['sha256'])
 teacher=r['teacher'];assert teacher['content_digest']==n['base_named_parameter_digest']==prior['base_named_parameter_digest']=='af4d2590defbcfa8deef89518f09ed5e368aef244d043b3ee20ca28624ada159'
 exact_int(teacher['cuda_update_entries'],25);exact_int(teacher['counters']['cuda_update_entries'],25)
 assert teacher['buffer_content_digest']==rj['teacher']['buffer_content_digest']
 assert teacher['history_caches']==rj['teacher']['caches'] and len(teacher['history_caches'])==37
 assert teacher['direct_saved_teacher_parameter_arrays'] is False and teacher['fixed_official_assets_and_content_digest'] is True
 for value in teacher['counters'].values():assert type(value) in (int,float) and math.isfinite(value) and value>=0
 rank_summary.append({'rank':rank,'pid':r['pid'],'state_sha256':r['state_sha256'],'local_unique_parameters':count,'changed_vs_official_fp32':changes,'full_HF_slice_keys_checked':435,'Adam_states':37,'Adam_step':400,'scheduler_last_epoch':25,'RNG_cuda_device':0,'teacher_cuda_updates':25})
assert len(rank_summary)==8 and len({r['pid'] for r in rank_summary})==8
driver=n['driver'];dj=get('full_state/global_step_25/driver.json')
assert driver['raw_saved_status']==dj['status']=='DRIVER_STATE_SERIALIZED_OK' and driver['pid']==dj['pid']==1533312
assert driver['state_sha256']==dj['state_sha256']=='dfbf01906077afbc6249ef2d4d57ecf665dcfa3f4f63c675d4d11a96608f9e22'
assert driver['state_digest']==dj['digest']=='d0898f45e262b582348bc9ab0edc700017b5aea3e0ed87422dd4e3e274a95cdf'
actual_small=load(raw(B/'independent-actual-sampler-pt-cpu-original.json')[0])
assert actual_small['records']['current25']['file_sha256']==driver['state_sha256'] and actual_small['records']['current25']['full_state_digest']==driver['state_digest']
assert driver['sampler']==actual_small['repaired_sampler_result']=={'consumed_cursor':12800,'epoch':0,'sampler_cursor':12800,'seed':None,'shuffle':True,'size':169028,'unchanged_epoch0_generator':True,'unchanged_epoch0_order':True}
assert driver['rng_content']==prior['driver']['rng_content'] and driver['rng_content']['cuda']==[]
completed=get(OUT+'inspection-completed-review-required.json','33a0443eaff6ca7b1a77b93edd7ae98c206da954e41c43dd42139ce89ffbd8c1')
assert completed['independent_acceptance_pending'] is True and completed['native_report_sha256']==NATIVE and completed['tools_manifest_sha256']==TOOLS
assert completed['request_sha256']==REQUEST and completed['inventory_sha256']==INVENTORY
for name,h in completed['files'].items():assert files[OUT+name][1]==h,(name,h)
request=get('evidence/checkpoint-cpu-25-repair-v2-launch.json',REQUEST)
assert request['original_request_sha256']=='feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f'
assert request['tools_manifest_sha256']==TOOLS and request['inventory_sha256']==INVENTORY
assert request['original_failed_exit_sha256']==completed['original_failure_exits_preserved']
for name,h in request['original_failed_exit_sha256'].items():
 old=get('evidence/checkpoint-25-cpu-inspection-once-v1/'+name,h);exact_int(old['returncode'],1)
assert 'evidence/checkpoint-25-cpu-inspection-once-v1/native.json' not in files
cpu=get(OUT+'cpu-step-original.json');ctx=cpu['execution_context'];identity=n['execution_identity'];proc=get(OUT+'native-process-original.json')
assert identity['node']==ctx['hostname']=='h100-10' and identity['uid']==0 and identity['uids']==[0]*4
assert identity['execution_job_id']==ctx['job_id']=='3261' and identity['execution_step_id']==ctx['step_id']=='4'
assert ctx['cuda_visible_devices']=='' and identity['stored_training_job_id']=='3261' and identity['python_executable']==ctx['python_executable']
exact_int(identity['pid'],1872025);exact_int(identity['start_ticks'],1830171886)
assert identity['cgroup']==proc['process']['cgroup_original']==ctx['process']['cgroup_original']
assert '/job_3261/step_4\n' in identity['cgroup'] and proc['pid']==identity['pid'] and proc['process']['start_ticks']==str(identity['start_ticks'])
exits={}
for role,pid,parent,pin in [('native',1872025,1871226,'1e5137233330d58ee71b46dd492b20ea4c6c95aa2a96548635d9deda002482f9'),('srun',1871037,1870561,'0cf9f6535da37694f23f5f77070bda34df21c743e4411a72f66f1d6fa8a45938'),('controller',1870561,1870258,'19bd59c9a766c6972404f58bd8112aca763a1a7add90af205ab00d819285f981')]:
 e=get(OUT+role+'-exit-original.json',pin);exact_int(e['returncode'],0);exact_int(e['pid'],pid);exact_int(e['waited_by_pid'],parent)
 assert get(OUT+role+'-process-original.json')['pid']==pid
 exits[role]=dict(e,original_sha256=pin)
assert exits['native']['utc']<completed['utc']<exits['srun']['utc']<exits['controller']['utc']
srun=get(OUT+'srun-command-original.json')['argv']
assert srun[:8]==['srun','--jobid=3261','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10']
for token in ['CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1']:assert token in srun
command=get(OUT+'native-command-original.json')['argv']
assert command[1]==ROOT+'/tools/final-acceptance-f5c72c942641/inspect_author200_checkpoint_cpu.py'
assert command[command.index('--step')+1]=='25' and command[command.index('--execution-step-id')+1]=='4'
assert command[command.index('--inventory-sha256')+1]==INVENTORY
stdout=get(OUT+'native-stdout-original.txt');assert stdout['output_sha256']==NATIVE and stdout['passed'] is True and stdout['cuda_is_initialized'] is False
for role in ('native','srun','controller','supervisor'):assert files[OUT+role+'-stderr-original.txt'][0]==b''
assert summary['processes']['1533312']['start_ticks']=='1829085316'
assert summary['commands']['source_head']['stdout'].strip()==SOURCE and summary['commands']['source_porcelain']['stdout']==''
print(json.dumps({'version':1,'status':'PASS_ACTUAL_REPAIRED25_CPU_STORED_CONTENT_AND_WAITED_EXITS_NOT_TRAINING_RESTORE_OR_EFFECT',
 'reviewer':'author200_review','utc':datetime.now(timezone.utc).isoformat(),'snapshot':str(SNAP),'snapshot_summary_sha256':summary_sha,'transport_original_sha256':transport_sha,
 'actual_copied_originals_sha_verified':286,'native_report_sha256':NATIVE,'native_report_bytes':len(files[OUT+'native.json'][0]),
 'source_commit':SOURCE,'config_sha256':CONFIG,'training_job_id':'3261','checkpoint_step':25,'tools_manifest_sha256':TOOLS,'inventory_sha256':INVENTORY,'request_sha256':REQUEST,
 'execution_identity':identity,'native_runtime':n['cpu_runtime'],'all271_input_pin_records_joined_to210_closure31_current_members_and_accepted5_anchors':True,
 'raw_statuses_preserved':n['original_raw_statuses'],'clock':n['clock'],'rank_content_summary':rank_summary,'full_HF_all435_keys_all8_rank_slices':True,
 'driver':driver,'actual_small_driver_PT_prior_independent_replay_join':True,'actual_native_srun_controller_waited_exits':exits,
 'old8441_native_srun_controller_failure_exit1_preserved':True,'snapshot_training_driver_still_same_pid':1533312,'snapshot_training_driver_still_same_start_ticks':1829085316,
 'limitations':['Independent reviewer locally joined all286 copied originals and reviewed exact executed inspector/math code; heavy8 native PT/HF shards were read in actual same-node Torch2.4 inspector, not redundantly downloaded or read by reviewer.',
 'New report proves stored435-key model/HF slices,37 Adam states, clocks,RNG,teacher history buffers and driver cursor; raw SERIALIZED status remains unchanged.',
 'No live/new-process restore or named-to-flat FSDP handle reconstruction was performed.',
 'Teacher full parameter arrays are reconstructed from fixed official assets/content digest; saved payload includes history buffers rather than all teacher parameter arrays.',
 'No HF25 whole independent CPU model load, efficacy improvement, final200 evaluation, Slurm terminal or GPU resource-release acceptance claimed.',
 'The old8441 failed inspection remains failure; this separately identified successful repair does not rewrite it.'],
 'blockers':[]},indent=2,allow_nan=False))
