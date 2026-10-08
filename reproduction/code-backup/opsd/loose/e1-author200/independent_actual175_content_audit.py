"""Independent local joins of actual CPU175 originals and new90 observer frame, without remote actions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, re, types
B=Path(__file__).resolve().parent
SNAP=B/'actual-snapshots/3261-20261006T040534209402Z'
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
RUN=ROOT+'/runs/formal-3261'
OUT='evidence/checkpoint-175-cpu-inspection-once-v1/'
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
NATIVE='53d7b0d96689a611033141f19ea7776729cb93dc63e2abbd3145cf8a3395fc05'
INSPECTOR='f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a'
TOOLS='90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
INVENTORY='427a0ae33fedc030316903f8e5e651addd191c99e822aca005e8a6a5c49650d7'
REQUEST='ebd297533e5d30c4570e06ada9d8e36b336ca35169fb6acd01873847927a91c4'
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
assert len(files)==550
CPU175=B/'actual175-cpu-output-originals-20261006T040923678349Z'
capsule_raw,capsule_sha=raw(CPU175/'transport-original.json','c3d142243a49f5343ea26499d695d1585cc1231b4c2ca3f4a9bf620c49118396')
capsule=load(capsule_raw);capsule_summary_raw,capsule_summary_sha=raw(CPU175/'summary.json');capsule_summary=load(capsule_summary_raw)
import base64
stripped=dict(capsule);stripped['files']={k:{f:v for f,v in row.items() if f!='base64'} for k,row in capsule['files'].items()}
assert stripped==capsule_summary and len(capsule['files'])==29
assert capsule['scope']=='ACTUAL175_CPU_OUTPUT_ORIGINALS_NOT_CONTENT_ACCEPTANCE'
for label in ['allocation_before','allocation_after']:
 allocation=capsule[label];p=allocation['driver']
 assert p['present'] is True and p['pid']==1533312 and p['start_ticks']=='1829085316' and p['uids']==[0]*4
 fields=dict(t.split('=',1) for t in allocation['scontrol_original'].split() if '=' in t)
 assert (fields['JobId'],fields['JobState'],fields['Restarts'],fields['NodeList'])==('3261','RUNNING','0','h100-10')
for name,rec in capsule['files'].items():
 p=CPU175/'original-files'/name;r,h=raw(p,rec['sha256'])
 assert r==base64.b64decode(rec['base64'],validate=True) and len(r)==rec['bytes']
 assert rec['physical_path']==RUN+'/'+name and rec['uid']==0 and rec['unchanged_during_read'] is True
 assert name=='evidence/checkpoint-cpu-175-launch.json' or name.startswith(OUT)
 files[name]=(r,h)
assert raw(CPU175/'transport-original.stderr')[0]==b''
def get(name,pin=None):
 r,h=files[name];assert pin is None or h==pin,(name,h);return load(r)
n=get(OUT+'native.json',NATIVE)
inventory=get(OUT+'observation/observed-checkpoint-inventory.json',INVENTORY)
prior=get('evidence/first5-cpu-inspection-once-v1/native.json','b50c6ed9995f6148ac250e75f20721bafc95d952f4d841c24542dce31f4adcb2')
tools_raw,_=raw(B/'final-acceptance-90c2695a7f46/manifest.json',TOOLS);tools=load(tools_raw)
raw(B/'final-acceptance-90c2695a7f46/inspect_author200_checkpoint_cpu.py',INSPECTOR)
assert n['scope']=='NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION' and n['passed'] is True
exact_int(n['version'],1);exact_int(n['checkpoint_step'],175)
assert n['source_commit']==SOURCE and n['config_sha256']==CONFIG and n['job_id']=='3261' and n['run_root']==RUN
assert n['inspector_source_sha256']==tools['tools']['inspect_author200_checkpoint_cpu.py']==INSPECTOR
assert n['immutable_math_helper_sha256']=='e949ebab98a49b12ac67fd6531089ef34513e276f169d9fd240c6aaccf623db4'
assert n['inventory_sha256']==INVENTORY and n['inventory_path']==RUN+'/'+OUT+'observation/observed-checkpoint-inventory.json'
assert n['cuda_is_initialized'] is False and n['cpu_runtime']['torch_version']=='2.4.0+cu121'
exact_int(n['native_closure_module_count'],210);exact_int(n['actual_import_function_count'],89)
assert n['native_import_closure_sha256']=='3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
assert n['formal_manifest_sha256']=='bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c'
assert n['clock']==dict(adam_steps=2800,cursor=89600,epoch=0,global_step=175,lr=5e-6,next_global_step=176,scheduler_last_epoch=175,scheduler_step_count=176,teacher_cuda_updates=175)
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
index=get('checkpoints/actor/global_step_175/model.safetensors.index.json')
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
official=get('bindings/model_manifest.json','7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44')
assert official['repo']=='Qwen/Qwen2.5-3B-Instruct' and official['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
base_dir='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
base_pins={base_dir+'/'+r['file']:r['sha256'] for r in official['files']}
base_pins[base_dir+'/VERIFIED-MODEL-MANIFEST.json']=files['bindings/model_manifest.json'][1]
assert len(base_pins)==12
for path,h in n['all_files'].items():
 digest(h)
 if path in tools['dependencies']:assert tools['dependencies'][path]==h
 elif path in base_pins:assert base_pins[path]==h
 elif path==RUN+'/'+OUT+'observation/observed-checkpoint-inventory.json':assert h==INVENTORY
 elif path in {r['physical_path'] for r in inventory['files'].values()}:pass
 else:raise AssertionError('Unjoined input pin: '+path)
for path,h in base_pins.items():assert n['all_files'][path]==h
rank_summary=[]
for rank,r in enumerate(n['ranks']):
 old=prior['ranks'][rank];rj=get(f'checkpoints/actor/global_step_175/training_state/rank_{rank}.json');assert rj==inventory['rank_reports'][rank]
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
 assert r['optimizer_state_entries']==37 and r['optimizer_step_values']==[2800] and len(r['optimizer_moments'])==37
 assert [(a['parameter_id'],a['shape'],a['dtype']) for a in r['optimizer_moments']]==[(a['parameter_id'],a['shape'],a['dtype']) for a in old['optimizer_moments']]
 for opt in r['optimizer_moments']:
  exact_int(opt['step'],2800);assert opt['dtype']=='torch.float32';digest(opt['exp_avg_sha256']);digest(opt['exp_avg_sq_sha256'])
 assert r['scheduler']=={'last_epoch':175,'step_count':176,'base_lrs':[5e-6],'last_lr':[5e-6]}
 rng=r['rng_content'];assert rng['python_state_valid'] is True and rng['numpy_algorithm']=='MT19937'
 digest(rng['numpy_state_sha256']);digest(rng['torch_cpu_state_sha256'])
 assert [x['device'] for x in rng['cuda']]==[x['device'] for x in old['rng_content']['cuda']]==[0]
 for x in rng['cuda']:exact_int(x['bytes'],16);digest(x['sha256'])
 teacher=r['teacher'];assert teacher['content_digest']==n['base_named_parameter_digest']==prior['base_named_parameter_digest']=='af4d2590defbcfa8deef89518f09ed5e368aef244d043b3ee20ca28624ada159'
 exact_int(teacher['cuda_update_entries'],175);exact_int(teacher['counters']['cuda_update_entries'],175)
 assert teacher['buffer_content_digest']==rj['teacher']['buffer_content_digest']
 assert teacher['history_caches']==rj['teacher']['caches'] and len(teacher['history_caches'])==37
 assert teacher['direct_saved_teacher_parameter_arrays'] is False and teacher['fixed_official_assets_and_content_digest'] is True
 for value in teacher['counters'].values():assert type(value) in (int,float) and math.isfinite(value) and value>=0
 rank_summary.append({'rank':rank,'pid':r['pid'],'state_sha256':r['state_sha256'],'local_unique_parameters':count,'changed_vs_official_fp32':changes,'full_HF_slice_keys_checked':435,'Adam_states':37,'Adam_step':2800,'scheduler_last_epoch':175,'RNG_cuda_device':0,'teacher_cuda_updates':175})
assert len(rank_summary)==8 and len({r['pid'] for r in rank_summary})==8
driver=n['driver'];dj=get('full_state/global_step_175/driver.json');assert dj==inventory['driver_report']
assert driver['raw_saved_status']==dj['status']=='DRIVER_STATE_SERIALIZED_OK' and driver['pid']==dj['pid']==1533312
assert driver['state_sha256']==dj['state_sha256']==inventory['files']['driver/driver.pt']['sha256']
assert driver['state_digest']==dj['digest'];digest(driver['state_digest']);digest(driver['sampler_digest']);digest(driver['algorithm_digest'])
assert driver['sampler']=={'consumed_cursor':89600,'epoch':0,'sampler_cursor':89600,'seed':None,'shuffle':True,'size':169028,'unchanged_epoch0_generator':True,'unchanged_epoch0_order':True}
assert driver['rng_content']['python_state_valid'] is True and driver['rng_content']['cuda']==[]
digest(driver['rng_content']['torch_cpu_state_sha256']);digest(driver['rng_content']['numpy_state_sha256'])
algorithm=get('checkpoints/algorithm/global_step_175/state.json')
assert algorithm['global_step']==175 and algorithm['method']=='pure_opsd' and algorithm['outcome_rewards_train_policy'] is False
assert algorithm['distillation']==actor['distillation']
completed=get(OUT+'inspection-completed-review-required.json','f1bb25e6b1b6ac34b4942c2f61ab02a3e7e0b93232f65abaf4543c576726400f')
assert completed['independent_acceptance_pending'] is True and completed['native_report_sha256']==NATIVE and completed['source_commit']==SOURCE
assert completed['request_sha256']==REQUEST and completed['inventory_sha256']==INVENTORY and completed['checkpoint_step']==175
assert completed['whole_hf200_report_sha256'] is None
for name,h in completed['files'].items():assert files[OUT+name][1]==h,(name,h)
request=get('evidence/checkpoint-cpu-175-launch.json',REQUEST)
assert request['tools_manifest_sha256']==TOOLS and request['training_job_id']=='3261' and request['checkpoint_step']==175 and request['source_commit']==SOURCE
assert request['output_directory']==RUN+'/'+OUT.rstrip('/')
prelaunch=load(raw(B/'actual3261-checkpoint175-framing-cpu-start-once-v1/prelaunch-prefixes-original.json','75932c5a785471f69b90edf553cbf64e7dd07b1316a374498ceababe861a8c5c')[0])
assert request['log_prefixes']==prelaunch['log_prefixes']
cp=B/'final-acceptance-90c2695a7f46/author200_observation.py'
cr,_=raw(cp,'9360f3e80b8525a7291c649fe60f38a4e283b087db4baa482276447915055b5e')
common=types.ModuleType('independent_actual175_common');common.__file__=str(cp);exec(compile(cr,str(cp),'exec',dont_inherit=True),common.__dict__)
traw,_=raw(B/'review_progress_typed.py','d9c1e303b32e74c747bf9158aefdbc5bb173d22df49eae14ab44d12a69af2eca')
typed=types.ModuleType('independent_actual175_typed_context');typed.__file__=str(B/'review_progress_typed.py');exec(compile(traw,typed.__file__,'exec',dont_inherit=True),typed.__dict__)
tgate,tcommon,tmanifest,trecord=typed.runtime_helpers();captured_context,_=typed.source_context(SNAP,summary,tgate,tcommon,tmanifest,trecord)
fsdp=inventory['log_join']['warning_source_bindings']['fsdp'];local= captured_context['fsdp_warning_source']
assert fsdp['kind']==local['kind']=='EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE'
assert fsdp['native_closure_sha256']==local['native_closure_sha256']==n['native_import_closure_sha256']
assert fsdp['torch_anchor']==local['torch_anchor'] and fsdp['warning_line']==689 and fsdp['warning_source_line']==local['warning_source_line']
for k in ['physical_path','sha256','bytes']:assert fsdp['source_record'][k]==local['source_record'][k]
packet=load(raw(B/'actual-fsdp-warning-source-20261005T235824197037Z/summary.json','45dc1263d876c9614bd8015742d6a97322601fd48aa663330b7687f05e717ddc')[0])
for k in ['device','inode','uid']:assert fsdp['source_record'][k]==packet['files']['fully_sharded_data_parallel.py']['before'][k]
assert fsdp['external_tools_manifest']['sha256']==TOOLS and fsdp['source_record']['sha256']=='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
assert fsdp['source_record']['physical_path'] in tools['dependencies'] and tools['dependencies'][fsdp['source_record']['physical_path']]==fsdp['source_record']['sha256']
assert fsdp['source_record']['physical_path'] not in n['all_files']
context={'gate':gate,'workers':inventory['accepted5']['workers'], 'checkpoint_warning_path':Path(closure['modules']['torch.utils._contextlib']['path']).parent/'checkpoint.py','fsdp_warning_source':fsdp}
assert context['workers']==captured_context['workers']
streams={}
for role,name in [('driver','ray-stdout-pid-1533312.out'),('training','train.log')]:
 spec=request['log_prefixes'][role];captured=files[name][0];prefix=captured[:spec['bytes']]
 assert len(prefix)==spec['bytes'] and prefix.endswith(b'\n') and hashlib.sha256(prefix).hexdigest()==spec['sha256']
 rec=inventory[role+'_prefix'];assert rec['sha256']==spec['sha256'] and rec['bytes']==spec['bytes'] and rec['physical_path']==spec['path']
 assert summary['files'][name]['source_offset']==0 and summary['files'][name]['complete_original_at_start'] is True and summary['files'][name]['unchanged_during_read'] is True
 streams[role]=common.parse_records(prefix,context)
fixed_join=common.join_log_prefixes(streams['driver'],streams['training'],context,175)
assert fixed_join==inventory['log_join'] and fixed_join['authoritative_last_outer']==175 and fixed_join['aggregate_last_outer']==175
coverage=common.check_evaluation_sequence(streams['driver'],175)
assert [(e['kind'],e['step'],e['count']) for e in coverage['events']]==[('dev',0,512),('stage',5,64),('dev',50,512),('stage',50,64),('dev',100,512),('stage',100,64),('dev',150,512),('stage',150,64)]
frames=fixed_join['interleaved_frames']['aggregate']
assert len(frames)==1 and frames[0]['record_sha256']=='2653535246fd9c4d4fe193fbc58c4f5502822c1cf7b189f0d40021f6cafb92a2'
assert fixed_join['interleaved_frames']['driver']==[] and frames[0]['source_binding']==fsdp
assert inventory['common_helper_sha256']=='9360f3e80b8525a7291c649fe60f38a4e283b087db4baa482276447915055b5e'
assert inventory['observer_source_sha256']==tools['tools']['observe_author200_checkpoint.py']=='77f7099ee028ef2cc2b8fa0875f00940663265ec2285337e7900a5b946c38f7a'
assert inventory['dataset_bindings']['manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert {role:info['manifest_rows'] for role,info in inventory['dataset_bindings']['actual_files'].items()}=={'train':169028,'validation':512,'stage':64,'final':51713}
cpu=get(OUT+'cpu-step-original.json');ctx=cpu['execution_context'];identity=n['execution_identity'];proc=get(OUT+'native-process-original.json')
assert cpu['checkpoint_step']==175 and cpu['request_sha256']==REQUEST
assert identity['node']==ctx['hostname']=='h100-10' and identity['uid']==0 and identity['uids']==[0]*4
assert identity['execution_job_id']==ctx['job_id']=='3261' and identity['execution_step_id']==ctx['step_id']=='10'
assert ctx['cuda_visible_devices']=='' and identity['stored_training_job_id']=='3261' and identity['python_executable']==ctx['python_executable']
exact_int(identity['pid'],2253706);exact_int(identity['start_ticks'],1831985697)
assert identity['cgroup']==proc['process']['cgroup_original']==ctx['process']['cgroup_original']
assert '/job_3261/step_10\n' in identity['cgroup'] and proc['pid']==identity['pid'] and proc['process']['start_ticks']==str(identity['start_ticks'])
exits={}
for role,pid,parent,key,pin in [('observer',2253197,2253196,'waited_by_pipeline_pid','361e6e602194d42f84590285caf6449c3d3806ed4bc7aed2133dc9217b78d706'),('native',2253706,2253196,'waited_by_pipeline_pid','4fbb8a4f2415bb688ee0469ccd2f637dc5b0ea47e088be1b67f355a0d4e119a6'),('srun',2253182,2253177,'waited_by_controller_pid','f6e04317bfe69113449c3ed7ad6e0e1905cae2ad0b98e6c30eb0fc40673f5b4c'),('controller',2253177,2253170,'waited_by_supervisor_pid','4617d343a5c48335f7362eb12488f97b8ae78f3fd83c0dbac8f709660e47e3c5')]:
 e=get(OUT+role+'-exit-original.json',pin);exact_int(e['returncode'],0);exact_int(e['pid'],pid);exact_int(e[key],parent)
 if role in ('observer','native'):assert get(OUT+role+'-process-original.json')['pid']==pid
 exits[role]=dict(e,original_sha256=pin)
assert exits['observer']['utc']<exits['native']['utc']<exits['srun']['utc']<exits['controller']['utc']
# Controller and supervisor command originals bind the actual wait hierarchy.
controller=get(OUT+'controller-start-original.json');supervisor=get(OUT+'supervisor-start-original.json')
assert controller['process']['pid']==2253177 and supervisor['process']['pid']==2253170
assert supervisor['utc']<controller['utc']<exits['observer']['utc']
srun=controller['command']
assert srun[:8]==['srun','--jobid=3261','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10']
for token in ['CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1']:assert token in srun
command=get(OUT+'native-command-original.json')['argv']
assert command[1]==ROOT+'/tools/final-acceptance-90c2695a7f46/inspect_author200_checkpoint_cpu.py'
assert command[command.index('--step')+1]=='175' and command[command.index('--execution-step-id')+1]=='10'
assert command[command.index('--inventory-sha256')+1]==INVENTORY
stdout=get(OUT+'native-stdout-original.txt');assert stdout['output_sha256']==NATIVE and stdout['passed'] is True and stdout['cuda_is_initialized'] is False
for role in ('observer','native','srun','controller','supervisor'):assert files[OUT+role+'-stderr-original.txt'][0]==b''
for name in ('native-exit-original.json','srun-exit-original.json','controller-exit-original.json'):
 assert get('evidence/checkpoint-25-cpu-inspection-once-v1/'+name)['returncode']==1
assert summary['processes']['1533312']['start_ticks']=='1829085316'
assert summary['commands']['source_head']['stdout'].strip()==SOURCE and summary['commands']['source_porcelain']['stdout']==''
print(json.dumps({'version':1,'status':'PASS_ACTUAL175_CPU_STORED_CONTENT_AND_WAITED_EXITS_NOT_TRAINING_RESTORE_OR_EFFECT',
 'reviewer':'author200_review','utc':datetime.now(timezone.utc).isoformat(),'snapshot':str(SNAP),'snapshot_summary_sha256':summary_sha,'transport_original_sha256':transport_sha,
 'earlier_inflight_originals_sha_verified':550,'new_complete_CPU_output_originals_sha_verified':29,'new_CPU_capsule_sha256':capsule_sha,'new_CPU_capsule_summary_sha256':capsule_summary_sha,'native_report_sha256':NATIVE,'native_report_bytes':len(files[OUT+'native.json'][0]),
 'source_commit':SOURCE,'config_sha256':CONFIG,'training_job_id':'3261','checkpoint_step':175,'tools_manifest_sha256':TOOLS,'inventory_sha256':INVENTORY,'request_sha256':REQUEST,
 'execution_identity':identity,'native_runtime':n['cpu_runtime'],'all271_input_pin_records_joined_to210_closure31_current_members_and_accepted5_anchors':True,
 'raw_statuses_preserved':n['original_raw_statuses'],'clock':n['clock'],'rank_content_summary':rank_summary,'full_HF_all435_keys_all8_rank_slices':True,
 'driver':driver,'fixed_actual_prefixes_replayed_from_later_originals':True,'fixed_prefix_join':fixed_join,'evaluation_coverage':coverage,'extra_FSDP_source_provenance':fsdp,'extra_source_not_falsely_added_to_native271':True,
 'actual_observer_native_srun_controller_waited_exits':exits,
 'old8441_native_srun_controller_failure_exit1_preserved':True,'snapshot_training_driver_still_same_pid':1533312,'snapshot_training_driver_still_same_start_ticks':1829085316,
 'limitations':['Independent reviewer locally joined all400 earlier in-flight copied originals plus29 later completed actualCPU originals and reviewed exact executed inspector/math code; heavy8 native PT/HF shards were read in actual same-node Torch2.4 inspector, not redundantly downloaded or read by reviewer.',
 'New report proves stored435-key model/HF slices,37 Adam states, clocks,RNG,teacher history buffers and driver cursor; raw SERIALIZED status remains unchanged.',
 'No live/new-process restore or named-to-flat FSDP handle reconstruction was performed.',
 'Teacher full parameter arrays are reconstructed from fixed official assets/content digest; saved payload includes history buffers rather than all teacher parameter arrays.',
 'No HF100 whole independent CPU model load, efficacy improvement, final200 evaluation, Slurm terminal or GPU resource-release acceptance claimed.',
 'The old8441 failed inspection remains failure; successful25 repair and separate100 check do not rewrite it.'],
 'blockers':[]},indent=2,allow_nan=False))
