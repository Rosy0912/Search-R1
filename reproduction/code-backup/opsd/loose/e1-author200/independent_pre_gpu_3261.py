"""Independent read-only review of actual3261 originals and current identities.

This program neither imports CUDA nor writes anything remotely nor signs.
"""
import base64, datetime, hashlib, importlib.util, json, re, shlex, subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
RUN=ROOT+'/runs/formal-3261'
SNAP=HERE/'actual-snapshots/3261-20261005T194802469223Z'
REPO=HERE.parent/'opsd-e1-300'
ROLES={'allocation','source','import_closure','data_manifest','model_manifest','tool_manifest','environment'}
PUBLIC='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON');d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def original(name):
    raw=(SNAP/'original-files'/name).read_bytes();rec=observation['files'][name]
    need(sha(raw)==rec['sha256'] and len(raw)==rec['bytes'],'original byte mismatch '+name)
    need(rec['source_offset']==0 and rec['complete_original_at_start'] and rec['same_inode'] and rec['unchanged_during_read'],'incomplete/mutable original '+name)
    need(rec['before']['uid']==rec['after']['uid']==0 and rec['physical_path']==RUN+'/'+name,'unowned or foreign original '+name)
    return unique(raw),sha(raw)

observation=unique((SNAP/'observed-remote-original.json').read_bytes())
summary=unique((SNAP/'observation-summary.json').read_bytes())
need(sha((SNAP/'observed-remote-original.json').read_bytes())==summary['transport_original_sha256'],'transport mismatch')
need(observation['errors']==[] and observation['job_id']=='3261' and observation['reader_uid']==0 and observation['node']=='h100-10','snapshot errors/identity')
for name,rec in observation['files'].items():
    raw=base64.b64decode(rec['base64'],validate=True)
    need(sha(raw)==rec['sha256'] and raw==(SNAP/'original-files'/name).read_bytes(),'transport copied mismatch')
ready,ready_sha=original('approval-ready.json')
binding,binding_sha=original('bindings/manifest.json')
need(ready['source_commit']==SOURCE and ready['job_id']=='3261' and ready['uid']==0 and ready['run_root']==RUN,'ready actual identity')
need(ready['binding_sha256']==binding_sha and binding['source_commit']==SOURCE and binding['run_root']==RUN and set(binding['files'])==ROLES,'seven roles identity')
objects={};hashes={}
for role,rec in binding['files'].items():
    d,pin=original('bindings/'+role+'.json');need(rec=={'path':RUN+'/bindings/'+role+'.json','sha256':pin},'bound role byte/path mismatch')
    objects[role]=d;hashes[role]=pin
pub,pub_sha=original('bindings/public.json')
need(pub_sha==PUBLIC==ready['public_key_sha256'] and pub['algorithm']=='RSA-SHA256' and pub['e']==65537,'public key mismatch')
source=objects['source'];spec=unique((HERE/'runtime-tools-bound-188f4e5e7b63.json').read_bytes())
need(source['source_commit']==SOURCE and source['repo']==ROOT+'/repo-'+SOURCE[:12] and source['clean'] is True,'source identity')
formal=unique((HERE/'formal-bb5f4872142f/manifest.json').read_bytes())
need(source['source_files']==formal['source_files'] and len(source['source_files'])==16 and source['formal_manifest_sha256']=='bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c','source bundle binding')
for name,pin in source['source_files'].items():
    raw=subprocess.check_output(['git','-C',str(REPO),'show',SOURCE+':'+name]);need(sha(raw)==pin,'committed source differs '+name)
closure=objects['import_closure']['modules'];proof,proof_sha=original('evidence/pre-gpu-import-check.json')
need(len(closure)==210 and proof['module_count']==83 and proof['function_code_count']==689 and proof['cuda_is_initialized'] is False,'actual native counts/CUDA')
need(proof['job_id']=='3261' and proof['source_commit']==SOURCE and proof['repo']==source['repo'] and proof['uid']==0 and proof['closure_sha256']==hashes['import_closure'],'actual native identity')
need(len(proof['actual_loaded_modules']['modules'])==83,'actual module count differs')
for name,rec in proof['actual_loaded_modules']['modules'].items():need(closure[name]==rec,'actual loaded origin differs')
for name,rec in closure.items():
    if name.startswith(('search_r1.','verl.','reproduction.')):
        rel=Path(rec['path']).relative_to(source['repo']);raw=subprocess.check_output(['git','-C',str(REPO),'show',SOURCE+':'+str(rel)]);need(sha(raw)==rec['sha256'],'native project source mismatch '+name)
profile,profile_sha=original('evidence/pre-gpu-profile.json')
a=profile['actor_rollout_ref']['actor'];roll=profile['actor_rollout_ref']['rollout'];d=a['distillation'];cfg=profile['baselines']['opsd_e1']
need(a['baseline_method']=='opsd' and a['use_kl_loss'] is False and a['kl_loss_coef']==0 and a['entropy_coeff']==0 and profile['algorithm']['kl_ctrl']['kl_coef']==0,'pure OPSD objective')
need(a['ppo_mini_batch_size']==32 and a['ppo_micro_batch_size']==16 and a['ppo_epochs']==1 and a['grad_clip']==.1 and a['optim']['lr']==5e-6 and a['optim']['lr_warmup_steps']==0 and a['optim']['warmup_style']=='constant','new optimizer config')
need(roll['n']==roll['n_agent']==roll['tensor_model_parallel_size']==1 and roll['temperature']==1.1 and roll['top_p']==.95 and roll['top_k']==20 and roll['log_prob_micro_batch_size']==32,'new rollout config')
need(d['component_clip']==.05 and d['beta']==0 and d['temperature']==1 and d['teacher_prompt_mode']=='user_context' and d['teacher_device']=='cuda','OPSD teacher/component config')
modelroot='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
need(d['teacher_path']==profile['actor_rollout_ref']['model']['path']==modelroot,'new official initialization')
need(profile['trainer']['total_training_steps']==200 and profile['trainer']['nnodes']==1 and profile['trainer']['n_gpus_per_node']==8 and profile['trainer']['save_freq']==25 and profile['trainer']['test_freq']==50 and profile['trainer']['val_before_train'] is True,'horizon/checkpoints')
need(profile['data']['train_batch_size']==profile['data']['val_batch_size']==512 and profile['max_turns']==4 and profile['data']['max_prompt_length']==4096 and profile['data']['max_start_length']==2048 and profile['data']['max_response_length']==profile['data']['max_obs_length']==500,'data/budget')
need(cfg['profile']=='author200' and cfg['gate_required'] is True and cfg['acceptance_step']==5 and cfg['binding_manifest_sha256']==binding_sha and cfg['run_root']==RUN and cfg['stage_num']==64 and cfg['stage_freq']==50,'actual first5 gate config')
need(profile['retriever']=={'url':'http://127.0.0.1:18084/retrieve','topk':3},'same node loopback retrieval')
data=objects['data_manifest'];need(hashes['data_manifest']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','common manifest')
need({k:v['rows'] for k,v in data['splits'].items()}=={'train':169028,'validation':512,'stage':64,'final':51713},'common split counts')
lines=(SNAP/'original-files/formal-wrapper.log').read_text().splitlines()
loaded=unique(next(x.split('=',1)[1] for x in lines if x.startswith('BASELINE_PREBUILT_MANIFEST_JSON=')))
need(loaded['manifest']==data and all(v=={'ids':0,'questions':0} for v in loaded['overlaps'].values()),'actual common identity/overlap check')
model=objects['model_manifest'];need(model['repo']=='Qwen/Qwen2.5-3B-Instruct' and model['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1' and {v['file'] for v in model['files']}=={'LICENSE','README.md','config.json','generation_config.json','merges.txt','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json'},'official model manifest')
tools=objects['tool_manifest'];need(tools['status']=='FULL_TOOL_ASSET_CONTENT_VERIFIED' and tools['hostname']=='h100-10' and len(tools['results']['e5'])==8,'tool manifest')
env=objects['environment']['values'];need(env['PYTHONPATH']==source['repo'] and env['SEARCH_R1_REPO']==source['repo'] and env['SEARCH_R1_BASELINE_RUN_ROOT']==RUN and env['OPSD_E1_RETRIEVER_URL']=='http://127.0.0.1:18084/retrieve' and env['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7','actual wrapper environment')
need(env['OPSD_E1_BINDING_MANIFEST_SHA256'] is None,'environment self hash should be pre-binding null')
allocation=objects['allocation'];need(allocation['wrapper_process']==ready['process'] and allocation['job_id']=='3261' and allocation['uid']==0 and allocation['hostname']=='h100-10','allocation identity')
for snap in allocation['nvml_double_guard']:
    need(snap['applications']==[] and len(snap['inventory'])==8 and {v['uuid'] for v in snap['inventory']}==set(ready['nvml_uuids']) and all(v['used']==v['util']==0 for v in snap['inventory']),'original allocation double idle')
current=observation['processes'][str(ready['pid'])]
need(current['start_ticks']==ready['process_start_ticks'] and current['uids']==[0,0,0,0] and current['state'] in ['R','S','D','I'] and current['cgroup']==ready['process']['cgroup'],'snapshot actual same live process')

# Independent fresh observer deliberately does not reuse publisher validation.
literal={'run':RUN,'ready_sha':ready_sha,'pid':ready['pid'],'start':ready['process_start_ticks'],'binding':binding_sha,'source':SOURCE,'roles':hashes,'uuids':ready['nvml_uuids']}
code='D='+repr(literal)+'\n'+r'''
import csv,hashlib,io,json,os,pathlib,re,socket,subprocess,time,datetime
def need(ok,why):
 if not ok:raise ValueError(why)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=pathlib.Path(D['run']);need(os.getuid()==0 and socket.gethostname()=='h100-10','actual wrong node')
need(h(run/'approval-ready.json')==D['ready_sha'] and h(run/'bindings/manifest.json')==D['binding'],'actual original changed')
for role,pin in D['roles'].items():need(h(run/'bindings'/ (role+'.json'))==pin,'actual bound original changed')
p=pathlib.Path('/proc')/str(D['pid']);status=(p/'status').read_text();cg=(p/'cgroup').read_text();st=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().split(b'\0')
need(re.search(r'^Uid:\s+0\s+0\s+0\s+0$',status,re.M) is not None and st[19]==D['start'] and st[0] in ['R','S','D','I'] and '/job_3261/' in cg,'actual current PID foreign or stale')
need(b'--prepare' in argv and (D['run'].split('/runs/')[0]+'/tools/formal-bb5f4872142f/formal_supervisor.py').encode() in argv,'actual current argv')
def c(args):return subprocess.check_output(args,text=True,timeout=20)
slurm=c(['scontrol','show','job','3261','-o']);kv=dict(x.split('=',1) for x in slurm.split() if '=' in x)
need(kv['JobState']=='RUNNING' and kv['NodeList']=='h100-10' and kv['TresPerNode']=='gres:gpu:h100:8' and kv['OverSubscribe']=='NO' and kv['NumCPUs']=='192' and kv['MinMemoryNode']=='1T' and kv['Restarts']=='0','current allocation changed')
repo=pathlib.Path(D['run'].split('/runs/')[0])/'repo-188f4e5e7b63';need(c(['git','-C',str(repo),'rev-parse','HEAD']).strip()==D['source'] and c(['git','-C',str(repo),'status','--porcelain'])=='','current source changed')
reads=[]
for i in range(2):
 inv=c(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']);apps=c(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']);rows=list(csv.reader(io.StringIO(inv)))
 need(apps=='' and len(rows)==8 and {r[1].strip() for r in rows}==set(D['uuids']) and all('H100' in r[2] and float(r[3])==float(r[5])==0 for r in rows),'current unfiltered GPU not idle')
 reads.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory_original':inv,'applications_original':apps})
 if i==0:time.sleep(2)
print(json.dumps({'scope':'INDEPENDENT_READ_ONLY_CURRENT_3261_INITIALIZATION_IDENTITY_ONLY','uid':os.getuid(),'node':socket.gethostname(),'pid':D['pid'],'start':st[19],'status_original':status,'cgroup_original':cg,'argv':[x.decode() for x in argv],'slurm_original':slurm,'nvml_double_reads':reads},sort_keys=True))
'''
spec_loader=importlib.util.spec_from_file_location('independent_transport',HERE/'transport.py');t=importlib.util.module_from_spec(spec_loader);spec_loader.loader.exec_module(t)
result=t.remote('python3 -c '+shlex.quote(code),timeout=60)
with (HERE/'independent-3261-pre-gpu-live-original.json').open('xb') as f:f.write(result.stdout)
with (HERE/'independent-3261-pre-gpu-live-original.stderr').open('xb') as f:f.write(result.stderr)
live=unique(result.stdout)
review={'status':'ACTUAL_PRE_GPU_INDEPENDENT_REVIEW_PASS_INITIALIZATION_ONLY','source':SOURCE,'job_id':'3261','wrapper_pid':ready['pid'],'process_start_ticks':ready['process_start_ticks'],'nonce':ready['nonce'],'ready_sha256':ready_sha,'binding_sha256':binding_sha,'public_key_sha256':PUBLIC,'native_closure_sha256':hashes['import_closure'],'all_seven_binding_hashes':hashes,'uuids':ready['nvml_uuids'],'snapshot_transport_original_sha256':summary['transport_original_sha256'],'independent_live_original_sha256':sha(result.stdout),'actual_profile_sha256':profile_sha,'actual_import_proof_sha256':proof_sha,'native_inventory_count':210,'actual_loaded_module_count':83,'actual_function_code_count':689,'cuda_initialized_during_import':False,'actual_completed_outer_iterations':0,'first5_acceptance':False,'scope':'Only observed3261 pre-GPU initialization eligibility. Full original model/data/tool bytes have been verified by the source-bound supervisor before ready. Fresh8rank GPU path/actual16Adam per rank/fullstate/HF/reload/behavior remain future first5 gates. No efficacy or training initialization claim.','limitations':['sacct accounting storage disabled; actual scheduler scontrol was used.','Environment binding intentionally predates its own binding SHA; resolved profile carries the actual binding SHA.','Raw .05 component clipping matches the newly authorized algorithm but can reverse KL gradient directions; no convergence guarantee.']}
with (HERE/'ACTUAL3261_PRE_GPU_INDEPENDENT_REVIEW.json').open('x') as f:json.dump(review,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':review['status'],'review_sha256':sha((HERE/'ACTUAL3261_PRE_GPU_INDEPENDENT_REVIEW.json').read_bytes()),'live_sha256':sha(result.stdout)},sort_keys=True))
