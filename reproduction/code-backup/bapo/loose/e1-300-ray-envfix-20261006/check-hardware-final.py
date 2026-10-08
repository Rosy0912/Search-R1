"""Independent local review of real raw evidence; missing large .pt remain remote."""
from pathlib import Path
import json,hashlib,math,re,ast,collections,types,yaml,copy,subprocess
B=Path(__file__).resolve().parent;ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876';A=None;RUN=None;R=None
def h(p):
 if not p.exists()and A/'repo'in p.parents:p=B.parent/'bapo-e1-300-ray-envfix'/p.relative_to(A/'repo')
 return hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
class ArtifactPath(type(Path())):
 def local(self):return A/Path(str(self)).relative_to(ROOT)
 def read_bytes(self):return self.local().read_bytes()
 def read_text(self,*a,**kw):return self.local().read_text(*a,**kw)
 def open(self,*a,**kw):return self.local().open(*a,**kw)
 def exists(self):return self.local().exists()
 def is_dir(self):return self.local().is_dir()
 def rglob(self,pattern):
  assert self.local().resolve()==self.local()
  for p in self.local().rglob(pattern):
   yield ArtifactPath(str(ROOT/p.relative_to(A)))
 def is_file(self):return self.local().is_file()
 def is_symlink(self):return self.local().is_symlink()
 def stat(self,*a,**kw):return self.local().stat(*a,**kw)
 def resolve(self,*a,**kw):
  assert self.local().resolve()==self.local()
  return self
 def iterdir(self):
  for p in self.local().iterdir():yield self/p.name
def check(archive,job):
 global A,RUN,R
 assert type(job)is int and job>0
 A=Path(archive).resolve();RUN=ROOT/'runs'/f'bapo-instruct300-hardware-{job}';R=A/'runs'/RUN.name
 assert A.is_dir()and R.is_dir()
 assert subprocess.check_output(['git','-C',str(B.parent/'bapo-e1-300-ray-envfix'),'rev-parse','HEAD'],text=True).strip()==SOURCE
 inv=read(R/'actual-fullhash-inventory.json');items=inv['evidence'];assert items and len({i['path']for i in items})==len(items) and inv['job_id']==job and inv['source_commit']==SOURCE and inv['run_root']==str(RUN)
 bound={x['path']:x for x in items};missing=[]
 for x in items:
  p=A/Path(x['path']).relative_to(ROOT)
  assert type(x['bytes'])is int and x['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',x['sha256'])
  if p.exists():assert not p.is_symlink()and p.stat().st_size==x['bytes']and h(p)==x['sha256']
  else:missing.append(x)
 assert {x['path']for x in missing}=={str(RUN/'recovery-native-v3-roundtrip'/f'rank_{i}.pt')for i in range(8)}
 large_bytes=sum(x['bytes']for x in missing);assert large_bytes>0
 s=(R/'slurm-terminal.txt').read_text()
 for k,v in [('JobId',str(job)),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:assert re.search(r'(?:^|\s)'+k+'='+re.escape(v)+r'(?:\s|$)',s)
 assert read(R/'own-process-cleanup.json')['remaining']==[]
 for n in ('source-commit.txt','cpu-gate-source.txt'):assert(R/n).read_text().strip()==SOURCE
 cpu=read(R/'cpu-gate.json');assert cpu['passed']==395 and cpu['failed']==cpu['skipped']==cpu['exit_code']==0 and cpu['sys_prefix']==str(ROOT/'env')and cpu['source_commit']==SOURCE
 assert h(A/'logs/cpu-gate-durable.log')==cpu['log_sha256'];tools=read(R/'tools-manifest.json');assert len(tools['files'])==36 and h(R/'tools-manifest.json')=='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46'
 for n,sha in tools['files'].items():assert h(A/'tools'/n)==sha
 manifest=read(R/'recovery-manifest.json');assert manifest['target_job_id']==job and manifest['target_source_commit']==SOURCE and manifest['target_run_root']==str(RUN)
 c=read(B.parent/'evidence/formal3231-checkpoint25-fullhash-cpu-candidate.json');oldreports={x['rank']:x for x in c['native_reports']}
 # Candidate native_reports is a list of true raw producer reports.
 rawold=B.parent/'evidence/formal3231-fault-raw-20261005T0918/runs/bapo-instruct300-formal300-3231'
 driverold=read(rawold/'checkpoints/full_state/global_step_25/driver-reload.json');algo=read(rawold/'checkpoints/algorithm/global_step_25/state.json');oldcfg=yaml.safe_load((rawold/'resolved-config.yaml').read_text());cfg=yaml.safe_load((R/'resolved-config.yaml').read_text())
 treecfg=ast.parse((B.parent/'bapo-e1-300-ray-envfix/search_r1/baselines/recovery.py').read_text());fn=next(x for x in treecfg.body if isinstance(x,ast.FunctionDef)and x.name=='config_identity');identity={'copy':copy};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-config-identity','exec'),identity);assert identity['config_identity'](cfg)==identity['config_identity'](oldcfg)
 assert cfg['trainer']['resume_verify_only']is True and cfg['trainer']['resume_full_state']==str(RUN/'recovery-manifest.json')
 for cfgx in (cfg,oldcfg):assert cfgx['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1 and cfgx['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfgx['trainer']['total_training_steps']==301 and cfgx['baselines']['bapo']['stage_validation_size']==64
 ranks=[read(R/'recovery-witness'/f'rank_{i}.json')for i in range(8)];d=read(R/'recovery-witness/driver.json');v3s=[]
 for i,w in enumerate(ranks):
  old=oldreports[i];e=next(x for x in manifest['ranks']if x['rank']==i)
  assert w['status']=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'and w['rank']==i and w['world_size']==8 and w['step']==w['scheduler_last_epoch']==25 and w['next_step']==26
  assert w['distributed_backend']=='nccl'and w['model_state_dict_type']=='SHARDED_STATE_DICT'and w['model_dtensor_entries']==old['model_dtensor_entries']==435 and w['optimizer_state_entries']==old['optimizer_state_entries']==37
  assert w['old_pid']==old['pid']and w['new_pid']!=old['pid']and w['state_digest']==old['state_digest']and w['old_state_sha256']==e['state']['sha256']
  assert w['model_class']=='Qwen2ForCausalLM'and w['model_config']==dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,vocab_size=151936,architectures=['Qwen2ForCausalLM'])
  assert all(math.isclose(lr,1e-6*25/114,rel_tol=1e-12)for lr in w['lr'])
  v=read(R/'recovery-native-v3-roundtrip'/f'rank_{i}.json');v3s.append(v)
  assert v['status']=='EXACT_FULL_STATE_RELOAD_OK'and v['native_state_version']==3 and v['roundtrip']is True and v['step']==v['scheduler_last_epoch']==25 and v['rank']==i and v['world_size']==8 and v['pid']==w['new_pid']
  assert v['path']==str(RUN/'recovery-native-v3-roundtrip'/f'rank_{i}.pt')and v['sha256']==bound[v['path']]['sha256']
  assert v['state_digest']==v['reloaded_state_digest']and v['generation_rng_before_digest']==v['generation_rng_after_digest']
  assert v['model_dtensor_entries']==435 and v['optimizer_state_entries']==37
  top=v['generation_rng_topology'];assert top['version']==1 and top['device_mesh']==dict(shape=[8,1],names=['dp','infer_tp'],ranks=[[i]for i in range(8)])
  for name in ('torch_random_states','gen_random_states'):assert type(top[name+'_numel'])is int and top[name+'_numel']>0 and top[name+'_dtype']=='torch.uint8'and top[name+'_shape']==[top[name+'_numel']]
 assert d['status']=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and d['verify_only']is True and d['no_generation_or_optimizer_update']is True and d['old_pid']==driverold['pid']and d['new_pid']!=driverold['pid']and d['loaded_state_digest']==driverold['state_digest']and d['bapo']==algo['bapo']and d['rank_witnesses']==ranks
 assert d['step']==d['actual_outer_updates']==25 and d['next_step']==26 and d['epoch']==0 and d['sampler_cursor']==12800
 for n in ('rng','sampler'):assert d['saved_'+n+'_digest']==d['restored_'+n+'_digest']
 assert len({x['new_pid']for x in ranks+[d]})==9
 for w in ranks+[d]:
  assert w['target_source_commit']==SOURCE and w['target_run_root']==str(RUN)and w['sampling_continuity']=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'
  p=w['process'];assert p['pid']==w['new_pid']and type(p['start_ticks'])is int and p['start_ticks']>0 and p['actual_env']['SLURM_JOB_ID']==str(job)and p['actual_env']['BAPO_RUN_ID']==str(RUN)
  assert w['runtime_module_sha256']==h(A/Path(w['runtime_module_path']).relative_to(ROOT))
 train=(R/'train.log').read_text();assert 'BAPO_TRAIN_METRICS_JSON='not in train and 'E1_STEP5_PAUSED_JSON='not in train
 for marker,expected,sort in [('BAPO_RECOVERY_RANK_JSON=',ranks,True),('BAPO_RECOVERY_DRIVER_JSON=',[d],False),('BAPO_RECOVERY_V3_RANK_JSON=',v3s,True)]:
  rows=[json.JSONDecoder().raw_decode(l.split(marker,1)[1].lstrip())[0]for l in train.splitlines()if marker in l];assert (sorted(rows,key=lambda x:x['rank'])if sort else rows)==expected
 # Execute actual frozen math validation functions with only SHA I/O relocated.
 source=(A/'tools/verify_hardware_receipt.py').read_text();tree=ast.parse(source);funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('finite','validate_native_counts','validate_native_stability')]
 ns={'math':math,'sha':lambda p:h(A/Path(p).relative_to(ROOT))};exec(compile(ast.Module(body=funcs,type_ignores=[]),'frozen-fixture-oracles','exec'),ns)
 for i in range(8):
  ns['validate_native_counts'](read(R/'native-count'/f'rank-{i}.json'),i,SOURCE,ROOT);ns['validate_native_stability'](read(R/'native-stability'/f'rank-{i}.json'),i,SOURCE,ROOT)
  for kind in ('native-actor','native-state','numeric-fixture','flash-native'):
   x=read(R/kind/f'rank-{i}.json');ns['finite'](x);assert x['rank']==i
   if kind=='native-actor':
    assert x['source_commit']==SOURCE and len(x['records'])==62 and collections.Counter(z['case']for z in x['records'])=={'actual_actor_micro_invariance':48,'actual_actor_global_empty_step':2,'actual_actor_2560_rows_20_updates':2,'actual_actor_production_bf16_fp32_micro_invariance':8,'actual_actor_production_bf16_20_updates':2}
    for z in x['records']:
     flat=[abs(t)for g in z['gradients']for t in g];tol=(.03 if z['production_bf16_fp32']else 2e-5)*max(flat,default=0)+(2e-4 if z['production_bf16_fp32']else 2e-7);assert 0<=z['max_gradient_error']<=1.1*tol and z['optimizer_steps']==(20 if z['global_rows']==2560 else 1)
    probe=x['production_forward_precision'];assert probe['status']=='PASS_REAL_ACTOR_FORWARD_PRECISION_PROBE'and len(probe['branch_records'])==2
    for z in probe['branch_records']:assert z['finite_forward_backward']is True and z['nonzero_gradient']is True and z['logits_dtype']=='torch.bfloat16'and z['logprob_dtype']=='torch.float32'and z['gradient_dtypes']==['torch.float32']and z['actual_device']==f'cuda:{i}'
   elif kind=='native-state':
    assert x['world_size']==8 and len(x['records'])==2
    for z in x['records']:
     assert z['native_state_version']==3 and z['next_update_full_state_exact_equal']is True and z['generation_rng_saved_and_restored']is True and z['num_warmup_steps']==114 and z['total_actual_optimizer_calls']==28 and z['oracle_sixth_optimizer_calls']==z['reloaded_sixth_optimizer_calls']==4
     for component in z['next_update_components'].values():assert component['exact_equal']is True and component['no_roundtrip_sha256']==component['reloaded_sha256']
     for first,last in [('initial_snapshot','actual_worker_roundtrip'),('oracle_snapshot','reloaded_snapshot')]:assert z[first]['state_digest']==z[last]['state_digest']
   elif kind=='numeric-fixture':
    assert x['source_commit']==SOURCE and len(x['records'])==22
    for z in x['records']:
     if z['case']=='normal_native_and_fullvector_oracle':assert math.isclose(z['norm'],math.sqrt(5120),rel_tol=1e-12)
     elif z['case']=='empty_one_rank_sharded_gradients':assert math.isclose(z['norm'],math.sqrt(480),rel_tol=1e-12)
     elif z['case']=='finite_gradient_fp32_square_overflow':assert math.isclose(z['norm'],4.525483490288562e21,rel_tol=1e-12)
     elif z['case']=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':assert z['points']==408 and z['max_value_error']<=2e-5 and z['max_grad_error']<=2e-5
   else:assert x['finite_forward_backward']is True and x['max_abs_reference_error']<=.02
 model=read(R/'model-gpu.json');assert model['status']=='REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK'and model['source_commit']==SOURCE and model['token_ids']
 comp=read(R/'retrieval-comparison.json');assert comp['same_asset_identity']is True and comp['asset_identity_verified']is True
 for mode in ('cpu','gpu'):
  x=read(R/f'retrieval-{mode}.json');assert x['asset_identity_verified']is True and x['source']['slurm_job_id']==str(job)and x['source']['code_commit']==SOURCE and x['index_metadata']['d']==768 and x['index_metadata']['ntotal']==21015324 and x['index_metadata']['gpu_shard']is(mode=='gpu')and len(x['queries'])==14
 reserved=inv['slurm_reserved_wall_seconds'];assert type(reserved)is int and reserved>0 and math.isclose(inv['slurm_reserved_gpu_hours'],reserved*8/3600)
 from datetime import datetime
 start=re.findall(r'(?:^|\s)StartTime=([^\s]+)',s);end=re.findall(r'(?:^|\s)EndTime=([^\s]+)',s);assert len(start)==len(end)==1 and (datetime.fromisoformat(end[0])-datetime.fromisoformat(start[0])).total_seconds()==reserved
 summary=read(R/'guard-summary.json');assert summary['samples']>0 and summary['abort_events']==0 and summary['unfinished_recovery']is False
 watches=[json.loads(l)for l in (R/'resource-watch.jsonl').read_text().splitlines()];foreign=set();maxsm=0.;maxmem=0.
 for x in watches:
  for process in x['external_processes']:foreign.add(process['pid'])
  if x['pause']:continue  # Recovery decisions are independently recomputed by the full strict replay below.
  for totals in x['external_totals'].values():
   value=totals['external_sm_percent']if totals['external_sm_percent']is not None else totals['external_sm_upper_bound_percent'];assert value is not None;maxsm=max(maxsm,value);maxmem=max(maxmem,totals['external_memory_percent'])
  for p in x['external_processes']:foreign.add(p['pid'])
 assert maxsm<=15 and maxmem<=15
 assert not foreign.intersection(read(R/'own-process-cleanup.json')['stopped_pids'])
 epochs={p:set()for p in foreign}
 for row in watches:
  for pid in foreign:
   
   if str(pid)not in row['pid_start_ticks']:continue
   tick=row['pid_start_ticks'][str(pid)];assert tick['before']==tick['after'];epochs[pid].add(tick['before'])
 assert all(len(x)==1 for x in epochs.values())
 
 
 # Artifact I/O relocation only: retain remote identity strings in the exact bound functions.
 # Recompute actual raw GPU fixture with all remote bound paths preserved.
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_freeze_scalability')
 ns2={'pathlib':__import__('pathlib'),'json':json,'math':math,'hashlib':hashlib,'sha':lambda p:h(A/Path(str(p)).relative_to(ROOT)),'finite':ns['finite']}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-scale-raw-validator','exec'),ns2)
 scale=ns2['validate_freeze_scalability'](ArtifactPath(str(RUN)),ArtifactPath(str(ROOT)),SOURCE,job,{p:v['sha256']for p,v in bound.items()},summary['allocated_uuids'])
 # Exact reviewed helper replays raw using read-only local file relocation; not a target worker gate.
 helper=types.ModuleType('bound_actual_helper');helper.__file__=str(ROOT/'tools/guard_journal.py');exec(compile((A/'tools/guard_journal.py').read_bytes(),helper.__file__,'exec'),helper.__dict__);helper.Path=ArtifactPath
 import builtins
 def artifact_exec(code,namespace):
  builtins.exec(code,namespace)
  if namespace.get('__file__')==str(ROOT/'tools/resource_watch.py'):namespace['Path']=ArtifactPath
 helper.__dict__['exec']=artifact_exec
 replayed=helper.replay(str(RUN/'resource-watch.jsonl'),job,str(RUN),SOURCE,str(ROOT/'tools/resource_watch.py'))
 assert replayed==summary==inv['guard_summary']
 

 # Independently parse all CPU title and actual production Ray originals.
 def load_bound(path,name):
  remote=Path(str(path));local=A/remote.relative_to(ROOT)
  module=types.ModuleType(name);module.__file__=str(remote);exec(compile(local.read_bytes(),str(remote),'exec'),module.__dict__)
  if 'Path'in module.__dict__:module.Path=ArtifactPath
  if 'pathlib'in module.__dict__:module.pathlib=types.SimpleNamespace(Path=ArtifactPath)
  return module
 identity=load_bound(ROOT/'tools/identity_evidence_validator.py','independent_bound_identity_validator');identity.load_module=load_bound
 hashes={p:v['sha256']for p,v in bound.items()}
 cpuidentity=identity.validate_identity_linux_fixture(ArtifactPath(str(RUN)),ArtifactPath(str(ROOT)),SOURCE,job,hashes,summary['allocated_uuids'])
 rayidentity=identity.validate_production_ray_identity(ArtifactPath(str(RUN)),ArtifactPath(str(ROOT)),SOURCE,job,hashes,summary['allocated_uuids'])
 assert cpuidentity==inv['identity_linux_original_proof']and rayidentity==inv['production_ray_original_proof']
 # Original head IDs, intent and controller hashes are rederived, not inferred from inventory flags.
 import base64
 proof=read(R/'head-terminal-ledger-proof.json');assert set(proof)=={'files'}
 head=ROOT/'scheduler-head';names=('hardware-job-id.txt','hardware-submit-intent.txt','submit.sh','hardware.sbatch','approved-launch-hashes.json',f'hardware-{job}-terminal.txt');assert set(proof['files'])==set(names)
 decoded={}
 for name,x in proof['files'].items():
  raw=base64.b64decode(x['base64'],validate=True);assert x['path']==str(head/name)and type(x['bytes'])is int and len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256'];decoded[name]=raw
 assert decoded['hardware-job-id.txt'].decode().strip()==str(job)and decoded['hardware-submit-intent.txt'].decode().strip()=='hardware'
 assert decoded[f'hardware-{job}-terminal.txt']==(R/'slurm-terminal.txt').read_bytes()
 approved=json.loads(decoded['approved-launch-hashes.json'])
 for name in ('submit.sh','hardware.sbatch'):assert hashlib.sha256(decoded[name]).hexdigest()==approved[name]==tools['files'][name]
 assert inv['terminal_head_ledger_bound']is True
 result=dict(status='PASS_LOCAL_RAW_RECOMPUTATION_NOT_GPU_ACCEPTANCE_OR_PERMISSION',job_id=job,source_commit=SOURCE,run_root=str(RUN),inventory_sha256=h(R/'actual-fullhash-inventory.json'),inventory_count=len(items),small_files_rehashed=len(items)-len(missing),large_v3_files_remote_bound=len(missing),large_v3_bytes_remote_bound=large_bytes,restore_witnesses=9,v3_reports=8,math_fixture_ranks=8,scale=scale,scale_fixture_wall_seconds=read(R/'freeze-scalability-gpu/result.json')['wall_seconds'],scale_freeze_seconds=read(R/'freeze-scalability-gpu/result.json')['freeze_seconds'],guard=replayed,identity_cpu=cpuidentity,production_ray=rayidentity,foreign_pids=sorted(foreign),max_observed_safe_external_sm_percent=maxsm,max_observed_safe_external_memory_percent=maxmem,reserved_wall_seconds=reserved,reserved_gpu_hours=reserved*8/3600,fullhash_wall_seconds=inv['wall_seconds'],fullhash_cpu_seconds=inv['process_cpu_seconds'],actual_gpu_accepted=False,permission_signed=False,target_worker_complete_receipt_gate_required=True)
 return result

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()and ROOT not in out.resolve().parents;r=check(a.archive,a.job_id)
 with out.open('x')as f:f.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps(r))
if __name__=='__main__':main()
