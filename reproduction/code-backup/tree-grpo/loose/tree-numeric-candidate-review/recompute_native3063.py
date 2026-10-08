from pathlib import Path
import json,hashlib,math,struct
R=Path(__file__).resolve().parents[2];D=Path(__file__).parent;P=R/'outputs/evidence/tree-native-numeric-3063-bundle.json';x=json.loads(P.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();f=x['files']
for n,v in f.items():assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'] and len(v['text'].encode())==v['bytes'],n
m=R/'work/tree-native-numeric-296217a/manifest.json';assert x['manifest_sha256']==sha(m)=='f1d34d7fec865f2c0ccb3c572b3cd17a896c295ca9fcd702c52070b63b01285e'
result=json.loads(f['result.json']['text']);assert result['status']=='NATIVE_FIXTURES_COMPLETE_PENDING_REVIEW' and result['code_commit']=='296217a4b676a26a92fc729f5d9882b785dbfa96' and result['manifest_sha256']==sha(m) and result['gpu_exit_code']==0 and result['cleanup_residual']=={} and result['guard_healthy_at_fixture_end']
assert (result['allocated_cpus'],result['requested_cpus'],result['allocated_gpus'])==(192,32,8)
raw=f['allocation.txt']['text'];assert all(s in raw for s in ['JobState=RUNNING','NodeList=h100-9','OverSubscribe=NO','TresPerNode=gres:gpu:8','CPUs/Task=32'])
records=[json.loads(f[f'gpu-records/rank-{i}.json']['text']) for i in range(8)];assert records==result['records'];storage=[]
for i,v in enumerate(records):
 assert v['rank']==i and v['world']==8 and v['status']=='PASS' and v['torch']=='2.4.0+cu121' and v['gradient_norm_file']==str(Path(x['run']).parents[1]/'candidates/tree-native-numeric-296217a/repo/search_r1/baselines/gradient_norm.py') and v['sys_prefix']==str(Path(x['run']).parents[1]/'env')
 assert len(v['records'])==20 and len(v['diagnostics_records'])==2
 expected={'normal_native_and_fullvector_oracle':math.sqrt(5120),'finite_gradient_fp32_square_overflow':struct.unpack('f',struct.pack('f',2e20))[0]*math.sqrt(512),'empty_one_rank_sharded_gradients':math.sqrt(480),'pure_NO_SHARD_all_empty_gradients':0.}
 for case in v['records']:
  if case['case'] in expected:assert math.isclose(case['norm'],expected[case['case']],rel_tol=1e-12,abs_tol=1e-12)
  else:assert case['case'] in ['one_rank_nonsharded_nan_collective_rejection','pure_NO_SHARD_one_rank_inf_rejection']
 off,on=v['diagnostics_records'];assert off['enabled'] is False and on['enabled'] is True and off['unique_storage_bytes']==0 and on['unique_storage_bytes']==4980752 and on['logical_tensor_bytes']==4718608
 assert on['after_clear_allocated']==on['before_allocated'] and off['after_clear_allocated']==off['retained_allocated']==off['before_allocated'] and on['retained_allocated']>on['before_allocated']
 storage.append(dict(rank=i,retained_allocation_increment=on['retained_allocated']-on['before_allocated'],unique_storage_bytes=on['unique_storage_bytes'],logical_bytes=on['logical_tensor_bytes'],on_wall=on['wall_seconds'],off_wall=off['wall_seconds']))
guard=[json.loads(z) for z in f['resources.jsonl']['text'].splitlines()];assert len(guard)==3 and result['guard_samples']==3
for row in guard:
 assert row['job_id']=='3063' and 'event' not in row and len(row['gpus'])==8
 for gpu in row['gpus']:assert gpu['foreign_memory_fraction']<=.15 and gpu['foreign_sm_upper']<=15
baseline={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446}
for key in ['initial_gpu','final_gpu']:assert {int(s.split(',')[1]) for s in result[key]['processes']}==baseline
for s in result['final_gpu']['gpus']:assert s.split(',')[-1].strip()=='0' and s.split(',')[2].strip()=='1131'
terminal=R/'outputs/evidence/tree-native-numeric-3063-first-status.txt';text=terminal.read_text();assert all(z in text for z in ['JobState=COMPLETED','ExitCode=0:0','RunTime=00:00:17','NumCPUs=192','TresPerNode=gres:gpu:8','OverSubscribe=NO'])
out=dict(status='PASS',stage='native_numeric_gate_only',candidate_commit=result['code_commit'],job_id='3063',manifest_sha256=sha(m),wrapper_sha256=sha(R/'work/start_tree_native_numeric_296217a.sh'),scope='ActualTorch2.4/CUDA/NCCL8rank syntheticFSDPnorm and syntheticdetachedTree-storage gate only; no real3B/new2/5/500/effect acceptance or2971rootcause',actual_rank_count=8,cases_per_rank=20,total_rank_cases=160,guard_samples=3,cleanup_residual={},background_PIDs_preserved=True,slurm_exit_code='0:0',scheduler_seconds=17,allocated_gpu=8,allocated_cpu=192,requested_cpu=32,scheduler_gpu_hours=17*8/3600,scheduler_cpu_hours=17*192/3600,storage=storage,foreign_memory_fraction_peak=max(g['foreign_memory_fraction'] for z in guard for g in z['gpus']),foreign_sm_upper_peak=max(g['foreign_sm_upper'] for z in guard for g in z['gpus']),native_FSDP_norm_gate_accepted=True,actual3B_peak_verified=False,actual_actor_optimization_or_new_reward_verified=False,lowvarKL_GPU_backward_tested=False,old308_rootcause_proven=False,evidence=[dict(path=x['run']+'/'+n,sha256=v['sha256']) for n,v in f.items()]+[dict(local_path=str(P),sha256=sha(P)),dict(local_path=str(terminal),sha256=sha(terminal))])
(D/'NATIVE3063_ACTUAL_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['evidence','storage']},indent=2))
