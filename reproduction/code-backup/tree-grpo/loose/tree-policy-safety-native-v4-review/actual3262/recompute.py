import base64,collections,gzip,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
P=ROOT/'outputs/tree-instruct-300/PPO_SAFETY_NATIVE3262_RAW_FIRST.json'
D=json.loads(P.read_bytes());raw={}
for name,v in D['files'].items():
 b=gzip.decompress(base64.b64decode(v['gzip_base64']));assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'],name;raw[name]=b
assert len(raw)==766
R=json.loads(raw['result.json']);assert R['job_id']=='3262' and R['code_commit']=='0980ec343e90fdd39a91ecc91ba61e594281c063'
assert R['manifest_sha256']=='0935f213c61b70ea06faa2737ce31211693a4f9e848429a3b1c27ae7d76f4b1e'
assert R['status']=='NATIVE_FIXTURES_COMPLETE_PENDING_REVIEW' and R['fixtures_passed'] is True and R['cleanup_residual']=={} and R['gpu_exit_code']==0
slurm=D['commands']['scontrol show job 3262'];assert slurm['rc']==0
for s in ['JobId=3262 ','JobState=COMPLETED ','ExitCode=0:0','Restarts=0','RunTime=00:00:23','NodeList=h100-9','BatchHost=h100-9','NumCPUs=192','CPUs/Task=32','TresPerNode=gres:gpu:8','OverSubscribe=NO']:assert s in slurm['stdout'],s
cases=['bounded_extreme']*3+['off','on','padding','one_rank_invalid','post_write_brake','startup_mismatch','active_clip','nonempty_negative_global_kl_tail_quantiles','one_rank_before_exp_kl_bound'];ranks=[]
for i in range(8):
 r=json.loads(raw[f'gpu-records/rank-{i}.json']);assert r['status']=='PASS' and r['rank']==i and r['world']==8 and r['selected_visible_gpu']==str(i) and r['cuda_local_device']==0
 assert [x['case'] for x in r['records']]==cases
 for dtype,x in zip(['torch.float16','torch.bfloat16','torch.float32'],r['records'][:3]):
  assert x['dtype']==dtype and x['loss']==[[3.,0.,0.]] and x['gradient']==[[0.,0.,0.]],x
 values={x['case']:x for x in r['records'][3:10]}
 for case in ['off','on','padding','active_clip']:
  x=values[case];assert x['actual_optimizer_steps']==10 and x['refused'] is False and x['populated_adam_offload_reload_equal'] is True
 assert values['on']['parameter']==values['off']['parameter']
 for case,n in [('on',10),('padding',10),('active_clip',10),('one_rank_invalid',0),('post_write_brake',1),('startup_mismatch',0)]:
  x=values[case];assert x['committed']==n and x['refused']==(case not in ['on','padding','active_clip'])
 for x in r['records'][3:10]:assert x['rng_preserved'] is True and all(math.isfinite(v) for row in x['parameter'] for v in row)
 assert values['one_rank_invalid']['parameter']==values['startup_mismatch']['parameter']==[[0.009999999776482582]]
 ranks.append(r)
 assert R['records'][i]==r
E=collections.defaultdict(list)
for path,b in raw.items():
 if '/policy-safety/' in path:
  x=json.loads(b);case=path.split('/')[1];E[case,x['rank']].append(x)
allnorm=[];globalgroups=collections.defaultdict(list)
for (case,rank),ev in E.items():
 assert 0<=rank<8
 phase=collections.Counter(x['phase'] for x in ev)
 if case in ['on','padding','active_clip']:assert phase=={'before_optimizer':10,'optimizer_committed':10,'after_optimizer':10}
 elif case=='post_write_brake':assert phase=={'before_optimizer':1,'optimizer_committed':1,'after_optimizer':1}
 elif case=='one_rank_invalid':assert phase=={'before_optimizer':1}
 else:raise AssertionError(case)
 for x in ev:
  assert x['outer_step']==111 and x['local_outer_rows']==480 and x['monitor_rows']==32 and len(set(x['monitor_local_row_indices']))==32
  assert x['outer_completed'] is False and x['scheduler_advanced'] is False and x['automatic_rollback'] is False
  phase=x['phase'];n=x['committed_optimizer_calls'];assert x['micro']==n*24
  if phase=='before_optimizer':assert n in (range(10) if case in ['on','padding','active_clip'] else [0])
  else:
   assert n in (range(1,11) if case in ['on','padding','active_clip'] else [1]);pre=x['measured_gradient_preclip'];post=x['measured_gradient_postclip'];assert math.isfinite(pre) and math.isfinite(post) and 0<post<=1.00001 ;allnorm.append(pre)
   if case=='active_clip':assert pre>1 and abs(post-1)<1e-5
   else:assert pre==post
  decision=x['decision'];assert decision['allowed']==(not(case=='one_rank_invalid' or case=='post_write_brake' and phase=='after_optimizer'))
  if case=='one_rank_invalid':assert decision['invalid_count']==1 and decision['sampled_old_policy_kl'] is None
  if case=='post_write_brake' and phase=='after_optimizer':assert decision['reasons']==['sampled_old_policy_kl'] and decision['sampled_old_policy_kl']>1e-12
  if 'kl_sum' in decision and decision['kl_computation_valid']:assert abs(decision['sampled_old_policy_kl']-decision['kl_sum']/decision['mass'])<1e-15
  globalgroups[case,phase,n].append(decision)
for key,decisions in globalgroups.items():assert len(decisions)==8 and all(x==decisions[0] for x in decisions),key
oracle=json.loads((ROOT/'work/tree-policy-safety-native-v4-review/LOCAL_CPU_ORACLE.json').read_text())
def equal_numeric(actual,expected):
 if isinstance(expected,dict):
  assert actual.keys()==expected.keys()
  for k in expected:equal_numeric(actual[k],expected[k])
 elif isinstance(expected,(int,float)) and not isinstance(expected,bool):assert math.isfinite(actual) and abs(actual-expected)<1e-12
 else:assert actual==expected
for r in ranks:
 x=r['records'][10];assert x['case']=='nonempty_negative_global_kl_tail_quantiles'
 for actual,key in [('allowed','allowed'),('quantiles','quantiles'),('kl_refusal','low_kl_refusal'),('tail_refusal','low_tail_refusal')]:equal_numeric(x[actual],oracle[key])
 equal_numeric(r['records'][11]['decision'],oracle['one_rank_bound_refusal'])
 assert r['records'][9]['measured_preclip']>1 and abs(r['records'][9]['measured_postclip']-1)<1e-5
samples=[json.loads(l) for l in raw['resources.jsonl'].splitlines()];assert len(samples)==R['guard_samples'] and R['guard_healthy_at_fixture_end'] is True
for x in samples:
 assert not x.get('event') and x['job_id']=='3262' and len(x['gpus'])==8
 for g in x['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
assert R['guard_last_sample']==samples[-1] and R['end_epoch']-samples[-1]['epoch']<15
for pid,stat in D['background_proc_stat'].items():
 parts=stat.split(') ',1)[1].split();assert int(parts[19])==(1502352181 if int(pid)<1690000 else 1502398762)
finalpid={int(x.split(',')[1]) for x in R['final_gpu']['processes']};assert finalpid==set(map(int,D['background_proc_stat'])) and len(finalpid)==8
out={'status':'ACTUAL3262_SYNTHETIC_NATIVE_RECOMPUTE_PASS','raw_bundle_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'raw_files_verified':len(raw),'ranks':8,'cases_per_rank':12,'dtype_oracles':24,'safety_events':sum(map(len,E.values())),'global_decision_groups':len(globalgroups),'finite_pre_post_norm_range':[min(allnorm),max(allnorm)],'active_clip_pre_norm_range':[min(x['measured_gradient_preclip'] for (case,rank),ev in E.items() if case=='active_clip' for x in ev if x['phase']=='optimizer_committed'),max(x['measured_gradient_preclip'] for (case,rank),ev in E.items() if case=='active_clip' for x in ev if x['phase']=='optimizer_committed')],'active_clip_post_norm_range':[min(x['measured_gradient_postclip'] for (case,rank),ev in E.items() if case=='active_clip' for x in ev if x['phase']=='optimizer_committed'),max(x['measured_gradient_postclip'] for (case,rank),ev in E.items() if case=='active_clip' for x in ev if x['phase']=='optimizer_committed')],'norm_clipping_active_tested':True,'guard_samples':len(samples),'wall_seconds':R['wall_seconds'],'slurm_seconds':23,'allocated_gpus':8,'allocated_cpus':192,'requested_cpus':32,'allocated_gpu_hours':23*8/3600,'allocated_cpu_hours':23*192/3600,'actual_process_cpu_time_unknown':True,'startup_mismatch_published_events':0,'nonempty_negative_advantage_global_quantile_tested_on_gpu':True}
pathlib.Path(__file__).with_name('RECOMPUTED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
