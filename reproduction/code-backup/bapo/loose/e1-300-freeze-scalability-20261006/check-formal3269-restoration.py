from pathlib import Path
import json,base64,hashlib,math,yaml,ast,copy
B=Path(__file__).resolve().parent;E=B.parent/'evidence';p=E/'freeze-scalability-formal3269-actual-restoration-files.json';raw=json.loads(p.read_text());ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006';RUN=ROOT+'/runs/bapo-instruct300-formal300-3269';SOURCE='b7119791b86951727cea36a1881e8a525c932494';h=lambda b:hashlib.sha256(b).hexdigest();f={}
assert raw['job_id']==3269 and raw['run_root']==RUN
for n,v in raw['files'].items():
 b=base64.b64decode(v['base64'],validate=True);assert h(b)==v['sha256']and len(b)==v['bytes']and b.decode()==v['text']and v['path']==RUN+'/'+n;f[n]=b
assert f['recovery-permission.json']==(B/'formal3269-resume25-permission.json').read_bytes()
permit=json.loads(f['recovery-permission.json']);manifest=json.loads(f['recovery-manifest.json']);assert permit['manifest_sha256']==h(f['recovery-manifest.json'])and permit['target_job_id']==3269
pc=json.loads(f['recovery-permission-check.json']);assert pc['status']=='ACTUAL_ALLOCATION_RECOVERY_PERMISSION_VALIDATED_NOT_GPU_PROOF'and pc['verify_only']is False and pc['job_id']==3269 and pc['step']==25 and pc['next_step']==26 and pc['source_commit']==SOURCE and pc['run_root']==RUN
old=E/'formal3231-fault-raw-20261005T0918/runs/bapo-instruct300-formal300-3231';oldreports={x['rank']:x for x in json.loads((E/'formal3231-checkpoint25-fullhash-cpu-candidate.json').read_text())['native_reports']};old_driver=json.loads((old/'checkpoints/full_state/global_step_25/driver-reload.json').read_text());algo=json.loads((old/'checkpoints/algorithm/global_step_25/state.json').read_text())
ranks=[json.loads(f[f'recovery-witness/rank_{i}.json'])for i in range(8)];d=json.loads(f['recovery-witness/driver.json'])
for i,w in enumerate(ranks):
 o=oldreports[i];entry=next(x for x in manifest['ranks']if x['rank']==i)
 assert w['status']=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'and w['rank']==i and w['world_size']==8 and w['step']==w['scheduler_last_epoch']==25 and w['next_step']==26
 assert w['old_pid']==o['pid']and w['new_pid']!=o['pid']and w['state_digest']==o['state_digest']and w['old_state_sha256']==entry['state']['sha256']
 assert w['distributed_backend']=='nccl'and w['model_state_dict_type']=='SHARDED_STATE_DICT'and w['model_dtensor_entries']==o['model_dtensor_entries']==435 and w['optimizer_state_entries']==o['optimizer_state_entries']==37
 assert w['model_class']=='Qwen2ForCausalLM'and w['model_config']==dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,vocab_size=151936,architectures=['Qwen2ForCausalLM'])
 assert all(math.isclose(x,1e-6*25/114,rel_tol=1e-12,abs_tol=1e-18)for x in w['lr'])and len(w['reconstructed_generation_rng_digest'])==64
assert d['status']=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and d['old_pid']==old_driver['pid']and d['new_pid']!=old_driver['pid']and d['loaded_state_digest']==old_driver['state_digest']and d['rank_witnesses']==ranks
assert d['step']==d['actual_outer_updates']==25 and d['next_step']==26 and d['epoch']==0 and d['sampler_cursor']==12800 and d['bapo']==algo['bapo']and d['verify_only']is False and d['no_generation_or_optimizer_update']is False
for k in ('sampler','rng'):assert len(d['saved_'+k+'_digest'])==64 and d['saved_'+k+'_digest']==d['restored_'+k+'_digest']
for w in ranks+[d]:
 assert w['target_source_commit']==SOURCE and w['target_run_root']==RUN and w['sampling_continuity']=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'and w['process']['pid']==w['new_pid']and type(w['process']['start_ticks'])is int and w['process']['start_ticks']>0
 assert w['process']['actual_env']['SLURM_JOB_ID']=='3269'and w['process']['actual_env']['BAPO_RUN_ID']==RUN
 rel='verl/trainer/ppo/ray_trainer.py'if w is d else'verl/workers/fsdp_workers.py';assert w['runtime_module_path']==ROOT+'/repo/'+rel and w['runtime_module_sha256']==h((B.parent/'bapo-e1-300-freeze-scalability'/rel).read_bytes())
assert len({w['new_pid']for w in ranks+[d]})==9
cfg=yaml.safe_load(f['resolved-config.yaml']);ocfg=yaml.safe_load((old/'resolved-config.yaml').read_text());tree=ast.parse((B.parent/'bapo-e1-300-freeze-scalability/search_r1/baselines/recovery.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='config_identity');ns={'copy':copy};exec(compile(ast.Module(body=[fn],type_ignores=[]),'b711-config-identity','exec'),ns);assert ns['config_identity'](cfg)==ns['config_identity'](ocfg)
assert cfg['trainer']['resume_verify_only']is False and cfg['trainer']['resume_full_state']==RUN+'/recovery-manifest.json'and cfg['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfg['trainer']['total_training_steps']==301 and cfg['baselines']['bapo']['stage_validation_size']==64
loc=json.loads(f['retrieval-assets-locality.json']);assert loc['host']=='h100-5'and len(loc['paths'])==3 and all('/dataset'in x['mount']for x in loc['paths'])and loc['proxy_env']==[]
print(json.dumps({'status':'PASS_INDEPENDENT_FORMAL3269_ACTUAL_RESTORE_WITNESS_REVIEW','job_id':3269,'source_commit':SOURCE,'witnesses':9,'rank_model_dtensor_entries':435,'rank_optimizer_entries':37,'restored_actual_outer_updates':25,'next_step':26,'scheduler_last_epoch':25,'lr':1e-6*25/114,'sampler_cursor':12800,'verify_only':False,'sampling_continuity':d['sampling_continuity'],'captured_raw_sha256':h(p.read_bytes()),'completed_new_outer_not_established_by_restore_witness':True}))
