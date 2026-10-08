import pathlib,json,hashlib,importlib.util,math,yaml
root=pathlib.Path.cwd();e=root/'outputs/evidence';d=root/'work/aux-gate-review';prefix='tree-two-2879-'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((e/(prefix+'stage-validation-2881.json')).read_text());h=json.loads((e/(prefix+'stage-validation-2881-hf-reload.json')).read_text())
s=importlib.util.spec_from_file_location('v',root/'work/validate_tree_stage.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
records=v.parse_console(e/(prefix+'stage.log'),2);assert records==r['records'];assert all(x['metrics']['actor/optimizer_steps']==12 and x['metrics']['actor/skipped_empty_minibatches']==0 for x in records)
mapfiles={'stage.log':prefix+'stage.log','cost.json':prefix+'cost.json','resolved.yaml':prefix+'resolved.yaml','state.json':prefix+'output__checkpoints__tree_grpo-qwen2.5-3b__algorithm__global_step_2__state.json','stage-validation-2881-hf-reload.json':prefix+'stage-validation-2881-hf-reload.json','validate_tree_stage.py':None,'check_real_checkpoint.py':None}
for remote,digest in r['source_sha256'].items():
 name=pathlib.Path(remote).name;local=root/'work'/name if mapfiles[name] is None else e/mapfiles[name];assert sha(local)==digest,(name,'SHA changed')
cfg=yaml.safe_load((e/(prefix+'resolved.yaml')).read_text());assert cfg['trainer']['total_training_steps']==3 and cfg['trainer']['smoke_only'] is True and cfg['data']['train_batch_size']==512 and cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size']==256 and cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size']==8
state=json.loads((e/mapfiles['state.json']).read_text());assert state==r['algorithm_state'] and state['global_step']==2 and state['method']=='tree_grpo'
assert r['tree_audit_required'] and r['actual_outer_updates']==2
for pair in r['tree_audit_artifacts']:
 for k,suffix in [('rollout',''),('policy','_policy_batch')]:
  st=pair['step'];assert sha(e/(prefix+f'step_{st}_rollout_{st}{suffix}.json'))==pair[k+'_sha256']
assert h['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and h['all_parameters_finite'] and h['cpu_forward_logits_finite'] and h['changed_parameters']>0 and h['changed_tensors']>0 and math.isfinite(h['max_abs_parameter_delta']) and h['max_abs_parameter_delta']>0
assert not any(h['hf_loading_info'].values()) and h['tokenizer_vocab_equal'] and h['bos_eos_equal'];assert h['checkpoint']==r['checkpoint']
gpu=[json.loads(x) for x in (e/(prefix+'resource-samples.jsonl')).read_text().splitlines()];assert all(x.get('job_id')=='2879' and len(x['gpus'])==8 for x in gpu);assert all(g['foreign_memory_fraction']<.15 and g['foreign_sm_upper']<15 for x in gpu for g in x['gpus'])
cpu=[json.loads(x) for x in (e/(prefix+'retriever-cpu-samples.jsonl')).read_text().splitlines()];assert len({x['pid'] for x in cpu})==1 and all(x['job_id']=='2879' for x in cpu)
files=list(e.glob(prefix+'*'));hashes={str(p):sha(p) for p in files if p.is_file()}
summary={'decision':'INDEPENDENT_TWO_UPDATE_SMOKE_ACCEPTED','stage':'two','job':2879,'production_commit':'f4f457df55fbfe99691d466d5c6f1d3fd7175c65','actual_outer_updates':2,'optimizer_steps_per_outer':[12,12],'records_equal_raw_console':True,'all_bound_sha_verified':True,'hf_reload_report':h,'gpu_guard_samples':len(gpu),'max_foreign_memory_fraction':max(g['foreign_memory_fraction'] for x in gpu for g in x['gpus']),'max_foreign_sm_upper':max(g['foreign_sm_upper'] for x in gpu for g in x['gpus']),'cpu_samples':len(cpu),'retriever_peak_rss_bytes':max(x['peak_rss_bytes'] for x in cpu),'retriever_cpu_first':cpu[0]['process_cpu_seconds'],'retriever_cpu_last':cpu[-1]['process_cpu_seconds'],'slurm_gpu_hours':1988*8/3600,'slurm_cpu_hours':1988*192/3600,'stage_body_gpu_hours':(r['cost']['end_epoch']-r['cost']['start_epoch'])*8/3600,'cpu_validation_cpu_hours':21*8/3600,'evidence_file_sha256':hashes,'scope':'two-update infrastructure/math/serialization smoke only; not five, 500-update convergence, heldout effect or exact optimizer/RNG/dataloader resume; checkpoint bytes validated by independent CPU helper job2881, not copied/reloaded on Mac; shared auxiliary GPU lifetime cost separate'}
(d/'real-two-2879-independent-results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:x for k,x in summary.items() if k not in ('hf_reload_report','evidence_file_sha256')},indent=2))
