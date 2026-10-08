from pathlib import Path
import hashlib,json,re,math,importlib.util
W=Path(__file__).resolve().parent;S=W/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());idx=read(S/'index.json');lookup={}
for rec in idx['records']:
 p=Path(rec['local_path']);assert p.parent==S;assert p.stat().st_size==rec['bytes'] and H(p)==rec['sha256'];lookup[rec['path']]=p
root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210';out=idx['output'];assert idx['job_id']=='3210' and idx['manifest_sha256']=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b' and out==root+'/evidence/first5-cpu-inspection-v5-20261004T173404Z'
m=read(lookup[root+'/full_state/global_step_5/manifest.json']);assert H(lookup[root+'/full_state/global_step_5/manifest.json'])==idx['manifest_sha256'];native=read(lookup[out+'/native.json']);hf=read(lookup[out+'/hf.json']);reports=[read(lookup[m['files'][f'actor/training_state/rank_{i}.json']['path']]) for i in range(8)]
for i,r in enumerate(reports):assert H(lookup[m['files'][f'actor/training_state/rank_{i}.json']['path']])==m['files'][f'actor/training_state/rank_{i}.json']['sha256'];assert r['rank']==i
P=W/'first5-tools-a096-raydedup-candidate/receipt_payload.py';assert H(P)=='85bef7b214bb8678885923a18cbdfbe0408aa8e9811b130241cd97593b551714';sp=importlib.util.spec_from_file_location('reviewed_external_consumer',P);consumer=importlib.util.module_from_spec(sp);sp.loader.exec_module(consumer);consumer.check_cpu_evidence(native,hf,m,idx['manifest_sha256'],reports)
assert native['inspector_source_sha256']=='b017a443f1256d9eedf3d4cdfa6aca1073596fd5e5cf416cd41a8150a58a7623' and hf['inspector_source_sha256']=='38bc089be3e35f193d6192e200beb4155f02583fc1993996bf85ff2198d98aec'
for role,rec in m['files'].items():assert native['all_files'][rec['path']]==rec['sha256']
assert native['all_files'][root+'/full_state/global_step_5/manifest.json']==idx['manifest_sha256'];assert native['base_named_parameter_digest']==hf['base_named_parameter_digest'];assert len(hf['hf_map'])==435 and len(hf['base_map'])==434 and set(hf['hf_map'])==set(hf['base_map'])|{'lm_head.weight'}
assert hf['tied_embeddings'] is True and hf['actual_unique_parameters']==3085938688 and hf['all_loaded_parameters_finite'] is True and hf['actual_cpu_forward_finite'] is True and hf['config_computation_equal'] is True
assert len(hf['loading_info'])==2
for info in hf['loading_info']:assert set(info)=={'error_msgs','mismatched_keys','missing_keys','unexpected_keys'} and all(v==[] for v in info.values())
assert set(hf['parameter_changes'])==set(hf['base_map']) and sum(v['elements'] for v in hf['parameter_changes'].values())==3085938688 and sum(v['changed_elements'] for v in hf['parameter_changes'].values())==hf['total_changed_elements']>0
for v in hf['parameter_changes'].values():assert 0<=v['changed_elements']<=v['elements'] and math.isfinite(v['max_abs_delta']) and math.isfinite(v['mean_abs_delta']) and v['max_abs_delta']>=v['mean_abs_delta']>=0
rank_summary=[]
for r,original in zip(sorted(native['ranks'],key=lambda z:z['rank']),reports):
 rank=r['rank'];assert r['unique_local_model_numel']==3085938688//8 and len(r['model_layouts'])==435 and len(r['optimizer_moments'])==r['optimizer_state_entries']==37
 assert sum(math.prod(v['shape']) for v in r['optimizer_moments'])==3085938688//8 and {v['step'] for v in r['optimizer_moments']}=={100}
 for v in r['optimizer_moments']:assert v['dtype']=='torch.float32' and re.fullmatch('[a-f0-9]{64}',v['exp_avg_sha256']) and re.fullmatch('[a-f0-9]{64}',v['exp_avg_sq_sha256'])
 for key,l in r['model_layouts'].items():
  assert l['mesh_device_type']=='cuda' and l['mesh_shape']==[8] and l['mesh_ranks']==list(range(8)) and l['mesh_dim_names']==['fsdp'] and l['mesh_coordinate']==[rank] and l['placements']==['S(0)']
  assert l['global_dtype']==l['local_dtype']=='torch.float32' and l['global_shape'][0]==l['local_shape'][0]*8 and l['global_shape'][1:]==l['local_shape'][1:] and re.fullmatch('[a-f0-9]{64}',l['local_bytes_sha256'])
 assert r['model_layouts']['lm_head.weight']['local_bytes_sha256']==r['model_layouts']['model.embed_tokens.weight']['local_bytes_sha256'];assert set(r['component_digests'])=={'model','optimizer','scheduler','rng','teacher'}
 assert r['teacher_counters']==original['teacher']['counters'] and r['teacher_buffer_digest']==original['teacher']['buffer_content_digest'] and r['teacher_content_digest']==hf['base_named_parameter_digest'];assert r['rng_content']['python_state_valid'] is True and r['rng_content']['numpy_algorithm']=='MT19937'
 assert len(r['rng_content']['cuda'])==1 and r['rng_content']['cuda'][0]['bytes']==16 and r['rng_content']['cuda'][0]['device']==0
 changes=sum(v for k,v in r['changed_elements_vs_original_fp32'].items() if k!='lm_head.weight');assert changes>0
 rank_summary.append(dict(rank=rank,pid=r['pid'],object_ids=r['object_ids'],state_sha256=r['state_sha256'],state_digest=r['state_digest'],unique_local_numel=r['unique_local_model_numel'],optimizer_states=37,adam_step100=True,scheduler_last_epoch=5,last_lr=r['scheduler_last_lr'],model_keys_exact435=True,hf_slice_exact=True,changed_local_elements=changes,rng_verified=True,teacher_official_exact=True))
driver=read(lookup[m['files']['driver/driver.json']['path']]);assert native['driver']['state_sha256']==driver['state_sha256']==m['files']['driver/driver.pt']['sha256'] and native['driver']['state_digest']==driver['digest'] and native['driver']['rng_verified'] is True
assert native['driver']['pid']==m['driver_pid']==591546 and native['driver']['sampler_cursor']==native['driver']['consumed_cursor']==2560 and native['driver']['size']==169028 and native['driver']['epoch']==0
step=read(lookup[out+'/cpu-step-original.json']);assert step['job_id']=='3210' and step['uid']==0 and step['hostname'].split('.')[0]=='h100-10' and step['cuda_visible_devices']=='' and step['cuda_is_initialized'] is False and re.search(r'(?:^|/)job_3210(?:/|$)',step['cgroup'])
exits={}
for name in ['native','hf','srun']:
 p=lookup[out+'/'+name+'-exit-original.json'];e=read(p);assert e['returncode']==0
 if name!='srun':assert e['output_exists'] is True
 exits[name]=e
for name,obj in [('native',native),('hf',hf)]:
 p=lookup[out+'/'+name+'-stdout-original.txt'];d=json.loads(p.read_text().splitlines()[-1]);assert d['output']==out+'/'+name+'.json' and d['output_sha256' if name=='native' else 'sha256']==H(lookup[d['output']]) and d['passed'] is True
completion=read(lookup[out+'/inspection-completed-review-required.json']);assert completion['manifest_sha256']==idx['manifest_sha256'] and completion['cuda_is_initialized'] is False
for path,p in lookup.items():
 if path.startswith(out+'/') and p.name not in ['index.json'] and Path(path).name in completion['files']:assert H(p)==completion['files'][Path(path).name]
assert native['cuda_is_initialized'] is hf['cuda_is_initialized'] is False and hf['node'].split('.')[0]=='h100-10' and hf['uid']==0
result={'scope':'ACTUAL3210_NATIVE8_AND_FULL_HF5_CONTENT_INDEPENDENT_AUDIT_NOT_SIGNATURE_OR_RELEASE','source_commit':m['source_commit'],'job_id':'3210','checkpoint_manifest_sha256':idx['manifest_sha256'],'native_json_sha256':H(lookup[out+'/native.json']),'hf_json_sha256':H(lookup[out+'/hf.json']),'actual_cpu_step':step['pid'],'native_srun_hf_actual_exits':exits,'ranks':rank_summary,'driver':native['driver'],'teacher_base_parameter_digest':hf['base_named_parameter_digest'],'hf_unique_parameters':hf['actual_unique_parameters'],'hf_changed_elements':hf['total_changed_elements'],'full_HF435_allrank_slice_association':True,'all_manifest_file_SHA_rechecked_by_actual_native':True,'raw_files_retained_remote':True,'cuda_uninitialized':True,'passed_content_scope':True,'pending':'Separate actual source/resource/retriever process-start ownership audit and root+independent review/receipt; not signed or released; no10 claim'}
(W/'ACTUAL3210_NATIVE_HF_CONTENT_INDEPENDENT_REVIEW.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['ranks','driver','native_srun_hf_actual_exits']}))
