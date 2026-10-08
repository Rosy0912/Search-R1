from pathlib import Path
import json,hashlib,importlib.util,math,re
W=Path(__file__).resolve().parent;H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());A=W/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z';B=W/'formal3210-pause-resource-independent-20261004T171020Z';root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210';specp=W/'actual3210-final5role-evidence-spec-root.json';assert H(specp)=='d5b585feedcec814eef8416c073e85b636b621a190a210c0f68476c1c58e749f';spec=read(specp);assert spec==read(W/'actual3210-cpu-runner-v5/actual-five-role-evidence-spec.json')
paths={rec['path']:Path(rec['local_path']) for rec in read(A/'index.json')['records']}
for role,rec in spec.items():
 if role not in ('rank_audit','hf_reload'):paths[rec['path']]=B/'run'/rec['path'].removeprefix(root+'/')
 assert H(paths[rec['path']])==rec['sha256']
m=read(A/'032-manifest.json');reports=[read(A/f'{i+6:03}-rank_{i}.json') for i in range(8)];native=read(paths[spec['rank_audit']['path']]);hf=read(paths[spec['hf_reload']['path']]);service=read(paths[spec['service']['path']])['owner'];retrieval=read(paths[spec['retrieval']['path']]);collection=read(B/'collection-original.json');audit=read(B/'pause-resource-independent-audit.json')
p=W/'first5-tools-a096-raydedup-candidate/receipt_payload.py';assert H(p)=='85bef7b214bb8678885923a18cbdfbe0408aa8e9811b130241cd97593b551714';s=importlib.util.spec_from_file_location('consumer',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);r.check_cpu_evidence(native,hf,m,H(A/'032-manifest.json'),reports);join=r.resource_join(paths[spec['resources']['path']],m,reports,service);assert r.same_service_identity(retrieval['service'],service)
assert retrieval['job_id']=='3210' and len(retrieval['queries'])==len(retrieval['response']['result'])==24
for row in retrieval['response']['result']:
 assert len(row)==3
 for item in row:assert item['document']['contents'] and math.isfinite(float(item['score']))
assert collection['source_commit']==m['source_commit'] and collection['source_clean'] is True and collection['job_id']=='3210' and collection['uid']==0 and collection['node'].split('.')[0]=='h100-10' and collection['receipt_exists'] is False
slurm=collection['slurm'];assert all(v in slurm for v in ['JobId=3210','JobState=RUNNING','NodeList=h100-10','UserId=root(0)','NumCPUs=192','TresPerNode=gres:gpu:h100:8'])
expected={v['pid']:v for v in reports};starts={v['pid']:v['start'] for v in audit['rank_identity']}
assert audit['driver_pid']==m['driver_pid']==591546 and audit['driver_start']=='1819179500' and audit['runtime15percent_reserve_proven'] is False
for pid in [*expected,service['pid'],m['driver_pid']]:
 process=collection['current_processes'][str(pid)];assert process['pid']==pid and process['uids']==['0']*4 and process['state'] in ('R','S','D','I') and process['in_slurm_job'] is True and re.search(r'(?:^|/)job_3210(?:/|$)',process['cgroup'])
 if pid in expected:assert process['process_start_ticks']==starts[pid]
 elif pid==m['driver_pid']:assert process['process_start_ticks']=='1819179500'
 else:assert process['process_start_ticks']==service['start']=='1819168667'
for sample in join['samples']:
 for pid in expected:assert sample['identity'][pid]['start']==starts[pid]
assert len(join['uuid_inventory'])==8 and set(join['uuid_inventory'])=={v['uuid'] for v in audit['rank_identity']}
servproc=collection['current_processes'][str(service['pid'])];cmd=servproc['cmdline'];assert all(v in cmd for v in ['repo','--device cuda','--faiss_gpu','--host 127.0.0.1','--port 18084','--topk 3']) if 'repo' in cmd else all(v in cmd for v in ['--device cuda','--faiss_gpu','--host 127.0.0.1','--port 18084','--topk 3'])
result={'scope':'ACTUAL3210_COMPLETE_FIVE_ROLE_ARTIFACT_AND_RESOURCE_JOIN_REVIEW_NOT_SIGNATURE','evidence':spec,'spec_sha256':H(specp),'checkpoint_manifest_sha256':H(A/'032-manifest.json'),'source_commit':m['source_commit'],'job_id':'3210','driver_pid':m['driver_pid'],'pause_nonce':m['pause_nonce'],'resource_native_join':join,'driver_observed_start':'1819179500','service_pid':service['pid'],'service_start':service['start'],'http_elapsed_seconds':retrieval['seconds'] if 'seconds' in retrieval else audit['http_seconds'],'runtime15percent_reserve_proven':False,'sampled_max_gpu_used_fraction':audit['peak_used_fraction'],'native_and_HF_content_review':'ACTUAL3210_NATIVE_HF_CONTENT_INDEPENDENT_REVIEW.json','all_five_role_hashes_exact':True,'passed_artifact_join_scope':True,'pending_fresh_driver_before_signature':True,'limitations':['Resource samples not continuous peak; actual history violates15percent runtime reserve; noOOM observed is not future safety guarantee','CPU stored-state inspection not full new-process RL restore','No10/checkpoint10 or accuracy improvement claim','No signatures/release']}
(W/'ACTUAL3210_FIVE_ROLE_JOIN_INDEPENDENT_REVIEW.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['resource_native_join','evidence','limitations']}))
