from pathlib import Path
import json,base64,hashlib,importlib.util,math,re,gzip,datetime
W=Path(__file__).resolve().parent; B=W/'formal3210-actual10-resource-recovered-20261004T184157Z'; A=W/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); J=lambda p:json.loads(p.read_text()); d=J(B/'collection-original.json')
assert H(B/'collection-original.json')=='2ce0a7f77a17c76aa32f55d650a8711302766599d988a1c49a2a6a72f05fdb37'
raw={}
for path,v in d['files'].items():
 b=base64.b64decode(v['base64'],validate=True);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];raw[path]=b
one=lambda suffix:next((p,b) for p,b in raw.items() if p.endswith(suffix))
m=J(A/'032-manifest.json');reports=[J(A/f'{i+6:03}-rank_{i}.json') for i in range(8)]
assert one('global_step_5/manifest.json')[1]==(A/'032-manifest.json').read_bytes()
for i in range(8):assert one(f'training_state/rank_{i}.json')[1]==(A/f'{i+6:03}-rank_{i}.json').read_bytes()
assert hashlib.sha256(one('acceptance-5-receipt.json')[1]).hexdigest()=='25b41a15d3e70958b6335f43350ece1c63472478f5d23d9a8c6f78b1a4af433f'
log=J(W/'ACTUAL3210_TEN_CONTINUITY_INDEPENDENT_LOG_REVIEW.json');assert H(W/'ACTUAL3210_TEN_CONTINUITY_INDEPENDENT_LOG_REVIEW.json')=='746e9fd902b7b0a230c7194b3480a46a3419b027e30e034930d9b59db43b8858';assert log['actual_completed_outers']==10 and log['actual10_logs_complete']
assert d['checkpoint_manifest_sha256']==log['manifest_sha256'] and d['source_commit']==log['source_commit'] and d['source_clean'] is True and d['job_id']=='3210' and d['uid']==0 and d['node'].split('.')[0]=='h100-10' and d['receipt_exists'] is True
assert all(s in d['slurm'] for s in ('JobId=3210 ','JobState=RUNNING ','Restarts=0 ','NodeList=h100-10 ','UserId=root(0)','gres:gpu:h100:8'))
assert one('resolved-profile.json')[1]==(W/'formal3210-pause-resource-independent-20261004T171020Z/run/resolved-profile.json').read_bytes()
p=W/'first5-tools-a096-raydedup-candidate/receipt_payload.py';assert H(p)=='85bef7b214bb8678885923a18cbdfbe0408aa8e9811b130241cd97593b551714';s=importlib.util.spec_from_file_location('reviewconsumer',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
service=json.loads(one('/service.json')[1])['owner']; resourcepath=one('pause-resources.ndjson')[0];localresource=B/'run'/resourcepath.split('/runs/formal-3210/')[1];assert localresource.read_bytes()==raw[resourcepath]
join=r.resource_join(localresource,m,reports,service);assert len(join['samples'])==2 and len(join['uuid_inventory'])==8
old=J(W/'formal3210-pause-resource-independent-20261004T171020Z/pause-resource-independent-audit.json'); starts={v['pid']:v['start'] for v in old['rank_identity']};uuids={v['pid']:v['uuid'] for v in old['rank_identity']}
for pid in [*starts,591546,590120]:
 q=d['current_processes'][str(pid)];assert q['pid']==pid and q['uids']==['0']*4 and q['state'] in ('R','S','D','I') and q['in_slurm_job'] and '/job_3210/' in q['cgroup'];assert q['process_start_ticks']==starts.get(pid,'1819179500' if pid==591546 else '1819168667')
for sample,nvml in zip(d['samples'],d['raw_nvml_snapshots']):
 assert sample['time']==nvml['time']; apps=[tuple(x.strip() for x in line.split(',')) for line in nvml['apps_raw'].splitlines() if line.strip()];assert len(apps)==16
 expected={(str(pid),uuid) for pid,uuid in uuids.items()}|{('590120',uuid) for uuid in uuids.values()};assert {(pid,uuid) for uuid,pid,mem in apps}==expected
 actual={(str(x['pid']),x['uuid']) for x in sample['gpu_processes']};assert actual==expected
 for x in sample['gpu_processes']:assert x['uids']==['0']*4 and x['state'] in ('R','S','D','I') and '/job_3210/' in x['cgroup'] and x['process_start_ticks']==starts.get(x['pid'],'1819168667')
http=json.loads(one('fresh-http-top3-24.json')[1]);assert http['job_id']=='3210' and http['url']=='http://127.0.0.1:18084/retrieve' and http['topk']==3 and len(http['queries'])==len(http['response']['result'])==24
for q in (http['service_before'],http['service_after']):
 assert q['pid']==service['pid'] and q['process_start_ticks']==service['start'] and q['uid']==str(service['uid']) and q['uids']==['0']*4 and q['cgroup']==service['cgroup'] and q['state'] in ('R','S','D','I') and q['in_slurm_job'] is True
for row in http['response']['result']:
 assert len(row)==3
 for x in row:assert x['document']['contents'] and math.isfinite(float(x['score']))
cmd=d['current_processes']['590120']['cmdline'];assert all(x in cmd for x in ('--device cuda','--faiss_gpu','--host 127.0.0.1','--port 18084','--topk 3'))
# Replay entire lossless captured resource prefix; no reliance on peer PASS boolean.
gz=B/'resources-history.ndjson.gz';assert H(gz)==d['history']['gzip_sha256'];hasher=hashlib.sha256();count=0;size=0;last=None;peak=0.;gap=0.
with gzip.open(gz,'rb') as f:
 for b in f:
  hasher.update(b);size+=len(b);q=json.loads(b);t=datetime.datetime.fromisoformat(q['time']);assert t.utcoffset()==datetime.timedelta(0)
  if last is not None:assert t>last;gap=max(gap,(t-last).total_seconds())
  last=t;count+=1;assert not q.get('query_error');assert len(q['gpus'])==8
  peak=max(peak,max(float(g['used_mib']) for g in q['gpus']))
assert hasher.hexdigest()==d['history']['raw_sha256'] and size==d['history']['raw_bytes'] and count==3219 and peak==80977 and gap<5
native=J(W/'ACTUAL3210_NATIVE_HF_CONTENT_INDEPENDENT_REVIEW.json');assert native['passed_content_scope'] is True and native['full_HF435_allrank_slice_association'] is True
result={'status':'PASS_ACTUAL_TEN_CONTINUITY_AND_FIRST5_STORED_EVIDENCE','source_commit':log['source_commit'],'job_id':'3210','actual_completed_outers':10,'driver_pid':591546,'driver_start_ticks':'1819179500','config_sha256':log['config_sha256'],'checkpoint5_manifest_sha256':log['manifest_sha256'],'signed_first5_release_receipt_sha256':log['signed_actual_release_receipt_sha256'],'log_review_sha256':H(W/'ACTUAL3210_TEN_CONTINUITY_INDEPENDENT_LOG_REVIEW.json'),'fresh_collection_sha256':H(B/'collection-original.json'),'native_content_review_sha256':H(W/'ACTUAL3210_NATIVE_HF_CONTENT_INDEPENDENT_REVIEW.json'),'live_same_all8_pid_start_uuid':True,'all1to10_each_rank20_teacher_and_lr_continuity':True,'native5_all8_Adam100_scheduler5_RNG_cursor2560_HF435_slice_exact':True,'fresh_http24_top3_seconds':http['seconds'],'history_rows':count,'history_raw_sha256':hasher.hexdigest(),'history_gzip_sha256':H(gz),'sampled_max_gpu_mib':peak,'sampled_max_gpu_fraction':peak/81559,'runtime15percent_reserve_proven':False,'limitations':['10-round log continuity; checkpoint state audited at5 only. No HF10, stored Adam200 or stored cursor5120 claim.','No complete new-process RL recovery tested.','Resource peak is sampled, not a continuous upper bound. Observed99.286% violates15%runtime reserve; no future memory safety guarantee.','Loss/reward on varying training batches is not heldout improvement. Formal300 and final51713 evaluation remain incomplete.'],'signature_created':False,'training_mutated':False}
out=W/'ACTUAL3210_TEN_FINAL_INDEPENDENT_REVIEW.json'
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(result))
