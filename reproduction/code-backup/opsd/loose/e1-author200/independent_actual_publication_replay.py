"""Public RSA and actual publication/release replay, no private key access."""
from pathlib import Path
import json,hashlib,hmac,csv,io,importlib.util
W=Path(__file__).resolve().parent
D=W/'actual3261-first5-signed-publication-once-v1'
S=W/'actual-snapshots/3261-20261005T205829353886Z'
F=S/'original-files'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
envelope=read(D/'signed-envelope.json')
payload=read(W/'actual3261-unsigned-collected-20261005T205139510221Z/unsigned-step5-payload.json')
need(envelope['payload']==payload and set(envelope)=={'payload','signature_hex'},'Signed canonical payload changed')
pubpath=F/'bindings/public.json'
need(sha(pubpath)=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c','Actual task public key changed')
pub=read(pubpath);need(pub['algorithm']=='RSA-SHA256' and pub['e']==65537,'Actual public key algorithm')
n=int(pub['n_hex'],16);k=(n.bit_length()+7)//8;sig=bytes.fromhex(envelope['signature_hex'])
need(len(sig)==k and int.from_bytes(sig,'big')<n,'Actual signature integer encoding')
canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
tail=bytes.fromhex('3031300d060960864801650304020105000420')+hashlib.sha256(canonical).digest()
expected=b'\x00\x01'+b'\xff'*(k-len(tail)-3)+b'\x00'+tail
actual=pow(int.from_bytes(sig,'big'),65537,n).to_bytes(k,'big')
need(hmac.compare_digest(actual,expected),'Independent mathematical PKCS1/SHA256 public RSA verification')
receipt=sha(D/'signed-envelope.json')
need(receipt=='812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b' and sha(F/'acceptance-5-receipt.json')==receipt,'Actual exclusively published receipt original')
a=read(D/'publish-remote-original.json');check=read(D/'check-remote-original.json')
need(a['published_exclusively'] is True and a['receipt_sha256']==receipt,'Actual publication response')
r=read(W/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json')
starts={x['pid']:x['start_ticks'] for x in r['actual_rank_results']}
uuids={x['gpu_uuid'] for x in r['actual_rank_results']}
for event in [check,a['before'],a['after']]:
 need(event['source_commit']==payload['source_commit'] and event['job_id']=='3261','Current publication source/job')
 need(event['driver']['pid']==1533312 and event['driver']['start']=='1829085316','Same original driver')
 need({x['pid']:x['start'] for x in event['workers']}==starts and event['service']['pid']==1531763 and event['service']['start']=='1829074494','Same original eight workers/retriever')
 for proc in [event['driver'],*event['workers'],event['service']]:
  need(proc['uids']==[0]*4 and '/job_3261/step_0' in proc['cgroup'] and proc['state'] in ['R','S','D','I'],'Live owned physical process')
 sl=event['slurm_original']
 need(all(v in sl for v in ['JobId=3261','JobState=RUNNING','Restarts=0','NodeList=h100-10','TresPerNode=gres:gpu:h100:8']),'Actual exclusive allocation/restarts')
 inventory=list(csv.reader(io.StringIO(event['gpu_inventory_original'])))
 need(len(inventory)==8 and {v[1].strip() for v in inventory}==uuids,'Current all eight physical H100 UUIDs')
 apps=list(csv.reader(io.StringIO(event['gpu_unfiltered_apps_original'])))
 need({int(x[1]) for x in apps}==set(starts)|{1531763},'No foreign current GPU application')
need(check['require_paused'] is True and a['before']['require_paused'] is True and a['after']['require_paused'] is False,'Actual publication paused/live sequencing')
for alias,orig in [('independent_review','ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json'),('root_review','ACTUAL3261_FIRST5_ROOT_REVIEW_RECORD.json'),('evidence_spec','ACTUAL3261_FIRST5_FIVE_ROLE_EVIDENCE_SPEC.json')]:
 need(sha(D/(alias+'-local-original.json'))==sha(W/orig),'Signed input record changed '+alias)
need(sha(D/'manifest-local-original.json')==payload['checkpoint_manifest_sha256'] and sha(D/'unsigned_payload-local-original.json')=='30439f4a4a352ce2299e369b8c03b3719c01f4219c627e144673075aa01aa928','Actual signed input pins')
gp=W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
need(sha(gp)=='c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb','Frozen release consumer source')
sp=importlib.util.spec_from_file_location('independent_release_source',gp);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
direct=(F/'ray-stdout-pid-1533312.out').read_text();events=[];releases=[]
for line in direct.splitlines():
 if 'OPSD_E1_STEP5_RELEASED_JSON=' in line:
  releases.append(g.parse_log_marker(line,'OPSD_E1_STEP5_RELEASED_JSON=',driver_pid=1533312,job_id='3261',worker_pids=tuple(starts)))
 if 'OPSD_E1_OUTER_JSON=' in line:
  events.append(g.parse_log_marker(line,'OPSD_E1_OUTER_JSON=',driver_pid=1533312,job_id='3261',worker_pids=tuple(starts)))
need(releases==[{'driver_pid':1533312,'actual_outer_updates':5,'next_global_step':6,'receipt_sha256':receipt}],'Actual unique same-driver release marker')
need([x['actual_outer_updates'] for x in events]==[1,2,3,4,5],'Do not infer unfinished sixth outer from release')
observed=read(S/'observed-remote-original.json')
need(observed['processes']['1533312']['start_ticks']=='1829085316' and observed['processes']['1533312']['in_observed_job'],'Release original same live driver identity')
for pid,start in starts.items():need(observed['processes'][str(pid)]['start_ticks']==start and observed['processes'][str(pid)]['in_observed_job'],'Release original eight worker continuity')
out={'status':'PASS_ACTUAL_EXCLUSIVE_FIRST5_PUBLICATION_PUBLIC_RSA_AND_RELEASE_ONLY','source_commit':payload['source_commit'],'job_id':'3261','driver_pid':1533312,'driver_start_ticks':'1829085316','receipt_sha256':receipt,'public_key_sha256':sha(pubpath),'independent_RSA_public_mathematical_verification':True,'actual_publication_original_sha256':sha(D/'publish-remote-original.json'),'actual_precheck_original_sha256':sha(D/'check-remote-original.json'),'exclusive_publication':True,'unchanged_eight_workers_and_same_node_service':True,'observed_job_restarts':0,'actual_original_driver_stdout_sha256':sha(F/'ray-stdout-pid-1533312.out'),'actual_release_marker':releases[0],'actual_completed_outer_count':5,'actual_outer6_accepted':False,'effect_improvement_proven':False,'final200_complete':False,'scope':'Real same-driver RELEASED5-to6 after valid external signed actual-state gate; sixth outer underway, not completed. No private key read, signing, source edit, submission, cancellation or GPU action by reviewer.'}
p=W/'ACTUAL3261_FIRST5_PUBLICATION_INDEPENDENT_REVIEW.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'review_sha256':sha(p),'status':out['status'],'receipt_sha256':receipt}))
