"""Independent local byte/infra review; no remote/GPU imports."""
from pathlib import Path
import ast,hashlib,json,math,os,re,stat
H=Path(__file__).resolve().parent;D=H/'job3304-complete-probe-originals'
S='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6';J='3304'
ISH='3f03ccd612498ad013571d4963b3fe3e540b2c3c0035f1f3cc590ea779879907'
R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
checks=[];pins={};sourcepins={}
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(b):
 def pairs(xs):
  out={}
  for k,v in xs:
   if k in out:raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def read(p):
 need(p.is_absolute() and p.resolve(strict=True)==p,'physical local '+p.name)
 a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid(),'owned regular local '+p.name)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);chunks=[]
  while True:
   b=os.read(fd,1024*1024)
   if not b:break
   chunks.append(b)
  b=b''.join(chunks);f=lambda v:(v.st_dev,v.st_ino,v.st_uid,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
  need(f(a)==f(z)==f(os.fstat(fd))==f(p.lstat()) and len(b)==z.st_size,'stable raw same-FD '+p.name);return b
 finally:os.close(fd)
ib=read(D/'inventory.json');need(sha(ib)==ISH,'exact actual capsule inventory SHA');inv=unique(ib)
need(inv['source_commit']==S and inv['job_id']==J and inv['run_root']==R and inv['hostname']=='h100-10' and type(inv['uid']) is int and inv['uid']==0,'actual capsule source job node UID')
need(inv['kind']==inv['scope']=='BOUNDED_OWN_RUN_PROBE_BYTES_SNAPSHOT_NOT_ACCEPTANCE_OR_TERMINAL_PROOF' and len(inv['files'])==52 and inv['missing_optional']==[],'actual52-member snapshot byte scope')
def member(n,parse=True):
 b=read(D/n);r=inv['files'][n];need(sha(b)==r['sha256'] and len(b)==r['bytes'],'full original SHA/length '+n)
 need(r['physical']==R+'/'+n and type(r['uid']) is int and r['uid']==0,'remote physical UID '+n)
 a,z=r['stat_before'],r['stat_after'];need(len(a)==len(z)==6 and all(type(v) is int for v in a+z) and a[:3]==z[:3]==[r['device'],r['inode'],0] and a[3]==len(b),'remote stable inode/length '+n)
 if n.endswith('.log'):need(z[3]>=len(b) and r['reading_scope']=='VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED','nonclosed log prefix '+n)
 else:need(a==z and r['reading_scope']=='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF','stable member observation '+n)
 pins[n]={'path':str(D/n),'sha256':sha(b),'bytes':len(b),'remote_metadata':r};return unique(b) if parse else b
bb=read(H/'runtime-tools-bound-fcecbea5d939-v5.json');need(sha(bb)=='8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228','V5 exact source binding');bound=unique(bb)
ready=member('approval-ready.json');manifest=member('bindings/manifest.json');src=member('bindings/source.json');alloc=member('bindings/allocation.json');env=member('bindings/environment.json');assets=member('bindings/tool_manifest.json');data=member('bindings/data_manifest.json')
need(pins['bindings/manifest.json']['sha256']==ready['binding_sha256'],'ready actual manifest exact pin')
for role in ['allocation','source','environment','tool_manifest','data_manifest']:
 n='bindings/'+role+'.json';need(manifest['files'][role]=={'path':R+'/'+n,'sha256':pins[n]['sha256']},'manifest role '+role)
need(src['source_commit']==ready['source_commit']==manifest['source_commit']==S and ready['job_id']==alloc['job_id']==J and ready['run_root']==manifest['run_root']==R and src['clean'] is True and src['source_files']==bound['source_files'] and src['formal_manifest_sha256']==bound['formal']['manifest_sha256'],'actual source/job/run/clean19/formal join')
need(pins['bindings/data_manifest.json']['sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','shared common data manifest exact SHA')
phys=alloc['nvml_double_guard'][0]['inventory'];need(phys==alloc['nvml_double_guard'][1]['inventory'] and len(phys)==8 and len({v['uuid'] for v in phys})==8 and all(v['name']=='NVIDIA H100 80GB HBM3' for v in phys) and alloc['selected_cuda_devices']==list(map(str,range(8))),'actual pre-init physical8H100 inventory/selectors')
f=dict(v.split('=',1) for v in alloc['scontrol'].split() if '=' in v);need(f['JobId']==J and f['JobState']=='RUNNING' and f['NodeList']=='h100-10' and f['TresPerNode']=='gres:gpu:h100:8' and f['OverSubscribe']=='NO','actual exclusive one-node8H100 Slurm original')
formal=Path(bound['formal']['local_directory']);fmraw=read(formal/'manifest.json');need(sha(fmraw)==bound['formal']['manifest_sha256'],'actual formal manifest pin');fm=unique(fmraw)
for n in ['hardware_rank.py','formal_supervisor.py','retriever_entry.py','owned_runtime.py']:
 b=read(formal/n);need(sha(b)==fm['tools'][n]==bound['formal']['files'][n],'pinned execution tool '+n);sourcepins[n]={'path':str(formal/n),'sha256':sha(b),'bytes':len(b)}
repo=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'
for n in ['reproduction/h20/retriever_local.py','search_r1/search/retrieval_server.py']:
 b=read(repo/n);need(sha(b)==src['source_files'][n],'pinned retrieval source '+n);sourcepins[n]={'path':str(repo/n),'sha256':sha(b),'bytes':len(b)}
def cgroup(t):return re.search(r'(?:^|/)job_3304(?:/|$)',t,re.M) is not None and re.search(r'(?:^|/)step_0(?:/|$)',t,re.M) is not None
ranks=[]
for r in range(8):
 v=member(f'evidence/hardware-rank-{r}.json');need(all(type(v[k]) is int for k in ['rank','local_rank','world_size','pid','uid']) and v['rank']==v['local_rank']==r and v['world_size']==8 and v['pid']>0 and v['uid']==0 and v['job_id']==J and v['source_commit']==S and cgroup(v['cgroup']),'actual owned rank/local/world/PID '+str(r))
 need(v['uuid']==v['actual_cuda_uuid']==phys[r]['uuid'] and v['gpu_name']=='NVIDIA H100 80GB HBM3','actual CUDA UUID/selectors '+str(r))
 need(v['allreduce_value']==36.0 and v['flash_finite'] is True and v['torch']=='2.4.0+cu121','NCCL BF16 sum36 and native Flash finite '+str(r));ranks.append(v)
need(len({v['pid'] for v in ranks})==len({v['uuid'] for v in ranks})==8,'distinct8workers and CUDA contexts')
he=member('evidence/hardware-exit.json');need(type(he['returncode']) is int and he['returncode']==0,'hardware real child exit0');member('evidence/hardware-torchrun.log',False)
tree=ast.parse((formal/'hardware_rank.py').read_text());calls=[v for v in ast.walk(tree) if isinstance(v,ast.Call)]
need(any(isinstance(v.func,ast.Attribute) and v.func.attr=='init_process_group' and v.args and isinstance(v.args[0],ast.Constant) and v.args[0].value=='nccl' for v in calls),'hardware actual source NCCL backend')
need(any(isinstance(v.func,ast.Name) and v.func.id=='flash_attn_func' for v in calls),'hardware actual source Flash forward')
service=member('evidence/service.json');http=member('evidence/http-top3-24.json');o=service['owner'];o2=http['service']
need(type(o['pid']) is int and o['pid']>0 and type(o['uid']) is int and o['uid']==0 and type(o['start']) is str and o['start'].isdigit() and cgroup(o['cgroup']) and o['state'] not in ['Z','X'],'own actual retriever PID/start/UID/cgroup')
need({k:o[k] for k in ['pid','start','uid','cgroup']}=={k:o2[k] for k in ['pid','start','uid','cgroup']} and o2['state'] not in ['Z','X'],'HTTP captured same live service identity')
need(len(service['uuids'])==8 and set(service['uuids'])==set(ready['nvml_uuids'])=={v['uuid'] for v in ranks},'service allocated GPU UUID set matches actual8contexts')
a=assets['results'];e5dir=str(Path(a['e5'][0]['path']).parent);command=[env['python'],'-u',bound['formal']['remote_directory']+'/retriever_entry.py','--index_path',a['index']['path'],'--corpus_path',a['corpus']['path'],'--retriever_model',e5dir,'--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18084','--threads','8','--topk','3']
need(service['command']==command,'actual CUDA E5/GPU FAISS command exact asset paths top3 loopback')
need(service['pythonpath']==src['repo']+':/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib' and env['values']['OPSD_E1_RETRIEVER_URL']=='http://127.0.0.1:18084/retrieve','actual local library path and loopback training route')
need(assets['status']=='FULL_TOOL_ASSET_CONTENT_VERIFIED' and len(a['e5'])==8 and all(Path(v['path']).parent==Path(e5dir) and v['original']['repo']=='intfloat/e5-base-v2' and v['original']['revision']=='f52bf8ec8c7124536f0efb74aca902b2995e5bcd' for v in a['e5']),'eight fixed E5 asset records')
need(a['index']['sha256']=='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166' and a['corpus']['sha256']==a['gzip_decompressed']['sha256']=='85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9' and len(a['index']['segment_sha256'])==2,'exact common Wiki2018 corpus/index pinned full SHA')
qs=http['queries'];rows=http['response']['result'];need(http['job_id']==J and len(qs)==len(rows)==24 and len(set(qs))==8 and qs[:8]*3==qs,'24 query rows are8unique repeated3times')
need(type(http['seconds']) in [int,float] and math.isfinite(http['seconds']) and http['seconds']>0,'positive actual retrieval elapsed')
for n,row in enumerate(rows):
 need(len(row)==3 and all(type(v['score']) in [int,float] and math.isfinite(v['score']) and type(v['document']['contents']) is str and bool(v['document']['contents']) and type(v['document']['id']) is str and v['document']['id'].isdigit() for v in row),'actual3scored docs '+str(n));need(all(row[j]['score']>=row[j+1]['score'] for j in range(2)),'actual descending scores '+str(n))
need(all(rows[n]==rows[n+8]==rows[n+16] for n in range(8)),'repeated raw retrieval responses exactly agree')
wrapper=member('formal-wrapper.log',False);ret=member('retriever.log',False)
need(('Started server process ['+str(o['pid'])+']').encode() in ret and b'Uvicorn running on http://127.0.0.1:18084' in ret and b'POST /retrieve HTTP/1.1" 200 OK' in ret,'raw retriever prefix actual serverPID/loopback/HTTP200')
need(len({(v['remote_metadata']['device'],v['remote_metadata']['inode']) for v in pins.values()})==len(pins),'infra original physical inode identities distinct')
result={'version':1,'status':'PASS_ACTUAL3304_LOCAL_GPU_RETRIEVAL_INFRA_EVIDENCE','scope':'Actual small native8 hardware and local GPU retrieval original conformance only; no capacity math/first5/effect/terminal/release acceptance','source_commit':S,'job_id':J,'snapshot_inventory_sha256':ISH,'snapshot_observed_utc':inv['observed_utc'],'infra_originals':pins,'execution_source_pins':sourcepins,'check_count':len(checks),'check_labels':checks,'hardware_rank_records':ranks,'hardware_exit':he,'retrieval':{'service':service,'same_http_service_identity':o2,'query_count':24,'unique_queries':8,'documents_per_query':3,'document_entries':72,'unique_document_ids':len({v['document']['id'] for row in rows for v in row}),'seconds':http['seconds'],'score_min':min(v['score'] for row in rows for v in row),'score_max':max(v['score'] for row in rows for v in row),'index':{k:a['index'][k] for k in ['path','sha256','bytes']},'corpus':{k:a['corpus'][k] for k in ['path','sha256','bytes']},'e5_assets':[{k:v[k] for k in ['path','sha256','bytes']} for v in a['e5']]},'limitations':['Hardware scope is small NCCL BF16 allreduce plus native Flash BF16 forward; no Flash backward or OPSD numerical-gradient claim','service uuids are allocation-assignment records, not per-application NVML process-to-UUID telemetry in this subset','Pinned command/source/API support CUDA E5 and GPU FAISS route; no live tensor/index-object introspection by this reviewer','Asset SHA records are bound manifest plus pinned wrapper pre-GPU full-content validation path; reviewer performed no remote full-asset reread','actual-faiss-runtime.json absent from52-member capsule; exact live FAISS imported-module provenance not freshly audited here','24queries contain8unique repeats; not retrieval accuracy benchmark','Wrapper/retriever are nonclosed captured prefixes, not lifetime logs or termination/resource-release evidence','No long-run noOOM, first5, final200/51713 or effect-improvement claim']}
print(json.dumps(result,sort_keys=True,allow_nan=False))
