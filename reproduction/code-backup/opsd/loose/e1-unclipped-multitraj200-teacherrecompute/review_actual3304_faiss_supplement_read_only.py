"""Read-only local supplement for actual FAISS origin and historical NVML service contexts."""
from pathlib import Path
import ast,csv,hashlib,io,json,os,re,stat
H=Path(__file__).resolve().parent;D=H/'job3304-faiss-runtime-original';C=H/'job3304-complete-probe-originals';checks=[]
def need(ok,msg):
 if not ok:raise ValueError(msg)
 checks.append(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(b):
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d:raise ValueError('duplicate key')
   d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def raw(p):
 need(p.resolve(strict=True)==p and p.is_absolute(),'physical '+p.name);a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid(),'owned regular '+p.name);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);b=b''
  while True:
   v=os.read(fd,1048576)
   if not v:break
   b+=v
  f=lambda v:(v.st_dev,v.st_ino,v.st_uid,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
  need(f(a)==f(z)==f(os.fstat(fd))==f(p.lstat()) and len(b)==z.st_size,'stable same FD '+p.name);return b
 finally:os.close(fd)
old=raw(H/'ACTUAL3304_LOCAL_GPU_RETRIEVAL_INFRA_INDEPENDENT_REVIEW.json');need(sha(old)=='055048fc637a0c1ce32be30d4bc5bdbda6dba9e9eaa7a1779985e22b427ceb9f','original055048 untouched');prior=unique(old)
b=raw(D/'actual-faiss-runtime.json');need(sha(b)=='2f9db4a3970f040d95bf0bb7e640338e5861abf7eaf260802423efeca4fae3bc','actual origin JSON full SHA');r=unique(b)
pb=raw(D/'physical-original.json');p=unique(pb);need(p['sha256']==sha(b) and p['bytes']==len(b)==2298 and type(p['uid']) is int and p['uid']==0,'physical source UID0/length/fullSHA')
need(p['physical']==prior['infra_originals']['evidence/service.json']['remote_metadata']['physical'].rsplit('/evidence/service.json',1)[0]+'/actual-faiss-runtime.json' and p['stat_before']==p['stat_after']==[p['device'],p['inode'],0,len(b),p['stat_before'][4],p['stat_before'][5]] and p['reading_scope']=='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF','origin physical run/stable inode record')
service=unique(raw(C/'evidence/service.json'));need(sha(raw(C/'evidence/service.json'))==prior['infra_originals']['evidence/service.json']['sha256'],'same actual service original')
need(type(r['pid']) is int and r['pid']==service['owner']['pid']==2543106 and r['pythonpath']==service['pythonpath'],'loaded origin actual servicePID/path join')
root='/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib';need(r['runtime_root']==root and r['faiss_version']=='1.8.0','actual loaded runtime root/version')
modules=r['modules'];expected={'faiss','faiss._swigfaiss','faiss.array_conversions','faiss.class_wrappers','faiss.extra_wrappers','faiss.gpu_wrappers','faiss.loader','faiss.shared_library_preload_trigger','faiss.shared_library_preloader','faiss.swigfaiss'}
need(set(modules)==expected,'actual10 imported FAISS modules retained')
for name,row in modules.items():
 pp=Path(row['path']);need(pp.is_absolute() and Path(root) in pp.parents and '..' not in pp.parts and set(row)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'loaded root and recorded fullSHA '+name)
need(modules['faiss._swigfaiss']['path']==root+'/faiss/_swigfaiss.cpython-310-x86_64-linux-gnu.so' and modules['faiss.gpu_wrappers']['path']==root+'/faiss/gpu_wrappers.py','actual loaded native extension and GPU wrappers paths')
entry=H/'formal-71d2501d4512/retriever_entry.py';eb=raw(entry);need(sha(eb)==prior['execution_source_pins']['retriever_entry.py']['sha256']=='be0a6228de6affa84b6cb0f2106ea8309d159991c0c37aefe31941bbdf548d60','unchanged frozen entry source SHA')
t=ast.parse(eb);need(any(isinstance(n,ast.Import) and any(a.name=='faiss' for a in n.names) for n in ast.walk(t)) and b'for name,module in list(sys.modules.items())' in eb and b'assert runtime in path.parents' in eb and b"hashlib.sha256(path.read_bytes()).hexdigest()" in eb and b"'actual-faiss-runtime.json').open('x')" in eb,'entry records actually imported modules/fullbytes within asserted runtime root before unchanged retriever')
http=unique(raw(C/'evidence/http-top3-24.json'));need(sha(raw(C/'evidence/http-top3-24.json'))==prior['infra_originals']['evidence/http-top3-24.json']['sha256'] and http['service']['pid']==r['pid'] and http['service']['start']==service['owner']['start'],'actual API same loaded process start identity')
need(len(http['queries'])==len(http['response']['result'])==24 and all(len(x)==3 for x in http['response']['result']),'actual 24top3 query result joins')
sp=H/'job3304-snapshot-3/transport.stdout.original';sb=raw(sp);need(sha(sb)=='f98b60ddf1f0b6de88826b56634dfdd132e6ebb38f2cc7ac9f551926ab5a347f','existing actual snapshot3 wire exactSHA');snap=unique(sb)
need(str(snap['job_id'])=='3304' and snap['run_root']==prior['infra_originals']['approval-ready.json']['remote_metadata']['physical'].rsplit('/approval-ready.json',1)[0],'snapshot3 actualjob/run')
g=snap['live']['gpu_apps'];need(type(g['returncode']) is int and g['returncode']==0 and g['stderr']=='' and type(g['stdout']) is str,'actual historical unfiltered GPUapps query successful')
rows=list(csv.reader(io.StringIO(g['stdout'])));need(all(len(x)==3 for x in rows),'complete raw NVML rows')
parsed=[{'uuid':x[0].strip(),'pid':int(x[1]),'used_mib':int(x[2])} for x in rows]
owned=[x for x in parsed if x['pid']==r['pid']];need(len(owned)==8 and len({x['uuid'] for x in owned})==8 and {x['uuid'] for x in owned}==set(service['uuids']) and all(x['used_mib']>0 for x in owned),'historical actual servicePID GPU contexts across all8 allocated UUIDs')
result={'version':1,'status':'PASS_ACTUAL3304_FAISS_ORIGIN_AND_HISTORICAL_GPU_CONTEXT_SUPPLEMENT','scope':'Independent existing actual loaded module-origin record and instantaneous historical NVML service context supplement; no new remote/GPU or fresh binary reread','original_infra_review_sha256':sha(old),'source_commit':prior['source_commit'],'job_id':'3304','faiss_origin':r,'origin_original':{'path':str(D/'actual-faiss-runtime.json'),'sha256':sha(b),'bytes':len(b)},'physical_original':{'path':str(D/'physical-original.json'),'sha256':sha(pb),'metadata':p},'frozen_entry_source':{'path':str(entry),'sha256':sha(eb)},'snapshot3_wire':{'path':str(sp),'sha256':sha(sb),'bytes':len(sb),'utc':snap['utc'],'original_gpu_apps_query':g,'service_rows':owned},'checks':checks,'check_count':len(checks),'limitations':['Loaded module SHA values are full-byte hashes recorded by actual frozen entry process; this reviewer did not freshly read any remote Python/binary module file to recompute those hashes','Extension filename and GPU wrapper imports bind the intended GPU-capable runtime origin; this is not direct live FAISS class/index-object introspection','snapshot3 NVML raw rows establish service contexts at that recorded historical instant, not present liveness, continuous assignment or final resource release','The original055048 snapshot did not include this origin JSON; it remains unchanged and its limitations described its own evidence subset','Command/pinned CUDA-E5 and sharded-GPUFAISS source plus local API and historical8GPU contexts show no CPU-fallback or cross-cluster RPC evidence in the reviewed path; no claim about every unobserved action','No first5/full model-capacity numerical gate/long-run OOM/training effect/final200 acceptance claim']}
print(json.dumps(result,sort_keys=True,allow_nan=False))
