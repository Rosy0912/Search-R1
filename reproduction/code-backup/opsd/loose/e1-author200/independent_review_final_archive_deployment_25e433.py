"""Read-only actual software publication originals; no publisher or remote execution."""
import ast,datetime,hashlib,json,os,re,stat
from pathlib import Path
W=Path(__file__).resolve().parent; D=W/'final-archive-deployment-25e433c0a1d3-originals'; B=W/'final-archive-25e433c0a1d3'
PIN='25e433c0a1d32e9a13f228537e13bda337089186a7a4be0349923580a97a9354';SOURCE='188f4e5e7b63cf479290d0888a4a075499886750';RUNTIME='d83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006');RUN=ROOT/'runs/formal-3261';REPO=ROOT/'repo-188f4e5e7b63';SNAP=W/'actual-snapshots/3261-20261005T232501896059Z/original-files'
records={}
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def meta(s):return {k:getattr(s,k)for k in ('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
def read(p,pin=None):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts,'Absolute physical path required');d=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for n in p.parts[1:-1]:
   nxt=os.open(n,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=d);os.close(d);d=nxt
  before=os.stat(p.name,dir_fd=d,follow_symlinks=False);fd=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=d)
  try:
   s=os.fstat(fd);need(s.st_uid==os.getuid()and s.st_nlink==1 and stat.S_ISREG(s.st_mode),'Owned nonalias original required');chunks=[]
   while True:
    b=os.read(fd,1048576)
    if not b:break
    chunks.append(b)
   raw=b''.join(chunks);need(meta(before)==meta(s)==meta(os.fstat(fd))==meta(os.stat(p.name,dir_fd=d,follow_symlinks=False))and len(raw)==s.st_size,'Original changed')
   need(pin is None or sha(raw)==pin,'Original SHA differs');records[str(p)]={'path':str(p),'sha256':sha(raw),'bytes':len(raw),'metadata':meta(s)};return raw
  finally:os.close(fd)
 finally:os.close(d)
def unique(b):
 def pairs(xs):
  v={}
  for k,x in xs:need(k not in v,'Duplicate JSON key');v[k]=x
  return v
 def bad(v):raise ValueError('Nonfinite JSON')
 v=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);json.dumps(v,allow_nan=False);return v
def fields(raw):
 items=re.findall(r'(?:^|\s)([A-Za-z0-9/:_]+)=([^\s]*)',raw);need(len(items)==len(dict(items)),'Repeated scheduler field');return dict(items)
manifest=unique(read(B/'manifest.json',PIN));freeze=unique(read(W/'FINAL_ARCHIVE_25E433_IMMUTABLE_BUNDLE_INDEPENDENT_REVIEW.json','0cee3b526ee32d38bade63a70239bf0257321b2e206dc4e631bd4c8ff57206ee'))
need(freeze['status']=='PASS_INDEPENDENT_FINAL_ARCHIVE_IMMUTABLE_BUNDLE_REVIEW'and freeze['manifest_sha256']==PIN and freeze['collector_runtime_sha256']==RUNTIME and freeze['source_commit']==SOURCE,'Actual prior freeze review differs')
need({p.name for p in D.iterdir()}=={'publish-once-stdout.original.json','publish-once-stderr.original','publish-once-exit-original.json','publish-once-intent.json','publisher-remote-source.py'},'Exact one actual attempt originals required')
intent=unique(read(D/'publish-once-intent.json'));code=read(D/'publisher-remote-source.py','9bc954a405f08d8c164af964a855a280a38c48b24cdb2319a9ee3a1cec1ea8df')
need(intent['manifest_sha256']==PIN and intent['independent_freeze_review_sha256']==records[str(W/'FINAL_ARCHIVE_25E433_IMMUTABLE_BUNDLE_INDEPENDENT_REVIEW.json')]['sha256']and intent['remote_publisher_source_sha256']==sha(code),'Actual intent binding differs')
publisher=read(B/'PREPARATION_ONLY_publish_final_archive_bundle.py','25d11d5957c7297ec1d8479d150803ced19985140721e8668a09b00e465a6067')
p_ast=ast.parse(publisher);values=[ast.literal_eval(n.value)for n in p_ast.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CODE'for t in n.targets)]
need(len(values)==1 and values[0].encode()==code,'Exact reviewed portable publisher CODE differs')
exit_record=unique(read(D/'publish-once-exit-original.json'));stderr=read(D/'publish-once-stderr.original')
need(type(exit_record['returncode'])is int and exit_record['returncode']==0 and stderr==b'','Actual transport exit/stderr differs')
actual=unique(read(D/'publish-once-stdout.original.json','fb26e48bc71949ac0dadcc4f25a627e5b04d98f0189f7fe121dc3a4fcea3b366'))
need(actual['scope']=='ACTUAL_NEW_ARCHIVE_SOFTWARE_PUBLICATION_NOT_COLLECTOR_OR_FINAL_ACCEPTANCE'and actual['manifest_sha256']==PIN and actual['directory']==str(ROOT/'tools'/B.name)and actual['directory_mode']=='0o555'and actual['manifest_last']is True and actual['source_commit']==SOURCE and actual['collector_executed']is False and type(actual['runtime_count'])is int and actual['runtime_count']==1 and type(actual['existing_dependency_count'])is int and actual['existing_dependency_count']==31,'Actual software publication scope/identity differs')
expected={**manifest['tools'],**manifest['review_records'],'manifest.json':PIN};published=actual['published']
need(type(published)is list and len(published)==23 and {r['name']for r in published}==set(expected) and len({r['name']for r in published})==23 and published[-1]['name']=='manifest.json','Actual23 member/order differs')
for r in published:
 need(set(r)=={'name','bytes','sha256','mode'}and type(r['bytes'])is int and r['bytes']>=0 and r['sha256']==expected[r['name']]and r['mode']=='0o444','Actual member fullSHA/mode differs')
 raw=read(B/r['name'],r['sha256']);need(len(raw)==r['bytes'],'Actual member fullsize differs')
verified=actual['verified_dependencies'];need(set(verified)==set(manifest['dependencies'])and len(verified)==31,'Actual fresh31 path set differs')
for path,pin in manifest['dependencies'].items():
 r=verified[path];need(set(r)=={'sha256','bytes'}and r['sha256']==pin and type(r['bytes'])is int and r['bytes']>=0,'Actual dependency fullSHA/size differs')
 local=[v for v in freeze['all_read_original_records'].values()if v['sha256']==pin]
 need(local and all(v['bytes']==r['bytes']for v in local),'Fresh dependency size differs from previously pinned original')
# Successful exact CODE returns only after the second whole-byte dictionary
# equals the first. The output carries one hash map; it does not serialize both.
code_ast=ast.parse(code);func=next(n for n in code_ast.body if isinstance(n,ast.FunctionDef)and n.name=='publish')
source_lines=code.decode().splitlines()
required=["verified={p:read(c,p,h) for p,h in m['dependencies'].items()}","before=source_and_allocation(c,verified)","verified_after={p:read(c,p,h) for p,h in m['dependencies'].items()}","need(verified_after==verified,'Existing dependencies changed during publication')","after=source_and_allocation(c,verified_after)"]
indices=[]
for line in required:need(source_lines.count(' '+line)==1,'Required reviewed guard missing');indices.append(source_lines.index(' '+line))
need(indices==sorted(indices)and indices[-1]<func.end_lineno-1,'Whole31 before/after guards/order differ')
old=unique(read(SNAP/'bindings/allocation.json',manifest['dependencies'][str(RUN/'bindings/allocation.json')]))
old_fields=fields(old['scontrol']); cgroup='1:freezer:/slurm_h100-10/uid_0/job_3261/step_0\n0::/system.slice/slurmd.service\n'
scopes=[]
for label in ('allocation_before','allocation_after'):
 v=actual[label];need(v['source_commit']==SOURCE and v['source_clean']is True and v['scope']=='Fresh deployment identity only; no collector or terminal acceptance','Fresh source observation differs')
 f=fields(v['scontrol_original']);fixed={'JobId':'3261','JobState':'RUNNING','UserId':'root(0)','NodeList':'h100-10','BatchHost':'h100-10','NumNodes':'1','NumCPUs':'192','TresPerNode':'gres:gpu:h100:8','OverSubscribe':'NO','Restarts':'0','MinMemoryNode':'1T','WorkDir':str(REPO),'StdOut':str(ROOT/'logs/formal-3261.out'),'StdErr':str(ROOT/'logs/formal-3261.err')}
 need(all(f.get(k)==x for k,x in fixed.items())and all(f[k]==old_fields[k]for k in ('StartTime','SubmitTime')),'Actual RUNNING8GPU/exclusive192cpu1T original allocation differs')
 identity=v['original_driver_current_identity'];need(type(identity['pid'])is int and identity['pid']==1533312 and identity['start_ticks']==identity['expected_start_ticks']=='1829085316'and identity['status']=='ORIGINAL_OR_CURRENT_PROCESS_ALIVE'and identity['state']not in('Z','X')and identity['cgroup']==cgroup and identity['in_training_job']is True and identity['uids']==[0]*4 and all(type(u)is int for u in identity['uids']),'Actual original driver differs')
 for name in ('stat_before','stat_after'):
  rawstat=identity['raw'][name];need(rawstat.startswith('1533312 (ray::main_task)'),'Actual proc stat PID/name differs');parts=rawstat.rsplit(')',1)[1].split();need(parts[19]=='1829085316'and parts[0]not in('Z','X'),'Actual proc start/state differs')
 uid=re.search(r'^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',identity['raw']['status'],re.M);need(uid is not None and list(map(int,uid.groups()))==[0]*4 and identity['raw']['cgroup']==cgroup,'Actual raw UID/cgroup differs')
 scopes.append({'label':label,'allocation_fields':f,'process_identity':identity})
report={'version':1,'status':'PASS_INDEPENDENT_ACTUAL_FINAL_ARCHIVE_SOFTWARE_DEPLOYMENT_ONLY','reviewer_role':'INDEPENDENT_READ_ONLY_ACTUAL_DEPLOYMENT_ORIGINALS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':PIN,'collector_runtime_sha256':RUNTIME,'source_commit':SOURCE,'actual_result_sha256':records[str(D/'publish-once-stdout.original.json')]['sha256'],'actual_returncode':0,'directory':actual['directory'],'checks':{'actual_one_attempt_intent_source_stderr_exit0':True,'23_members_exact_fullSHA_size0444_directory0555_manifest_last':True,'portable_CODE_equals_exact_reviewed25d1_constant':True,'fresh31_wholeSHA_with_before_after_fullbyte_equality_enforced_by_successful_exact_CODE':True,'before_after_original3261_RUNNING8H100_exclusive192cpu1T_NoRestart_StartSubmit':True,'before_after_driver1533312_start1829085316_actual_UID0_cgroup_and_stat':True,'fresh_source188_clean_and_old90_oldformal_dependency_SHA_unchanged':True,'runtime_not_executed':True},'actual_published_members':published,'actual_fresh_dependency_hash_map':verified,'actual_allocation_and_process_observations':scopes,'original_records':records,'publisher_or_remote_execution_performed_by_reviewer':False,'training_or_GPU_mutation_performed_by_reviewer':False,'collector_executed':False,'final_training_effect_or_resource_release_accepted':False,'scope':'ACTUAL_SOFTWARE_PUBLICATION_BYTES_AND_GUARDS_ONLY_NOT_COLLECTOR_EXECUTION_OR_FINAL_ACCEPTANCE','limitations':['The exact successful portable publisher code reads all31 whole originals twice and compares the two raw byte maps; the saved output serializes one full hash/length map, not two distinct raw dependency snapshots.','Member hashes/size/modes/order are actual publisher readback observations, not another independent remote read. Reviewer performed zero remote actions.','RUNNING and current UID/start/cgroup samples are bounded observations at deployment; later state or release is not inferred.','No final collector observation, checkpoint/HF200, final51713, child/wrapper/Slurm terminal, effect or resource release acceptance is asserted.']}
out=W/'ACTUAL_FINAL_ARCHIVE_25E433_DEPLOYMENT_INDEPENDENT_REVIEW.json'
with out.open('x')as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'report':str(out),'sha256':sha(out.read_bytes()),'status':report['status']}))
