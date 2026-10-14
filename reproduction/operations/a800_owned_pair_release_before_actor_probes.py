
"""Release exactly the two reviewed capacity holders; never start a trainer."""
import contextlib,hashlib,io,json,os,signal,subprocess,sys,time,socket
from pathlib import Path
assert sys.flags.optimize==0 and hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
assert Path(sys.prefix).resolve()==Path('/data/search-g1/env').resolve()
p=json.load(sys.stdin);base=Path(p['base']);repo=base/'repo'
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-cb042832b1814fad86a9734bbdd4e94c'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id'] and socket.gethostname()==p['hostname']
assert hashlib.sha256((base/'logs/actor-probe-production-deployment-review.json').read_bytes()).hexdigest()=='882f9907a8bfee0b6b11e2f4e763460be5f45f9c08c48d2a0d5f8bdd56016f35'
assert hashlib.sha256((repo/'reproduction/a800/owned_processes.py').read_bytes()).hexdigest()=='df83d747f3b86a187eb2dc4267ac146a6afeab7338f274ab5ecd8e7de9e2969e'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='c4d156de124de7a94652d75858bcbabc8c56083c'
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import identity,signal_owned
expected={2435117:(601538171,[0],'e1d46b43e3b96bbb43268b9531a5d633934964ead29854570ccfba1df96772f9','743d055d4d6e868b34f0280e010d7bd67a516fed6f33779d986f7756f23a6e82'),2340358:(601041566,[1,2,3],'861a00c09acf3a438563e9a159e99b88da00bb9d2c0316ab326efb119bd25cee','7284c1eaee74d06b245072fec114646d5d9649baba70d6282ec58a47fca0d4a4')}
assert len(p['keepers'])==2 and {x['receipt']['identity']['pid'] for x in p['keepers']}==set(expected)
records=[];owners={}
for item in p['keepers']:
 keeper=item['receipt'];pid=keeper['identity']['pid'];ticks,indices,sha,reader_sha=expected[pid]
 assert keeper['identity']['start_ticks']==ticks and keeper['identity']['sid']==keeper['identity']['pgid']==pid and keeper['identity']['uid']==keeper['identity']['euid']==os.getuid()==0
 assert keeper['physical_indices']==indices and keeper['source_sha256']==sha and keeper['bytes_per_gpu']==60*1024**3
 assert keeper['boot_id']==p['boot_id'] and keeper['hostname']==p['hostname']
 assert hashlib.sha256(item['reader'].encode()).hexdigest()==reader_sha
 original=sys.stdin;output=io.StringIO()
 try:
  sys.stdin=io.StringIO(json.dumps(keeper))
  with contextlib.redirect_stdout(output):exec(compile(item['reader'],'<published-strong-owned-read>','exec'),{'__name__':'__main__'})
 finally:sys.stdin=original
 observation=json.loads(output.getvalue());records.append(observation)
 assert observation['actual_owned_60GiB_each_verified'] and observation['post_READY_retention_sample_verified'] and observation['process']['alive']
 assert observation['boot_id']==p['boot_id'] and observation['hostname']==p['hostname'] and observation['state']['status']=='READY'
 assert observation['state']['failure'] is None and observation['state']['release'] is None
 owners[pid]=keeper['identity']
unknown=[]
for observation in records:
 for app in observation['foreign_training_cuda_applications']+observation['foreign_auxiliary_cuda_applications']:
  owner=owners.get(app['pid'])
  if owner is None or not all(app.get(k)==owner[k] for k in ('pid','start_ticks','sid','uid')):unknown.append(app)
print(json.dumps(dict(role='actual_pair_release_current_preflight',checked_unix=time.time(),keepers=[r['process'] for r in records],records_sha256=[hashlib.sha256((json.dumps(r,sort_keys=True)+'\n').encode()).hexdigest() for r in records],true_foreign_assigned=unknown)),flush=True)
assert not unknown,'Assigned foreign context: no claims/signals/trainer'
claims=[]
for item in p['keepers']:
 keeper=item['receipt'];owner=keeper['identity'];actual=identity(owner['pid']);assert all(actual[k]==owner[k] for k in ('pid','start_ticks','sid','uid'))
 path=Path(keeper['namespace'])/'release-for-reviewed-A9-pair-once.json'
 data=dict(role='exact_owned_capacity_pair_release_before_reviewed_A9',base=str(base),identity=owner,boot_id=p['boot_id'],hostname=p['hostname'],source_sha256=keeper['source_sha256'],claimed_unix=time.time(),signal='SIGTERM',no_foreign_signal=True,no_automatic_retry=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(data,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 claims.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=owner))
for claim in claims:
 signal_owned(claim['identity'],signal.SIGTERM)
 print(json.dumps(dict(role='exact_owned_pair_SIGTERM_sent',identity=claim['identity'],claim_sha256=claim['sha256'],checked_unix=time.time())),flush=True)
def inventory():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=10)
 mapping={row.split(',')[1].strip().lower():int(row.split(',')[0]) for row in raw.splitlines()}
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 rows=[]
 for row in raw.splitlines():
  fields=[x.strip() for x in row.split(',')];index=mapping[fields[0].lower()]
  if index in [0,1,2,3,5,6]:rows.append(dict(gpu_index=index,pid=int(fields[1]),used_mib=fields[2],uuid=fields[0]))
 return rows
start=time.monotonic();samples=[]
while True:
 live=[]
 for owner in owners.values():
  try:
   actual=identity(owner['pid']);same=all(actual[k]==owner[k] for k in ('pid','start_ticks','sid','uid'))
   if same and actual['state'] not in ('Z','X'):live.append(actual)
  except (FileNotFoundError,ProcessLookupError):pass
 rows=inventory();owned=[r for r in rows if r['pid'] in owners]
 samples.append(dict(checked_unix=time.time(),owned_live_births=len(live),owned_CUDA_contexts=len(owned),assigned_CUDA_contexts=len(rows)))
 if not live and not owned:
  print(json.dumps(dict(role='actual_pair_owned_exit_and_CUDA_drain',checked_unix=time.time(),owned_births_exited=True,owned_CUDA_released=True,all_assigned_contexts_absent=not rows,remaining_assigned_contexts=rows,bounded_samples=samples,new_trainer_started=False,native_algorithm_accepted=False)),flush=True);break
 assert time.monotonic()-start<60,'Owned actual drain incomplete within bound; do not launch'
 time.sleep(2)
