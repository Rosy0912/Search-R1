python3 - <<'PY'
import pathlib,json,hashlib,datetime,subprocess,re,os,collections
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
B=R/'logs/ACTION-PREFIX-NATIVE-BINDING-2262942-v1.json'
b=B.read_bytes();assert hashlib.sha256(b).hexdigest()=='c5e184a458cc6ae10cb3c924c845564f1684e0347dfc9fb0391c2fd1b7bd7c76'
x=json.loads(b)
def command(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return dict(args=a,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def fmeta(p,content=False):
 p=pathlib.Path(p);v={'path':str(p),'exists':p.exists()}
 if p.is_file():
  b=p.read_bytes();v.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());
  if content:v['text']=b.decode('utf8')
 return v
aliases=[];summaries={}
for field in ('files','evidence'):
 groups=collections.defaultdict(list);raw=collections.Counter()
 for e in x.get(field,[]):
  p=pathlib.Path(e['path']);raw[str(p)]+=1
  st=p.stat();rp=str(p.resolve(strict=True))
  groups[rp].append(dict(path=str(p),bytes=e.get('bytes'),sha256=e['sha256'],actual_bytes=st.st_size,realpath=rp,dev=st.st_dev,ino=st.st_ino,mtime_ns=st.st_mtime_ns))
 dup=[v for v in groups.values() if len(v)>1]
 summaries[field]=dict(entries=len(x.get(field,[])),raw_duplicate_paths=[p for p,n in raw.items() if n>1],resolved_duplicate_groups=len(dup))
 for v in dup:
  aliases.append(dict(field=field,items=v,bound_bytes_sha_identical=len({(t['bytes'],t['sha256']) for t in v})==1,actual_sizes_match=all(t['bytes']==t['actual_bytes'] for t in v)))
log=R/'logs/action-prefix-native-3297.log'
pattern=re.compile(r'(?<!\d)3297(?!\d)');daemon=[]
for p in sorted(pathlib.Path('/var/log/slurm').glob('*slurmd*')):
 if not p.is_file():continue
 matches=[]
 try:
  with p.open(errors='replace') as f:
   for ln,line in enumerate(f,1):
    if pattern.search(line):matches.append({'line_number':ln,'text':line.rstrip('\n')})
  if matches:daemon.append({'path':str(p),'matching_original_lines':matches})
 except Exception as e:daemon.append({'path':str(p),'read_error':repr(e)})
uid=os.getuid();cg=pathlib.Path(f'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_3297')
other_cg=pathlib.Path(f'/sys/fs/cgroup/freezer/slurm/uid_{uid}/job_3297')
value=dict(status='ACTUAL_STARTUP_FAILURE_METADATA_NOT_GPU_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job='3297',node=command(['hostname']),binding=fmeta(B),binding_alias_summary=summaries,resolved_aliases=aliases,original_log=fmeta(log,True),original_receipt=fmeta(R/'logs/ACTION-PREFIX-NATIVE-SUBMIT-RECEIPT-2262942-v1.json',True),original_intent=fmeta(R/'logs/ACTION-PREFIX-NATIVE-SUBMIT-INTENT-2262942-v1.json',True),scontrol=command(['scontrol','show','job','3297','-o']),sacct=command(['sacct','-j','3297','--format=JobID,State,ExitCode,NodeList','-n','-P']),queue=command(['squeue','-w','h100-8','-h','-o','%i %T %j %N']),gpu=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader']),slurmd_original_matches=daemon,native_run={'path':str(R/'runs/gigpo-e1-action-prefix-native-job3297'),'exists':(R/'runs/gigpo-e1-action-prefix-native-job3297').exists()},cgroup_candidates=[{'path':str(p),'exists':p.exists(),'scope':'Standard candidate path, not header-bound native scope'} for p in (cg,other_cg)],formal_updates_contributed=0)
out=R/'logs/ACTION-PREFIX-NATIVE-FAILURE3297-METADATA-v1.json';data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();assert len(data)<16*1024*1024
with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'alias_summary':summaries,'run_exists':value['native_run']['exists'],'gpu':value['gpu'],'slurmd':daemon},ensure_ascii=False))
PY
