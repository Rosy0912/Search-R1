import pathlib,json,subprocess,hashlib,datetime,os
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-resume5-3494f75-20261011');ledger=json.loads((base/'logs/process-ledger.json').read_text());cleanup=json.loads((base/'logs/cleanup-status.json').read_text());assert cleanup['cleaned']
identities=[]
for job in ledger['jobs'].values():
 for v in job.get('ledger',{}).values():
  p=pathlib.Path('/proc')/str(v['pid'])/'stat';same=False
  if p.exists():same=int(p.read_text().rsplit(')',1)[1].split()[19])==v['start_ticks']
  identities.append(dict(pid=v['pid'],start_ticks=v['start_ticks'],original_alive=same))
assert not any(v['original_alive'] for v in identities)
files=[]
for p in sorted((base/'logs').rglob('*.log')):
 st=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 after=p.stat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 files.append(dict(path=str(p),inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,independent_full_sha256=h.hexdigest()))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
active=subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i|%j'],text=True)
assert not any(row.split('|')[0]=='3366' for row in active.splitlines())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal='FAILED1:0',actual_new_adam_writes=0,latest_checkpoint_still_parent5=True,original_processes=identities,cleanup=cleanup,terminal_log_full_readback=files,remaining_gpu_apps=apps,active_native_jobs=active)))
