import json,pathlib,subprocess,sys
job=int(sys.argv[1]);root=sys.argv[2]
assert root.startswith('/dataset/shaoshitong/search-r1-rl/topd/logs/') and '..' not in pathlib.PurePosixPath(root).parts
code="""import json,pathlib,time,subprocess
r=pathlib.Path(ROOT);d={'epoch':time.time(),'job_id':JOB,'queue':subprocess.check_output(['squeue','-h','-j',str(JOB),'-o','%i|%T|%M'],text=True),'files':[]}
if r.exists():
 for p in sorted(r.rglob('*')):
  if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log') and 'binding' not in p.name:
   st=p.stat();item={'path':str(p.relative_to(r)),'bytes':st.st_size,'modified_epoch':st.st_mtime}
   if p.suffix=='.log' or p.name in ('initial-complete.json','cleanup.json','retriever-startup-failures.jsonl'):
    with p.open('rb') as f:f.seek(max(0,st.st_size-2000));item['tail']=f.read().decode('utf-8',errors='replace')
   d['files'].append(item)
print(json.dumps(d))
""".replace('ROOT',repr(root)).replace('JOB',str(job))
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'LIVE'\n"+code+'\nLIVE\n',text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
d=json.loads(p.stdout.strip().splitlines()[-1]);out=pathlib.Path('outputs/remote-evidence/g1-live-'+str(job)+'.jsonl')
with out.open('a') as f:f.write(json.dumps(d)+'\n')
print(json.dumps(d))
