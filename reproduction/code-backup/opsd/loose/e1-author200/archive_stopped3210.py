import pathlib,subprocess,shlex,json,datetime
code=r'''import pathlib,json,hashlib,gzip,datetime,subprocess,os
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210'); stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=r/('stop-evidence-'+stamp);out.mkdir()
s=subprocess.check_output(['scontrol','show','job','3210','-o'],text=True);assert 'JobState=CANCELLED' in s and 'JobId=3210' in s
(out/'slurm-terminal.txt').write_text(s); records=[]
paths=[r/'train.log',r/'formal-wrapper.log',r/'resolved-profile.json',r/'acceptance-5-receipt.json',r/'bindings/manifest.json',r/'resources.ndjson']
for q in (r/'full_state').glob('global_step_*/manifest.json'): paths.append(q)
for name in ['formal-3210.out','formal-3210.err']:paths.append(r.parent.parent/'logs'/name)
for q in paths:
 if not q.is_file():continue
 stat=q.stat(); dest=out/(hashlib.sha256(str(q).encode()).hexdigest()[:12]+'-'+q.name+'.gz');h=hashlib.sha256();n=0
 with q.open('rb') as src,gzip.open(dest,'xb',compresslevel=1) as dst:
  for b in iter(lambda:src.read(1024*1024),b''):h.update(b);n+=len(b);dst.write(b)
 assert q.stat().st_size==stat.st_size==n
 gh=hashlib.sha256()
 with dest.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):gh.update(b)
 rh=hashlib.sha256();rn=0
 with gzip.open(dest,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):rh.update(b);rn+=len(b)
 assert rh.hexdigest()==h.hexdigest() and rn==n
 records.append({'path':str(q),'bytes':n,'sha256':h.hexdigest(),'gzip_path':str(dest),'gzip_sha256':gh.hexdigest(),'gzip_bytes':dest.stat().st_size,'full_decompression_verified':True})
text=(r/'train.log').read_text(errors='replace');rows=[]
for line in text.splitlines():
 if 'OPSD_E1_OUTER_JSON=' in line:
  try:rows.append(json.JSONDecoder().raw_decode(line.split('OPSD_E1_OUTER_JSON=',1)[1])[0])
  except ValueError:pass
assert [x['actual_outer_updates'] for x in rows]==list(range(1,len(rows)+1))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
value={'job_id':'3210','reason':'latest explicit human stop and fresh200 restart; not fault/recovery','slurm':s,'full_logged_outers':len(rows),'latest':{'step':rows[-1]['actual_outer_updates'],'reward':rows[-1]['metrics'].get('critic/score/mean')} if rows else None,'full_state_dirs':sorted(p.name for p in (r/'full_state').glob('global_step_*')),'gpu_compute_apps_unfiltered':gpu,'files':records,'scope':'old source and checkpoints retained; directory existence is not fresh content/HF reload acceptance','archive_dir':str(out)}
(out/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
w=pathlib.Path(__file__).resolve().parent;q=subprocess.run(['python3',str(w.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=205,env={**__import__('os').environ,'OPSD_REMOTE_TIMEOUT':'180'});(w/'stopped3210-archive.stderr').write_bytes(q.stderr);(w/'stopped3210-archive.stdout').write_bytes(q.stdout)
if q.returncode:print(q.stderr.decode());raise SystemExit(q.returncode)
d=json.loads(q.stdout);(w/'stopped3210-archive-original.json').write_bytes(q.stdout);print(json.dumps({k:d[k] for k in ['job_id','full_logged_outers','latest','gpu_compute_apps_unfiltered','archive_dir','full_state_dirs']},indent=2))
