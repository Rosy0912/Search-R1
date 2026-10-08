import base64,hashlib,json,pathlib,subprocess,sys
job=int(sys.argv[1]);assert job>0
code="""import base64,hashlib,json,pathlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs')
paths=[root/'g1-initial-static-cpu-JOB.log']
for record in root.glob('*cpu-submission.json'):
 try:obj=json.loads(record.read_text())
 except (ValueError,OSError):continue
 if obj.get('job_id')==JOB:paths.append(record)
d=root/'g1-initial-static-cpu-JOB/recipe'
if d.exists():paths+=list(d.glob('*.json'))
a=[]
for p in paths:
 if p.exists():
  b=p.read_bytes();a.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps(a))
""".replace('JOB',str(job))
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'FETCH'\n"+code+'\nFETCH\n',text=True,capture_output=True)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
out=pathlib.Path('outputs/remote-evidence')/('g1-initial-recipe-'+str(job));out.mkdir(exist_ok=True)
refs=[]
for rec in json.loads(p.stdout):
 b=base64.b64decode(rec.pop('base64'));assert hashlib.sha256(b).hexdigest()==rec['sha256'];dest=out/pathlib.Path(rec['path']).name;dest.write_bytes(b);rec['local_path']=str(dest);refs.append(rec)
(out/'fetch-refs.json').write_text(json.dumps(refs,indent=2));print(json.dumps(refs))
