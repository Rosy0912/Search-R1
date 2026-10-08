import subprocess,json,base64,pathlib,hashlib
code="""import pathlib,json,base64,hashlib,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v2/evidence');rows=[];excluded=[]
for p in sorted(r.rglob('*')):
 if 'cache' in p.relative_to(r).parts or not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log'):continue
 if p.stat().st_size>16*1024**2:excluded.append(str(p));continue
 b=p.read_bytes();rows.append({'path':str(p),'relative_path':str(p.relative_to(r)),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps({'scope':'READ_ONLY_STANDARD_R1_PROFILE3155_RAW','epoch':time.time(),'rows':rows,'excluded_large_files':excluded}))
"""
p=subprocess.run(['python3','work/remote_topd.py'],input="python3 - <<'READ'\n"+code+'\nREAD\n',text=True,capture_output=True,timeout=3600)
if p.returncode:raise RuntimeError(p.stderr)
o=json.loads(p.stdout);d=pathlib.Path('outputs/remote-evidence/r1-standard32b-profile3155');d.mkdir(exist_ok=True);refs=[]
for row in o['rows']:
 b=base64.b64decode(row.pop('base64'));assert hashlib.sha256(b).hexdigest()==row['sha256'];f=d/row['relative_path'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);refs.append(row)
(d/'fetch-refs.json').write_text(json.dumps({'epoch':o['epoch'],'refs':refs,'excluded_large_files':o['excluded_large_files']},indent=2));print(json.dumps({'files':len(refs),'excluded':o['excluded_large_files']}))
