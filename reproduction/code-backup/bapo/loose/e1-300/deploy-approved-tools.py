import pathlib,json,hashlib,base64,subprocess,sys
root='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004'
local=pathlib.Path('work/e1-300')
m=json.loads((local/'tools-manifest.json').read_text())
approval=json.loads((local/'launch-approval.json').read_text())
assert approval['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW' and approval['root']==root
files={'tools/'+p.name:p.read_bytes() for p in (local/'tools').iterdir() if p.is_file()}
assert {pathlib.Path(n).name:hashlib.sha256(b).hexdigest() for n,b in files.items()}==m['files']
assert hashlib.sha256((local/'tools-manifest.json').read_bytes()).hexdigest()==approval['tools_manifest_sha256']
report=pathlib.Path('outputs/e1-300/INDEPENDENT-LAUNCH-TOOLS-REVIEW.md')
assert hashlib.sha256(report.read_bytes()).hexdigest()==approval['independent_review']['sha256']
assert approval['independent_review']['path']==root+'/logs/'+report.name
for name in ('launch-approval.json','tools-manifest.json'):
 files['logs/'+name]=(local/name).read_bytes()
files['logs/'+report.name]=report.read_bytes()
encoded={n:base64.b64encode(b).decode() for n,b in files.items()}
code='import pathlib,base64,json\nroot=pathlib.Path('+repr(root)+')\nfiles='+repr(encoded)+'''\nassert root.is_dir() and not root.is_symlink()
assert not list((root/'runs').iterdir()),'Run exists: no mutation'
for name,payload in files.items():
 p=root/name;assert '..' not in pathlib.PurePosixPath(name).parts and not p.is_symlink()
 data=base64.b64decode(payload)
 if p.exists():assert p.read_bytes()==data,'Different existing fixed file '+str(p)
 else:
  with p.open('xb')as f:f.write(data)
 if p.suffix in ('.sh','.sbatch'):p.chmod(0o755)
print(json.dumps({'status':'APPROVED_TOOLS_DEPLOYED_NOT_HARDWARE','files':len(files),'root':str(root)}))
'''
script=local/'remote-deploy-approved-tools.py';script.write_text(code)
subprocess.run([sys.executable,'work/h100_exec.py','--socks','--timeout','180','--stdin-file',str(script),'python3 -'],check=True)
head={n:files['tools/'+n] for n in ('submit.sh','hardware.sbatch','formal300.sbatch')}
head['approved-launch-hashes.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in head.items()},indent=2)+'\n').encode()
encoded={n:base64.b64encode(b).decode() for n,b in head.items()}
code='import pathlib,base64,json\np=pathlib.Path('+repr(root+'/scheduler-head')+')\np.mkdir(parents=True,exist_ok=True)\nassert not p.is_symlink()\nfiles='+repr(encoded)+'''\nfor name,payload in files.items():
 q=p/name;assert not q.is_symlink();data=base64.b64decode(payload)
 if q.exists():assert q.read_bytes()==data,str(q)
 else:
  with q.open('xb')as f:f.write(data)
 if q.suffix in ('.sh','.sbatch'):q.chmod(0o755)
print('DURABLE_HEAD_PREPARED_NO_SUBMISSION')
'''
script=local/'remote-deploy-approved-head.py';script.write_text(code)
subprocess.run([sys.executable,'work/h100_exec.py','--socks','--node','h100-16','--timeout','120','--stdin-file',str(script),'python3 -'],check=True)
