import ast,pathlib,tempfile,subprocess,hashlib,json,os
p=pathlib.Path('work/e1-300/publish_actual3210_first5_release.py');raw=p.read_bytes();value=next(n.value for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='publishcode' for t in n.targets));out=[]
for case in ['normal','existing_target','target_symlink','temp_regular','temp_symlink']:
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td).resolve(); env=b'{"unsigned_test":"no_private_keys_or_real_receipt"}'
  code=eval(compile(ast.Expression(value),'<publishcodeonly>','eval'),{'R':str(r),'hashlib':hashlib,'env':env}).replace('r.stat().st_uid==0',f'r.stat().st_uid=={os.getuid()}')
  if case=='existing_target':(r/'acceptance-5-receipt.json').write_bytes(b'keep')
  if case=='target_symlink':
   (r/'other').write_bytes(b'keep');(r/'acceptance-5-receipt.json').symlink_to(r/'other')
  if case.startswith('temp_'):
   code=code.replace("secrets.token_hex(16)","'testfixedtoken'");t=r/'.acceptance-5-testfixedtoken.tmp'
   if case=='temp_regular':t.write_bytes(b'keep')
   else:(r/'other').write_bytes(b'keep');t.symlink_to(r/'other')
  result=subprocess.run(['python','-c',code],input=env,capture_output=True)
  if case=='normal':assert result.returncode==0,(result.stderr.decode());assert (r/'acceptance-5-receipt.json').read_bytes()==env and (r/'acceptance-5-receipt.json').stat().st_mode&0o777==0o600
  else:
   assert result.returncode!=0
   if case=='existing_target':assert (r/'acceptance-5-receipt.json').read_bytes()==b'keep'
   if case in ('target_symlink','temp_symlink'):assert (r/'other').read_bytes()==b'keep'
   if case=='temp_regular':assert t.read_bytes()==b'keep'
  out.append({'case':case,'passed':True,'returncode':result.returncode,'error_last_line':result.stderr.decode().splitlines()[-1] if result.stderr else None})
a={'scope':'CPU exact publication primitive only; localUID adaptation and controlled temp nonce; no signer/privatekey/remote execution','script_sha256':hashlib.sha256(raw).hexdigest(),'cases':out};pathlib.Path('work/e1-300/actual3210-signing-publication-independent-cpu.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
