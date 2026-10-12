import pathlib,json,os,time
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'bapo-aligned-cohost-node11-' in cmd and ('python' in cmd or 'ray' in cmd or 'train' in cmd):
   rows.append(dict(pid=p.name,cmd=cmd[:1000],cwd=os.readlink(p/'cwd')))
 except OSError:pass
print(json.dumps(dict(epoch=time.time(),processes=rows)))
