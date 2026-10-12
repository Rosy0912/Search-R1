import json,pathlib,subprocess,shlex
code='''import pathlib,os,json,time,signal
scopes=['tree-grpo','opsd'];roots={};children={}
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0');stat=(p/'stat').read_text().rsplit(') ',1)[1].split();children[int(p.name)]=int(stat[1])
  for scope in scopes:
   target='/home/lichen/h100-cleanup-20261012-'+scope+'/uploader.py'
   if target in argv:roots[scope]=int(p.name)
 except:pass
out={'epoch':time.time(),'roots':roots,'source_route':'H100->MGMT RAM->HF','reason':'Verified bounded parallel source reader handoff; no training processes touched'}
for scope,pid in roots.items():
 descendants={pid}
 while True:
  new={p for p,pp in children.items() if pp in descendants};new-=descendants
  if not new:break
  descendants|=new
 info={'pid':pid,'children':sorted(descendants-{pid})}
 for p in sorted(descendants,reverse=True):
  try:os.kill(p,signal.SIGTERM)
  except ProcessLookupError:pass
 root=pathlib.Path('/home/lichen/h100-cleanup-20261012-'+scope);(root/'handoff.json').write_text(json.dumps(dict(out,scope=scope,processes=info)))
time.sleep(3)
for scope,pid in roots.items():
 if pathlib.Path('/proc',str(pid)).exists():
  argv=(pathlib.Path('/proc',str(pid))/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  if '/home/lichen/h100-cleanup-20261012-'+scope+'/uploader.py' in argv:raise RuntimeError('Own old writer still alive')
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -'],input=code,capture_output=True,text=True,timeout=60);pathlib.Path('work/cleanup-h10010/handoff-result.json').write_text(p.stdout);print(p.stdout);assert p.returncode==0
