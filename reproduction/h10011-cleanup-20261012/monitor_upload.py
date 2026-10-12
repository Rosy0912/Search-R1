import pathlib,subprocess,shlex,json,time
code='''import pathlib,json,time,os
r=pathlib.Path('/home/lichen/h100-cleanup-20261012-bapo');j={'epoch':time.time(),'progress':[]}
for p in r.glob('progress-*.json'):j['progress'].append(json.loads(p.read_text()))
j['errors']=[json.loads(p.read_text()) for p in r.glob('file-*-error.json')]
for fn in ['controller.json','verified-progress.json','complete.json']:
 p=r/fn
 if p.exists():j[fn]=json.loads(p.read_text())
c=j['controller.json'];j['children']=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  st=(p/'stat').read_text().split(') ')[1].split()
  if int(st[1])==c['pid']:j['children'].append({'pid':p.name,'state':st[0]})
 except (OSError,ValueError):pass
print(json.dumps(j))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30)
print(p.stdout,p.stderr)
if p.returncode==0:pathlib.Path('outputs/cleanup-h10011/upload-progress-%d.json'%int(time.time())).write_text(p.stdout)
