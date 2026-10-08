import json,pathlib,shlex,subprocess,sys
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code=r'''import pathlib,subprocess,json,socket,datetime
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006');state=root/'logs/fresh200-startuprepair3323-once-v1'
if socket.gethostname().split('.')[0]!='h100-8':raise RuntimeError('Wrong node')
out={'hostname':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state_exists':state.exists(),'state_entries':[p.name for p in state.iterdir()] if state.exists() else [],'metadata':{}}
for name in ['SUBMIT-INTENT.json','SUBMIT-RECEIPT.json']:
 p=state/name
 if p.is_file():out['metadata'][name]=json.loads(p.read_bytes())
p=root/'logs/startuprepair3323-once-v1-error.log'
if p.exists():out['receiver_log']=p.read_text()[-10000:]
r=subprocess.run(['squeue','-h','-w','h100-8','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=15);out['queue']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out))
'''
inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
a=subprocess.run(route,capture_output=True,text=True,timeout=60)
if a.returncode:raise RuntimeError(a.stderr)
p=pathlib.Path('outputs/gigpo-e1-instruct200/FRESH200-STARTUP3323-ACTUAL-SUBMISSION-RECONCILIATION-'+sys.argv[1]+'.json');p.write_text(a.stdout);print(a.stdout)
