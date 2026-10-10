"""Move only the exact live CPU memory observer outside controller ancestry; no training signal or restart."""
import json,os,signal,time,shlex,subprocess,hashlib,shutil
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010')
cfgpath=B/'receipts/training-config.json';raw=cfgpath.read_bytes();cfg=json.loads(raw);import sys;sys.path.insert(0,cfg['paths']['helper_root']);import gates
binding=json.loads((B/'receipts/memory-binding.json').read_text());old=binding['observer_identity'];actual=json.loads((B/'receipts/scoped-cpu-processes.json').read_text())['roles'][0]['identity'];assert int(Path('/proc/'+str(old['pid'])+'/stat').read_text().split()[21])==old['starttick'] and actual['uid']==1005;assert actual['run_id']!=cfg['run_id'];source=Path(cfg['paths']['helper_root'])/'memory_observer.py';assert str(source) in actual['argv']
assert actual['pid'] not in {a['pid'] for a in gates.inventory()[1]},'Observer holds GPU; do not signal'
controller=json.loads((B/'runs/status.json').read_text())['owner'];assert actual['ppid']==controller['pid'],'Only the exact accidental child-observer case is authorized'
fd=os.pidfd_open(controller['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM) # Exact zero-Adam held controller only; releases orphan CPU zombie
try:
 fd=os.pidfd_open(actual['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM)
except ProcessLookupError:pass
for _ in range(30):
 try:v=gates.process(actual['pid'])
 except (FileNotFoundError,ValueError,PermissionError):break
 if not gates.same_process(v,old):break
 time.sleep(.2)
else:raise RuntimeError('Old observer did not terminate')
preserve=B/'preserved'/('observer-parent-release-'+str(old['pid']));preserve.mkdir()
for name in ['memory-binding.json','memory-live.json','memory-live.samples.jsonl','scoped-cpu-processes.json']:
 p=B/'receipts'/name
 if p.exists():shutil.move(str(p),str(preserve/name))
newmarker='j-bf16-memory-independent-'+str(int(time.time()));command='env E1_RUN_ID='+shlex.quote(newmarker)+" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "+shlex.quote(cfg['paths']['python'])+' -B '+shlex.quote(str(source))+' --config '+shlex.quote(str(cfgpath))+' --output '+str(B/'receipts/memory-live.json')+' --binding-output '+str(B/'receipts/memory-binding.json')+' >> '+str(B/'logs/memory-independent.log')+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s','e1-j-bf16-memory-independent-20261010',command],check=True)
for _ in range(30):
 if (B/'receipts/memory-binding.json').exists() and (B/'receipts/memory-live.json').exists():break
 time.sleep(1)
nb=json.loads((B/'receipts/memory-binding.json').read_text());owner=gates.process(nb['observer_identity']['pid']);assert owner['ppid']!=controller['pid'] and owner['run_id']==newmarker
role=dict(role='memory_observer',identity=owner,no_gpu_or_training=True,argv_sha256=hashlib.sha256(json.dumps(owner['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),input_config=dict(path=str(cfgpath),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),helper_source_manifest=cfg['helper_source_manifest'],publication=cfg['gates']['publication'],source=dict(path=str(source),bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
registry=dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=hashlib.sha256(raw).hexdigest(),roles=[role]);(B/'receipts/scoped-cpu-processes.json').write_text(json.dumps(registry,indent=2)+'\n')
proof={'accepted':True,'old':old,'new':owner,'preserved':str(preserve),'config_sha256':hashlib.sha256(raw).hexdigest(),'training_config_unchanged':True,'sampling_not_repeated':True,'only_observer_signaled':True,'observed_unix':time.time()};(B/'receipts/observer-detached.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
