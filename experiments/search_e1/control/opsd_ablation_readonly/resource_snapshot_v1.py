"""Read-only resource snapshot; no Node writes, model/CUDA load, signals or leases."""
import hashlib,json,os,sys,time,types,subprocess,shutil,socket
from pathlib import Path
INPUTS={'candidate_configs': [{'bytes': 8363, 'ctime_ns': 1791476860915695278, 'device': 2065, 'inode': 466421441, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C0/receipts/config-candidate.json', 'sha256': 'fb41381d184693ad4b2319ec5e709754b98169f5d916f83ba76c2e21780b60ee'}, {'bytes': 8363, 'ctime_ns': 1791476860915695278, 'device': 2065, 'inode': 466421442, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1/receipts/config-candidate.json', 'sha256': 'd320527eef38baa1844cbab5201d4f8fbfc901c1ca35dbdb2f934d62ce8eb653'}, {'bytes': 8357, 'ctime_ns': 1791476861027695718, 'device': 2065, 'inode': 466421444, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-M/receipts/config-candidate.json', 'sha256': '79c13e363737b356ced817f7363942d2875edcc146332e403e122c55a02d2d30'}, {'bytes': 8357, 'ctime_ns': 1791476861027695718, 'device': 2065, 'inode': 466421446, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-R/receipts/config-candidate.json', 'sha256': 'ea35daa4a4499850aebd78a2a8487e421dfc16180aab4fafa1f2062fc4214eb3'}, {'bytes': 8357, 'ctime_ns': 1791476861147696192, 'device': 2065, 'inode': 466421447, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-W/receipts/config-candidate.json', 'sha256': '245021a535e85c60a9fbc27037d8157468d455b3bc05ec974e42a875aa197236'}, {'bytes': 8364, 'ctime_ns': 1791476861027695718, 'device': 2065, 'inode': 466421445, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-P/receipts/config-candidate.json', 'sha256': '85b0410fb25cbac6cb7a2eb8e69da99e726828085f85ca017f6e073e51c00dda'}, {'bytes': 8364, 'ctime_ns': 1791476860915695278, 'device': 2065, 'inode': 466421443, 'mtime_ns': 1791476813000000000, 'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-J/receipts/config-candidate.json', 'sha256': '1d6ce25e79aaa438c53ca981a84fabbdf1b6bab26e81873c70d853f2b3bd70ba'}], 'gate_source_sha': '473f6b059104a44e7d946b157dd75a7ac4b51525197b9d99ce826a5f6178321e', 'helper_root': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/training-871dc46/opsd_ablation_v1', 'helper_source_manifest': {'path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/training-871dc46/opsd_ablation_v1/source-manifest.json', 'sha256': '8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41'}, 'old_config': {'path': '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/receipts/config.json', 'sha256': 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'}, 'oracle_admission': {'bytes': 799, 'ctime_ns': 1791450238985731306, 'device': 2065, 'inode': 466389695, 'mtime_ns': 1791450238985731306, 'path': '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/receipts/oracle18004-admission.json', 'sha256': 'f69c5aab88424323c1d63a82e6a0e399ec7f9ad221e07e540d8f5b3884ab810b'}, 'publication_path': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/receipts/source-publication-871dc46.json', 'source_commit': '871dc46f9ae436e5b4115dd33e51715b858ee57d', 'stop_preservation': {'bytes': 12203, 'ctime_ns': 1791474150646930491, 'device': 2065, 'inode': 466421430, 'mtime_ns': 1791474150646930491, 'path': '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/preserved/user-stop-for-seven-ablations-20261008-2347/checkpoint-preservation.json', 'sha256': 'eb31a475607561c67ba9748387296210e872bc13978c89a3f3b17fba307dcaef'}, 'stop_terminal': {'bytes': 13715, 'ctime_ns': 1791474151494914756, 'device': 2065, 'inode': 466421436, 'mtime_ns': 1791474151494914756, 'path': '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/preserved/user-stop-for-seven-ablations-20261008-2347/terminal.json', 'sha256': '8fd4eb76e9fbbd1bddf69278fccd512a2dc38415322683c90d40bfbdbf37eba9'}, 'study_base': '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008', 'study_id': 'reverse-seven-a1dacc9d8c77451c8b04140e8e355777'}

START=time.time()
def require(value,reason):
 if not value:raise RuntimeError(reason)
def proof(path,expected=None):
 path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed during read: '+str(path))
 actual=h.hexdigest();require(expected is None or actual==expected,'SHA mismatch: '+str(path))
 return dict(path=str(path),sha256=actual,bytes=after.st_size,inode=after.st_ino,device=after.st_dev,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns)
def checked(reference):
 p=proof(reference['path'],reference['sha256']);return json.loads(Path(p['path']).read_text()),p
def snap(pid,expected=None):
 path=Path('/proc')/str(pid)
 try:
  stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
  uid=next(line.split()[1:] for line in (path/'status').read_text().splitlines() if line.startswith('Uid:'))
  actual=dict(pid=pid,uid=int(uid[0]),euid=int(uid[1]),starttick=int(fields[19]),ppid=int(fields[1]),pgid=int(fields[2]),state=fields[0])
  if expected is not None:actual['relation_to_known']='SAME_IDENTITY' if all(actual[k]==expected[k] for k in ('pid','uid','starttick')) else 'PID_REUSED_OR_UID_CHANGED'
  return actual
 except FileNotFoundError:return dict(pid=pid,state='ABSENT',relation_to_known='ABSENT')
 except Exception as error:return dict(pid=pid,state='UNKNOWN',error_type=type(error).__name__,reason=str(error))
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Explicit read-only CPU environment missing')
configs=[];raw_configs=[]
for ref in INPUTS['candidate_configs']:
 c,cp=checked(ref);raw_configs.append(c)
 require(c['study_id']==INPUTS['study_id'] and c['source_commit']==INPUTS['source_commit'],'Wrong candidate study/source')
 require(c['gates']['storage'] is None and c['preparation_status']['ready'] is False,'Candidate authorization status changed; inspect independently')
 configs.append(dict(variant=c['variant'],run_id=c['run_id'],config=cp,helper_source_manifest=c['helper_source_manifest'],owner=c['owner'],retriever_url=c['paths']['retriever_url'],gates=c['gates'],preparation_status=c['preparation_status'],output_exists=Path(c['paths']['output_root']).exists()))
require([c['variant'] for c in configs]==['C0','C1','M','R','W','P','J'],'Seven candidate order changed')
manifest,mp=checked(INPUTS['helper_source_manifest']);source=Path(INPUTS['helper_root'])/'gates.py';entry=manifest['files']['gates.py']
sp=proof(source,INPUTS['gate_source_sha']);require(sp['sha256']==entry['sha256'] and sp['bytes']==entry['bytes'],'Gate source/manifest differs')
publication,pp=checked(dict(path=INPUTS['publication_path'],sha256=hashlib.sha256(Path(INPUTS['publication_path']).read_bytes()).hexdigest()))
require(publication['accepted'] is True and publication['source_commit']==publication['remote_readback_sha']==INPUTS['source_commit'],'Published source commit/readback differs')
rows=[r for r in publication['files'] if r['path']=='experiments/search_e1/control/opsd_ablation_v1/gates.py']
require(len(rows)==1 and rows[0]['equal'] is True and rows[0]['sha256']==sp['sha256'] and rows[0]['bytes']==sp['bytes'],'Native gate source missing from publication')
g=types.ModuleType('published_readonly_ablation_gates');g.__file__=str(source);exec(compile(source.read_bytes(),str(source),'exec'),g.__dict__)
old,ocp=checked(INPUTS['old_config']);admission,ap=checked(INPUTS['oracle_admission'])
require(old['gates']['retriever_admission']==INPUTS['oracle_admission'],'Old config oracle ref changed')
gate,gp=checked(dict(path=admission['full_gpu_gate_path'],sha256=admission['full_gpu_gate_sha256']))
retrieval=g.retriever_admission(raw_configs[0],admission)
require(gate['accepted'] is True,'Original full oracle gate not accepted')
term,tp=checked(INPUTS['stop_terminal']);preservation,pvp=checked(INPUTS['stop_preservation'])
require(term['accepted'] is True and term['run_id']==old['run_id'] and term['actual_tmux_status']==143 and term['signal_termination_not_exit0'] is True,'Old stop is not actual signal terminal')
require(term['preservation']['sha256']==pvp['sha256'] and preservation['optimizer_updates']==374,'Old checkpoint preservation identity differs')
known=[term['owner']]+term['old_children'];pid_status=[dict(expected={k:v[k] for k in ('pid','uid','starttick')},actual=snap(v['pid'],v)) for v in known]
history_ref=next(row['copy'] for row in preservation['compact_copies'] if row['copy']['path'].endswith('/child-history.jsonl'))
history_proof=proof(history_ref['path'],history_ref['sha256']);history=[json.loads(line) for line in Path(history_ref['path']).read_text().splitlines() if line.strip()]
historical_unique={(row['identity']['pid'],row['identity']['starttick']):row['identity'] for row in history}
historical_status=[dict(expected={k:v[k] for k in ('pid','uid','starttick')},actual=snap(v['pid'],v)) for v in historical_unique.values()]
old_live=[];opaque=[];scanned=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 s=snap(int(proc.name))
 if s.get('uid')!=1005:continue
 scanned+=1
 try:
  env=(proc/'environ').read_bytes().split(b'\0')
  marker=next((item[len(b'E1_RUN_ID='):].decode(errors='replace') for item in env if item.startswith(b'E1_RUN_ID=')),None)
  if marker==old['run_id']:old_live.append(s)
 except FileNotFoundError:pass
 except Exception as error:opaque.append(dict(identity=s,error_type=type(error).__name__,scope='Environment unreadable; no claim of old-run absence for this process'))
devices,applications=g.inventory();oracle=retrieval['owner'];foreign=[]
for app in applications:
 if app['pid']==oracle['pid']:
  require(app['gpu_uuid']==devices[7]['uuid'] and app['used_mib']<=1024,'Old shared oracle GPU/memory differs')
 else:foreign.append(dict(application=app,process=snap(app['pid'])))
require([d['uuid'] for d in devices]==raw_configs[0]['owner']['gpu_uuids'],'Actual UUID mapping changed')
storage=[]
for path in (INPUTS['study_base'],'/data','/mnt/shared'):
 p=Path(path);exists=p.exists();anchor=p
 while not anchor.exists() and anchor!=anchor.parent:anchor=anchor.parent
 usage=shutil.disk_usage(anchor)
 command=['findmnt','--json','-T',str(anchor),'-o','TARGET,SOURCE,FSTYPE']
 actual=subprocess.run(command,text=True,capture_output=True)
 storage.append(dict(requested_path=path,exists=exists,actual_anchor=str(anchor),available_bytes=usage.free,total_bytes=usage.total,used_bytes=usage.used,
  findmnt_argv=command,findmnt_returncode=actual.returncode,findmnt=json.loads(actual.stdout) if actual.returncode==0 else None,findmnt_stderr=actual.stderr))
require('torch' not in sys.modules,'Read-only snapshot imported Torch')
result=dict(schema='opsd-ablation-turn2-resource-readonly-v1',accepted_evidence=True,started_unix=START,ended_unix=time.time(),node=socket.gethostname(),actual_uid=os.getuid(),study_id=INPUTS['study_id'],
 candidates=configs,published_gate=dict(source=sp,manifest=mp,publication=pp,source_commit=INPUTS['source_commit']),old_config=ocp,
 retriever=dict(admission=admission,admission_file=ap,full_gate_file=gp,full_gate_accepted=gate['accepted'],candidate_health=gate['candidate_health'],actual_live=retrieval),
 old_stop=dict(terminal=tp,actual_tmux_status=term['actual_tmux_status'],signal_termination_not_exit0=True,preservation=pvp,optimizer_updates=preservation['optimizer_updates'],
 canonical_checkpoint_manifest=preservation['checkpoint']['manifest'],checkpoint_content_not_rehashed_this_snapshot=True,current_known_pid_status=pid_status,
 historical_child_history=history_proof,historical_unique_count=len(historical_unique),historical_status=historical_status,selected_same_uid_environment_scan_count=scanned,old_exact_run_environment_matches=old_live,environment_unreadable=opaque),
 gpu=dict(devices=devices,compute_applications=applications,foreign_compute_apps=foreign,foreign_compute_none=len(foreign)==0,minimum_actual_free_mib=min(d['actual_free_mib'] for d in devices)),storage=storage,
 scope=dict(storage_permission='PENDING; old online200 exception is not inherited',instantaneous_only=True,not_future_launch_admission=True,oracle_pid_change_requires_new_full_gate=True,
 no_node_writes=True,no_signals=True,no_locks=True,no_gpu_workload=True,no_model_or_torch=True,http_only_original18004_health=True,no_retrieve_scan=True,no_new_observer=True))
print(json.dumps(result,sort_keys=True),flush=True)
