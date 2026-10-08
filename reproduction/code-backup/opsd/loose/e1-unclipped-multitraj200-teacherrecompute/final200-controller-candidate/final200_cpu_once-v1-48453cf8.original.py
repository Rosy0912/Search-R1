"""Preparation-only, once CPU orchestration; never training, GPU, restore or receipt.

Production start is intentionally limited to the still-live original3304 allocation.
The sealed inspectors distinguish their actual execution job from stored training3304.
No missing wait or partial output is converted to a successful inspection.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import types

SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
STORED_JOB='3304'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
REPO=ROOT/('repo-'+SOURCE[:12]);RUN=ROOT/'runs'/('formal-'+STORED_JOB)
PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
OUTPUT=RUN/'evidence/final200-cpu-inspection-once-v1'
HERE=Path(__file__).absolute()
RECEIPT='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
PUBLIC='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
COMMON='e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
PINS={'common.py':COMMON,
 'observe_final200.py':'9ac3015f60dc98012678203049985b0fb575cd44ca2e8a287e1b61cc54c6ee31',
 'inspect_final200_native.py':'03dcdc1e0cd84eb77780d1e52193c7b904e1e084467e8fb130e184ce008777d9',
 'load_final200_hf.py':'9fbe4ae0eaca5bcbbf75611ae47b2f0c8e94745c123138e6fee004558adb87cb',
 'check_final200_archives.py':'7fc3e7b96465a681390aae1a59ca9d3aae381b95ef5ea80914425f13ada8f4ed'}
MATH=ROOT/'tools/first5-native-dc1905562f2f/inspect_payloads.py'
MATH_SHA='77afa00b2b3e0fdcf674f5022f8b6cc39e12818bc7e825f8e58128fecab0f68b'
HF=ROOT/'tools/first5-1b16c2227caf/hf_cpu_reload.py'
HF_SHA='b2e02be33d04d925af04a6a8f4efea4ba93997146197d54729d0a9715754c419'
SCOPE='CPU_FINAL200_ONCE_CONTENT_INSPECTION_NOT_TRAINING_RESTORE_EFFECT_RECEIPT_OR_TERMINAL'
NATIVE_SCOPE='NATIVE_CPU_FINAL200_STORED_CHECKPOINT_CONTENT_INSPECTION_NOT_LIVE_RESTORE'
HF_SCOPE='HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE'
ARCHIVE_SCOPE='ALL200_RAW_PAIRED_TRAINING_ARCHIVES_AND_DATA_CURSOR_CONTENT_INSPECTION_NOT_FINAL_EVALUATION'
MAX_DRIVER_PREFIX=128*1024**2
FSDP_WARNING_PATH=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py')
FSDP_WARNING_SHA='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
CHECKPOINT_WARNING_PATH=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')


def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pin(value):
 require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'Exact actual SHA required');return value
def typed(value,target,message):require(type(value) is type(target) and value==target,message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def parse_json(raw):
 def pairs(items):
  result={}
  for key,value in items:require(key not in result,'Duplicate JSON key');result[key]=value
  return result
 def number(value):
  result=float(value);require(math.isfinite(result),'Nonfinite JSON number');return result
 return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def physical(path,directory=False):
 p=Path(path);require(p.is_absolute() and p.resolve(strict=True)==p,'Physical absolute path required')
 a=p.lstat();require(a.st_uid==os.getuid() and (stat.S_ISDIR(a.st_mode) if directory else stat.S_ISREG(a.st_mode)),'Owned physical type required');return p
def identity(a):return (a.st_dev,a.st_ino,a.st_uid,a.st_mode,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
def stable_raw(path,expected=None,limit=32*1024**2):
 p=physical(path);before=p.lstat();require(before.st_size<=limit,'Bounded file exceeded');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);require(identity(before)==identity(opened) and stat.S_ISREG(opened.st_mode),'Opened file identity differs')
  chunks=[];count=0
  while True:
   raw=os.read(fd,1024**2)
   if not raw:break
   count+=len(raw);require(count<=limit,'Bounded file grew beyond limit');chunks.append(raw)
  raw=b''.join(chunks)
  require(identity(before)==identity(os.fstat(fd))==identity(p.lstat()) and len(raw)==before.st_size,'Same-FD file changed')
  if expected is not None:typed(sha(raw),pin(expected),'Exact file SHA differs')
  return raw
 finally:os.close(fd)
def sync_dir(path):
 fd=os.open(physical(path,True),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_bytes_once(path,raw):
 p=Path(path);physical(p.parent,True);require(type(raw) is bytes,'Raw bytes required')
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  require(os.fstat(fd).st_uid==os.getuid(),'Output owner differs')
  with os.fdopen(fd,'wb',closefd=False) as stream:stream.write(raw);stream.flush();os.fsync(fd)
 finally:os.close(fd)
 sync_dir(p.parent)
def write_once(path,value):write_bytes_once(path,canonical(value))
def reserve_once(path):
 p=Path(path);physical(p.parent,True);require(not os.path.lexists(p),'Existing or unknown once attempt never retried')
 p.mkdir(mode=0o700);sync_dir(p.parent);sync_dir(p)
def host():require(sys.platform=='linux' and socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0,'Actual root on h100-10 required')


def process(pid,job=None,step=None):
 require(type(pid) is int and pid>0,'Actual positive PID required');p=Path('/proc')/str(pid)
 first=(p/'stat').read_text();status=(p/'status').read_text();cg=(p/'cgroup').read_text();last=(p/'stat').read_text()
 start=lambda raw:raw.rsplit(')',1)[1].split()[19]
 require(start(first)==start(last) and re.fullmatch('[1-9][0-9]*',start(first)),'PID start changed')
 match=re.search(r'^Uid:\s+(.*)$',status,re.M);require(match is not None,'Process UID missing')
 uids=[int(v) for v in match.group(1).split()];require(uids==[os.getuid()]*4,'Actual four UIDs differ')
 state=re.search(r'^State:\s+(\S+)',status,re.M).group(1);require(state in ('R','S','D','I'),'Process not live')
 if job is not None:
  require(type(job) is str and re.fullmatch('[1-9][0-9]*',job),'Actual numeric job required')
  if step is not None:require(type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)',step),'Actual step required')
  suffix=(r'(?:\.scope)?/step[_-]'+re.escape(step)+r'(?:\.scope)?(?:/|$)') if step is not None else r'(?:\.scope)?(?:/|$)'
  require(re.search(r'(?:^|/)job[_-]'+re.escape(job)+suffix,cg,re.M),'Same-path job/step identity differs')
 return dict(pid=pid,start_ticks=int(start(first)),uid=os.getuid(),uids=uids,state=state,cgroup=cg,node=socket.gethostname(),status_original=status,utc=utc())


def bundle(manifest_sha,controller_sha):
 host();pin(manifest_sha);pin(controller_sha);directory=ROOT/'tools'/('final200-'+manifest_sha[:12]);physical(directory,True)
 require(HERE==directory/'final200_cpu_once.py','Only isolated fixed bundle controller permitted')
 stable_raw(HERE,controller_sha);raw=stable_raw(directory/'manifest.json',manifest_sha);manifest=parse_json(raw)
 require(type(manifest) is dict and set(manifest)=={'version','kind','source_commit','tools'},'Bundle schema differs')
 typed(manifest['version'],1,'Bundle version differs');typed(manifest['kind'],'FINAL200_READ_ONLY_RUNTIME_BUNDLE','Bundle kind differs');typed(manifest['source_commit'],SOURCE,'Bundle source differs')
 tools=manifest['tools'];require(type(tools) is dict and set(PINS)|{'final200_cpu_once.py'}<=set(tools) and len(tools)<=128,'Bundle execution member inventory differs')
 require(set(p.name for p in directory.iterdir())==set(tools)|{'manifest.json'},'Unmanifested bundle members present')
 typed(tools['final200_cpu_once.py'],controller_sha,'Caller controller pin differs')
 for name,expected in tools.items():
  require(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name) and name!='manifest.json','Unsafe member name')
  stable_raw(directory/name,pin(expected));require(stat.S_IMODE((directory/name).stat().st_mode)==0o444,'Immutable bundle file mode differs')
 for name,expected in PINS.items():typed(tools[name],expected,'Sealed runtime pin differs: '+name)
 require(stat.S_IMODE(directory.stat().st_mode)==0o555 and stat.S_IMODE((directory/'manifest.json').stat().st_mode)==0o444,'Immutable bundle modes differ')
 stable_raw(MATH,MATH_SHA);stable_raw(HF,HF_SHA)
 source=stable_raw(directory/'common.py',COMMON);c=types.ModuleType('final200_once_common');c.__file__=str(directory/'common.py')
 exec(compile(source,c.__file__,'exec',dont_inherit=True,optimize=0),c.__dict__)
 return directory,c


def read_driver_prefix(driver):
 """Exact complete-LF prefix of a growing owned stdout, not lifetime closure."""
 pid=driver['pid'];before=process(pid,STORED_JOB);typed(before['start_ticks'],int(driver['start']),'Original driver PID reused')
 fdpath=Path('/proc')/str(pid)/'fd/1';origin=Path(os.readlink(fdpath));physical(origin)
 require(Path('/tmp/ray') in origin.parents and origin.name.endswith('.out'),'Owned Ray driver stdout path differs')
 a=origin.lstat();require(a.st_size<=MAX_DRIVER_PREFIX,'Driver prefix exceeded fixed bound');fd=os.open(origin,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);require((opened.st_dev,opened.st_ino,opened.st_uid)==(a.st_dev,a.st_ino,a.st_uid) and stat.S_ISREG(opened.st_mode),'Driver stdout FD identity differs')
  raw=b''
  while len(raw)<a.st_size:
   part=os.read(fd,min(1024**2,a.st_size-len(raw)));require(part,'Driver stdout prefix shortened');raw+=part
  boundary=raw.rfind(b'\n')+1;require(boundary>0,'No complete driver stdout line');complete=raw[:boundary]
  os.lseek(fd,0,os.SEEK_SET);repeat=b''
  while len(repeat)<boundary:
   part=os.read(fd,min(1024**2,boundary-len(repeat)));require(part,'Driver stdout prefix shortened');repeat+=part
  after=os.fstat(fd);path_after=origin.lstat()
  for value in (after,path_after):require((value.st_dev,value.st_ino,value.st_uid)==(a.st_dev,a.st_ino,a.st_uid) and value.st_size>=a.st_size,'Driver stdout changed identity/truncated')
  require(repeat==complete and Path(os.readlink(fdpath))==origin,'Driver stdout fixed prefix/FD changed')
  typed(process(pid,STORED_JOB)['start_ticks'],before['start_ticks'],'Driver start changed during prefix read')
  return complete,dict(origin_path=str(origin),writer=before,first_observed_bytes=a.st_size,complete_prefix_bytes=boundary,complete_prefix_sha256=sha(complete),incomplete_tail_bytes=len(raw)-boundary,incomplete_tail_sha256=sha(raw[boundary:]),device=a.st_dev,inode=a.st_ino,scope='INFLIGHT_COMPLETE_LF_PREFIX_NOT_CLOSED_WRITER_LIFETIME')
 finally:os.close(fd)


def validate_driver_prefix(raw,context,c):
 # Native211 inventories imported project callables, not all Torch files.
 # Use the exact already-reviewed native warning origin/pin separately; do not
 # invent extra entries or require nonexistent Torch keys in that inventory.
 gate=context['gate'];fsdp_raw=stable_raw(FSDP_WARNING_PATH,FSDP_WARNING_SHA);physical(CHECKPOINT_WARNING_PATH);rows=[];released=[]
 kwargs=dict(driver_pid=context['manifest']['driver_pid'],job_id=STORED_JOB,worker_pids=[w['pid'] for w in context['workers']],checkpoint_warning_path=str(CHECKPOINT_WARNING_PATH),fsdp_warning_source_bytes=fsdp_raw)
 for line in raw.decode('utf-8',errors='strict').splitlines():
  if 'OPSD_E1_OUTER_JSON=' in line:
   row=gate.parse_log_marker(line,'OPSD_E1_OUTER_JSON=',**kwargs);step=len(rows)+1
   require(set(row)=={'actual_outer_updates','global_step','driver_pid','job_id','source_commit','config_sha256','metrics'},'Outer marker fields differ')
   for key,target in dict(actual_outer_updates=step,global_step=step,driver_pid=context['manifest']['driver_pid'],job_id=STORED_JOB,source_commit=SOURCE,config_sha256=CONFIG).items():typed(row[key],target,'Actual complete outer identity differs: '+key)
   require(step<=200,'Executed201 or later is forbidden');gate.validate_outer(row['metrics'],world=8,profile='multitraj200');rows.append(row)
  if 'OPSD_E1_STEP5_RELEASED_JSON=' in line:released.append(gate.parse_log_marker(line,'OPSD_E1_STEP5_RELEASED_JSON=',**kwargs))
 require(len(rows)==200,'Real complete200 outer prefix required')
 require(len(released)==1 and set(released[0])=={'driver_pid','actual_outer_updates','next_global_step','receipt_sha256'},'Exactly one historical5 release required')
 for key,target in dict(driver_pid=context['manifest']['driver_pid'],actual_outer_updates=5,next_global_step=6,receipt_sha256=RECEIPT).items():typed(released[0][key],target,'Original signed5 release differs')
 return dict(real_complete_outers=200,no201_in_read_prefix=True,last_outer=rows[-1],release=released[0],native_warning_bindings=dict(fsdp_path=str(FSDP_WARNING_PATH),fsdp_source_sha256=FSDP_WARNING_SHA,fsdp_source_bytes=len(fsdp_raw),checkpoint_warning_path=str(CHECKPOINT_WARNING_PATH),scope='Typed log framing only; separate from native211 computation import closure'),scope='Launch prerequisite only; not final logs or writer lifetime closure')


def scheduler():
 argv=['scontrol','show','job',STORED_JOB,'-o'];r=subprocess.run(argv,capture_output=True,text=True,timeout=20)
 require(r.returncode==0,'Original scheduler query failed');fields=dict(x.split('=',1) for x in r.stdout.split() if '=' in x)
 for key,target in dict(JobId=STORED_JOB,JobState='RUNNING',NodeList='h100-10',TresPerNode='gres:gpu:h100:8',OverSubscribe='NO',UserId='root(0)').items():typed(fields.get(key),target,'Own original allocation differs: '+key)
 return dict(argv=argv,stdout_original=r.stdout,stderr_original=r.stderr,returncode=r.returncode,waited=True,utc=utc())


def precondition(directory,c,out):
 context=c.accepted_context(RECEIPT);driver=process(context['manifest']['driver_pid'],STORED_JOB)
 typed(driver['start_ticks'],int(context['driver_identity']['start']),'Original driver start differs')
 raw,prefix=read_driver_prefix(context['driver_identity']);write_bytes_once(out/'driver-200-prefix.original.log',raw)
 joined=validate_driver_prefix(raw,context,c)
 for rank,w in enumerate(context['workers']):
  p=c.checkpoint_paths()[f'actor/training_state/rank_{rank}.json'];report,_=c.read_json(p,root=RUN);c.rank_report(report,rank,200,'FULL_STATE_SERIALIZED_OK',w['pid']);require(report['object_ids']==w['object_ids'],'Saved200 live object continuity differs')
 d,_=c.read_json(c.checkpoint_paths()['driver/driver.json'],root=RUN)
 for key,target in dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,pid=driver['pid'],uid=0,job_id=STORED_JOB).items():typed(d.get(key),target,'Saved200 driver header differs')
 for path in list(c.checkpoint_paths().values())+list(c.archive_paths().values()):physical(path)
 result=dict(version=1,scope=SCOPE,source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id=STORED_JOB,driver=driver,driver_prefix=prefix,outer_prerequisite=joined,checkpoint_member_count=31,archive_member_count=4000,scheduler=scheduler(),accepted5=context['accepted5'],bundle=str(directory),utc=utc())
 write_once(out/'launch-prerequisites.original.json',result);return result


def waited_code(rc):return rc if rc>=0 else 128-rc
def wait_child(argv,out,prefix,identity_reader=process,cwd=None):
 """Genuine wait is retained even if capturing child/parent identity fails."""
 require(type(argv) is list and argv and all(type(v) is str for v in argv),'Explicit argv required')
 with (out/(prefix+'-stdout.original')).open('xb') as stdout,(out/(prefix+'-stderr.original')).open('xb') as stderr:
  try:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=cwd,close_fds=True)
  except BaseException as error:
   write_once(out/(prefix+'-spawn-failure.original.json'),dict(argv=argv,exception_type=type(error).__name__,exception=str(error),waited=False,scope=SCOPE,utc=utc()));raise
  errors=[];started=dict(argv=argv,child_pid=p.pid,parent_pid=os.getpid(),child=None,parent=None,scope=SCOPE,utc=utc())
  try:
   for key,pid in [('parent',os.getpid()),('child',p.pid)]:
    try:started[key]=identity_reader(pid)
    except BaseException as error:errors.append(dict(stage=key,exception_type=type(error).__name__,exception=str(error)))
   started['capture_errors']=errors;write_once(out/(prefix+'-child-start.original.json'),started)
  finally:
   rc=p.wait();stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
   result=dict(returncode=rc,waited=True,waited_by_pid=os.getpid(),child_pid=p.pid,capture_errors=errors,scope=SCOPE,utc=utc())
   write_once(out/(prefix+'-exit.original.json'),result)
 return result,started


def role_command(role,launch,lsha):return [str(PYTHON),'-B',str(HERE),role,'--launch',str(launch),'--launch-sha256',lsha]
def srun_command(launch,lsha,execution_job):
 typed(execution_job,STORED_JOB,'V1 only permits own live original3304')
 return ['srun','--jobid='+execution_job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=none','--nodelist=h100-10','--chdir='+str(REPO),'env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(REPO)]+role_command('cpu-step',launch,lsha)
def launch_fields(c):
 require(type(c) is dict and set(c)=={'version','scope','stored_training_job_id','execution_job_id','source_commit','config_sha256','accepted5_receipt_sha256','public_key_sha256','controller_sha256','bundle_manifest_sha256','driver_pid','driver_start_ticks','output','precondition_sha256'},'Once intent schema differs')
 for key,target in dict(version=1,scope=SCOPE,stored_training_job_id=STORED_JOB,execution_job_id=STORED_JOB,source_commit=SOURCE,config_sha256=CONFIG,accepted5_receipt_sha256=RECEIPT,public_key_sha256=PUBLIC,output=str(OUTPUT)).items():typed(c[key],target,'Once intent fixed field differs: '+key)
 for key in ('controller_sha256','bundle_manifest_sha256','precondition_sha256'):pin(c[key])
 require(type(c['driver_pid']) is int and c['driver_pid']>0 and type(c['driver_start_ticks']) is int and c['driver_start_ticks']>0,'Typed actual original driver identity required')
def load_launch(path,expected):
 physical(OUTPUT,True);typed(str(path),str(OUTPUT/'once-intent.original.json'),'Only fixed once intent allowed')
 c=parse_json(stable_raw(path,expected));launch_fields(c);directory,common=bundle(c['bundle_manifest_sha256'],c['controller_sha256'])
 proof=parse_json(stable_raw(OUTPUT/'launch-prerequisites.original.json',c['precondition_sha256']))
 typed(proof['driver']['pid'],c['driver_pid'],'Intent/proof driver differs');typed(proof['driver']['start_ticks'],c['driver_start_ticks'],'Intent/proof driver start differs')
 return c,directory,common
def live_original(c):
 driver=process(c['driver_pid'],STORED_JOB);typed(driver['start_ticks'],c['driver_start_ticks'],'Original driver gone/reused');return dict(driver=driver,scheduler=scheduler())


def helper_commands(directory,out,job,step,records):
 base=['--common-sha256',COMMON,'--accepted5-receipt-sha256',RECEIPT,'--execution-job-id',job,'--execution-step-id',step]
 inventory=['--inventory',str(out/'observed200.original.json'),'--inventory-sha256',records['observe']['sha256']] if 'observe' in records else []
 native=['--native-report',str(out/'native200.original.json'),'--native-report-sha256',records['native']['sha256'],'--native-inspector',str(directory/'inspect_final200_native.py'),'--native-inspector-sha256',PINS['inspect_final200_native.py']] if 'native' in records else []
 return {'observe':[str(PYTHON),'-B',str(directory/'observe_final200.py')]+base+['--output',str(out/'observed200.original.json')],
 'native':[str(PYTHON),'-B',str(directory/'inspect_final200_native.py')]+base+inventory+['--math-helper',str(MATH),'--output',str(out/'native200.original.json')],
 'hf':[str(PYTHON),'-B',str(directory/'load_final200_hf.py')]+base+inventory+native+['--hf-helper',str(HF),'--output',str(out/'hf200.original.json')],
 'archives':[str(PYTHON),'-B',str(directory/'check_final200_archives.py')]+base+inventory+native+['--output',str(out/'archives200.original.json')]}


def qualify(stage,value,c,execution,child,records):
 require(type(value) is dict,'Actual result must be an object')
 for key,target in dict(version=1,source_commit=SOURCE,config_sha256=CONFIG,job_id=STORED_JOB,run_root=str(RUN),checkpoint_step=200,common_source_sha256=COMMON).items():typed(value.get(key),target,'Actual report binding differs: '+key)
 require(type(child) is dict,'Actual child identity unavailable');got=value['execution_identity']
 for key,target in dict(execution_job_id=execution['execution_job_id'],execution_step_id=execution['execution_step_id'],stored_training_job_id=STORED_JOB,pid=child['pid'],start_ticks=child['start_ticks'],uid=0,node=execution['node'],cgroup=child['cgroup'],uids=child['uids'],python=str(PYTHON)).items():typed(got.get(key),target,'Actual helper process identity differs: '+key)
 require(got['pid']!=execution['pid'],'Helper cannot impersonate controller');typed(value['accepted5']['receipt_sha256'],RECEIPT,'Actual accepted5 continuity receipt differs');typed(value['accepted5']['public_key_sha256'],PUBLIC,'Actual accepted5 public key differs')
 if stage=='observe':
  require('passed' not in value,'Inventory cannot claim acceptance');c.validate_inventory(value);typed(value['observer_source_sha256'],PINS['observe_final200.py'],'Observer source differs');return
 typed(value.get('passed'),True,'Exit0 without content PASS rejected');typed(value.get('cuda_is_initialized'),False,'Helper initialized CUDA')
 expected={'native':('inspect_final200_native.py',NATIVE_SCOPE),'hf':('load_final200_hf.py',HF_SCOPE),'archives':('check_final200_archives.py',ARCHIVE_SCOPE)}[stage]
 typed(value.get('scope'),expected[1],'Content scope differs');typed(value.get('inspector_source_sha256'),PINS[expected[0]],'Inspector source differs');typed(value.get('inventory_sha256'),records['observe']['sha256'],'Observed200 prerequisite differs')
 if stage=='native':require(c.canonical(value['clock'])==c.canonical(c.clock()) and value.get('hf_all_rank_slices_exact') is True and len(value['ranks'])==8,'Native200 clock/allrank content differs')
 else:
  typed(value.get('native_report_sha256'),records['native']['sha256'],'Native200 prerequisite differs');typed(value.get('native_inspector_source_sha256'),PINS['inspect_final200_native.py'],'Native source prerequisite differs')
 if stage=='hf':
  for key in ('all_loaded_parameters_finite','actual_cpu_forward_finite','tied_embeddings'):typed(value.get(key),True,'HF content differs: '+key)
  typed(value.get('hf_helper_sha256'),HF_SHA,'HF helper pin differs')
 if stage=='archives':
  for key,target in dict(outer_count=200,member_count=4000,student_count=512000,teacher_count=102400,unique_common_question_count=102400,official_tokenizer_replayed=True,archived_questions_exact_stored200_sampler_and_common_parquet=True).items():typed(value.get(key),target,'All200 archive content differs: '+key)


def run_pipeline(out,command_builder,accept_report,identity_reader=process,cwd=None,finalize=None):
 """Small sequential engine; test fixtures cannot access production main guards."""
 records={};results=[]
 for stage,filename in [('observe','observed200.original.json'),('native','native200.original.json'),('hf','hf200.original.json'),('archives','archives200.original.json')]:
  try:
   argv=command_builder(stage,records);write_once(out/(stage+'-command.original.json'),dict(argv=argv,stored_training_job_id=STORED_JOB,scope=SCOPE,utc=utc()))
   waited,started=wait_child(argv,out,stage,identity_reader,cwd);results.append(dict(stage=stage,wait=waited))
  except BaseException as error:
   write_once(out/(stage+'-control-failure.original.json'),dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE,utc=utc()))
   write_once(out/'pipeline-result.original.json',dict(version=1,status='FAILED_OR_UNKNOWN',completed_records=records,stages=results,scope=SCOPE,utc=utc()));return 1
  if waited['returncode']!=0 or waited['capture_errors']:
   write_once(out/'pipeline-result.original.json',dict(version=1,status='FAILED_OR_UNKNOWN',completed_records=records,stages=results,scope=SCOPE,utc=utc()));return waited_code(waited['returncode']) or 1
  try:records[stage]=accept_report(stage,out/filename,started['child'],records)
  except BaseException as error:
   write_once(out/(stage+'-content-failure.original.json'),dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE,utc=utc()))
   write_once(out/'pipeline-result.original.json',dict(version=1,status='FAILED_CONTENT',completed_records=records,stages=results,scope=SCOPE,utc=utc()));return 1
 try:
  if finalize is not None:finalize(records)
 except BaseException as error:
  write_once(out/'final-record-recheck-failure.original.json',dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE,utc=utc()))
  write_once(out/'pipeline-result.original.json',dict(version=1,status='FAILED_FINAL_RECORD_RECHECK',completed_records=records,stages=results,scope=SCOPE,utc=utc()));return 1
 write_once(out/'pipeline-result.original.json',dict(version=1,status='FOUR_STORED_CONTENT_SCOPES_PASSED',completed_records=records,stages=results,scope=SCOPE,utc=utc(),limitations=['Not final logs/no201 writer lifetime/evaluation/terminal/resource release acceptance','Does not restore live training objects','Original job may terminate during long archive inspection; no automatic retry']))
 return 0


def cpu_step(config,directory,c):
 host();job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID');typed(job,config['execution_job_id'],'Actual execution job differs')
 require(type(step) is str and re.fullmatch('[0-9]+',step),'Actual numeric overlap step required')
 for key,target in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO)).items():typed(os.environ.get(key),target,'Explicit CPU environment differs: '+key)
 typed(str(Path.cwd()),str(REPO),'Explicit native source working directory differs');execution=c.execution_identity(job,step)
 live=live_original(config);mem=Path('/proc/meminfo').read_text();match=re.search(r'^MemAvailable:\s+([0-9]+) kB$',mem,re.M);require(match and int(match.group(1))>=128*1024**2,'At least128GiB current CPU memory required')
 write_once(OUTPUT/'cpu-step-start.original.json',dict(execution_identity=execution,controller=process(os.getpid(),job,step),live_original=live,meminfo_original=mem,scope=SCOPE,utc=utc()))
 def commands(stage,records):
  bundle(config['bundle_manifest_sha256'],config['controller_sha256']);return helper_commands(directory,OUTPUT,job,step,records)[stage]
 def accept(stage,path,child,records):
  value,record=c.read_json(path,root=OUTPUT);qualify(stage,value,c,execution,child,records);bundle(config['bundle_manifest_sha256'],config['controller_sha256']);return record
 def finish(records):
  c.recheck(records);bundle(config['bundle_manifest_sha256'],config['controller_sha256'])
 return run_pipeline(OUTPUT,commands,accept,cwd=str(REPO),finalize=finish)


def controller(config,launch,lsha):
 write_once(OUTPUT/'controller-start.original.json',dict(process=process(os.getpid()),before=live_original(config),execution_job_id=config['execution_job_id'],stored_training_job_id=STORED_JOB,scope=SCOPE,utc=utc()))
 waited,_=wait_child(srun_command(launch,lsha,config['execution_job_id']),OUTPUT,'srun',cwd=str(REPO));return waited_code(waited['returncode']) or int(bool(waited['capture_errors']))
def supervise(config,launch,lsha):
 write_once(OUTPUT/'supervisor-start.original.json',dict(process=process(os.getpid()),before=live_original(config),scope=SCOPE,utc=utc()))
 waited,_=wait_child(role_command('controller',launch,lsha),OUTPUT,'controller',cwd=str(REPO));return waited_code(waited['returncode']) or int(bool(waited['capture_errors']))
def start(args):
 directory,c=bundle(args.bundle_manifest_sha256,args.controller_sha256);typed(args.execution_job_id,STORED_JOB,'V1 alive-original allocation only');physical(REPO,True);physical(OUTPUT.parent,True)
 reserve_once(OUTPUT)
 launch_attempted=False;launched_pid=None
 try:
  proof=precondition(directory,c,OUTPUT);proof_sha=sha(stable_raw(OUTPUT/'launch-prerequisites.original.json'))
  config=dict(version=1,scope=SCOPE,stored_training_job_id=STORED_JOB,execution_job_id=args.execution_job_id,source_commit=SOURCE,config_sha256=CONFIG,accepted5_receipt_sha256=RECEIPT,public_key_sha256=PUBLIC,controller_sha256=args.controller_sha256,bundle_manifest_sha256=args.bundle_manifest_sha256,driver_pid=proof['driver']['pid'],driver_start_ticks=proof['driver']['start_ticks'],output=str(OUTPUT),precondition_sha256=proof_sha)
  launch_fields(config);launch=OUTPUT/'once-intent.original.json';write_once(launch,config);lsha=sha(stable_raw(launch));argv=role_command('supervise',launch,lsha)
  with (OUTPUT/'supervisor-stdout.original').open('xb') as stdout,(OUTPUT/'supervisor-stderr.original').open('xb') as stderr:
   launch_attempted=True
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True,cwd=str(REPO))
   launched_pid=child.pid
   write_once(OUTPUT/'durable-launch.original.json',dict(argv=argv,supervisor=process(child.pid),launcher=process(os.getpid()),execution_job_id=args.execution_job_id,stored_training_job_id=STORED_JOB,scope=SCOPE,utc=utc()))
  print(canonical(dict(launched=True,scope=SCOPE,once_intent_sha256=lsha,output=str(OUTPUT),supervisor_pid=child.pid,execution_job_id=args.execution_job_id,stored_training_job_id=STORED_JOB)).decode())
 except BaseException as error:
  write_once(OUTPUT/'start-failure-or-unknown.original.json',dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE,utc=utc(),launch_attempted=launch_attempted,observed_supervisor_pid=launched_pid,rule='Existing once directory is never automatically retried'));raise


def main():
 parser=argparse.ArgumentParser();roles=parser.add_subparsers(dest='role',required=True);start_parser=roles.add_parser('start')
 for name in ('controller-sha256','bundle-manifest-sha256','execution-job-id'):start_parser.add_argument('--'+name,required=True)
 for role in ('supervise','controller','cpu-step'):
  q=roles.add_parser(role);q.add_argument('--launch',type=Path,required=True);q.add_argument('--launch-sha256',required=True)
 args=parser.parse_args()
 if args.role=='start':start(args);return
 config,directory,c=load_launch(args.launch,args.launch_sha256)
 if args.role=='supervise':sys.exit(supervise(config,args.launch,args.launch_sha256))
 if args.role=='controller':sys.exit(controller(config,args.launch,args.launch_sha256))
 sys.exit(cpu_step(config,directory,c))


if __name__=='__main__':main()
