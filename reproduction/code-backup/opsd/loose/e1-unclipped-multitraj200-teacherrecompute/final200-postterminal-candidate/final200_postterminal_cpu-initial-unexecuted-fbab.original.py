"""Reviewed future CPU-only fallback preparation; no automatic retries or GPU work.

Stored training3304 is immutable. Actual execution job/step are supplied by Slurm.
Old successful scopes retain their original execution identity when explicitly reused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import types

SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
STORED_JOB='3304'
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
RUN=ROOT/'runs/formal-3304';REPO=ROOT/('repo-'+SOURCE[:12])
PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
BUNDLE_SHA='f11580b6a7bc0ae418e196bf13a3bcd920fe63114106b95057008af1ea9af2be5'
BUNDLE=ROOT/'tools'/('final200-'+BUNDLE_SHA[:12])
OLD_OUTPUT=RUN/'evidence/final200-cpu-inspection-once-v1'
OUTPUT=RUN/'evidence/final200-postterminal-cpu-inspection-once-v1'
SELF=Path(__file__).absolute()
CONTROLLER_SHA='cd0062891f64100e46f01f2d9e1e7ef1a402cb926f39a20c4cdddc7d8f8f6f1d'
LOG_SHA='509df9332effd1196c6cb07a6a9d67612d9efdd75498f39a2bdb36726dcd7306'
COMMON_SHA='e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
RECEIPT='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
SCOPE='POSTTERMINAL_CPU_ONLY_STORED3304_CONTENT_NOT_LIVE_RESTORE_TERMINAL_RECEIPT_OR_EFFECT'
STAGES=('observe','native','hf','archives')
FILENAMES=dict(observe='observed200.original.json',native='native200.original.json',hf='hf200.original.json',archives='archives200.original.json')
REQUEST_KIND='REVIEWED_POSTTERMINAL_CPU_ONLY_REQUEST_NOT_COMPLETION_RECEIPT'
REQUEST_FIELDS={'version','kind','source_commit','config_sha256','stored_training_job_id','accepted5_receipt_sha256','bundle_manifest_sha256','execution_job_id','output','logs','terminal','previous_attempt','reuse_prefix','reviews','explicit_decision'}
REVIEW_STATUS='PASS_POSTTERMINAL_CPU_FALLBACK_INPUTS_NOT_FINAL_ACCEPTANCE'


def require(value,message):
 if not value:raise ValueError(message)
def typed(value,target,message):require(type(value) is type(target) and value==target,message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def pin(value):require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'Exact SHA required');return value
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def strict_json(raw):
 def pairs(values):
  result={}
  for key,value in values:require(key not in result,'Duplicate JSON key');result[key]=value
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')),
  parse_float=lambda x:finite_float(x))
def finite_float(value):
 import math
 number=float(value);require(math.isfinite(number),'Nonfinite JSON number');return number
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def identity(value):return tuple(getattr(value,k) for k in ('st_dev','st_ino','st_uid','st_mode','st_size','st_mtime_ns','st_ctime_ns'))
def physical(path,directory=False):
 p=Path(path);require(p.is_absolute() and p.resolve(strict=True)==p,'Owned physical absolute path required')
 st=p.lstat();require(st.st_uid==os.getuid() and (stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)),'Owned physical type differs');return p
def read(path,expected=None,limit=32*1024**2):
 p=physical(path);before=p.stat();require(before.st_size<=limit,'Read bound exceeded');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  require(identity(os.fstat(fd))==identity(before),'Opened identity differs');parts=[];count=0
  while True:
   chunk=os.read(fd,1024**2)
   if not chunk:break
   count+=len(chunk);require(count<=limit,'Read grew beyond bound');parts.append(chunk)
  raw=b''.join(parts);require(identity(os.fstat(fd))==identity(p.stat())==identity(before) and count==before.st_size,'Read changed')
  if expected is not None:typed(digest(raw),pin(expected),'Read SHA differs')
  return raw
 finally:os.close(fd)
def sync_directory(path):
 fd=os.open(physical(path,True),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def write(path,raw):
 physical(Path(path).parent,True);require(type(raw) is bytes,'Literal bytes required')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  with os.fdopen(fd,'wb',closefd=False) as f:f.write(raw);f.flush();os.fsync(fd)
 finally:os.close(fd)
 sync_directory(Path(path).parent)
def write_json(path,value):write(path,canonical(value))
def host():require(sys.platform=='linux' and socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0 and str(Path(sys.executable).resolve())==str(PYTHON),'Native root CPU on h100-10 required')
def exact_module(path,expected,name):
 raw=read(path,expected);module=types.ModuleType(name);module.__file__=str(path)
 exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),module.__dict__);read(path,expected);return module


def dependencies(self_sha):
 host();read(SELF,pin(self_sha));physical(BUNDLE,True)
 require(stat.S_IMODE(BUNDLE.stat().st_mode)==0o555,'Sealed bundle directory mode differs')
 manifest=strict_json(read(BUNDLE/'manifest.json',BUNDLE_SHA));typed(manifest.get('source_commit'),SOURCE,'Sealed source differs')
 require(set(manifest)=={'version','kind','source_commit','tools'} and manifest['version']==1 and type(manifest['version']) is int and manifest['kind']=='FINAL200_READ_ONLY_RUNTIME_BUNDLE','Sealed bundle schema differs')
 require(set(p.name for p in BUNDLE.iterdir())==set(manifest['tools'])|{'manifest.json'},'Sealed membership differs')
 for name,expected in manifest['tools'].items():
  require(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name),'Unsafe sealed member')
  read(BUNDLE/name,pin(expected));require(stat.S_IMODE((BUNDLE/name).stat().st_mode)==0o444,'Sealed mode differs')
 typed(manifest['tools']['final200_cpu_once.py'],CONTROLLER_SHA,'Sealed controller changed')
 typed(manifest['tools']['common.py'],COMMON_SHA,'Sealed common changed')
 typed(manifest['tools']['consume_final200_logs_terminal.py'],LOG_SHA,'Sealed logs consumer changed')
 ctl=exact_module(BUNDLE/'final200_cpu_once.py',CONTROLLER_SHA,'sealed_postterminal_old_controller')
 c=exact_module(BUNDLE/'common.py',COMMON_SHA,'sealed_postterminal_common')
 logs=exact_module(BUNDLE/'consume_final200_logs_terminal.py',LOG_SHA,'sealed_postterminal_logs')
 for name,p in ctl.PINS.items():typed(manifest['tools'][name],p,'Sealed helper changed: '+name)
 read(ctl.MATH,ctl.MATH_SHA);read(ctl.HF,ctl.HF_SHA)
 return ctl,c,logs


def fingerprint(request):
 return digest(canonical({k:request[k] for k in sorted(REQUEST_FIELDS-{'reviews'})}))
def request_headers(value):
 require(type(value) is dict and set(value)==REQUEST_FIELDS,'Unknown request fields')
 for key,target in dict(version=1,kind=REQUEST_KIND,source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id=STORED_JOB,accepted5_receipt_sha256=RECEIPT,bundle_manifest_sha256=BUNDLE_SHA,execution_job_id=None,output=str(OUTPUT),explicit_decision='ONE_NEW_CPU_ONLY_ALLOCATION_AFTER_ACTUAL3304_TERMINAL').items():typed(value[key],target,'Request field differs: '+key)
 require(set(value['logs'])=={'driver','aggregate'} and set(value['terminal'])=={'training_child_exit','wrapper_exit','scheduler_original'},'Raw terminal/log roles differ')
 require(type(value['reuse_prefix']) is dict and tuple(value['reuse_prefix']) in ((),('observe',),('observe','native'),('observe','native','hf')),'Only a qualified contiguous incomplete prefix is reusable')
 previous=value['previous_attempt'];require(type(previous) is dict and set(previous)=={'directory','state','files'},'Prior attempt schema differs')
 typed(previous['directory'],str(OLD_OUTPUT),'Prior directory differs')
 require(previous['state'] in {'NO_PRIOR_ATTEMPT','ACTUAL_FAILED_CLOSED','ACTUAL_ALLOCATION_INTERRUPTED_WITH_REVIEWED_MISSING_WAIT'},'Unknown/successful old attempt cannot automatically rerun')
 require(type(previous['files']) is dict and len(previous['files'])<=128,'Prior artifact inventory differs')
 require(set(value['reviews'])=={'root','independent'},'Two actual input reviews required')
 return value


def qualify_reviews(request,c,records):
 fp=fingerprint(request)
 for role,rec in request['reviews'].items():
  p=c.verify_record(rec);review,r=c.read_json(p,rec['sha256']);records[r['physical_path']]=r
  require(type(review) is dict and set(review)=={'version','kind','role','status','source_commit','stored_training_job_id','request_fingerprint_sha256','prior_attempt_state'},'Input review schema differs')
  for key,target in dict(version=1,kind='POSTTERMINAL_CPU_FALLBACK_INPUT_REVIEW',role='ROOT' if role=='root' else 'INDEPENDENT_REVIEWER',status=REVIEW_STATUS,source_commit=SOURCE,stored_training_job_id=STORED_JOB,request_fingerprint_sha256=fp,prior_attempt_state=request['previous_attempt']['state']).items():typed(review.get(key),target,'Input review binding differs: '+key)
 require(request['reviews']['root']['physical_path']!=request['reviews']['independent']['physical_path'] and (request['reviews']['root']['device'],request['reviews']['root']['inode'])!=(request['reviews']['independent']['device'],request['reviews']['independent']['inode']),'Input reviews alias')


def reuse_report(stage,entry,c,ctl,qualified,records):
 require(type(entry) is dict and set(entry)=={'report','child_start','exit','cpu_step_start'},'Reuse provenance roles differ')
 raw={}
 for role,rec in entry.items():
  require(Path(rec['physical_path']).is_relative_to(OLD_OUTPUT),'Reuse original is outside old once directory')
  p=c.verify_record(rec);value,r=c.read_json(p,rec['sha256']);raw[role]=value;records[r['physical_path']]=r
 waited=raw['exit'];require(type(waited['returncode']) is int and waited['returncode']==0 and waited['waited'] is True and waited['capture_errors']==[],'Reuse requires literal true zero wait')
 child=raw['child_start']['child'];execution=raw['cpu_step_start']['execution_identity']
 require(type(child) is dict and type(child['pid']) is int and child['pid']>0 and type(child['start_ticks']) is int and child['start_ticks']>0,'Reuse child identity missing')
 typed(waited['child_pid'],child['pid'],'Reuse exit/child differs');typed(raw['child_start']['child_pid'],child['pid'],'Reuse start/child differs')
 typed(execution['execution_job_id'],STORED_JOB,'Old overlap execution job differs');typed(execution['stored_training_job_id'],STORED_JOB,'Old stored job differs')
 require(type(execution['execution_step_id']) is str and re.fullmatch('[0-9]+',execution['execution_step_id']),'Old actual numeric CPU step required')
 require(type(execution['pid']) is int and execution['pid']>0 and type(execution['start_ticks']) is int and execution['start_ticks']>0,'Old execution controller identity missing')
 typed(raw['child_start']['parent_pid'],execution['pid'],'Reuse historical parent differs');typed(waited['waited_by_pid'],execution['pid'],'Reuse waiter differs')
 require(raw['child_start']['capture_errors']==[] and raw['child_start']['parent']['start_ticks']==execution['start_ticks'],'Reuse start capture differs')
 ctl.qualify(stage,raw['report'],c,execution,child,qualified)
 rec=entry['report'];require(rec['physical_path']==str(OLD_OUTPUT/FILENAMES[stage]),'Reuse report role/path differs')
 for key,value in raw['report'].get('verified_input_records',{}).items():c.verify_record(value);records[key]=value
 return rec


def precondition(request,ctl,c,logs):
 request_headers(request);context=c.accepted_context(RECEIPT);records=dict(context['records']);qualify_reviews(request,c,records)
 raw_terminal={}
 for role,rec in request['terminal'].items():
  require(type(rec) is dict,'Actual terminal originals required')
  p=c.verify_record(rec);raw,r=c.read_file(p,rec['sha256'],keep_bytes=True);records[r['physical_path']]=r;raw_terminal[role]=raw
 typed(request['terminal']['training_child_exit']['physical_path'],str(RUN/'evidence/train-process-exit.json'),'Actual train exit origin differs')
 typed(request['terminal']['wrapper_exit']['physical_path'],str(RUN/'job-exit.txt'),'Actual wrapper exit origin differs')
 terminal=logs.terminal_layers(raw_terminal,c)
 typed(terminal['training_child_exit']['returncode'],0,'Actual full training child zero exit required')
 typed(terminal['scheduler']['status'],'OBSERVED_LITERAL_SCHEDULER_TERMINAL','Actual terminal scheduler original required')
 fsdp,rec=c.read_file(ctl.FSDP_WARNING_PATH,ctl.FSDP_WARNING_SHA,keep_bytes=True);records[rec['physical_path']]=rec
 parsed={}
 for role,entry in request['logs'].items():
  require(type(entry) is dict and set(entry)=={'file','origin_path','writer_pid'},'Actual log provenance schema differs')
  typed(entry['writer_pid'],context['manifest']['driver_pid'],'Stored log writer differs')
  typed(entry['origin_path'],str(ctl.DRIVER_STDOUT if role=='driver' else RUN/'train.log'),'Stored physical log origin differs')
  p=c.verify_record(entry['file']);raw,rec=c.read_file(p,entry['file']['sha256'],keep_bytes=True,limit=256*1024**2);records[rec['physical_path']]=rec
  parsed[role]=logs.parse_log(raw,context,context['gate'],fsdp);logs.check_log(parsed[role],context,context['gate'])
 require([(r['marker'],r['value']) for r in parsed['driver']['records']]==[(r['marker'],r['value']) for r in parsed['aggregate']['records']],'Full driver/aggregate event streams differ')
 previous=request['previous_attempt'];old=Path(previous['directory'])
 if previous['state']=='NO_PRIOR_ATTEMPT':require(not os.path.lexists(old) and previous['files']=={} and request['reuse_prefix']=={},'Absent prior must be truly absent')
 else:
  physical(old,True);actual={str(p.relative_to(old)):p for p in old.rglob('*') if not p.is_dir()}
  require(set(actual)==set(previous['files']),'Old once artifacts not completely preserved/bound')
  for name,rec in previous['files'].items():
   typed(rec['physical_path'],str(actual[name]),'Old artifact physical path differs');c.verify_record(rec);records[rec['physical_path']]=rec
  if previous['state']=='ACTUAL_FAILED_CLOSED':
   require('pipeline-result.original.json' in previous['files'],'Closed prior failure result missing')
   prior,_=c.read_json(old/'pipeline-result.original.json',previous['files']['pipeline-result.original.json']['sha256']);require(prior['status'].startswith('FAILED'),'Already successful prior pipeline must not rerun')
  else:
   require('once-intent.original.json' in previous['files'] and 'pipeline-result.original.json' not in previous['files'],'Reviewed interrupted prior must preserve its missing completion literally')
 checkpoints=c.checkpoint_paths();archives=c.archive_paths();require(len(checkpoints)==31 and len(archives)==4000,'Actual member role inventory differs')
 for rank,w in enumerate(context['workers']):
  r,rec=c.read_json(checkpoints[f'actor/training_state/rank_{rank}.json'],root=RUN);records[rec['physical_path']]=rec;c.rank_report(r,rank,200,'FULL_STATE_SERIALIZED_OK',w['pid']);require(r['object_ids']==w['object_ids'],'Stored200 rank identity continuity differs')
 driver,rec=c.read_json(checkpoints['driver/driver.json'],root=RUN);records[rec['physical_path']]=rec
 for key,target in dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,pid=context['manifest']['driver_pid'],uid=0,job_id=STORED_JOB).items():typed(driver.get(key),target,'Stored200 driver header differs')
 for p in list(checkpoints.values())+list(archives.values()):physical(p)
 qualified={}
 for stage,entry in request['reuse_prefix'].items():qualified[stage]=reuse_report(stage,entry,c,ctl,qualified,records)
 c.recheck(records)
 return dict(scope=SCOPE,source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id=STORED_JOB,terminal_literal=terminal,accepted5=context['accepted5'],checkpoint_member_count=31,archive_member_count=4000,reused_records=qualified,input_records=records,prior_attempt_state=previous['state'],current_job_or_step=None,limitations=['No resource release or lifecycle acceptance; bad66 and actual wrapper result unchanged','Stored historical3304 signatures/objects; never live process restore'])


def allocation(stdout,job):
 require(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job!=STORED_JOB,'New actual CPU job required')
 fields={}
 for item in stdout.split():
  if '=' in item:
   key,value=item.split('=',1);require(key not in fields,'Repeated allocation field');fields[key]=value
 for key,target in dict(JobId=job,JobState='RUNNING',UserId='root(0)',NodeList='h100-10',NumNodes='1').items():typed(fields.get(key),target,'Own CPU allocation differs: '+key)
 require(re.fullmatch('[1-9][0-9]*',fields.get('NumCPUs','')) and int(fields['NumCPUs'])>=4,'CPU allocation needs four CPUs')
 for key in ('ReqTRES','AllocTRES'):
  value=fields.get(key);require(type(value) is str and value and value!='(null)','Actual resource TRES missing')
  tokens=value.split(',');mem=[v.split('=',1)[1] for v in tokens if v.startswith('mem=')]
  require(len(mem)==1 and memory_bytes(mem[0])>=128*1024**3,'CPU allocation needs128GiB')
  for token in tokens:
   if 'gpu' in token.lower():require(re.fullmatch(r'gres/gpu(?::[A-Za-z0-9_-]+)?=0',token),'CPU allocation must have exactly zero GPUs')
 require(fields.get('TresPerNode') in (None,'(null)','gres:gpu:0','gres:gpu:h100:0'),'Unexpected GPU/resource request')
 return fields

def memory_bytes(value):
 match=re.fullmatch(r'([1-9][0-9]*)([KMGT]?)',value);require(match is not None,'Memory TRES must be explicit integer units')
 return int(match[1])*1024**({'':0,'K':1,'M':2,'G':3,'T':4}[match[2]])
def parse_submission(stdout,rc):
 require(type(rc) is int and rc==0,'Unknown/nonzero submission cannot bind job')
 require(type(stdout) is str and re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?',stdout),'Only one actual parsable sbatch ID is accepted')
 job=stdout.rstrip('\n').split(';',1)[0];require(job!=STORED_JOB,'New CPU job must not impersonate3304');return job


def commands(stage,records):
 base=['--common-sha256',COMMON_SHA,'--accepted5-receipt-sha256',RECEIPT,'--execution-job-id',os.environ['SLURM_JOB_ID'],'--execution-step-id',os.environ['SLURM_STEP_ID']]
 inventory=['--inventory',records['observe']['physical_path'],'--inventory-sha256',records['observe']['sha256']] if 'observe' in records else []
 native=['--native-report',records['native']['physical_path'],'--native-report-sha256',records['native']['sha256'],'--native-inspector',str(BUNDLE/'inspect_final200_native.py'),'--native-inspector-sha256','03dcdc1e0cd84eb77780d1e52193c7b904e1e084467e8fb130e184ce008777d9'] if 'native' in records else []
 extra={'observe':[], 'native':['--math-helper',str(ROOT/'tools/first5-native-dc1905562f2f/inspect_payloads.py')], 'hf':['--hf-helper',str(ROOT/'tools/first5-1b16c2227caf/hf_cpu_reload.py')], 'archives':[]}[stage]
 filename={'observe':'observe_final200.py','native':'inspect_final200_native.py','hf':'load_final200_hf.py','archives':'check_final200_archives.py'}[stage]
 return [str(PYTHON),'-B',str(BUNDLE/filename)]+base+(inventory if stage!='observe' else [])+(native if stage in ('hf','archives') else [])+extra+['--output',str(OUTPUT/FILENAMES[stage])]


def pipeline(ctl,c,execution,reused,command_builder=commands):
 records=dict(reused);stages=[]
 for stage in STAGES:
  if stage in reused:
   stages.append(dict(stage=stage,status='REUSED_ORIGINAL_FINISHED_SCOPE_NOT_NEW_EXECUTION',original=reused[stage]));continue
  try:
   argv=command_builder(stage,records);write_json(OUTPUT/(stage+'-command.original.json'),dict(argv=argv,scope=SCOPE,execution_identity=execution))
   waited,started=ctl.wait_child(argv,OUTPUT,stage,lambda pid:ctl.process(pid,execution['execution_job_id'],execution['execution_step_id']),str(REPO))
   stages.append(dict(stage=stage,status='ACTUAL_NEW_EXECUTION_WAITED',wait=waited))
   require(waited['returncode']==0 and not waited['capture_errors'],'Actual helper failed or identity capture unavailable')
   value,record=c.read_json(OUTPUT/FILENAMES[stage],root=OUTPUT);ctl.qualify(stage,value,c,execution,started['child'],records);records[stage]=record
   c.recheck(records)
  except BaseException as error:
   write_json(OUTPUT/(stage+'-failure.original.json'),dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE))
   write_json(OUTPUT/'pipeline-result.original.json',dict(status='FAILED_OR_UNKNOWN',scope=SCOPE,completed_records=records,stages=stages,actual_final_completion=None));return 1
 c.recheck(records)
 write_json(OUTPUT/'pipeline-result.original.json',dict(status='FOUR_STORED_CONTENT_SCOPES_QUALIFIED_NOT_FINAL_ACCEPTANCE',scope=SCOPE,completed_records=records,stages=stages,execution_identity=execution,stored_training_job_id=STORED_JOB,actual_final_completion=None))
 return 0


def environment():
 return ['CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(REPO)]
def role_command(role,intent,sha):return [str(PYTHON),'-B',str(SELF),role,'--intent',str(intent),'--intent-sha256',sha]
def batch_script(intent,sha):
 import shlex
 return ('#!/bin/sh\nset -eu\nexec env '+shlex.join(environment()+role_command('step-launch',intent,sha))+'\n').encode()
def sbatch_command(script):
 return ['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=128G','--gres=none','--nodelist=h100-10','--export=NONE','--chdir='+str(REPO),'--output='+str(OUTPUT/'slurm-%j.stdout.original'),'--error='+str(OUTPUT/'slurm-%j.stderr.original'),str(script)]
def srun_command(job,intent,sha):
 return ['srun','--jobid='+job,'--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=none','--nodelist=h100-10','--chdir='+str(REPO),'env']+environment()+role_command('worker',intent,sha)


def prepare(args):
 ctl,c,logs=dependencies(args.controller_sha256);raw=read(args.request,pin(args.request_sha256));request=strict_json(raw)
 physical(OUTPUT.parent,True);require(not os.path.lexists(OUTPUT),'Existing or unknown new once attempt never retried');OUTPUT.mkdir(mode=0o700);sync_directory(OUTPUT.parent);sync_directory(OUTPUT)
 write(OUTPUT/'request.original.json',raw)
 try:
  proof=precondition(request,ctl,c,logs);write_json(OUTPUT/'precondition.original.json',proof)
  intent=dict(version=1,scope=SCOPE,source_commit=SOURCE,stored_training_job_id=STORED_JOB,execution_job_id=None,controller_sha256=args.controller_sha256,request_sha256=args.request_sha256,precondition_sha256=digest(read(OUTPUT/'precondition.original.json')),output=str(OUTPUT))
  write_json(OUTPUT/'intent.original.json',intent);isha=digest(read(OUTPUT/'intent.original.json'))
  write(OUTPUT/'batch.original.sh',batch_script(OUTPUT/'intent.original.json',isha));print(canonical(dict(prepared=True,scope=SCOPE,intent_sha256=isha,execution_job_id=None)).decode())
 except BaseException as error:
  write_json(OUTPUT/'preparation-failure.original.json',dict(exception_type=type(error).__name__,exception=str(error),scope=SCOPE,execution_job_id=None));raise


def load_intent(path,expected):
 typed(str(path),str(OUTPUT/'intent.original.json'),'Only own once intent is permitted');intent=strict_json(read(path,pin(expected)))
 require(type(intent) is dict and set(intent)=={'version','scope','source_commit','stored_training_job_id','execution_job_id','controller_sha256','request_sha256','precondition_sha256','output'},'Intent schema differs')
 for key,target in dict(version=1,scope=SCOPE,source_commit=SOURCE,stored_training_job_id=STORED_JOB,execution_job_id=None,output=str(OUTPUT)).items():typed(intent[key],target,'Intent field differs: '+key)
 ctl,c,logs=dependencies(intent['controller_sha256']);request=strict_json(read(OUTPUT/'request.original.json',intent['request_sha256']));proof=strict_json(read(OUTPUT/'precondition.original.json',intent['precondition_sha256']))
 current=precondition(request,ctl,c,logs);typed(digest(canonical(current)),digest(canonical(proof)),'Actual stored prerequisites changed');return intent,ctl,c,request,proof


def submit(args):
 intent,ctl,c,request,proof=load_intent(args.intent,args.intent_sha256)
 require(not os.path.lexists(OUTPUT/'submission-attempt.original.json'),'Unknown submission cannot be retried')
 script=OUTPUT/'batch.original.sh';typed(read(script),batch_script(args.intent,args.intent_sha256),'Batch source differs')
 argv=sbatch_command(script);write_json(OUTPUT/'submission-attempt.original.json',dict(argv=argv,scope=SCOPE,execution_job_id=None,utc=utc(),parent=ctl.process(os.getpid())))
 waited,started=ctl.wait_child(argv,OUTPUT,'sbatch',cwd=str(REPO))
 job=None;error=None
 try:job=parse_submission(read(OUTPUT/'sbatch-stdout.original').decode(),waited['returncode']);require(not waited['capture_errors'],'Submission metadata unavailable')
 except BaseException as exc:error=dict(exception_type=type(exc).__name__,exception=str(exc))
 write_json(OUTPUT/'submission-result.original.json',dict(scope=SCOPE,status='ACTUAL_ID_BOUND_NOT_CONTENT_ACCEPTANCE' if job is not None and error is None else 'FAILED_OR_UNKNOWN_NO_AUTOMATIC_RETRY',execution_job_id=job,wait=waited,error=error,stdout_sha256=digest(read(OUTPUT/'sbatch-stdout.original')),stderr_sha256=digest(read(OUTPUT/'sbatch-stderr.original')),utc=utc()))
 require(error is None,'Submission failed/unknown; original attempt preserved')
 print(canonical(dict(scope=SCOPE,submitted=True,execution_job_id=job,actual_final_completion=None)).decode())


def actual_submission(wait_seconds=30):
 deadline=time.monotonic()+wait_seconds
 while not (OUTPUT/'submission-result.original.json').exists():
  require(time.monotonic()<deadline,'Actual submission outcome absent; never infer new ID');time.sleep(.25)
 value=strict_json(read(OUTPUT/'submission-result.original.json'))
 typed(value.get('status'),'ACTUAL_ID_BOUND_NOT_CONTENT_ACCEPTANCE','Actual submission outcome is unknown')
 typed(value['execution_job_id'],os.environ.get('SLURM_JOB_ID'),'Actual scheduler environment and reply differ');parse_submission(read(OUTPUT/'sbatch-stdout.original').decode(),value['wait']['returncode'])
 require(value['error'] is None and value['wait']['waited'] is True and value['wait']['capture_errors']==[],'Actual submission lacks true wait/identity')
 return value

def own_scheduler(job):
 result=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,timeout=20)
 require(result.returncode==0,'Actual own CPU scheduler query failed');fields=allocation(result.stdout.decode(),job)
 return dict(argv=['scontrol','show','job',job,'-o'],returncode=result.returncode,stdout_original=result.stdout.decode(),stderr_original=result.stderr.decode(),fields=fields,utc=utc())

def step_launch(args):
 intent,ctl,c,request,proof=load_intent(args.intent,args.intent_sha256);actual=actual_submission();job=actual['execution_job_id']
 scheduler=own_scheduler(job);write_json(OUTPUT/'batch-start.original.json',dict(process=ctl.process(os.getpid(),job),scheduler=scheduler,scope=SCOPE))
 waited,_=ctl.wait_child(srun_command(job,args.intent,args.intent_sha256),OUTPUT,'srun',cwd=str(REPO));return ctl.waited_code(waited['returncode']) or int(bool(waited['capture_errors']))

def worker(args):
 intent,ctl,c,request,proof=load_intent(args.intent,args.intent_sha256);actual=actual_submission();job=actual['execution_job_id'];step=os.environ.get('SLURM_STEP_ID')
 require(type(step) is str and re.fullmatch('[0-9]+',step),'Actual new numeric CPU step required')
 for setting in environment():key,value=setting.split('=',1);typed(os.environ.get(key),value,'CPU environment differs: '+key)
 typed(str(Path.cwd()),str(REPO),'CPU source working directory differs');execution=c.execution_identity(job,step);scheduler=own_scheduler(job)
 mem=Path('/proc/meminfo').read_text();match=re.search(r'^MemAvailable:\s+([0-9]+) kB$',mem,re.M);require(match and int(match[1])>=128*1024**2,'Current CPU memory below128GiB')
 write_json(OUTPUT/'cpu-step-start.original.json',dict(execution_identity=execution,scheduler=scheduler,meminfo_original=mem,stored_training_job_id=STORED_JOB,scope=SCOPE))
 return pipeline(ctl,c,execution,proof['reused_records'])


def main():
 p=argparse.ArgumentParser();roles=p.add_subparsers(dest='role',required=True)
 a=roles.add_parser('prepare')
 for name in ('request','request-sha256','controller-sha256'):a.add_argument('--'+name,required=True)
 for role in ('submit','step-launch','worker'):
  a=roles.add_parser(role);a.add_argument('--intent',type=Path,required=True);a.add_argument('--intent-sha256',required=True)
 a=p.parse_args()
 if a.role=='prepare':prepare(a)
 elif a.role=='submit':submit(a)
 elif a.role=='step-launch':sys.exit(step_launch(a))
 else:sys.exit(worker(a))


if __name__=='__main__':main()
