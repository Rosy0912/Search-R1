"""Prepared root-only fresh multitraj200 release; imports never sign or contact SSH."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import types

HERE=Path(__file__).resolve().parent
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
SOURCE='BLOCKED_UNBOUND_SOURCE_COMMIT'
TOOLS_SHA='BLOCKED_RECEIPT_MANIFEST_SHA'
GATE_SHA='BLOCKED_GATE_SOURCE_SHA'
CONFIG_SHA=None  # Supplied by externally pinned actual resolved first5 configuration.
PUBLIC_SHA='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
PRIVATE=HERE.parent/'e1-300/review-signing/private.pem'
PUBLIC=HERE.parent/'e1-300/review-signing/public.json'
TOOLS=HERE/'BLOCKED_LOCAL_RECEIPT_MANIFEST'
IDENTITY=('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates',
          'next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce')
ROLES={'resources','retrieval','hf_reload','rank_audit','service'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def unique(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    def finite_float(value):
        import math
        result=float(value);need(math.isfinite(result),'Overflowed JSON float');return result
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite_float,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def pinned_local(path,pin):
    path=Path(path)
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin) is not None,'Actual SHA256 required')
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
         and path.stat().st_uid==os.getuid(),'Local physical current-owned original required')
    raw=path.read_bytes();need(sha(raw)==pin,'Local pinned bytes changed');return raw


def record_path(record,run):
    need(type(record) is dict and set(record)=={'path','sha256'},'Exact remote file record required')
    need(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}',record['sha256']) is not None,'Invalid remote pin')
    need(type(record['path']) is str,'Remote physical path must be text')
    path=Path(record['path']);need(path.is_absolute() and '..' not in path.parts,'Invalid absolute remote path')
    path.relative_to(Path(run));return path


def check_inputs(args,manifest,payload,spec,root_review,independent):
    need(type(args.job_id) is str and re.fullmatch('[1-9][0-9]*',args.job_id) is not None and type(args.driver_pid) is int and args.driver_pid>0
         and re.fullmatch('[1-9][0-9]*',args.driver_start) is not None,'Actual observed current driver identity required')
    need(args.source==SOURCE and args.tools_sha256==TOOLS_SHA,'Reviewed fixed source/tools required')
    run=ROOT+'/runs/formal-'+args.job_id
    need(type(manifest.get('version')) is int and manifest.get('version')==1 and manifest.get('method')=='pure_opsd'
         and manifest.get('run_root')==run and manifest.get('source_commit')==SOURCE
         and manifest.get('driver_pid')==args.driver_pid and manifest.get('job_id')==args.job_id
         and type(manifest.get('uid')) is int and manifest.get('uid')==0
         and manifest.get('config_sha256')==CONFIG_SHA,'Actual manifest identity differs')
    for key,value in [('global_step',5),('actual_outer_updates',5),('next_global_step',6),
                      ('horizon',200),('warmup_steps',57),('world_size',8)]:
        need(type(manifest.get(key)) is int and manifest[key]==value,'Actual multitraj200 clock differs:'+key)
    need(type(manifest.get('pause_nonce')) is str and re.fullmatch('[0-9a-f]{64}',manifest['pause_nonce']) is not None
         and type(manifest.get('config_sha256')) is str
         and re.fullmatch('[0-9a-f]{64}',manifest['config_sha256']) is not None,'Actual nonce/config digest missing')
    need(set(payload)==set(IDENTITY)|{'status','checkpoint_manifest_sha256','checkpoint_files',
         'independent_review','evidence','train_log_prefix'},'Unexpected unsigned payload schema')
    need(all(type(payload[k]) is type(manifest[k]) and payload[k]==manifest[k] for k in IDENTITY),
         'Unsigned payload manifest identity differs')
    need(payload['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
         and payload['checkpoint_manifest_sha256']==args.manifest_sha256
         and payload['checkpoint_files']==manifest['files'],'Unsigned checkpoint association differs')
    need(type(spec) is dict and set(spec)==ROLES,'Exactly five externally reviewed evidence roles required')
    need(set(payload['evidence'])==ROLES|{'root_review','preparation_audit'}
         and all(payload['evidence'][role]==spec[role] for role in ROLES),'Unsigned evidence spec differs')
    for review,role,pin,rec in [
            (root_review,'ROOT_AUDITOR',args.root_review_sha256,payload['evidence']['root_review']),
            (independent,'INDEPENDENT_REVIEWER',args.independent_review_sha256,payload['independent_review'])]:
        need(rec['sha256']==pin and review.get('version')==1 and review.get('role')==role
             and review.get('status')=='PASS_ACTUAL_FIRST5_EVIDENCE'
             and review.get('source_commit')==SOURCE and review.get('job_id')==args.job_id
             and review.get('driver_pid')==args.driver_pid and review.get('pause_nonce')==manifest['pause_nonce']
             and review.get('checkpoint_manifest_sha256')==args.manifest_sha256
             and review.get('evidence')==spec,'Actual external review role/identity/evidence differs')
    records=list(manifest['files'].values())+list(payload['evidence'].values())+[payload['independent_review']]
    records.extend([{'path':run+'/full_state/global_step_5/manifest.json','sha256':args.manifest_sha256},
                    {'path':args.unsigned_payload_remote_path,'sha256':args.unsigned_payload_sha256},
                    {'path':args.evidence_spec_remote_path,'sha256':args.evidence_spec_sha256}])
    prefix=payload['train_log_prefix']
    need(set(prefix)=={'live_path','snapshot_path','sha256','bytes'} and prefix['live_path']==run+'/train.log'
         and prefix['snapshot_path']!=prefix['live_path'] and type(prefix['bytes']) is int
         and prefix['bytes']>0,'Actual immutable training-prefix identity differs')
    records.append({'path':prefix['snapshot_path'],'sha256':prefix['sha256']})
    for record in records:record_path(record,run)
    return run,records


LIVE_BODY=r'''
import base64,csv,datetime,hashlib,io,json,math,os,pathlib,re,secrets,socket,subprocess,sys,types
def need(ok,msg):
 if not ok:raise ValueError(msg)
def unique(raw):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'duplicateJSON');d[k]=v
  return d
 def finite(v):
  f=float(v);need(math.isfinite(f),'overflowJSON');return f
 return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfiniteJSON')))
request=unique(sys.stdin.buffer.read());D=request['data']
def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def physical(path,directory=False):
 p=pathlib.Path(path);need(p.is_absolute() and p.resolve(strict=True)==p and p.stat().st_uid==0,'physicalownpath')
 need(p.is_dir() if directory else p.is_file(),'wrongfilekind');return p
def process(pid,start=None):
 p=pathlib.Path('/proc')/str(pid);status=p.joinpath('status').read_text();cgroup=p.joinpath('cgroup').read_text()
 u=re.search(r'^Uid:\s+(.*)',status,re.M);need(u is not None and list(map(int,u.group(1).split()))==[0,0,0,0],'fourUIDchanged')
 need(re.search(r'(?:^|/)job_'+re.escape(D['job_id'])+r'(?:/|$)',cgroup,re.M) is not None,'processcgroupchanged')
 state=re.search(r'^State:\s+(\S+)',status,re.M);need(state is not None and state.group(1) in ('R','S','D','I'),'processnotlive')
 ticks=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
 if start is not None:need(ticks==str(start),'processPIDrecycled')
 return {'pid':pid,'start':ticks,'cgroup':cgroup,'uids':[0,0,0,0],'state':state.group(1)}
def command(args):return subprocess.check_output(args,text=True,timeout=20)
def validate_live(require_paused=True):
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','wrongnodeUID')
 run=physical(D['run_root'],True);repo=physical(D['repo'],True)
 need(command(['git','--no-optional-locks','-C',str(repo),'rev-parse','HEAD']).strip()==D['source_commit']
      and not command(['git','--no-optional-locks','-C',str(repo),'status','--porcelain']).strip(),'sourcechanged')
 tm=physical(D['tools_manifest']);need(hashfile(tm)==D['tools_sha256'],'toolsmanifestchanged')
 tools=unique(tm.read_bytes());need(tools['source_commit']==D['source_commit'],'toolsourcechanged')
 for name,pin in tools['tools'].items():need(hashfile(physical(tm.parent/name))==pin,'toolbyteschanged:'+name)
 for name,pin in tools['source_files'].items():need(hashfile(physical(repo/name))==pin,'sourcebyteschanged:'+name)
 native=physical(pathlib.Path(D['tools_manifest']).parent.parent/'BLOCKED_NATIVE_MANIFEST_RELATIVE')
 need(hashfile(native)==tools['native_payload_tools_manifest_sha256'],'nativeinspectormanifestchanged')
 native_manifest=unique(native.read_bytes());need(native_manifest['source_commit']==D['source_commit']
      and (native_manifest['horizon'],native_manifest['warmup_steps'],native_manifest['saved_adam_steps'],native_manifest['saved_lr'])==(200,57,50,1e-6*(5/57)),'nativeinspectorprofilechanged')
 for name,pin in native_manifest['tools'].items():need(hashfile(physical(native.parent/name))==pin,'nativeinspectorbyteschanged')
 need(native_manifest['tools']['inspect_payloads.py']==tools['native_payload_inspector_sha256'],'nativeinspectorassociationchanged')
 need(tools['source_files']['search_r1/baselines/checkpoint_gate.py']==D['gate_sha256'],'frozengatepinchanged')
 for rec in D['all_records']:
  p=physical(rec['path']);p.relative_to(run);need(hashfile(p)==rec['sha256'],'actualevidencebyteschanged:'+str(p))
 public=physical(run/'bindings/public.json');need(hashfile(public)==D['public_sha256'],'publicchanged')
 manifest=unique(physical(run/'full_state/global_step_5/manifest.json').read_bytes());need(manifest==D['manifest'],'manifestchanged')
 bound_source=unique(physical(manifest['files']['source']['path']).read_bytes())
 need(bound_source['source_commit']==D['source_commit'] and bound_source['repo']==str(repo)
      and bound_source['clean'] is True and bound_source['source_files']==tools['source_files'],'actualboundsourcechanged')
 formal=physical(pathlib.Path(D['run_root']).parent.parent/'BLOCKED_FORMAL_MANIFEST_RELATIVE')
 need(hashfile(formal)==bound_source['formal_manifest_sha256']=='BLOCKED_FORMAL_MANIFEST_SHA','formalmanifestchanged')
 formal_manifest=unique(formal.read_bytes());need(formal_manifest['source_commit']==D['source_commit']
      and formal_manifest['source_files']==tools['source_files'],'formalsourcechanged')
 for name,pin in formal_manifest['tools'].items():need(hashfile(physical(formal.parent/name))==pin,'formaltoolbyteschanged')
 closure=unique(physical(manifest['files']['import_closure']['path']).read_bytes())
 need(set(closure)=={'modules'} and bool(closure['modules']),'nativeclosuremissing')
 for name,rec in closure['modules'].items():
  path=physical(rec['path']);need(hashfile(path)==rec['sha256'],'nativeclosurebyteschanged:'+name)
  if name.startswith(('search_r1.','verl.','reproduction.')):path.relative_to(repo)
 unsigned=physical(D['unsigned_payload_remote_path']);need(unique(unsigned.read_bytes())==D['payload'],'unsignedpayloadchanged')
 spec=unique(physical(D['evidence_spec_remote_path']).read_bytes());need(spec==D['spec'],'evidencespecchanged')
 source=physical(repo/'search_r1/baselines/checkpoint_gate.py').read_bytes();need(hashlib.sha256(source).hexdigest()==D['gate_sha256'],'gatebyteschanged')
 gate=types.ModuleType('root_first5_frozen_gate');exec(compile(source,str(repo/'search_r1/baselines/checkpoint_gate.py'),'exec'),gate.__dict__)
 workers=gate.manifest_worker_pids(manifest,run);need(len(workers)==8,'savedrankPIDalias')
 prep=unique(physical(D['payload']['evidence']['preparation_audit']['path']).read_bytes())
 need(prep['version']==1 and prep['source_commit']==D['source_commit'] and prep['job_id']==D['job_id']
      and prep['manifest_sha256']==D['manifest_sha256'],'preparationidentitychanged')
 need(prep['actual_live_driver']['pid']==D['driver_pid']
      and str(prep['actual_live_driver']['start'])==D['driver_start'],'driverstartassociationchanged')
 driver=process(D['driver_pid'],D['driver_start'])
 saved_workers=prep['actual_live_workers'];need(len(saved_workers)==8 and {p['pid'] for p in saved_workers}==workers,'preparationrankPIDchanged')
 actual_workers=[process(p['pid'],p['start']) for p in saved_workers]
 service=unique(physical(D['spec']['service']['path']).read_bytes())['owner'];actual_service=process(service['pid'],service['start'])
 need(actual_service['pid']==prep['actual_live_service']['pid']
      and actual_service['start']==str(prep['actual_live_service']['start']),'retrieverservicechanged')
 s=command(['scontrol','show','job',D['job_id'],'-o']);fields=dict(x.split('=',1) for x in s.split() if '=' in x)
 need(fields.get('JobId')==D['job_id'] and fields.get('JobState')=='RUNNING'
      and fields.get('NodeList')=='h100-10' and fields.get('TresPerNode')=='gres:gpu:h100:8'
      and fields.get('NumNodes')=='1' and fields.get('OverSubscribe')=='NO'
      and re.fullmatch(r'\S+\(0\)',fields.get('UserId','')) is not None,'actualallocationchanged')
 prefix=D['payload']['train_log_prefix'];live=physical(prefix['live_path']);snapshot=physical(prefix['snapshot_path'])
 with live.open('rb') as f:need(hashlib.sha256(f.read(prefix['bytes'])).hexdigest()==prefix['sha256'],'liveprefixchanged')
 need(snapshot.stat().st_size==prefix['bytes'] and hashfile(snapshot)==prefix['sha256'],'immutableprefixchanged')
 rows=[];pauses=[];releases=[]
 for line in snapshot.read_text().splitlines():
  for marker,items in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_PAUSED_JSON=',pauses),('OPSD_E1_STEP5_RELEASED_JSON=',releases)]:
   if marker in line:items.append(gate.parse_log_marker(line,marker,driver_pid=D['driver_pid'],job_id=D['job_id'],worker_pids=workers,
     checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'))
 need([r['actual_outer_updates'] for r in rows]==[1,2,3,4,5] and len(pauses)==1 and not releases,'notuniquefivepausedsnapshot')
 pause=pauses[0];need(pause['manifest_path']==str(run/'full_state/global_step_5/manifest.json')
      and pause['manifest_sha256']==D['manifest_sha256'] and pause['driver_pid']==D['driver_pid']
      and pause['job_id']==D['job_id'] and pause['pause_nonce']==manifest['pause_nonce'],'pausedidentitychanged')
 for row in rows:
  need(all(row[k]==manifest[k] for k in ('driver_pid','job_id','source_commit','config_sha256'))
       and row['global_step']==row['actual_outer_updates'],'outeridentitychanged')
  gate.validate_outer(row['metrics'],profile='multitraj200');need(abs(row['metrics']['actor/lr']-1e-6*row['actual_outer_updates']/57)<1e-18,'actualpostouterwarmuplrchanged')
 if require_paused:
  text=live.read_text();need(text.count('OPSD_E1_STEP5_PAUSED_JSON=')==1
       and 'OPSD_E1_STEP5_RELEASED_JSON=' not in text
       and len([x for x in text.splitlines() if 'OPSD_E1_OUTER_JSON=' in x])==5,'currentrunalreadyleftpause')
  final=run/'acceptance-5-receipt.json';need(not final.exists() and not final.is_symlink(),'receiptexists')
 inv=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
 inventory=list(csv.reader(io.StringIO(inv)));uuids={r[1].strip() for r in inventory};need(len(inventory)==8 and len(uuids)==8 and all('H100' in r[2] for r in inventory),'actualeightH100missing')
 apps_raw=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
 apps=list(csv.reader(io.StringIO(apps_raw)));need(all(len(r)==3 for r in apps),'invalidunfilteredapps')
 need({int(r[1].strip()) for r in apps}==workers|{service['pid']},'foreignormissingGPUprocess')
 need({r[0].strip() for r in apps if int(r[1].strip())==service['pid']}==uuids,'retrieverGPUshardschanged')
 placements=[{r[0].strip() for r in apps if int(r[1].strip())==p} for p in workers]
 need(all(len(p)==1 for p in placements) and set().union(*placements)==uuids,'eightpolicyGPUplacementchanged')
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':D['job_id'],'driver':driver,
         'workers':actual_workers,'service':actual_service,'slurm_original':s,'gpu_inventory_original':inv,
         'gpu_unfiltered_apps_original':apps_raw,'source_commit':D['source_commit'],'require_paused':require_paused}
def exclusive(path,raw):
 path=pathlib.Path(path);parent=path.parent;need(parent.resolve(strict=True)==parent and parent.stat().st_uid==0,'foreignpublicationparent')
 temp=parent/('.root-first5-'+secrets.token_hex(16)+'.tmp')
 fd=os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(temp,path,follow_symlinks=False)
 finally:
  temp.unlink(missing_ok=True)
 fd=os.open(str(parent),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 need(path.read_bytes()==raw,'publishedbyteschanged')
if request['action']=='check':
 print(json.dumps(validate_live(),sort_keys=True))
elif request['action']=='publish':
 before=validate_live();raw=base64.b64decode(request['envelope_base64'],validate=True)
 need(hashlib.sha256(raw).hexdigest()==request['envelope_sha256'],'transportenvelopechanged')
 envelope=unique(raw);need(envelope['payload']==D['payload'] and set(envelope)=={'payload','signature_hex'},'envelopepayloadchanged')
 public=physical(pathlib.Path(D['run_root'])/'bindings/public.json');need(hashfile(public)==D['public_sha256'],'publicchanged')
 path=physical(pathlib.Path(D['repo'])/'search_r1/baselines/checkpoint_gate.py')
 source=path.read_bytes();need(hashlib.sha256(source).hexdigest()==D['gate_sha256'],'publishgatebyteschanged')
 gate=types.ModuleType('publish_frozen_gate');exec(compile(source,str(path),'exec'),gate.__dict__)
 gate.verify_rsa_signature(envelope['payload'],envelope['signature_hex'],unique(public.read_bytes()))
 final=pathlib.Path(D['run_root'])/'acceptance-5-receipt.json';exclusive(final,raw)
 after=validate_live(require_paused=False);need(hashfile(physical(final))==request['envelope_sha256'],'finalreceiptchanged')
 print(json.dumps({'published_exclusively':True,'receipt_sha256':request['envelope_sha256'],'before':before,'after':after},sort_keys=True))
else:raise ValueError('Unknown root release action')
'''


def local_gate():
    path=HERE.parent/'opsd-unclipped-multitraj200/search_r1/baselines/checkpoint_gate.py'
    raw=pinned_local(path,GATE_SHA);gate=types.ModuleType('root_frozen_author200_gate')
    exec(compile(raw,str(path),'exec'),gate.__dict__);return gate


def key_for_root_execution():
    need(PRIVATE.resolve(strict=True)==PRIVATE and PRIVATE.is_file() and PRIVATE.stat().st_uid==os.getuid()
         and stat.S_IMODE(PRIVATE.stat().st_mode)==0o600 and stat.S_IMODE(PRIVATE.parent.stat().st_mode)==0o700,
         'Task private key must remain local physical with restricted permissions')
    public=unique(pinned_local(PUBLIC,PUBLIC_SHA))
    modulus=subprocess.check_output(['openssl','rsa','-in',str(PRIVATE),'-noout','-modulus'],stderr=subprocess.PIPE,text=True).strip().split('=',1)[1]
    need(int(modulus,16)==int(public['n_hex'],16),'Local private and actual public key differ');return public


def main():
    global CONFIG_SHA
    parser=argparse.ArgumentParser();parser.add_argument('--config-sha256',required=True)
    for name in ('job-id','manifest','manifest-sha256','driver-start','unsigned-payload',
                 'unsigned-payload-sha256','unsigned-payload-remote-path','evidence-spec',
                 'evidence-spec-sha256','evidence-spec-remote-path','root-review','root-review-sha256',
                 'independent-review','independent-review-sha256','source','tools-sha256','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--driver-pid',required=True,type=int)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();CONFIG_SHA=args.config_sha256;need(args.publish,'Prepared only: explicit root --publish required')
    output=Path(args.output_dir)
    need(output.is_absolute() and output.parent.resolve(strict=True)==output.parent
         and output.parent.stat().st_uid==os.getuid()
         and not output.exists() and not output.is_symlink(),'New physical local audit directory required; never blindly retry')
    raw_inputs={name:pinned_local(getattr(args,name),getattr(args,name+'_sha256'))
                for name in ('manifest','unsigned_payload','evidence_spec','root_review','independent_review')}
    manifest,payload,spec,root_review,independent=[unique(raw_inputs[name])
          for name in ('manifest','unsigned_payload','evidence_spec','root_review','independent_review')]
    run,records=check_inputs(args,manifest,payload,spec,root_review,independent)
    pinned_local(TOOLS,TOOLS_SHA);gate=local_gate()
    data={'run_root':run,'job_id':args.job_id,'source_commit':SOURCE,'repo':ROOT+'/repo-'+SOURCE[:12],
          'driver_pid':args.driver_pid,'driver_start':args.driver_start,'manifest':manifest,
          'manifest_sha256':args.manifest_sha256,'payload':payload,'spec':spec,'all_records':records,
          'unsigned_payload_remote_path':args.unsigned_payload_remote_path,
          'evidence_spec_remote_path':args.evidence_spec_remote_path,
          'tools_manifest':ROOT+'/tools/BLOCKED_LOCAL_RECEIPT_MANIFEST',
          'tools_sha256':TOOLS_SHA,'gate_sha256':GATE_SHA,'public_sha256':PUBLIC_SHA}
    output.mkdir(mode=0o700)
    def save(name,raw):
        with (output/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    save('execution-intent.json',canonical({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id':args.job_id,'scope':'ACTUAL_FIRST5_RELEASE_INTENT_NOT_COMPLETION','input_pins':
        {name:sha(raw) for name,raw in raw_inputs.items()},'driver_pid':args.driver_pid,'driver_start':args.driver_start}))
    for name,raw in raw_inputs.items():save(name+'-local-original.json',raw)
    import transport
    def request_remote(action,extra=None):
        request={'action':action,'data':data};request.update(extra or {})
        try:
            result=transport.remote('python3 -c '+shlex.quote(LIVE_BODY),canonical(request),timeout=600)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            save(action+'-transport-failed.stdout',error.stdout or b'')
            save(action+'-transport-failed.stderr',error.stderr or b'')
            save(action+'-transport-failure.json',canonical({'scope':'UNKNOWN_PUBLICATION_OUTCOME_DO_NOT_RETRY' if action=='publish'
                else 'PRE_SIGN_CHECK_FAILURE_NO_PUBLICATION','error':type(error).__name__,'detail':str(error)}))
            raise
        save(action+'-remote-original.json',result.stdout);save(action+'-remote.stderr',result.stderr);return unique(result.stdout)
    request_remote('check')
    public=key_for_root_execution()
    signature=subprocess.run(['openssl','dgst','-sha256','-sign',str(PRIVATE)],input=canonical(payload),
                             capture_output=True,check=True).stdout
    gate.verify_rsa_signature(payload,signature.hex(),public)
    envelope=canonical({'payload':payload,'signature_hex':signature.hex()});save('signed-envelope.json',envelope)
    result=request_remote('publish',{'envelope_base64':base64.b64encode(envelope).decode(),'envelope_sha256':sha(envelope)})
    need(result.get('published_exclusively') is True and result.get('receipt_sha256')==sha(envelope),'Actual publication response differs')
    print(json.dumps({'job_id':args.job_id,'receipt_sha256':sha(envelope),'output':str(output),
      'scope':'ACTUAL_FIVE_RELEASE_ONLY_NOT_200_OR_EFFECT_ACCEPTANCE'},sort_keys=True))


if __name__=='__main__':main()
