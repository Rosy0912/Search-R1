"""CPU source preparation and root-authorized single known GPU submission.
No model loading; GPU work only occurs if the explicit submit command is invoked.
"""
import argparse,hashlib,json,os,pathlib,re,socket,subprocess,sys
from known_record import atomic,submit_once
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
PIPELINE=None
PIPELINE_SHA=None
CPU_PROOF=None

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def ownlog(path):
 p=pathlib.Path(path)
 if p.is_symlink() or ROOT/'logs' not in p.resolve().parents:raise ValueError('own logs nonalias required')
 return p

def terminal(path,job):
 data=json.loads(ownlog(path).read_text());text=data.get('scontrol_stdout','')
 if data.get('job_id')!=job or data.get('final_state_observed') is not True or data.get('scontrol_returncode')!=0:raise ValueError('Captured actual job terminal absent')
 for field in ('JobId='+str(job),'JobState=COMPLETED','ExitCode=0:0','NodeList=h100-11'):
  if not re.search(r'(?:^|\s)'+re.escape(field)+r'(?:\s|$)',text):raise ValueError('Actual successful own-node terminal absent')
 return data

def cleanup(evidence):
 p=ownlog(pathlib.Path(evidence)/'cleanup.json');data=json.loads(p.read_text())
 keys=['own_processes_remaining','baseline_identity_missing','original_gpu_baseline_missing','own_gpu_records_remaining','sampler_errors','guardian_errors']
 if any(data.get(key)!=[] for key in keys) or data.get('error') is not None or data.get('cleanup',{}).get('actual_cgroup_empty') is not True:raise ValueError('Prior actual owned cleanup/baseline not complete')
 return {'path':str(p),'sha256':sha(p)}

def source_gate(binding_path):
 global PIPELINE,PIPELINE_SHA,CPU_PROOF
 binding_path=ownlog(binding_path);binding=json.loads(binding_path.read_text())
 PIPELINE=pathlib.Path(binding['directory']);PIPELINE_SHA=binding['manifest_sha256']
 if PIPELINE.is_symlink() or ROOT/'tools' not in PIPELINE.resolve().parents:raise ValueError('Own tools nonalias pipeline required')
 if sha(PIPELINE/'files.json')!=PIPELINE_SHA:raise ValueError('Frozen pipeline manifest drift')
 manifest=json.loads((PIPELINE/'files.json').read_text());actual=set()
 for p in PIPELINE.rglob('*'):
  if p.is_symlink():raise ValueError('Pipeline source alias')
  if p.is_file():actual.add(str(p.relative_to(PIPELINE)))
 if actual!=set(manifest['files'])|{'files.json'}:raise ValueError('Pipeline source set drift')
 for name,value in manifest['files'].items():
  p=PIPELINE/name
  if p.stat().st_size!=value['size'] or sha(p)!=value['sha256']:raise ValueError('Pipeline source bytes drift')
 proof=binding['native_cpu'];log=ownlog(proof['log']['path'])
 if sha(log)!=proof['log']['sha256']:raise ValueError('Actual native CPU log drift')
 terminal_ref=proof['terminal'];tp=ownlog(terminal_ref['path'])
 if sha(tp)!=terminal_ref['sha256']:raise ValueError('Actual native terminal drift')
 terminal(tp,proof['job_id'])
 count=proof['passed']
 if count!=binding.get('native_expected'):raise ValueError('Native count differs from explicit final-source expected suite')
 if not isinstance(count,int) or isinstance(count,bool) or count<170:raise ValueError('Final pipeline native suite count absent')
 lines=log.read_text()
 if not re.search(r'(?m)^'+str(count)+r' passed in [0-9.]+s$',lines):raise ValueError('Actual native zero-skip suite absent')
 expected='EXACT_SOURCE_MANIFEST '+str(PIPELINE/'files.json')+' '+PIPELINE_SHA
 if expected not in lines.splitlines():raise ValueError('Native log belongs to another source manifest')
 CPU_PROOF={'binding':{'path':str(binding_path),'sha256':sha(binding_path)},**proof}
 for name in ('pipeline','supervise','checkpoint_inventory','ipc','prepare_run'):
  if name in sys.modules and pathlib.Path(sys.modules[name].__file__).resolve()!=PIPELINE/(name+'.py'):raise ValueError('Shadow pipeline module')
 sys.path.insert(0,str(PIPELINE))
 return manifest

def initial_gate(args):
 terminal(args.initial_terminal,args.initial_job)
 config=json.loads(ownlog(args.initial_config).read_text());evidence=ownlog(config['evidence_root'])
 if str(config.get('job'))!=str(args.initial_job):raise ValueError('Initial config belongs to another job')
 frozen=ownlog(args.initial_frozen);review=ownlog(args.initial_review)
 if frozen!=evidence/'initial/frozen-calibration.json':raise ValueError('Frozen initial receipt is from different evidence root')
 r=json.loads(review.read_text());f=json.loads(frozen.read_text())
 if r.get('status')!='PASS' or r.get('reviewed_frozen_calibration_sha256')!=sha(frozen) or f.get('outer_step')!=0:raise ValueError('Actual successful initial independent review pending')
 exitref=cleanup(evidence)
 return config,{'job_id':args.initial_job,'terminal':{'path':str(args.initial_terminal),'sha256':sha(args.initial_terminal)},'frozen':{'path':str(frozen),'sha256':sha(frozen)},'review':{'path':str(review),'sha256':sha(review)},'cleanup':exitref}

def prepare(args):
 source_gate(args.pipeline_binding);initial,refs=initial_gate(args)
 directory=ownlog(args.directory)
 if directory.exists():raise ValueError('Unique prepare destination already exists; inspect, do not regenerate')
 command=[str(ROOT/'env/bin/python'),str(PIPELINE/'prepare_run.py'),'--initial-config',args.initial_config,'--mode',args.mode,'--directory',str(directory),'--initial-frozen',args.initial_frozen,'--initial-review',args.initial_review]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run(command,env=env,check=True)
 config=directory/'static-config.json';value=json.loads(config.read_text())
 if value['base_model']!=initial['base_model'] or value['train_file']!=initial['train_file'] or value['retriever_health_url']!=initial['retriever_health_url'] or not value['retriever_health_url'].endswith('/docs'):raise ValueError('Original base/data/newhealth recipe changed')
 atomic(directory/'prepare-receipt.json',{'scope':'ACTUAL_CPU_CONFIG_PREPARED_NOT_GPU_AUTHORIZATION','mode':args.mode,'initial':refs,'source_native_cpu':CPU_PROOF,'pipeline_manifest_sha256':PIPELINE_SHA,'config':{'path':str(config),'sha256':sha(config)}})
 print(json.dumps({'prepared':str(config),'independent_gate_pending':True}))

def profile_run_binding(result_path,job):
 result_path=ownlog(result_path);root=result_path.parent
 if result_path.name!='profile2-result.json':raise ValueError('Actual profile result basename differs')
 result=json.loads(result_path.read_text());config_path=root/'effective-runtime-config.json';config=json.loads(config_path.read_text())
 if str(config.get('job'))!=str(job) or pathlib.Path(config.get('evidence_root',''))!=root:raise ValueError('Profile effective config belongs to another job/root')
 expected_authorization=root/'trainer-terminal-authorization.json'
 link=result['terminal_supervision_evidence']
 if link!={'path':str(expected_authorization),'sha256':sha(expected_authorization)}:raise ValueError('Profile terminal authorization from another evidence root')
 authorization=json.loads(expected_authorization.read_text());held_path=root/'trainer-terminal-held.json';held=json.loads(held_path.read_text())
 if str(authorization.get('job'))!=str(job) or str(held.get('job'))!=str(job) or authorization.get('status')!='AUTHORIZED_OWN_SHUTDOWN':raise ValueError('Profile terminal authorization/held job differs')
 if authorization.get('held_sha256')!=sha(held_path) or held.get('effective_config_sha256')!=sha(config_path):raise ValueError('Profile held/config binding differs')
 outer={'path':config['actual_outer_receipt_path'],'sha256':sha(ownlog(config['actual_outer_receipt_path']))}
 if any(x!=outer for x in (result.get('actual_outer_evidence'),authorization.get('actual_outer_evidence'),held.get('actual_outer_evidence'))):raise ValueError('Profile outer link belongs to another run')
 cleanup(root)
 return result

def submit(args):
 source_gate(args.pipeline_binding);config_path=ownlog(args.config);config=json.loads(config_path.read_text())
 from supervise import verify_gate,reviewed_recipe_sha
 gate_path=ownlog(args.gate);gate=json.loads(gate_path.read_text());gate_sha=sha(gate_path)
 config['independent_gate']={'path':str(gate_path),'sha256':gate_sha};verify_gate(config,args.mode)
 canonical=reviewed_recipe_sha(config);entry=PIPELINE/'production.sbatch';approved=config_path.parent/('approved-'+args.mode+'-'+canonical+'.json')
 if approved.exists():
  if json.loads(approved.read_text())!=config:raise ValueError('Approved immutable config differs')
 else:atomic(approved,config)
 prepared=json.loads((config_path.parent/'prepare-receipt.json').read_text())
 if prepared['config']!={'path':str(config_path),'sha256':sha(config_path)} or prepared['mode']!=args.mode:raise ValueError('Prepared actual initial/config binding differs')
 initial_ref=prepared['initial']
 if args.initial_job!=initial_ref['job_id']:raise ValueError('Requested initial job differs from prepared provenance')
 terminal(initial_ref['terminal']['path'],initial_ref['job_id'])
 if sha(initial_ref['frozen']['path'])!=initial_ref['frozen']['sha256'] or sha(initial_ref['review']['path'])!=initial_ref['review']['sha256']:raise ValueError('Actual initial provenance drift')
 if args.mode=='train100':
  if not args.profile_result or not args.profile_review or not args.profile_terminal:raise ValueError('Formal100 needs actual profile2 proof')
  result_path=ownlog(args.profile_result);review_path=ownlog(args.profile_review);p=json.loads(result_path.read_text());r=json.loads(review_path.read_text())
  terminal(args.profile_terminal,args.profile_job);profile_run_binding(result_path,args.profile_job)
  u=p.get('actual_update_evidence',{})
  if p.get('scope')!='ACTUAL_G1_PROFILE2_UNREVIEWED_NOT_100_OR_TOPD' or u.get('actual_outer_steps')!=2 or u.get('optimizer_steps_per_rank')!=6 or u.get('micro_forwards_per_rank')!=20 or u.get('ranks')!=8:raise ValueError('Real G1 profile2 update counters absent')
  if r.get('status')!='PASS' or r.get('reviewed_profile2_result_sha256')!=sha(result_path) or r.get('reviewed_initial_frozen_sha256')!=initial_ref['frozen']['sha256']:raise ValueError('Actual same-calibration profile2 independent proof pending')
  if gate['actual_raw_evidence']!={'path':str(result_path),'sha256':sha(result_path)} or gate['independent_review']!={'path':str(review_path),'sha256':sha(review_path)}:raise ValueError('Formal reviewed resource gate belongs to another profile')
 record=ROOT/'logs'/('g1-known-'+args.mode+'-'+canonical+'-submission.json')
 binding={'mode':args.mode,'recipe_canonical_sha256':canonical,'gate_sha256':gate_sha,'entry_sha256':sha(entry),'prepared_receipt_sha256':sha(config_path.parent/'prepare-receipt.json'),'pipeline_manifest_sha256':PIPELINE_SHA,'source_native_binding_sha256':CPU_PROOF['binding']['sha256']}
 def precheck():
  if pathlib.Path(config['evidence_root']).exists():raise ValueError('Existing output without known submission; reconcile actual run')
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  if any('g1-32b-' in row for row in queue.splitlines()):raise ValueError('Owned G1 initial/profile/formal job still active')
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  rows=gpu.splitlines()
  if len(rows)!=8:raise ValueError('Actual node GPU count changed')
  for row in rows:
   uuid,used,total,sm=row.split(',')
   if not (float(total)>0 and 0<=float(used)/float(total)<=.15 and 0<=float(sm)<=15):raise ValueError('Preallocation baseline not idle protected15')
  subprocess.run(['bash','-n',str(entry)],check=True)
  return {'queue':queue,'gpu':gpu,'initial_cleanup':initial_ref['cleanup']}
 env=dict(os.environ,G1_PIPELINE_CONFIG=str(approved),G1_PIPELINE_DIR=str(PIPELINE),G1_PIPELINE_MODE=args.mode,G1_PREFLIGHT_ENTRY=config['preflight_source']['path'])
 command=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name=g1-32b-'+args.mode,'--comment=g1-known:'+args.mode+':'+canonical,str(entry)]
 result=submit_once(record,binding,command,env,subprocess.run,precheck)
 control=record.with_name(record.stem+'-scontrol-initial.txt')
 if not control.exists():
  process=subprocess.run(['scontrol','show','job','-dd',str(result['job_id'])],text=True,capture_output=True)
  with control.open('x') as stream:stream.write(process.stdout+process.stderr);stream.flush();os.fsync(stream.fileno())
 print(json.dumps({'record':str(record),**result}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','submit']);ap.add_argument('--mode',required=True,choices=['profile2','train100'])
 for key in ('initial-config','initial-frozen','initial-review','initial-terminal','directory','config','gate','profile-result','profile-review','profile-terminal'):ap.add_argument('--'+key)
 ap.add_argument('--pipeline-binding',required=True);ap.add_argument('--initial-job',type=int,required=True);ap.add_argument('--profile-job',type=int);args=ap.parse_args()
 if socket.gethostname()!='h100-11':raise ValueError('Own h11 only')
 os.environ['PYTHONDONTWRITEBYTECODE']='1'
 for key in (('initial_config','initial_frozen','initial_review','initial_terminal','directory') if args.action=='prepare' else ('config','gate')):
  if not getattr(args,key):raise ValueError('Missing '+key)
 (prepare if args.action=='prepare' else submit)(args)
if __name__=='__main__':main()
