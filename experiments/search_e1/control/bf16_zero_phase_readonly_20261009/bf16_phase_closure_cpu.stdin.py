import ast,base64,collections,datetime,hashlib,json,os,pathlib,random,time,types
N=pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
S=N/'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
def sha(b):return hashlib.sha256(b).hexdigest()
def require(c,m):
 if not c:raise ValueError(m)
copies={};proofs=[]
def read(p,expected=None,keep=False):
 p=pathlib.Path(p);before=p.stat();raw=p.read_bytes();after=p.stat()
 fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
 require(all(getattr(before,k)==getattr(after,k) for k in fields),'changed during read '+str(p))
 proof=dict(path=str(p),sha256=sha(raw),bytes=len(raw),device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns)
 require(expected is None or proof['sha256']==expected,'SHA differs '+str(p))
 proofs.append(proof)
 if keep:copies[str(p)]=base64.b64encode(raw).decode()
 return raw,proof
def j(p,expected=None,keep=False):
 raw,pf=read(p,expected,keep);return json.loads(raw),pf
cfg,cf=j(S/'common-zero/evaluation-config.json','899ac3fe26716e45f100bb815af965b9395c910ec597fa0b47a284ae071fc6a2',True)
train,tf=j(cfg['training']['config']['path'],cfg['training']['config']['sha256'])
inventory,ip=j(cfg['helper_source_manifest']['path'],'2e55c809d91bd3b4654d0ec0c4b0532e3ee8c413aac69a19cfd8f28bf042c842')
H=pathlib.Path(cfg['paths']['helper_root'])
raw,sf=read(H/'summary.py',inventory['files']['summary.py']['sha256'])
ns=dict(Counter=collections.Counter,random=random,require=require)
tree=ast.parse(raw)
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('counts','bootstrap','paired')],type_ignores=[]),str(H/'summary.py'),'exec'),ns)
zero=pathlib.Path(cfg['paths']['output_root']);phase,pp=j(zero/'bf16_zero/phase-complete.json',keep=True)
require(phase['accepted'] is True and phase['run_id']==cfg['run_id'] and phase['version']=='bf16_zero' and phase['actual_worker_count']==phase['expected_worker_count']==8,'phase mismatch')
exits=phase['child_exit_evidence'];require(len(exits)==8 and all(x['actual_wait_returncode']==0 for x in exits),'not actual eight rc0')
data_raw,dp=read(train['validation_identity']['path'],train['validation_identity']['sha256'])
data=[json.loads(x) for x in data_raw.splitlines()]
require(len(data)==512 and collections.Counter(x['data_source'] for x in data)==collections.Counter(nq=256,hotpotqa=256),'cohort mismatch')
def objectsha(o):return sha(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
def collect(directory,version,runid,configsha):
 rows=[];workers=[]
 for rank in range(8):
  c,cp=j(directory/f'worker-{rank}/complete.json',keep=True)
  require(c['accepted'] is True and c['status']=='COMPLETE' and c['primary_rows']==64 and c['rank']==rank and c['version']==version and c['run_id']==runid and c['config']['sha256']==configsha,'worker differs')
  raw,rp=read(c['results']['path'],c['results']['sha256'],True);require(raw.endswith(b'\n'),'partial rows')
  rr=[json.loads(x) for x in raw.splitlines()];require(len(rr)==64,'not64')
  for row in rr:
   i=row['validation_index'];require(row['rank']==rank and i%8==rank and row['slot']==i//8 and row['version']==version and row['run_id']==runid and row['primary'] is True and row['status']=='complete','slot differs')
   require(row['question_id']==data[i]['question_id'] and row['dataset']==data[i]['data_source'] and row['row_semantic_sha256']==objectsha(data[i]),'heldout differs')
   require(type(row['protocol_em']) is int and row['protocol_em'] in (0,1),'score differs')
   require(row['failure_class']==(None if row['protocol_em'] else 'final_invalid' if not row['protocol_valid_final_action'] else 'valid_gold_mismatch'),'class differs')
  workers.append(dict(rank=rank,complete=cp,results=rp,process=c['process'],source_manifest=c['source']['manifest'],science_manifest=c['science']['manifest'],actual_adapter=c['actual_adapter'],actual_loaded_adapter=c['actual_loaded_adapter'],actual_generation_config=c['actual_generation_config']))
  rows.extend(rr)
 rows.sort(key=lambda x:x['validation_index']);require([x['validation_index'] for x in rows]==list(range(512)),'coverage differs')
 return rows,workers
a,aw=collect(zero/'bf16_zero','bf16_zero',cfg['run_id'],cf['sha256'])
c0cfg,c0cf=j(S/'00-C0/evaluation-config.json')
C0=pathlib.Path(c0cfg['paths']['output_root'])
c0summary,c0sp=j(C0/'summary.json',keep=True)
b,bw=collect(C0/'raw10_bf16','raw10_bf16',c0cfg['run_id'],c0cf['sha256'])
require(json.loads(json.dumps(ns['counts'](b)))==c0summary['models']['raw10_bf16']['overall'] and ns['counts'](b)['strict_hits']==227,'C0 previous accepted differs')
for rank,w in enumerate(aw):
 x=exits[rank]['identity'];require(all(x[k]==w['process'][k] for k in ('pid','uid','starttick')),'exit worker identity differs')
 require(w['source_manifest']['sha256']==train['source_model_manifest']['sha256'] and w['science_manifest']['sha256']==train['runtime_manifest']['sha256'],'source/runtime differs')
 require(w['actual_adapter']['tensor_sha256']==w['actual_loaded_adapter']['tensor_sha256']==train['common_initial_adapter_tensor_sha256'] and w['actual_adapter']['lora_B_all_zero'] is True,'zero adapter differs')
summary=dict(schema='opsd-ablation-common-zero-bf16-phase-only-readonly-v1',accepted=True,observed_unix=time.time(),observed_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),scope='Only closed BF16 zero phase; not overall common-zero controller completion; FP16 excluded',phase=pp,config=cf,summary_function=sf,heldout512=dp,worker_count=8,primary_slots=512,actual_worker_exit_codes=[x['actual_wait_returncode'] for x in exits],models={'bf16_zero':{'overall':ns['counts'](a),'datasets':{d:ns['counts']([r for r in a if r['dataset']==d]) for d in ('nq','hotpotqa')}},'C0':c0summary['models']['raw10_bf16']},paired={'bf16_zero->C0':ns['paired'](a,b)},source_model_manifest=train['source_model_manifest'],runtime_manifest=train['runtime_manifest'],common_initial_adapter_tensor_sha256=train['common_initial_adapter_tensor_sha256'],workers=aw,c0_summary=c0sp,fp16_directory_exists=(zero/'fp16_zero').exists(),overall_controller_complete_exists=(zero/'controller-complete.json').exists(),native_protocol_rescoring_pending=True)

# CPU-only original final-action rescoring from exact frozen source bytes.
import sys
runtime,rm=j(train['runtime_manifest']['path'],train['runtime_manifest']['sha256'])
runtime=runtime.get('files',runtime);root=pathlib.Path(train['paths']['runtime'])
def rh(name):
 v=runtime[name];return v if isinstance(v,str) else v['sha256']
qa_raw,qp=read(root/'upstream/Search-R1/search_r1/qa_protocol.py',rh('upstream/Search-R1/search_r1/qa_protocol.py'))
require(qp['sha256']=='9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd','wrong strict QA')
init_raw,qi=read(root/'upstream/Search-R1/search_r1/__init__.py',rh('upstream/Search-R1/search_r1/__init__.py'))
pkg=types.ModuleType('search_r1');pkg.__file__=qi['path'];pkg.__path__=[str(pathlib.Path(qi['path']).parent)];exec(compile(init_raw,qi['path'],'exec'),pkg.__dict__);sys.modules['search_r1']=pkg
qa=types.ModuleType('search_r1.qa_protocol');qa.__file__=qp['path'];exec(compile(qa_raw,qp['path'],'exec'),qa.__dict__);sys.modules['search_r1.qa_protocol']=qa
core_raw,corep=read(root/'search_e1/core.py',rh('search_e1/core.py'));core=types.ModuleType('bound_original_core');core.__file__=corep['path'];exec(compile(core_raw,corep['path'],'exec'),core.__dict__)
worker_raw,wp=read(H/'worker.py',inventory['files']['worker.py']['sha256'])
wns=dict(require=require);wt=ast.parse(worker_raw)
exec(compile(ast.Module(body=[x for x in wt.body if isinstance(x,ast.FunctionDef) and x.name=='protocol'],type_ignores=[]),str(H/'worker.py'),'exec'),wns)
def checked_json(ref):return j(ref['path'],ref['sha256'])
ns.update(object_sha=objectsha,checked_json=checked_json,science=lambda c:(core,qa,dict(qa=qp,core=corep)),worker=types.SimpleNamespace(protocol=wns['protocol']))
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='aggregate'],type_ignores=[]),str(H/'summary.py'),'exec'),ns)
restricted=json.loads(json.dumps(cfg));restricted['scientific']['versions']=['bf16_zero']
agg=ns['aggregate'](a,data,restricted,verify_traces=True)
require(json.loads(json.dumps(agg['models']['bf16_zero']))==json.loads(json.dumps(summary['models']['bf16_zero'])),'Rescored aggregate changed')
ticket,tp=j(zero/'bf16_zero/launch-ticket.json',keep=True);terminal,ap=j(zero/'bf16_zero/actual-terminal.json',keep=True)
require(ticket['accepted'] is True and ticket['version']=='bf16_zero' and ticket['run_id']==cfg['run_id'] and ticket['config_sha256']==cf['sha256'] and ticket['logical_rank_map']==list(range(8)),'ticket identity differs')
require(terminal['all_actual_wait_zero'] is True and terminal['child_exit_evidence']==exits,'actual terminal differs from phase')
starts=[]
for rank,w in enumerate(aw):
 start,sp=j(zero/'bf16_zero'/('child-start-'+str(rank)+'.json'),keep=True)
 child,cp=j(zero/'bf16_zero'/('child-'+str(w['process']['pid'])+'.json'),keep=True)
 require(start['rank']==rank and start['version']=='bf16_zero' and child['identity']==exits[rank]['identity'] and child['actual_wait_returncode']==0 and child['version']=='bf16_zero' and child['run_id']==cfg['run_id'] and start['identity']==exits[rank]['identity'],'actual child proof differs')
 wcomplete=json.loads(base64.b64decode(copies[str(zero/'bf16_zero'/f'worker-{rank}/complete.json')]))
 require(wcomplete['variant']==cfg['variant'] and wcomplete['study_id']==cfg['study_id'] and wcomplete['physical_gpu']==rank and wcomplete['gpu_uuid']==cfg['owner']['gpu_uuids'][rank] and wcomplete['rank_seed']==42 and wcomplete['privileged_reference'] is False and wcomplete['no_training'] is True,'worker scientific scope differs')
 require(wcomplete['model_kind']=='BF16_base_FP32_unmerged_LoRA' and wcomplete['actual_adapter']['tensors']==504 and wcomplete['actual_adapter']['dtype']=='torch.float32' and wcomplete['actual_adapter']['all_finite'] is True and wcomplete['actual_adapter']['changed_tensors_vs_common_initial']==0 and wcomplete['actual_loaded_adapter']['all_finite'] is True,'actual loaded adapter scope differs')
 require(wcomplete['adapter']['manifest']['sha256']==train['common_initial_adapter_manifest']['sha256'],'wrong common initial manifest')
 starts.append(dict(rank=rank,start=sp,actual_exit=cp))
public,pub=j(cfg['gates']['publication']['path'],cfg['gates']['publication']['sha256'])
require(public['source_commit']==cfg['source_commit'],'source publication commit mismatch')
source_manifest,sm=j(train['source_model_manifest']['path'],train['source_model_manifest']['sha256'])
initial,im=j(train['common_initial_adapter_manifest']['path'],train['common_initial_adapter_manifest']['sha256'])
require(source_manifest['source_grpo_step']==100 and len(source_manifest['files'])==12 and initial['adapter_tensor_sha256']==train['common_initial_adapter_tensor_sha256'],'source100 / common adapter manifest differs')
summary.update(native_protocol_rescoring_pending=False,native_strict_cpu_rescoring=dict(questions=512,passed=True,qa_source=qp,core_source=corep,worker_protocol_source=wp,summary_aggregate_source=sf,source_only_compile=True,no_torch_import='torch' not in sys.modules,no_model_or_generation=True),source_and_publication=dict(publication=pub,helper_manifest=ip,source_model_manifest=sm,runtime_manifest=rm,common_initial_adapter_manifest=im,full_source_model_weights_rehashed=False,adapter_tensors_reloaded=False,scope='fresh small identity manifests + actual worker load records; no repeated model hashing or numerical re-inference'),actual_terminal=ap,launch_ticket=tp,worker_start_and_actual_exit=starts,closed_trace_count=512,closed_trace_proofs=[x for x in proofs if x['path'].endswith('/native-capture.json')],paired_by_dataset={d:dict(lost=sum(x['protocol_em']==1 and y['protocol_em']==0 for x,y in zip(a,b) if x['dataset']==d),gained=sum(x['protocol_em']==0 and y['protocol_em']==1 for x,y in zip(a,b) if x['dataset']==d),preserved=sum(x['protocol_em']==1 and y['protocol_em']==1 for x,y in zip(a,b) if x['dataset']==d),still_wrong=sum(x['protocol_em']==0 and y['protocol_em']==0 for x,y in zip(a,b) if x['dataset']==d)) for d in ('nq','hotpotqa')})

print(json.dumps(dict(summary=summary,copies=copies,proofs=proofs),ensure_ascii=False))
