"""Local corrected final evaluation post-receipt candidate. CLI blocked78.
External pin fixes actual stage/job/UID/repo/GPU inventory independently of output.
"""
import argparse,datetime,hashlib,json,math,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'strict-tools'))
from training_evidence import loads,digest,verify_slurm,OWN,COMMIT
from strict_stage_prerequisites import bound_file,verify_model_files
EVAL_COMMIT='e3f78f13b58898f7ab2f5a99cac2b76055d47a5a'
UUID_PATTERN=r'GPU-[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
POST_FIELDS={'schema','job_id','uid','eval_repo','stage','input_manifest','transcript','model_before','model_after','slurm_terminal','resources','main_adapter','late_lease_capture','launch_capture','retriever_capture','runtime_log','exit','finished','cleanup','source_bundle','retrieval_http','retrieval_bulk','retrieval_comparison','independent_review'}
EXTERNAL_FIELDS={'version','phase','job_id','uid','eval_repo','eval_commit','stage_sha256','input_manifest_sha256','model_role','selected_gpu_uuids','post_verifier_sha256','launcher_bundle_sha256','canonical_template_manifest','bundle_directory','config_binding'}

def exact_external(record):
 if set(record)!=EXTERNAL_FIELDS or record['version']!=1 or record['phase']!='post-evaluation-external-binding':raise ValueError('Exact externally reviewed post binding required')
 if not re.fullmatch('[0-9]+',str(record['job_id'])) or record['job_id'] in ('2994','2996','2999','3007') or type(record['uid']) is not int:raise ValueError('New actual corrected eval job and own UID required')
 if record['eval_commit']!=EVAL_COMMIT or record['model_role'] not in ('base','hf500'):raise ValueError('Corrected source/model role differs')
 uuids=record['selected_gpu_uuids']
 if not isinstance(uuids,list) or len(uuids)!=8 or len(set(uuids))!=8 or any(not re.fullmatch(UUID_PATTERN,u) for u in uuids):raise ValueError('Eight external actual GPU UUIDs required')
 if not Path(record['bundle_directory']).is_absolute() or set(record['config_binding'])!={'path','sha256'} or not Path(record['config_binding']['path']).is_absolute() or not re.fullmatch('[a-f0-9]{64}',record['config_binding']['sha256']):raise ValueError('External original bundle/config path and SHA binding required')
 canonical=record['canonical_template_manifest']
 if set(canonical)!={'path','sha256'} or not Path(canonical['path']).is_absolute() or not re.fullmatch('[a-f0-9]{64}',canonical['sha256']):raise ValueError('External frozen canonical template binding required')
 for name in ('stage_sha256','input_manifest_sha256','post_verifier_sha256','launcher_bundle_sha256'):
  if not re.fullmatch('[a-f0-9]{64}',record[name]):raise ValueError('External SHA pin missing')

def normal_terminal(text,external):
 job=external['job_id'];verify_slurm(text,job)
 for key,wanted in [('JobId',job),('NodeList','h100-10'),('NumNodes','1'),('TresPerNode','gres:gpu:h100:8')]:
  if re.findall(r'\b'+key+r'=([^\s]+)',text)!=[wanted]:raise ValueError('Actual terminal own8GPU job identity differs')
 ids=re.findall(r'\bUserId=[^\s()]+\(([0-9]+)\)',text)
 if ids!=[str(external['uid'])]:raise ValueError('Actual scheduler UID differs from external binding')

def process_raw(process,job,uid):
 if type(process['pid']) is not int or process['pid']<=0 or process['uid']!=uid or process['uids']!=[uid]*4:raise ValueError('Actual own process four UID identity differs')
 stat=process['raw_stat'];fields=stat.rsplit(')',1)[1].split()
 if not stat.startswith(str(process['pid'])+' ') or fields[19]!=str(process['process_start_ticks']) or int(fields[1])!=process['ppid']:raise ValueError('Actual process stat differs')
 status=dict(line.split(':',1) for line in process['raw_status'].splitlines() if ':' in line)
 if [int(x) for x in status['Uid'].split()]!=[uid]*4:raise ValueError('Actual raw UID status differs')
 if process['cgroup']!=process['raw_cgroup'] or not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',process['cgroup']):raise ValueError('Actual process cgroup outside own job')
 if [p.decode() for p in bytes.fromhex(process['raw_cmdline_hex']).split(b'\0') if p]!=process['cmdline'] or not process['cmdline']:raise ValueError('Actual raw process command differs')

def runtime_crossbinding(transcript,stage,external):
 audit=transcript['zero_update_guard'];runtime=audit['corrected_runtime'];spec=runtime['worker_audit']['spec'];rows=runtime['worker_audit']['rows']
 if audit['attempted_update_dispatches']!=0 or 'actor_rollout_wg.update_actor' not in audit['guarded_dispatches'] or runtime['rejected_calls']!=0:raise ValueError('Actual zero dispatch/generation guard invalid')
 if spec!={'eval_repo':external['eval_repo'],'source_files':stage['source_files'],'job_id':external['job_id'],'stage_sha256':external['stage_sha256'],'uid':external['uid']}:raise ValueError('Worker spec not bound to external actual job/repo/source/UID/stage')
 if len(rows)!=8 or {r['rank'] for r in rows}!=set(range(8)) or len({r['pid'] for r in rows})!=8 or {r['gpu_uuid'] for r in rows}!=set(external['selected_gpu_uuids']):raise ValueError('Eight actual worker rank/PID/selected UUID set differs')
 for row in rows:
  if type(row['device_index']) is not int or row['device_index']<0 or not re.fullmatch(UUID_PATTERN,row['gpu_uuid']):raise ValueError('Actual worker device identity invalid')
  if row['node'].split('.')[0]!='h100-10' or row['job_id']!=external['job_id'] or row['stage_sha256']!=external['stage_sha256'] or row['uid']!=external['uid']:raise ValueError('Worker external node/job/stage/UID differs')
  process_raw(row['process'],external['job_id'],external['uid'])
 if transcript['count']!=51713 or len(transcript['files'])!=809 or [v['count'] for v in transcript['files']]!=[64]*808+[1]:raise ValueError('Full final exactly809 native batches (808x64+1) required')
 calls=runtime['generation_calls']
 if not calls or [r['sequence'] for r in calls]!=list(range(1,len(calls)+1)) or any(r['do_sample'] is not False or r['validate'] is not True or type(r['rows']) is not int or r['rows']%8 or not 8<=r['rows']<=64 for r in calls):raise ValueError('Actual greedy generation call budget invalid')
 return rows

def verify_launch_capture(capture,external):
 from activation_verification import verify_lease
 if set(capture)!={'job_id','stage_sha256','process','selected_gpu_preflight','global_lease','gpu_leases','active_gpu_processes','raw_compute_apps','runtime_binding'} or capture['job_id']!=external['job_id'] or capture['stage_sha256']!=external['stage_sha256']:raise ValueError('Actual own launch capture differs')
 process_raw(capture['process'],external['job_id'],external['uid'])
 verify_tool_binding(capture['runtime_binding'],external,'eval_job.py')
 command=capture['process']['cmdline']
 if str(Path(external['bundle_directory'])/'eval_job.py') not in command or '--config' not in command or command[command.index('--config')+1]!=external['config_binding']['path']:raise ValueError('Actual launch command not external bundle eval_job/config argv')
 verify_lease(capture['global_lease'],OWN/'gpu-job.lock',9)
 inventory=capture['selected_gpu_preflight']
 if len(inventory)!=8 or {g['uuid'] for g in inventory}!=set(external['selected_gpu_uuids']):raise ValueError('Actual preflight selected8 inventory differs')
 for g in inventory:
  values=[float(g[k]) for k in ('utilization','memory_used','memory_total')]
  if any(not math.isfinite(v) for v in values) or not 0<=values[0]<=15 or values[1]<0 or values[2]<=0 or values[1]/values[2]>.15:raise ValueError('Actual launch GPU occupancy exceeds unchanged15%')
 leases=capture['gpu_leases']
 if len(leases)!=8 or len({v['fd'] for v in leases})!=8 or {Path(v['lock_path']).stem for v in leases}!=set(external['selected_gpu_uuids']):raise ValueError('Actual eight perGPU leases differ')
 for lease in leases:
  if type(lease['fd']) is not int or lease['fd']<3 or lease['fd']==9:raise ValueError('Actual GPU descriptor differs')
  verify_lease(lease,Path('/tmp')/f"search-r1-gpu-leases-{external['uid']}"/Path(lease['lock_path']).name,lease['fd'])

def verify_tool_binding(binding,external,filename):
 expected=Path(external['bundle_directory'])/filename
 if binding['directory']!=external['bundle_directory'] or binding['config_binding']!=external['config_binding'] or binding['executing_file']!={'path':str(expected),'sha256':digest(expected)}:raise ValueError('Actual helper path/SHA/config not externally pinned runtime')
 if digest(Path(external['config_binding']['path']))!=external['config_binding']['sha256']:raise ValueError('Original actual config changed')

def verify_late_capture(capture,external,helper):
 from activation_verification import verify_lease
 if capture['job_id']!=external['job_id'] or capture['stage_sha256']!=external['stage_sha256']:raise ValueError('Late own job/stage lease capture differs')
 process_raw(capture['process'],external['job_id'],external['uid'])
 if 'runtime_binding' in capture:verify_tool_binding(capture['runtime_binding'],external,helper)
 selected=set(external['selected_gpu_uuids'])
 if len(capture['selected_gpu_inventory'])!=8 or {r['uuid'] for r in capture['selected_gpu_inventory']}!=selected:raise ValueError('Post-start selected8 actual UUID differs')
 verify_lease(capture['global_lease'],OWN/'gpu-job.lock',9)
 leases=capture['gpu_leases']
 if len(leases)!=8 or len({r['fd'] for r in leases})!=8 or {Path(r['lock_path']).stem for r in leases}!=selected:raise ValueError('Late selected8 inherited descriptor set differs')
 for record in leases:
  if type(record['fd']) is not int or record['fd']<3 or record['fd']==9:raise ValueError('Late actual GPU descriptor differs')
  verify_lease(record,Path('/tmp')/f"search-r1-gpu-leases-{external['uid']}"/Path(record['lock_path']).name,record['fd'])
 actual=[]
 for line in capture['raw_compute_apps'].splitlines():
  fields=[v.strip() for v in line.split(',')]
  if len(fields)!=2 or not fields[1].isdigit():raise ValueError('Late GPU PID inventory unreadable')
  if fields[0] in selected:actual.append((int(fields[1]),fields[0]))
 rows=capture['active_gpu_processes']
 if [(r['pid'],r['gpu_uuid']) for r in rows]!=actual:raise ValueError('Late GPU process capture differs from raw NVML PID/UUID inventory')
 for row in rows:process_raw(row,external['job_id'],external['uid'])

def verify_main_adapter(record,stage,external,bundle):
 import inspect
 from native_main_adapter import MainLeaseWindow,compiled_function_code
 root=Path(external['eval_repo']);original=root/'search_r1/baselines/preflight.py';replacement=Path(external['bundle_directory'])/'native_main_adapter.py'
 if record['schema']!='actual-corrected-main-lease-adapter-v1' or record['restored_original'] is not True or record['replacement_intact'] is not True or record['main_exception'] is not None or not record['calls']:raise ValueError('Actual native main adapter window/restoration differs')
 verify_tool_binding(record['runtime_binding'],external,'native_main_adapter.py')
 for role,path,qualname in [('original',original,'validate_inherited_gpu_leases'),('replacement',replacement,'MainLeaseWindow.replacement')]:
  row=record[role]
  if row['code_sha256']!=compiled_function_code(path,qualname.split('.')[-1]):raise ValueError('Actual runtime original/replacement bytecode differs from frozen source')
  if row['path']!=str(path) or row['code_filename']!=str(path) or row['qualname']!=qualname or row['sha256']!=digest(path):raise ValueError('Actual adapter original/replacement function/code source differs')
 if record['original']['sha256']!=stage['source_files']['search_r1/baselines/preflight.py'] or record['replacement']['sha256']!=bundle['file_sources']['native_main_adapter.py']['sha256']:raise ValueError('Adapter runtime source is not actual independently reviewed byte pin')
 if record['replacement']['function_source_sha256']!=hashlib.sha256(inspect.getsource(MainLeaseWindow.replacement).encode()).hexdigest():raise ValueError('Actual adapter replacement function text differs')
 # Extract original function text from the externally pinned file without importing GPU code.
 import ast
 tree=ast.parse(original.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_inherited_gpu_leases')
 lines=original.read_text().splitlines(keepends=True);raw=''.join(lines[function.lineno-1:function.end_lineno])
 if record['original']['function_source_sha256']!=hashlib.sha256(raw.encode()).hexdigest():raise ValueError('Actual original lease function text differs')
 for capture in record['calls']:verify_late_capture(capture,external,'native_main_adapter.py')

def resources_crossbinding(rows,workers,retriever,external):
 job=external['job_id'];selected=set(external['selected_gpu_uuids']);identities={r['pid']:str(r['process']['process_start_ticks']) for r in workers};identities[retriever['process']['pid']]=str(retriever['process']['process_start_ticks']);seen=set();previous=None
 if not rows:raise ValueError('No actual resource samples')
 for row in rows:
  try:stamp=datetime.datetime.fromisoformat(row['time'])
  except (TypeError,ValueError):raise ValueError('Actual resource timestamp differs')
  if 'query_error' in row or str(row['job_id'])!=job or stamp.tzinfo is None or stamp.utcoffset()!=datetime.timedelta(0) or (previous is not None and stamp<=previous):raise ValueError('Actual resource job/time differs')
  previous=stamp;gpus=row['gpus']
  if len(gpus)!=8 or {g['uuid'] for g in gpus}!=selected:raise ValueError('Resource selected8 UUID inventory differs')
  for g in gpus:
   values=[float(g[k]) for k in ('util','used_mib','total_mib')]
   if any(not math.isfinite(x) or x<0 for x in values) or values[0]>100 or values[1]>values[2] or values[2]<=0:raise ValueError('Nonfinite/invalid device resource')
  known={p['pid']:p for p in row['job_processes'] if not p.get('unavailable')}
  for p in known.values():
   if p['uid']!=str(external['uid']) or p['in_slurm_job'] is not True:raise ValueError('Resource actual own process differs')
  for p in row['gpu_processes']:
   if p['uuid'] not in selected:raise ValueError('Cross-node GPU resource')
   if p.get('unavailable') or p.get('in_slurm_job') is not True or p['uid']!=str(external['uid']) or p['pid'] not in known or p['process_start_ticks']!=known[p['pid']]['process_start_ticks']:raise ValueError('Unowned/unreadable actual GPU process')
   value=float(p['gpu_used_mib'])
   if not math.isfinite(value) or value<0:raise ValueError('Invalid process GPU memory')
   if p['pid'] in identities:
    if p['process_start_ticks']!=identities[p['pid']]:raise ValueError('PID reuse differs from captured worker/retriever')
    seen.add((p['pid'],p['uuid']))
  if not row['cgroup']:raise ValueError('No actual cgroup resource')
  for path,value in row['cgroup'].items():
   if not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',path):raise ValueError('Resource cgroup not own job')
   if Path(path).name=='memory.events':
    pairs=[line.split() for line in value.splitlines()]
    if not pairs or any(len(p)!=2 or not p[1].isdigit() for p in pairs) or len({p[0] for p in pairs})!=len(pairs):raise ValueError('Bad memory event resource')
   elif not isinstance(value,str) or not(value.isdigit() or value=='max' and Path(path).name=='memory.max'):raise ValueError('Bad cgroup resource')
 if any((r['pid'],r['gpu_uuid']) not in seen for r in workers) or any((retriever['process']['pid'],u) not in seen for u in selected):raise ValueError('Actual all8 worker and local GPU retriever residency not sampled')

def verify_receipt(path,sha,external,base=OWN):
 exact_external(external)
 if digest(Path(__file__))!=external['post_verifier_sha256']:raise ValueError('Post verifier source differs from external review')
 p=bound_file({'path':str(path),'sha256':sha},base);r=loads(p.read_text())
 if set(r)!=POST_FIELDS or r['schema']!='corrected-evaluation-post-v1':raise ValueError('Exact post receipt schema required')
 if any(r[k]!=external[k] for k in ('job_id','uid','eval_repo')):raise ValueError('Post self-report differs from external job/repo/UID')
 evidence={name:bound_file(r[name],base) for name in POST_FIELDS-{'schema','job_id','uid','eval_repo','transcript'}}
 if digest(evidence['stage'])!=external['stage_sha256'] or digest(evidence['input_manifest'])!=external['input_manifest_sha256']:raise ValueError('Post stage/input external pin differs')
 stage=loads(evidence['stage'].read_text())
 if stage['eval_commit']!=EVAL_COMMIT or stage['model_role']!=external['model_role']:raise ValueError('Actual corrected stage source/role differs')
 repo=Path(external['eval_repo']).resolve()
 if not repo.is_relative_to(base.resolve()) or subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=EVAL_COMMIT:raise ValueError('Actual isolated corrected repo differs')
 subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD'],check=True)
 normal_terminal(evidence['slurm_terminal'].read_text(),external)
 verify_launch_capture(loads(evidence['launch_capture'].read_text()),external)
 if evidence['exit'].read_text().strip()!='0' or evidence['finished'].read_text().strip()!='CORRECTED_FINAL_EVAL_EXITED_51713':raise ValueError('Actual eval did not finish normally')
 for name in ('model_before','model_after'):
  pins=loads(evidence[name].read_text())
  if pins!={'directory':stage['model_directory'],'files':stage['model_files']}:raise ValueError('Actual before/after model pins differ from external stage')
  verify_model_files(pins)
 capture=loads(evidence['retriever_capture'].read_text());verify_retriever(capture,external,stage)
 for name,field in [('retrieval_http','http_sha256'),('retrieval_bulk','bulk_sha256'),('retrieval_comparison','comparison_sha256')]:
  if digest(evidence[name])!=capture[field]:raise ValueError('Actual live retrieval object differs from capture')
 verify_live_retrieval(loads(evidence['retrieval_comparison'].read_text()),loads(evidence['retrieval_http'].read_text()),loads(evidence['retrieval_bulk'].read_text()),external['job_id'])
 transcript_path=bound_file(r['transcript'],base);transcript=loads(transcript_path.read_text())
 provenance=transcript['provenance']
 if provenance['reviewed_stage_sha256']!=external['stage_sha256'] or provenance['reviewed_stage_path']!=str(evidence['stage']):raise ValueError('Transcript not bound to actual external stage path/SHA')
 workers=runtime_crossbinding(transcript,stage,external)
 resources_crossbinding([loads(line) for line in evidence['resources'].read_text().splitlines()],workers,capture,external)
 raw=evidence['runtime_log'].read_text();markers=re.findall(r'CORRECTED_SCORING_VIEW_MANIFEST_SHA256=([a-f0-9]{64})',raw)
 if markers!=[digest(transcript_path)]:raise ValueError('Actual runtime transcript marker missing/duplicated/differs')
 originals=[loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for line in raw.splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in line]
 if originals!=[transcript['online_metrics']]:raise ValueError('Actual original online metric JSON differs')
 bundle=loads(evidence['source_bundle'].read_text())
 if digest(evidence['source_bundle'])!=external['launcher_bundle_sha256']:raise ValueError('Launcher source bundle external pin differs')
 from source_bundle_check import verify_bundle
 canonical=bound_file(external['canonical_template_manifest'],base)
 if bundle['directory']!=external['bundle_directory'] or bundle['config_binding']!=external['config_binding']:raise ValueError('Actual bundle/config differ from external runtime binding')
 verify_bundle(bundle,canonical.parent)
 from runtime_binding import verify_runtime
 verify_runtime(bundle,__file__,external['config_binding']['path'])
 verify_main_adapter(loads(evidence['main_adapter'].read_text()),stage,external,bundle)
 verify_late_capture(loads(evidence['late_lease_capture'].read_text()),external,'eval_supervisor.py')
 for name,pin in bundle['file_sources'].items():bound_file(pin,base)
 cleanup=loads(evidence['cleanup'].read_text())
 if cleanup!={'job_id':external['job_id'],'node':'h100-10','retained_pids':[], 'listening_ports':[]}:raise ValueError('Actual post own-process/port cleanup incomplete')
 # Invoke native source-pinned CPU replay; no model GPU load/update occurs.
 code="import pathlib,json,sys;from search_r1.baselines import corrected_evaluation as ce;from transformers import AutoTokenizer;assert pathlib.Path(ce.__file__).resolve()==pathlib.Path(sys.argv[1])/'search_r1/baselines/corrected_evaluation.py';t=AutoTokenizer.from_pretrained(sys.argv[6],local_files_only=True);print(ce.encoded(ce.replay(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],t)).decode())"
 import os
 result=subprocess.run([sys.executable,'-c',code,str(repo),str(transcript_path.parent),digest(transcript_path),str(evidence['input_manifest']),external['input_manifest_sha256'],stage['model_directory']],cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True,text=True)
 metrics=loads(result.stdout)
 if metrics!=transcript['online_metrics']:raise ValueError('Native full CPU replay differs')
 review=loads(evidence['independent_review'].read_text())
 pins={name:r[name]['sha256'] for name in r if name not in ('schema','job_id','uid','eval_repo','independent_review')}
 if review!={'status':'PASS','scope':'actual-corrected-evaluation-post','job_id':external['job_id'],'stage_sha256':external['stage_sha256'],'evidence_sha256':pins}:raise ValueError('Actual independently reviewed post evidence differs')
 return {'kind':'actual-corrected-final-evaluation','job_id':external['job_id'],'model_role':stage['model_role'],'stage_sha256':external['stage_sha256'],'input_manifest_sha256':external['input_manifest_sha256'],'generation_budget_sha256':stage['generation_budget_sha256'],'source_files':stage['source_files'],'eval_commit':stage['eval_commit'],'metrics':metrics,'receipt_sha256':sha,'questions':51713,'batches':809}

def verify_retriever(capture,external,stage):
 required={'job_id','node','url','process','listeners','source_frames','runtime_binding','http_sha256','bulk_sha256','comparison_sha256'}
 if set(capture)!=required or capture['job_id']!=external['job_id'] or capture['node']!='h100-10' or capture['url']!='http://127.0.0.1:18084/retrieve':raise ValueError('Actual retriever locality/job schema differs')
 process_raw(capture['process'],external['job_id'],external['uid'])
 verify_tool_binding(capture['runtime_binding'],external,'retrieval_audit_entry.py')
 command=capture['process']['cmdline'];joined=' '.join(command)
 if '--device cuda' not in joined or '--faiss_gpu' not in command or '--host 127.0.0.1' not in joined or '--port 18084' not in joined or str(Path(external['eval_repo'])/'reproduction/h20/retriever_local.py') not in command or str(Path(external['bundle_directory'])/'retrieval_audit_entry.py') not in command:raise ValueError('Actual same-node native CUDA+FAISS retriever command differs')
 if f"pid={capture['process']['pid']}," not in capture['listeners'] or '127.0.0.1:18084' not in capture['listeners'] or re.search(r'(?:0\.0\.0\.0|\[::\]|\*):18084',capture['listeners']):raise ValueError('Actual local port not owned by retriever')
 required_modules={'search_r1.search.retrieval_server','reproduction.h20.retriever_local'}
 if set(capture['source_frames'])!=required_modules:raise ValueError('Actual retriever source identity incomplete')
 for name,pin in capture['source_frames'].items():
  filename=name.replace('.','/')+'.py'
  if pin!={'path':str(Path(external['eval_repo'])/filename),'sha256':stage['source_files'][filename]} or digest(Path(pin['path']))!=pin['sha256']:raise ValueError('Retriever runtime source differs from actual stage')
 # Full live GPU comparison/HTTP/bulk objects are bound separately by immutable capture.
 for name in ('http_sha256','bulk_sha256','comparison_sha256'):
  if not re.fullmatch('[a-f0-9]{64}',capture[name]):raise ValueError('Live retriever evidence SHA missing')

def verify_live_retrieval(comparison,http,bulk,job):
 queries=comparison['queries']
 if len(queries)!=8 or set(http)!={'3','50'} or comparison['index_dimension']!=768 or comparison['index_vectors']<=0:raise ValueError('Actual live retrieval reference schema differs')
 for k in (3,50):
  reference=comparison['content'][str(k)]['gpu'];scores=comparison['gpu_scores'];data=http[str(k)]['result']
  if len(data)!=8 or len(reference)!=8 or any(len(row)!=k for row in data):raise ValueError('Live retrieval rows/topk differ')
  for i,row in enumerate(data):
   for j,item in enumerate(row):
    value=float(item['score'])
    if not math.isfinite(value) or hashlib.sha256(item['document']['contents'].encode()).hexdigest()!=reference[i][j] or abs(value-scores[i][j])>=1e-5:raise ValueError('Live local GPU result differs from original GPU reference')
  if comparison['content'][str(k)]['id_overlap']<.95 or comparison['sorted_score_max_abs_delta']>=.01:raise ValueError('Actual CPU/GPU reference mismatch')
 if bulk['job_id']!=job or bulk['query_entries']!=24 or bulk['unique_queries']!=8 or bulk['topk']!=3:raise ValueError('Live bulk24 differs from actual same-node GPU reference')
 if len(bulk['results'])!=24:raise ValueError('Live bulk24 count differs')
 for row,ref in zip(bulk['results'],http['3']['result']*3):
  if len(row)!=3:raise ValueError('Live bulk24 topk differs')
  for item,wanted in zip(row,ref):
   value=float(item['score'])
   if item['document']!=wanted['document'] or not math.isfinite(value) or abs(value-float(wanted['score']))>=1e-5:raise ValueError('Live bulk24 content/score differs')

def compare_pair(base,hf):
 if base['model_role']!='base' or hf['model_role']!='hf500' or base['job_id']==hf['job_id']:raise ValueError('Distinct actual base/HF500 roles/jobs required')
 for key in ('kind','input_manifest_sha256','generation_budget_sha256','source_files','eval_commit','questions','batches'):
  if base[key]!=hf[key]:raise ValueError('Paired actual input/budget/source/row scope differs')
 if set(base['metrics'])!=set(hf['metrics']):raise ValueError('Paired actual source metrics differ')
 return {'base':base['receipt_sha256'],'hf500':hf['receipt_sha256'],'input_manifest_sha256':base['input_manifest_sha256'],'delta':{k:hf['metrics'][k]-v for k,v in base['metrics'].items()}}

if __name__=='__main__':
 print('BLOCKED PREVIEW: actual corrected evaluation post receipts are UNFILLED',file=sys.stderr)
 raise SystemExit(78)
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--external-binding',required=True);parser.add_argument('--external-sha256',required=True)
 args=parser.parse_args();external_path=Path(args.external_binding)
 if digest(external_path)!=args.external_sha256:raise ValueError('External post binding SHA differs')
 print(json.dumps(verify_receipt(args.receipt,args.receipt_sha256,loads(external_path.read_text())),sort_keys=True))
