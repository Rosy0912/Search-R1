"""External CPU unsigned payload preparer; contains no signing or automatic release."""
import argparse,datetime,hashlib,importlib.util,json,math,os,re,sys
from pathlib import Path
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
COUNT=3085938688
IDENTITY=('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce')

def fail(ok,message):
 if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):
 def pairs(items):
  d={}
  for k,v in items:fail(k not in d,'Duplicate JSON key');d[k]=v
  return d
 return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def file(rec,root):
 fail(set(rec)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',rec['sha256']) is not None,'Invalid SHA record')
 p=Path(rec['path']);fail(p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and p.stat().st_uid==os.getuid(),'Physical current-owned evidence required');p.relative_to(root);fail(sha(p)==rec['sha256'],'Evidence content changed: '+str(p));return p

def frozen_gate(repo,expected_sha):
 path=repo/'search_r1/baselines/checkpoint_gate.py';fail(path.resolve(strict=True)==path and sha(path)==expected_sha,'Frozen gate changed')
 # Pinned source bytes, no editable import/pyc. Its validation functions use
 # standard library only; no worker/torch code is executed in this consumer.
 source=path.read_bytes();module=type(sys)('first5_frozen_gate');module.__file__=str(path);exec(compile(source,str(path),'exec',dont_inherit=True),module.__dict__);return module

def identity(manifest,root):
 fail(manifest['version']==1 and manifest['method']=='pure_opsd' and manifest['source_commit']==SOURCE and manifest['run_root']==str(root) and manifest['uid']==os.getuid(),'Wrong source/run/UID/method')
 fail((manifest['global_step'],manifest['actual_outer_updates'],manifest['next_global_step'],manifest['horizon'],manifest['warmup_steps'],manifest['world_size'])==(5,5,6,200,0,8),'Wrong actual five-step clock')
 fail(type(manifest['driver_pid']) is int and manifest['driver_pid']>0 and isinstance(manifest['job_id'],str) and manifest['job_id'].isdigit() and re.fullmatch('[0-9a-f]{64}',manifest['pause_nonce']) is not None,'Invalid actual PID/job/nonce')
 fail(re.fullmatch('[0-9a-f]{64}',manifest['config_sha256']) is not None,'Invalid config digest')


def log_prefix(live,snapshot,root):
 fail(live!=snapshot and snapshot.parent.resolve(strict=True)==snapshot.parent,'Immutable separate snapshot required');snapshot.relative_to(root)
 physical={'path':str(live),'sha256':sha(live)};file(physical,root)
 # Copy only the observed current live prefix with exclusive creation. The
 # trainer is paused, but no assumption of tee flush/order or fixed size is made.
 size=live.stat().st_size;fail(size>0,'Training log empty')
 h=hashlib.sha256();remaining=size
 with live.open('rb') as src,snapshot.open('xb') as out:
  while remaining:
   block=src.read(min(1024*1024,remaining));fail(bool(block),'Live prefix truncated');out.write(block);h.update(block);remaining-=len(block)
  out.flush();os.fsync(out.fileno())
 fail(snapshot.stat().st_size==size and sha(snapshot)==h.hexdigest(),'Snapshot copy differs')
 with live.open('rb') as src:fail(hashlib.sha256(src.read(size)).hexdigest()==h.hexdigest(),'Live prefix changed while copying')
 return {'live_path':str(live),'snapshot_path':str(snapshot),'sha256':h.hexdigest(),'bytes':size}


def live_state(state):
 return type(state) is str and state in ('R','S','D','I')

def same_service_identity(observed, bound):
 keys=('pid','start','uid','cgroup')
 fail(all(key in observed and key in bound and observed[key]==bound[key] for key in keys),'Retrieval service identity changed')
 fail(live_state(observed.get('state')) and live_state(bound.get('state')),'Service stopped or unknown observed state')
 return True

RANK_DEDUP_SUFFIX=re.compile(r'\x1b\[32m \[repeated ([1-7])x across cluster\]\x1b\[0m\Z')

def parse_observed_rank_console(line,manifest,gate,workers):
 # Only this writer marker accepts the exact observed Ray suppression suffix.
 # The suppression count never creates inferred rank records.
 fail('FULL_STATE_RANK_JSON=' in line,'Not a native writer marker')
 suffix=RANK_DEDUP_SUFFIX.search(line)
 original=line[:suffix.start()] if suffix else line
 report=gate.parse_log_marker(original,'FULL_STATE_RANK_JSON=',driver_pid=manifest['driver_pid'],job_id=manifest['job_id'],worker_pids=workers)
 fail(type(report.get('rank')) is int and 0<=report['rank']<8,'Invalid observed writer rank')
 return report


def check_log(snapshot,manifest,manifest_sha,gate):
 rows=[];pauses=[];state_reports=[];releases=[];evaluations=[]
 workers=gate.manifest_worker_pids(manifest,Path(manifest['run_root']))
 for line in snapshot.read_text().splitlines():
  for marker,items in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_PAUSED_JSON=',pauses),('FULL_STATE_RANK_JSON=',state_reports),('OPSD_E1_STEP5_RELEASED_JSON=',releases),('BASELINE_EVALUATION_METRICS_JSON=',evaluations)]:
   if marker in line:
    items.append(parse_observed_rank_console(line,manifest,gate,workers) if marker=='FULL_STATE_RANK_JSON=' else gate.parse_log_marker(line,marker,driver_pid=manifest['driver_pid'],job_id=manifest['job_id'],worker_pids=workers))
 fail(not releases,'Training was already released')
 fail(len(evaluations)==2,'Require actual initial512 and step5 stage64 evaluation records')
 gate.validate_evaluation(evaluations[0],512);gate.validate_evaluation(evaluations[1],64)
 fail([r['actual_outer_updates'] for r in rows]==[1,2,3,4,5],'Exactly actual five completed outers required')
 for row in rows:
  fail(row['global_step']==row['actual_outer_updates'] and all(row[k]==manifest[k] for k in ('driver_pid','job_id','source_commit','config_sha256')),'Actual outer identity changed');gate.validate_outer(row['metrics'], profile='author200')
  lr=row['metrics'].get('actor/lr');fail(type(lr) in (int,float) and math.isfinite(lr) and lr==5e-6,'Actual outer constant LR differs')
  for rank in range(8):
   m=row['metrics'];suffix='/rank_'+str(rank)
   fail(m.get('actor/old_log_probs_present'+suffix)==1 and m.get('actor/old_log_probs_finite'+suffix)==1 and m.get('actor/supervised_tokens'+suffix,0)>0,'Missing LP/valid positive supervision')
   fail(m.get('teacher/cuda_update_entries_cumulative_per_rank'+suffix)==row['actual_outer_updates'],'Teacher CUDA update clock differs')
 fail(len(pauses)==1 and pauses[0]['manifest_sha256']==manifest_sha and pauses[0]['driver_pid']==manifest['driver_pid'] and pauses[0]['job_id']==manifest['job_id'] and pauses[0]['pause_nonce']==manifest['pause_nonce'],'No unique actual paused boundary')
 gate.validate_evaluation({k.removeprefix('opsd_stage/'):v for k,v in rows[-1]['metrics'].items() if k.startswith('opsd_stage/')},64)
 # Manifest is produced before timing/data metrics are added. Only common fields
 # are compared; full dict equality would reject the correct production order.
 fail(all(rows[-1]['metrics'].get(k)==v for k,v in manifest['outer_metrics'].items()),'Manifest/log fifth metrics conflict')
 return rows,state_reports

def strict_pairs(pairs):
 result={}
 for k,v in pairs:fail(k not in result,'Duplicate log JSON key');result[k]=v
 return result


def check_reports(manifest,root,state_reports):
 reports=[];files=manifest['files']
 fail(1<=len(state_reports)<=8,'Missing observed native writer console')
 observed={}
 for row in state_reports:
  fail(type(row.get('rank')) is int and 0<=row['rank']<8 and row['rank'] not in observed,'Duplicate/invalid observed native writer rank')
  observed[row['rank']]=row
 for rank in range(8):
  rec=files['actor/training_state/rank_'+str(rank)+'.json'];path=file(rec,root);r=read(path);pt=file(files['actor/training_state/rank_'+str(rank)+'.pt'],root)
  fail(r['version']==3 and r['algorithm']=='pure_opsd' and r['status']=='EXACT_FULL_STATE_RELOAD_OK' and (r['rank'],r['world_size'],r['step'])==(rank,8,5) and r['model_state_dict_type']=='SHARDED_STATE_DICT' and r['model_dtensor_entries']>0 and r['optimizer_state_entries']>0 and r['scheduler_last_epoch']==5,'Rank actual full state incomplete')
  fail(r['path']==str(pt) and r['sha256']==r['state_sha256']==sha(pt) and set(r['object_ids'])=={'model','optimizer','scheduler','teacher'} and all(type(v) is int and v>0 for v in r['object_ids'].values()),'Rank file/object identity differs')
  # Eight authoritative proofs are original manifest-pinned reports + raw PT,
  # joined below to the actual native full-content inspection. Ray console
  # suppression is retained as transport evidence, never expanded to eight.
  if rank in observed:fail(canonical(observed[rank])==canonical(r),'Observed writer body differs from exact saved report')
  fail(r['teacher']['parameter_count']==COUNT and r['teacher']['execution_device']=='cuda' and r['teacher']['storage_device']=='cpu' and r['teacher']['counters']['cuda_update_entries']==5,'Teacher fixed identity/update clock differs')
  reports.append(r)
 fail(len({r['pid'] for r in reports})==8,'Rank worker PID alias')
 driver=read(file(files['driver/driver.json'],root));driverpt=file(files['driver/driver.pt'],root)
 fail(driver['status']=='EXACT_DRIVER_STATE_RELOAD_OK' and driver['step']==5 and driver['pid']==manifest['driver_pid'] and driver['uid']==manifest['uid'] and driver['job_id']==manifest['job_id'] and driver['state_sha256']==sha(driverpt),'Driver actual roundtrip differs')
 return reports,driver


def check_cpu_evidence(native,hf,manifest,manifest_sha,reports):
 for evidence in (native,hf):
  fail(evidence['passed'] is True and evidence['source_commit']==SOURCE and evidence['job_id']==manifest['job_id'] and evidence['run_root']==manifest['run_root'] and evidence['checkpoint_manifest_sha256']==manifest_sha and evidence['cuda_is_initialized'] is False,'CPU inspection belongs to another checkpoint or used CUDA')
 fail(hf['scope']=='ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA' and hf['actual_unique_parameters']==COUNT and hf['all_loaded_parameters_finite'] is True and hf['actual_cpu_forward_finite'] is True and hf['total_changed_elements']>0 and hf['config_computation_equal'] is True,'Incomplete actual full HF reload/change')
 fail(native['scope']=='NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE' and native['cpu_runtime']['torch_version'].split('+')[0]=='2.4.0','Native Torch2.4 actual stored-content scope required')
 fail(len(native['ranks'])==8 and {r['rank'] for r in native['ranks']}==set(range(8)),'Native actual8 inspection missing')
 expected_hf={rec['path']:rec['sha256'] for role,rec in manifest['files'].items() if role=='actor/model.safetensors.index.json' or (role.startswith('actor/') and role.endswith('.safetensors'))}
 actual_hf={rec['path']:rec['sha256'] for role,rec in hf['files'].items() if role=='actor/model.safetensors.index.json' or (role.startswith('actor/') and role.endswith('.safetensors'))}
 fail(bool(expected_hf) and native['hf_all_rank_slices_exact'] is True and native['hf_files']==actual_hf==expected_hf,'Native model/HF exact current-index/shards join absent or stale')
 hf_keys=sorted(hf['hf_map']);fail(len(hf_keys)==435,'Actual complete435 HF tensor keys required')
 for r in native['ranks']:
  original=reports[r['rank']]
  fail(r['model_hf_slice_exact'] is True and r['model_hf_slice_keys']==hf_keys and r['model_dtensor_entries']==435 and sorted(r['model_layouts'])==hf_keys,'Native/HF all435 rank slice association missing')
  fail(r['pid']==original['pid'] and r['object_ids']==original['object_ids'] and r['state_sha256']==original['state_sha256'] and r['state_digest']==original['state_digest'],'Native content report not same saved rank')
  fail(r['optimizer_step_values']==[80] and r['scheduler_last_epoch']==5 and r['scheduler_last_lr']==[5e-6] and r['rng_verified'] is True,'Native Adam/scheduler/RNG clock differs')
  fail(r['teacher_content_digest']==hf['base_named_parameter_digest'] and r['teacher_cuda_update_entries']==5,'Frozen teacher no longer official initial weights')
 driver=native['driver'];fail(driver['pid']==manifest['driver_pid'] and driver['sampler_cursor']==driver['consumed_cursor']==2560 and driver['size']==169028 and driver['epoch']==0,'Driver sampler/RNG data cursor differs')


def resource_join(path,manifest,reports,service):
 expected={r['pid']:r['rank'] for r in reports};processes={};samples=[];previous=None;uuids=set()
 with Path(path).open() as handle:
  for line in handle:
   row=json.loads(line,object_pairs_hook=strict_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite resource JSON')));fail(row['job_id']==manifest['job_id'],'Resource job differs');t=datetime.datetime.fromisoformat(row['time']);fail(t.tzinfo is not None and t.utcoffset()==datetime.timedelta(0),'UTC resource time required')
   if previous is not None:fail(t>previous,'Resource time duplicate/backward')
   previous=t
   fail('query_error' not in row,'Resource query failure requires manual repair/new complete prefix; never silently skipped')
   gpu=row['gpus'];fail(len(gpu)==8 and len({v['uuid'] for v in gpu})==8,'Resource eight UUIDs missing');current={v['uuid'] for v in gpu}
   if uuids:fail(uuids==current,'Resource UUID inventory changed')
   for v in gpu:
    fail(math.isfinite(float(v['util'])) and 0<=float(v['util'])<=100 and math.isfinite(float(v['used_mib'])) and float(v['used_mib'])>=0,'Invalid GPU resource scalar')
   uuids=current;apps=row['gpu_processes']
   for app in apps:fail(not app.get('unavailable') and app.get('in_slurm_job') is True and str(app['uid'])==str(manifest['uid']),'Foreign resource GPU PID')
   ownapps={v['pid'] for v in apps};wanted=set(expected)|{service['pid']}
   if ownapps!=wanted:continue
   workers=[v for v in apps if v['pid'] in expected]
   if {v['uuid'] for v in workers}!=uuids or any(len({v['uuid'] for v in workers if v['pid']==pid})!=1 for pid in expected):continue
   entries={pid:[v for v in apps if v['pid']==pid] for pid in wanted}
   if any(len({v['process_start_ticks'] for v in entries[pid]})!=1 for pid in wanted):continue
   identity={pid:{'start':entries[pid][0]['process_start_ticks'],'uuids':sorted({v['uuid'] for v in entries[pid]})} for pid in wanted}
   fail(identity[service['pid']]['start']==service['start'] and set(identity[service['pid']]['uuids'])==uuids,'Own service start/UUID changed')
   if processes:fail(processes==identity,'Sampled worker PID recycled/remapped')
   processes=identity
   if len(samples)<2:samples.append({'time':row['time'],'identity':identity})
 fail(len(samples)==2,'Missing two actual native-rank8 plus own retriever samples')
 return {'samples':samples,'native_rank_pids':expected,'uuid_inventory':sorted(uuids),'scope':'Actual saved rank PID + sampled start/UUID join; NVML sampling not continuous peak, RSS shared pages/cgroup caveats remain'}


def actual_live_process(pid,manifest,proc_root=Path('/proc')):
 p=proc_root/str(pid);status=(p/'status').read_text();cgroup=(p/'cgroup').read_text();uids=re.search(r'^Uid:\s+(.*)',status,re.M)
 fail(uids is not None and set(map(int,uids.group(1).split()))=={manifest['uid']},'Live process four UIDs differ')
 fail(re.search(r'(?:^|/)job_'+re.escape(manifest['job_id'])+r'(?:/|$)',cgroup,re.M) is not None,'Live process cgroup changed')
 state=re.search(r'^State:\s+(\S+)',status,re.M)
 fail(state is not None and live_state(state.group(1)),'Live process stopped or unknown state')
 return {'pid':pid,'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':cgroup,'uid':manifest['uid']}

def review_record(path,pin,root,manifest,manifest_sha,role,evidence):
 rec={'path':str(path),'sha256':pin};d=read(file(rec,root));fail(d['version']==1 and d['role']==role and d['status']=='PASS_ACTUAL_FIRST5_EVIDENCE' and d['source_commit']==SOURCE and d['job_id']==manifest['job_id'] and d['driver_pid']==manifest['driver_pid'] and d['pause_nonce']==manifest['pause_nonce'] and d['checkpoint_manifest_sha256']==manifest_sha and d['evidence']==evidence,'Actual independent/root review identity/evidence mismatch');return rec


def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--repo',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--evidence-spec',required=True);p.add_argument('--evidence-spec-sha256',required=True);p.add_argument('--root-review',required=True);p.add_argument('--root-review-sha256',required=True);p.add_argument('--independent-review',required=True);p.add_argument('--independent-review-sha256',required=True);p.add_argument('--tools-manifest',required=True);p.add_argument('--tools-manifest-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 fail(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Producer is CPU-only')
 root=Path(a.run_root);fail(root.is_absolute() and root.resolve(strict=True)==root and root.stat().st_uid==os.getuid(),'Wrong physical runroot')
 tm=read(file({'path':a.tools_manifest,'sha256':a.tools_manifest_sha256},Path(a.tools_manifest).parent));fail(tm['source_commit']==SOURCE,'Wrong fixed tools source')
 tools=Path(a.tools_manifest).parent
 fail(Path(__file__).resolve()==tools/'receipt_payload.py','Wrong actual reviewed producer path')
 for name,pin in tm['tools'].items():file({'path':str(tools/name),'sha256':pin},tools)
 repo=Path(a.repo)
 fail(repo.is_absolute() and repo.resolve(strict=True)==repo and repo.name=='repo-'+SOURCE[:12],'Wrong physical fixed repo')
 for name,pin in tm['source_files'].items():file({'path':str(repo/name),'sha256':pin},repo)
 gate=frozen_gate(repo,tm['source_files']['search_r1/baselines/checkpoint_gate.py']);import subprocess
 fail(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Fixed source changed')
 mp=file({'path':a.manifest,'sha256':a.manifest_sha256},root);m=read(mp);identity(m,root)
 for rec in m['files'].values():file(rec,root)
 output=Path(a.output);fail(output.name=='unsigned-step5-payload.json' and output.is_absolute() and output.parent.resolve(strict=True)==output.parent and not output.exists() and not output.is_symlink(),'New unsigned payload output required');output.relative_to(root)
 fail(not (root/'acceptance-5-receipt.json').exists() and not (root/'acceptance-5-receipt.json').is_symlink(),'Signed release already exists')
 spec=read(file({'path':a.evidence_spec,'sha256':a.evidence_spec_sha256},root));fail(set(spec)=={'resources','retrieval','hf_reload','rank_audit','service'},'Evidence roles incomplete')
 for rec in spec.values():file(rec,root)
 fail(Path(spec['resources']['path'])!=root/'resources.ndjson','Use an immutable resource prefix, not growing sampler file')
 native=read(Path(spec['rank_audit']['path']));hf=read(Path(spec['hf_reload']['path']));fail(hf['inspector_source_sha256']==tm['tools']['hf_cpu_reload.py'] and native['inspector_source_sha256']==tm['native_payload_inspector_sha256'],'CPU evidence was not produced by reviewed pinned inspectors');service=read(Path(spec['service']['path']))['owner']
 retrieval=read(Path(spec['retrieval']['path']));fail(retrieval['job_id']==m['job_id'] and same_service_identity(retrieval['service'],service) and len(retrieval['queries'])==24 and len(retrieval['response']['result'])==24,'Retrieval current own service differs')
 for row in retrieval['response']['result']:
  fail(len(row)==3,'Retrieval top3 count differs')
  for item in row:fail(bool(item['document']['contents']) and math.isfinite(float(item['score'])),'Retrieval content/score invalid')
 prefix=log_prefix(root/'train.log',output.parent/'immutable-first5-train-prefix.out',root);rows,state_reports=check_log(Path(prefix['snapshot_path']),m,a.manifest_sha256,gate);reports,driver=check_reports(m,root,state_reports);check_cpu_evidence(native,hf,m,a.manifest_sha256,reports);join=resource_join(Path(spec['resources']['path']),m,reports,service)
 # Reviews are externally provided/pinned AFTER they inspect the actual source,
 # native/HF/raw evidence. This tool never creates their PASS records.
 live_driver=actual_live_process(m['driver_pid'],m);live_workers=[actual_live_process(r['pid'],m) for r in reports];live_service=actual_live_process(service['pid'],m)
 fail(live_service['start']==service['start'],'Own retriever PID recycled')
 for worker in live_workers:
  for sample in join['samples']:fail(sample['identity'][worker['pid']]['start']==worker['start'],'Saved rank PID is a recycled live process')
 review_bindings={k:v for k,v in spec.items()}
 independent=review_record(Path(a.independent_review),a.independent_review_sha256,root,m,a.manifest_sha256,'INDEPENDENT_REVIEWER',review_bindings)
 root_review=review_record(Path(a.root_review),a.root_review_sha256,root,m,a.manifest_sha256,'ROOT_AUDITOR',review_bindings)
 auditpath=output.parent/'receipt-preparation-audit.json';audit={'version':1,'source_commit':SOURCE,'job_id':m['job_id'],'manifest_sha256':a.manifest_sha256,'root_review':root_review,'independent_review':independent,'resource_native_join':join,'actual_live_driver':live_driver,'actual_live_workers':live_workers,'actual_live_service':live_service,'native_console_observation':{'observed_count':len(state_reports),'observed_ranks':[v['rank'] for v in state_reports],'authoritative_saved_report_count':len(reports),'ray_repeat_tags':[int(match.group(1)) for line in Path(prefix['snapshot_path']).read_text().splitlines() if 'FULL_STATE_RANK_JSON=' in line for match in [RANK_DEDUP_SUFFIX.search(line)] if match],'scope':'Actual observed console subset only; repeat count does not infer hidden rank records'},'scope':'CPU unsigned payload preparation after actual external PASS reviews; not signing/release/200/effect proof'}
 with auditpath.open('x') as f:json.dump(audit,f,sort_keys=True,indent=2,allow_nan=False)
 payload={k:m[k] for k in IDENTITY};payload.update(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',checkpoint_manifest_sha256=a.manifest_sha256,checkpoint_files=m['files'],independent_review=independent,evidence={**spec,'root_review':root_review,'preparation_audit':{'path':str(auditpath),'sha256':sha(auditpath)}},train_log_prefix=prefix)
 # Recheck immutable inputs after the full read/join; no stale review/input drift.
 for rec in list(m['files'].values())+list(spec.values())+[independent,root_review]:file(rec,root)
 fail(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Source drifted during preparation')
 for name,pin in tm['source_files'].items():file({'path':str(repo/name),'sha256':pin},repo)
 # Same exact schema consumed by frozen CheckpointGate.verify_receipt. Signature
 # and final acceptance-5-receipt.json must be produced externally afterward.
 with output.open('xb') as f:f.write(canonical(payload));f.flush();os.fsync(f.fileno())
 print(json.dumps({'unsigned_payload':str(output),'sha256':sha(output),'scope':'External signing only; no release performed'},sort_keys=True))
if __name__=='__main__':main()
