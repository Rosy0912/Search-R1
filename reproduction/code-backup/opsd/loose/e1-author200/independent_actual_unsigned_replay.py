"""Replay actual once-only unsigned preparation originals locally; never sign."""
import gzip,hashlib,importlib.util,json,math
from pathlib import Path
W=Path(__file__).resolve().parent
COL=W/'actual3261-unsigned-collected-20261005T205139510221Z'
SNAP=W/'actual-snapshots/3261-20261005T203236002279Z'
F=SNAP/'original-files'
RUN=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
SOURCE='188f4e5e7b63cf479290d0888a4a075499886750'
SCOPE='UNSIGNED_FIRST5_PAYLOAD_PREPARATION_ONLY_NO_REVIEW_SIGNATURE_RELEASE_OR_GPU'
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
manifest=read(F/'full_state/global_step_5/manifest.json');ms=sha(F/'full_state/global_step_5/manifest.json')
peer=read(W/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json');spec=peer['evidence']
maprec=read(COL/'remote-original-map.json')
need(len(maprec)==25 and len({r['path'] for r in maprec})==25,'Actual original inventory')
for r in maprec:
 p=COL/Path(r['path']).name
 need(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'Original copied bytes '+p.name)
payloadpath=COL/'unsigned-step5-payload.json';payload=read(payloadpath)
need(sha(payloadpath)=='30439f4a4a352ce2299e369b8c03b3719c01f4219c627e144673075aa01aa928','Actual unsigned immutable hash')
need(set(payload)==set(('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce'))|{'status','checkpoint_manifest_sha256','checkpoint_files','independent_review','evidence','train_log_prefix'},'Exact unsigned payload schema/no signature')
for k in ('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce'):
 need(payload[k]==manifest[k],'Actual unsigned identity '+k)
need(payload['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW' and payload['checkpoint_manifest_sha256']==ms and payload['checkpoint_files']==manifest['files'],'Exact first5 checkpoint inventory')
need(payload['independent_review']['sha256']==sha(W/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json') and {k:payload['evidence'][k] for k in spec}==spec,'Actual independent role/five evidence unchanged')
root=read(W/'ACTUAL3261_FIRST5_ROOT_REVIEW_RECORD.json')
need(payload['evidence']['root_review']['sha256']==sha(W/'ACTUAL3261_FIRST5_ROOT_REVIEW_RECORD.json') and root['role']=='ROOT_AUDITOR' and root['status']=='PASS_ACTUAL_FIRST5_EVIDENCE' and root['evidence']==spec,'Actual root audit schema/evidence')
auditpath=COL/'receipt-preparation-audit.json';audit=read(auditpath)
need(payload['evidence']['preparation_audit']['sha256']==sha(auditpath)=='d80b2a171baa158554791c24510694e77c27fbfd2a3d4bb2cec740ccbc17c7bb','Actual preparation audit hash')
need(audit['independent_review']==payload['independent_review'] and audit['root_review']==payload['evidence']['root_review'] and audit['source_commit']==SOURCE and audit['job_id']=='3261' and audit['manifest_sha256']==ms,'Actual preparation audit binds actual reviewers')
prefix=payload['train_log_prefix'];lp=COL/'immutable-first5-train-prefix.out'
need(sha(lp)==prefix['sha256'] and lp.stat().st_size==prefix['bytes']==405957 and prefix['live_path']==str(RUN/'train.log'),'Actual exact immutable train prefix')
need((F/'train.log').read_bytes()[:prefix['bytes']]==lp.read_bytes(),'Actual immutable prefix equals previously reviewed original log prefix')
gp=W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
need(sha(gp)=='c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb','Frozen gate bytes')
sp=importlib.util.spec_from_file_location('unsigned_independent_frozen_gate',gp);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
workers=tuple(r['pid'] for r in peer['actual_rank_results']);events=[];pauses=[];evals=[]
for line in lp.read_text().splitlines():
 for marker,dest in [('OPSD_E1_OUTER_JSON=',events),('OPSD_E1_STEP5_PAUSED_JSON=',pauses),('BASELINE_EVALUATION_METRICS_JSON=',evals)]:
  if marker in line:dest.append(g.parse_log_marker(line,marker,driver_pid=1533312,job_id='3261',worker_pids=workers))
 need('OPSD_E1_STEP5_RELEASED_JSON=' not in line,'Premature release existed in unsigned proof')
need([r['actual_outer_updates'] for r in events]==[1,2,3,4,5] and len(pauses)==1 and pauses[0]['manifest_sha256']==ms and pauses[0]['pause_nonce']==manifest['pause_nonce'],'Actual unsigned strict first5/pause')
for row in events:
 need(row['source_commit']==SOURCE and row['config_sha256']==manifest['config_sha256'] and row['metrics']['actor/lr']==5e-6,'Actual unsigned source/LR')
 g.validate_outer(row['metrics'],profile='author200')
need(len(evals)==2,'Actual original initial and stage evaluation count')
for e,n in zip(evals,(512,64)):g.validate_evaluation(e,n)
need(all(events[-1]['metrics'][k]==v for k,v in manifest['outer_metrics'].items()),'Actual fifth metrics vs manifest')
expectedworkers={r['pid']:r for r in peer['actual_rank_results']}
for proc in audit['actual_live_workers']:
 r=expectedworkers[proc['pid']]
 need(proc['uid']==0 and '/job_3261/step_0' in proc['cgroup'] and proc['start']==r['start_ticks'],'Actual unchanged own worker start')
need(len(audit['actual_live_workers'])==8 and {p['pid'] for p in audit['actual_live_workers']}==set(expectedworkers),'Eight actual live native workers')
need(audit['actual_live_driver']['pid']==1533312 and audit['actual_live_driver']['start']=='1829085316' and audit['actual_live_service']['pid']==1531763 and audit['actual_live_service']['start']=='1829074494','Actual driver/service continuity')
raw=b''.join(gzip.decompress((SNAP/'resources-prefix-original.ndjson.gz').read_bytes()).splitlines(keepends=True)[:631]);need(hashlib.sha256(raw).hexdigest()==spec['resources']['sha256'],'Actual resource prefix pin')
resources={r['time']:r for r in (json.loads(line) for line in raw.splitlines())}
join=audit['resource_native_join'];need(join['native_rank_pids']=={str(r['pid']):r['rank'] for r in peer['actual_rank_results']} and len(join['samples'])==2,'Actual saved native rank sampled join')
for sample in join['samples']:
 row=resources[sample['time']];identity={}
 for pid in set(expectedworkers)|{1531763}:
  apps=[a for a in row['gpu_processes'] if a['pid']==pid]
  need(apps and all(a['in_slurm_job'] is True and a['uid']=='0' for a in apps),'Actual own application sample')
  starts={a['process_start_ticks'] for a in apps};need(len(starts)==1,'Actual sampled process recycling')
  identity[str(pid)]={'start':next(iter(starts)),'uuids':sorted({a['uuid'] for a in apps})}
 need(identity==sample['identity'],'Actual resource original identities differ')
need(audit['native_console_observation']['observed_count']==2 and audit['native_console_observation']['observed_ranks']==[6,0] and audit['native_console_observation']['authoritative_saved_report_count']==8,'Console subset never expanded')
launch=read(COL/'first5-unsigned-receipt-overlap-launch.json')
need(launch['runner_manifest_sha256']=='317dec4d4dd1ce2b2a535d37b0cf39543422fbfff93e7d322e7aabf5cdb29319' and launch['driver_start_ticks']=='1829085316' and launch['checkpoint_manifest_sha256']==ms and launch['pause_nonce']==manifest['pause_nonce'] and launch['source_commit']==SOURCE and launch['job_id']=='3261','Actual reviewed durable runner identity')
need(launch['independent_review']==payload['independent_review'] and launch['root_review']==payload['evidence']['root_review'],'Actual launcher review inputs')
cpu=read(COL/'cpu-step-original.json');need(cpu['cuda_is_initialized'] is False and cpu['cuda_visible_devices']=='' and cpu['omp_num_threads']=='1' and cpu['torch_version']=='2.4.0+cu121' and cpu['process']['uid']==0 and '/job_3261/step_2' in cpu['process']['cgroup_original'],'Actual no-GPU CPU preparation step')
exits={k:read(COL/(k+'-exit-original.json')) for k in ('producer','srun','controller')}
need(all(type(e['returncode']) is int and e['returncode']==0 and e['scope']==SCOPE for e in exits.values()),'Actual waited preparation success')
need(exits['producer']['waited_by_cpu_step_pid']==cpu['process']['pid']==1571248 and exits['producer']['producer_pid']==1571260 and exits['srun']['waited_by_controller_pid']==exits['controller']['controller_pid']==1571230 and exits['controller']['waited_by_supervisor_pid']==1571225,'Actual full waited subprocess chain')
cmd=read(COL/'producer-command-original.json');need(cmd['command']==launch['producer_command'] and cmd['scope']==SCOPE,'Actual producer argv differs')
srunargv=read(COL/'controller-start-original.json')['command']
need('--jobid=3261' in srunargv and '--nodelist=h100-10' in srunargv and '--gres=none' in srunargv and 'CUDA_VISIBLE_DEVICES=' in srunargv and all(k+'=1' in srunargv for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Same-job same-node CPU-only actual command')
completed=read(COL/'unsigned-preparation-completed-review-required.json');need(completed['cuda_is_initialized'] is False and completed['scope']==SCOPE and completed['payload']['sha256']==sha(payloadpath) and completed['after']['manifest_sha256']==ms,'Actual unsigned completed original')
out={'status':'PASS_ACTUAL_UNSIGNED_FIRST5_PREPARATION_ONLY','scope':'UNSIGNED_ACTUAL_IMPLEMENTATION_AND_STATE_EVIDENCE_NOT_SIGNATURE_RELEASE_EFFECT_OR_200','source_commit':SOURCE,'job_id':'3261','driver_pid':1533312,'driver_start_ticks':'1829085316','checkpoint_manifest_sha256':ms,'pause_nonce':manifest['pause_nonce'],'actual_original_directory':str(COL),'actual_original_map_sha256':sha(COL/'remote-original-map.json'),'actual_original_count':25,'unsigned_payload_sha256':sha(payloadpath),'preparation_audit_sha256':sha(auditpath),'root_review_sha256':payload['evidence']['root_review']['sha256'],'independent_review_sha256':payload['independent_review']['sha256'],'actual_waited_exits':{k:{'record_sha256':sha(COL/(k+'-exit-original.json')),**v} for k,v in exits.items()},'actual_cpu_step_pid':cpu['process']['pid'],'actual_evidence':spec,'actual_train_prefix':prefix,'actual_completed_sha256':sha(COL/'unsigned-preparation-completed-review-required.json'),'negative_effects_preserved':peer['negative_effect_observations'],'released':False,'effect_improvement_proven':False,'final200_complete':False,'reviewer_performed_signature_or_GPU':False}
p=W/'ACTUAL3261_UNSIGNED_PREPARATION_INDEPENDENT_REVIEW.json'
with p.open('x') as h:json.dump(out,h,indent=2,sort_keys=True);h.write('\n')
print(json.dumps({'status':out['status'],'sha256':sha(p),'payload_sha256':sha(payloadpath),'audited_original_count':25},sort_keys=True))
