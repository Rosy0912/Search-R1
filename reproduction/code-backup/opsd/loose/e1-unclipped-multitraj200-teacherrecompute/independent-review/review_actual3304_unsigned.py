from pathlib import Path
import json,hashlib,types,datetime,re
H=Path(__file__).resolve().parents[1];D=H/'first5-unsigned-preparation-originals-v1';R=H/'job3304-first5-complete-originals-v3/raw';SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):
 def pairs(kv):
  d={}
  for k,v in kv:assert k not in d;d[k]=v
  return d
 return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in()).throw(ValueError(s)))
def pin(p,s):assert sha(p.read_bytes())==s;return read(p)
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
inv=read(D/'inventory.original.json');verified={}
for name,rec in inv['records'].items():
 p=D/name;raw=p.read_bytes();assert len(raw)==rec['bytes']==rec['stat']['st_size'] and sha(raw)==rec['sha256'] and rec['stat']['st_uid']==0 and rec['stat']['st_nlink']==1;verified[name]=rec['sha256']
assert len(verified)==10 and inv['true_transport_exit']==0
m=pin(R/'full_state/global_step_5/manifest.json','30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577')
spec=pin(D/'evidence-spec.json','cf5b09ce192406a6721d92e860d3a6ed66d056f2b414f1b7efa0d5e80f818bf6')
u=pin(D/'unsigned-step5-payload.json','d7a28ee64ff323e27736d04cd50f07d46e63f20b4f502c5bb0ec585cbaf0b7b7');a=pin(D/'receipt-preparation-audit.json','2d56c9710b217ed03be96cbd50d8f3b955b97cc2c08f4a6e59d769e0b00c4c8a')
identity=('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce')
assert all(u[k]==m[k] for k in identity) and u['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW' and u['checkpoint_manifest_sha256']==verified.get('manifest.json',sha((R/'full_state/global_step_5/manifest.json').read_bytes()))
assert u['checkpoint_files']==m['files'] and len(u['checkpoint_files'])==141
assert set(u['evidence'])==set(spec)|{'root_review','preparation_audit'}
assert all(u['evidence'][k]==v for k,v in spec.items())
refs={'INDEPENDENT_REVIEWER':('independent-review.json','ce71235e4f9193ece01e08f76447f1b62c72bfa35535951de7b433e9167a7ca5'),'ROOT_AUDITOR':('root-review.json','dbec240fdc5797a8492b9069356891121a8d16a396c00d3dc0c6bdd293eba989')}
for role,(name,sp) in refs.items():
 v=pin(D/name,sp);assert v['version']==1 and type(v['version']) is int and v['role']==role and v['status']=='PASS_ACTUAL_FIRST5_EVIDENCE' and v['evidence']==spec
 for key in ['source_commit','job_id','driver_pid','pause_nonce']:assert v[key]==m[key]
 assert v['checkpoint_manifest_sha256']==u['checkpoint_manifest_sha256']
 rec=a['independent_review' if role=='INDEPENDENT_REVIEWER' else 'root_review'];assert rec['sha256']==sp and rec['path']==m['run_root']+'/evidence/first5-actual-review-v1/'+name
 assert rec==u['independent_review' if role=='INDEPENDENT_REVIEWER' else 'evidence']['root_review'] if role=='ROOT_AUDITOR' else rec==u['independent_review']
assert u['evidence']['preparation_audit']==dict(path=m['run_root']+'/evidence/first5-actual-review-v1/receipt-preparation-audit.json',sha256=sha((D/'receipt-preparation-audit.json').read_bytes()))
content=read(H/'ACTUAL3304_FIRST5_CONTENT_AND_ARCHIVES_INDEPENDENT_REVIEW.json');assert canon(a['trajectory_archive'])==canon(content['actual_archive'])
workers={r:read(R/f'checkpoints/actor/global_step_5/training_state/rank_{r}.json')['pid'] for r in range(8)}
assert [w['pid'] for w in a['actual_live_workers']]==[workers[r] for r in range(8)]
startmap={2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',2549230:'1834786112',2549233:'1834786115',2548319:'1834784786',2543106:'1834765439'}
for obs in a['actual_live_workers']+[a['actual_live_driver'],a['actual_live_service']]:
 assert type(obs['pid']) is int and obs['uid']==0 and obs['start']==startmap[obs['pid']] and '/job_3304/step_0\n' in obs['cgroup']
assert a['resource_native_join']['native_rank_pids']=={str(p):r for r,p in workers.items()}
for sm in a['resource_native_join']['samples']:
 assert set(sm['identity'])=={str(p) for p in list(workers.values())+[2543106]}
 for k,v in sm['identity'].items():assert v['start']==startmap[int(k)] and len(v['uuids'])==(8 if int(k)==2543106 else 1)
assert a['native_console_observation']['observed_ranks']==[4,7] and a['native_console_observation']['authoritative_saved_report_count']==8
pref=u['train_log_prefix'];raw=(D/'immutable-first5-train-prefix.out').read_bytes();assert len(raw)==pref['bytes'] and sha(raw)==pref['sha256'] and pref['snapshot_path']==m['run_root']+'/evidence/first5-actual-review-v1/immutable-first5-train-prefix.out' and pref['live_path']==m['run_root']+'/train.log'
assert (R/'train.log').read_bytes()[:len(raw)]==raw
# Same exact frozen parser validates original five records and the unique pause.
g=types.ModuleType('frozen_gate');gp=R/'source/search_r1/baselines/checkpoint_gate.py';b=gp.read_bytes();assert sha(b)=='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16';g.__file__=str(gp);exec(compile(b,str(gp),'exec',dont_inherit=True),g.__dict__)
warning=(R/'native-source/fully_sharded_data_parallel.py').read_bytes();outer=[];pause=[];release=[]
for line in raw.decode().splitlines():
 for marker,rows in [('OPSD_E1_OUTER_JSON=',outer),('OPSD_E1_STEP5_PAUSED_JSON=',pause),('OPSD_E1_STEP5_RELEASED_JSON=',release)]:
  if marker in line:rows.append(g.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id='3304',worker_pids=tuple(workers.values()),checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'),fsdp_warning_source_bytes=warning))
assert canon(outer)==canon(content['actual_outer_records']) and len(pause)==1 and not release
assert pause[0]['manifest_sha256']==u['checkpoint_manifest_sha256'] and pause[0]['pause_nonce']==m['pause_nonce']
for v in outer:g.validate_outer(v['metrics'],world=8,profile='multitraj200')
pe=read(D/'producer-exit.original.json');ps=read(D/'producer-start.original.json');assert type(pe['returncode']) is int and pe['returncode']==0 and pe['waited'] is True and pe['child_pid']==ps['child_pid']==2635540 and pe['waited_by_pid']==ps['bootstrap_pid']==2635539 and pe==inv['native_producer_true_wait']
assert ps['CPU_only'] is True and 'CUDA_VISIBLE_DEVICES=' in ps['command'] and '-B' in ps['command'] and not any('sign'==c or 'release'==c for c in ps['command'])
stdout=read(D/'producer.stdout.original');assert stdout['sha256']==sha((D/'unsigned-step5-payload.json').read_bytes()) and stdout['unsigned_payload']==m['run_root']+'/evidence/first5-actual-review-v1/unsigned-step5-payload.json'
assert a['source_commit']==SOURCE and a['job_id']=='3304' and a['manifest_sha256']==u['checkpoint_manifest_sha256']
result=dict(version=1,status='PASS_ACTUAL_UNSIGNED_FIRST5_GRAPH_ONLY_NOT_SIGNATURE_RELEASE_OR_OUTER6',reviewer='/root/author200_review',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE,job_id='3304',driver_pid=m['driver_pid'],pause_nonce=m['pause_nonce'],checkpoint_manifest_sha256=u['checkpoint_manifest_sha256'],unsigned_payload_sha256=sha((D/'unsigned-step5-payload.json').read_bytes()),preparation_audit_sha256=sha((D/'receipt-preparation-audit.json').read_bytes()),inventory_sha256=sha((D/'inventory.original.json').read_bytes()),verified_original_files=verified,checkpoint_role_count=141,evidence_roles=list(spec),actual_observed_console_ranks=[4,7],actual_workers=a['actual_live_workers'],actual_driver=a['actual_live_driver'],actual_service=a['actual_live_service'],producer_true_wait=pe,unique_paused=True,observed_release_count=0,completed_outer_count=5,trajectory_graph_equal_prior_independent_content=True,scope='Bounded exact unsigned graph/provenance/log prefix audit. Existing full raw/state/resource independent reviews reused; no repeated native/HF/archive/model tests.',limitations=['Recorded producer live observations are historical; root immutable publisher must fresh-recheck live ownership and all pins before external signing.','No signature, release, sixth data fetch/update or continuation is inferred.','Original ALL preflight resource bad row66 remains; only the independently reviewed contiguous training range satisfies first5 resources; whole formal/terminal qualification remains separate.'])
p=H/'ACTUAL3304_UNSIGNED_FIRST5_INDEPENDENT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
md=H/'ACTUAL3304_UNSIGNED_FIRST5_INDEPENDENT_REVIEW.md';assert not md.exists();md.write_text('实际未签名首5验收图通过独立复核。141个 checkpoint 角色、5项证据、根审与独审原件 SHA、原审归档图、8个实际 worker 和本机检索服务的 PID/start 观测全部一致。固定训练前缀只含真实1–5轮、唯一 PAUSED，未出现 RELEASED。生产者2635540由父2635539真实 wait 返回0。\n\n这只确认现存 unsigned payload。签名、活体再次校验、第6轮原位继续尚须后续实际原件；全资源 bad66 原件仍保留。\n')
print(json.dumps({'report':str(p),'sha256':sha(p.read_bytes()),'md_sha256':sha(md.read_bytes())}))
