from pathlib import Path
import hashlib,json,gzip,types,datetime,base64,math
from unittest.mock import patch
H=Path(__file__).resolve().parents[1];D=H/'job3304-first5-resource-phase-originals-v3';OUT=H/'independent-review/actual3304-resource-phase-reviewer-originals';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p,pin):
 raw=p.read_bytes();assert sha(raw)==pin
 m=types.ModuleType(p.stem);m.__file__=str(p);exec(compile(raw,str(p),'exec',dont_inherit=True),m.__dict__);return m
canon=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False)
mraw=(D/'manifest.json').read_bytes();assert sha(mraw)=='b2300073d9920e3d26a02edea8905c9fbfe374334e60c8c6b6a9c8c2a9f9d9ac';m=json.loads(mraw)
assert m['source_commit']=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and m['job_id']=='3304'
pins={};bodies={}
for name,rec in m['files'].items():
 p=D/name;raw=p.read_bytes();assert len(raw)==rec['bytes'] and sha(raw)==rec['sha256'];pins[name]=sha(raw);bodies[name]=raw
for name in ['all-resources.ndjson','training-resources.ndjson']:
 assert gzip.decompress(bodies[name+'.gz'])==bodies[name]
a=bodies['all-resources.ndjson'];t=bodies['training-resources.ndjson'];pl=m['plan'];assert a[pl['slice_start_offset']:pl['slice_end_offset']]==t and pl['slice_end_offset']==len(a)
phase=load(H/'resource-training-phase-candidate/snapshot_training_resources_once.py','ebd52ef36f644ce3d49b4e966f88314c0ddf40dc93e31c1b02308e38e281433e')
with patch.object(phase.os,'getuid',return_value=0):
 newt,newplan=phase.phase_plan(a,pl['boundary_ns'],pl['historical_prefix']['bytes'],pl['historical_prefix']['sha256'])
assert newt==t and canon(newplan)==canon(pl)
receipt=load(H/'first5-1b16c2227caf/receipt_payload.py','7042f5a93158541e9ff0044bf4c21849d8de16eed159c9787fdd7ae8406c564e')
checkpoint=json.loads((H/'job3304-first5-live-binding-original/manifest.json').read_bytes());assert sha((H/'job3304-first5-live-binding-original/manifest.json').read_bytes())==m['manifest_sha256']
# Actual saved rank mapping is ordered by rank in before_context; never sort PIDs.
reports=[{'rank':r,'pid':v['pid']} for r,v in enumerate(m['before_context']['workers'])]
service=json.loads((H/'job3304-complete-probe-originals/evidence/service.json').read_bytes())
actualjoin=receipt.resource_join(D/'training-resources.ndjson',checkpoint,reports,service);assert canon(actualjoin)==canon(m['peer_frozen_resource_join'])
try:receipt.resource_join(D/'all-resources.ndjson',checkpoint,reports,service)
except ValueError as e:assert str(e)=='Foreign resource GPU PID';all_error=str(e)
else:raise AssertionError('Original whole-prefix failure lost')
for key in ['controller','driver','workers']:
 assert m['before_context'][key]==m['after_context'][key]
for key in ['manifest','ready']:
 assert m['before_context'][key]['sha256']==m['after_context'][key]['sha256']
 assert m['before_context'][key]['opened']==m['after_context'][key]['opened']
assert (m['original_sampler']['opened']['device'],m['original_sampler']['opened']['inode'],m['original_sampler']['opened']['uid'])==(64768,138683997,0)
assert m['original_sampler']['unclosed_tail_bytes']==0
assert m['actual_boundary']['environment']['opened']['mtime_ns']==1791287254032340562
assert m['actual_boundary']['environment']['sha256']==phase.ENV_SHA
assert m['actual_boundary']['post_capacity_idle']['sha256']==phase.IDLE_SHA
assert m['actual_boundary']['capacity_exit']['sha256']==phase.EXIT_SHA
assert m['actual_boundary']['capacity_exit']['opened']['mtime_ns']<=m['actual_boundary']['post_capacity_idle']['opened']['mtime_ns']<=pl['boundary_ns']
for k in ['receipt','formal','gate']:
 assert m['dependencies'][k]['before']==m['dependencies'][k]['opened']==m['dependencies'][k]['after']==m['dependencies'][k]['path_after']
 assert m['dependencies'][k]['sha256']==getattr(phase,k.upper()+'_SHA')
assert m['caller_provenance']['runtime']['sha256']=='ebd52ef36f644ce3d49b4e966f88314c0ddf40dc93e31c1b02308e38e281433e'
assert m['caller_provenance']['execution_pid']==2605227 and m['caller_provenance']['execution_uid']==0
assert m['caller_provenance']['actual_environment_job_id'] is None and m['caller_provenance']['actual_environment_step_id'] is None
x=H/'job3304-resource-phase-once-execution-original';xs=(x/'stdout.original').read_bytes();xe=json.loads((x/'exit.original.json').read_bytes());assert xe['returncode']==0 and xe['waited'] is True and xe['stdout_sha256']==sha(xs) and xe['stderr_sha256']==sha((x/'stderr.original').read_bytes())
xrows=[json.loads(v) for v in xs.splitlines()];assert xrows[0]['manifest_sha256']==sha(mraw) and xrows[1]['actual_child_waited'] is True and xrows[1]['returncode']==0 and xrows[1]['argv'][2:]==m['caller_provenance']['argv']
collector=json.loads((D/'collection-exit.original.json').read_bytes());assert collector['returncode']==0 and collector['manifest_sha256']==sha(mraw)
peaks={};apps=0;rows=0
for line in t.splitlines():
 row=json.loads(line);rows+=1;assert row['job_id']=='3304' and 'query_error' not in row
 assert len(row['gpus'])==8 and sorted(v['uuid'] for v in row['gpus'])==actualjoin['uuid_inventory']
 for gpu in row['gpus']:
  assert type(gpu['used_mib']) in (int,float) and math.isfinite(gpu['used_mib']);peaks[gpu['uuid']]=max(peaks.get(gpu['uuid'],0),gpu['used_mib'])
 for app in row['gpu_processes']:
  apps+=1;assert app['in_slurm_job'] is True and not app.get('unavailable') and str(app['uid'])=='0'
assert rows==1901 and pl['all_lines']==1970 and len(pl['all_problem_rows'])==1 and pl['all_problem_rows'][0]['line_number']==66
result={'version':1,'status':'PASS_ACTUAL_FIRST5_TRAINING_RESOURCE_CONTIGUOUS_RANGE_ONLY','reviewer':'/root/author200_review','source_commit':m['source_commit'],'job_id':m['job_id'],'driver_pid':m['driver_pid'],'checkpoint_manifest_sha256':m['manifest_sha256'],'pause_nonce':m['pause_nonce'],'resource_manifest_sha256':sha(mraw),'files':pins,'plan':pl,'strict_join':actualjoin,'actual_all_strict_failure':all_error,'training_app_rows_checked':apps,'sampled_peak_mib':peaks,'caller_provenance':m['caller_provenance'],'local_replay_uid_note':'Only pure phase_plan os.getuid was mocked 0 to replay recorded remote UID0; actual local UID is not claimed remote. No remote reads or writer/main calls occurred.','execution_originals':{n:sha((x/n).read_bytes()) for n in ['caller-source.original.py','stdout.original','stderr.original','exit.original.json']},'limits':['First5 implementation/state/effect acceptance is not granted by this resource-only review.','Whole unfiltered prefix still rejects original line66 cgroupFalse. No historical state/cgroup reconstruction, normalization, whole-formal PASS or terminal resource release is claimed.','NVML samples establish recorded sampled ownership and memory; not continuous peak or future capacity.','Original approved writer recorded live pre/post identities. This local replay does not observe current remote processes.']}
p=H/'ACTUAL3304_FIRST5_RESOURCE_CONTIGUOUS_RANGE_INDEPENDENT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');(OUT/'result.json').write_text(json.dumps({'report':str(p),'sha256':sha(p.read_bytes()),'local_cpu_completed':True},indent=2)+'\n');print(json.dumps({'report':str(p),'sha256':sha(p.read_bytes()),'lines':rows,'all_strict_error':all_error}))
