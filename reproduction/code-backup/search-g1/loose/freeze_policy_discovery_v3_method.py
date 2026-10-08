"""Freeze preregistered criteria against the actual new zero-write identities.

Metadata/source/scalar reads only. No GPU kernels, trainer start, empirical
threshold choice, production admission, or alteration of frozen runtime source.
"""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
launch=json.loads((prep/'policy-discovery-v3-native-launch.json').read_text())
prereg_raw=(prep/'policy-calibration-v3-method-preregistration.json').read_bytes()
prereg=json.loads(prereg_raw)
observed_raw=(prep/'policy-discovery-v3-native-observation.json').read_bytes()
observed=json.loads(observed_raw)
assert launch['source_commit']==prereg['source_commit']=='a09a5530fd16fa0c04c0aa157ee56462aab825f4'
assert observed['job_id']==launch['job_id']=='3283'
assert observed['source_hashes_match'] and observed['child_actual_unstarted_verified'] and observed['parent_actual_unstarted_verified']
assert not (prep/'policy-calibration-v3-selection-method.json').exists(), 'Inspect the saved method, never replay freeze'
payload=dict(launch=launch,prereg=prereg,prereg_sha256=hashlib.sha256(prereg_raw).hexdigest(),observation_sha256=hashlib.sha256(observed_raw).hexdigest(),observed=observed)
code=r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);l=p['launch'];b=Path(l['base']);parent=Path(l['parent_base'])
sys.path.insert(0,str(b/'repo'))
from reproduction.common8.freeze_calibration_plan import _unstarted
from reproduction.common8.native_allocation import fields
from reproduction.common8.policy_calibration_plan import _identity,canonical_bytes
job=fields(subprocess.check_output(['scontrol','show','job','-o',l['job_id']],text=True))
assert job['JobId']==l['job_id']=='3283' and job['JobState']=='RUNNING' and job['NodeList']=='h100-8'
assert job['Command']==str(b/'repo/reproduction/common8/launch_calibration_native.sh')
raw=(b/'logs/run-manifest.json').read_bytes();m=json.loads(raw)
parent_raw=(parent/'logs/run-manifest.json').read_bytes();pm=json.loads(parent_raw)
assert m['run_id']==p['observed']['run-manifest.json']['run_id'] and pm['run_id']==p['observed']['parent_run_manifest']['run_id']
assert m['run_id']!=pm['run_id'] and m['parent_production_run_id']==pm['run_id']
assert m['policy_calibration_slurm_job_id']==l['job_id'] and m['policy_calibration_role']=='policy_safety_calibration_discovery'
assert m['source_commit']==pm['source_commit']==l['source_commit']==p['prereg']['source_commit']
assert m['source_sha256']==pm['source_sha256']==l['source_sha256']
assert canonical_bytes(_identity(m))==canonical_bytes(_identity(pm))
for base,run in ((b,m),(parent,pm)):
 assert base.resolve()==base and str(base).startswith('/mnt/data/shaoshitong/')
 _unstarted(base,run)
 for name,sha in l['source_sha256'].items():
  target=base/'repo'/name;assert target.resolve()==target and hashlib.sha256(target.read_bytes()).hexdigest()==sha
status=json.loads((b/'logs/pipeline-status.json').read_text())
assert status['run_id']==m['run_id'] and status['stage'] not in ('BOUNDED_POLICY_CALIBRATION_ONE_OUTER','FAILED','OWNED_CLEANUP_FAILED')
assert not p['prereg']['prospective_identity_binding_complete'] and p['prereg']['actual_discovery_observations_for_this_source']==0
assert p['prereg']['created_unix']<l['submitted_unix']
method=dict(role='root_explicit_policy_calibration_selection_method',source_commit=l['source_commit'],parent_production_run_id=pm['run_id'],discovery_run_id=m['run_id'],frozen_unix=time.time(),before_completed_discovery_update=True,before_any_optimizer_write=True,health_rule_document=p['prereg']['health_rule_document'],negative_tail_selection=p['prereg']['negative_tail_selection'],fixed_candidates=p['prereg']['fixed_candidates'],negative_tail_limit=None,production_admission=False,healthy_checkpoint_accepted=False,sampling_distribution=p['prereg']['sampling_distribution'],prospective_binding=dict(native_job_id=l['job_id'],discovery_base=str(b),parent_base=str(parent),child_manifest_sha256=hashlib.sha256(raw).hexdigest(),parent_manifest_sha256=hashlib.sha256(parent_raw).hexdigest(),zero_update_observation_sha256=p['observation_sha256'],preregistration_sha256=p['prereg_sha256'],preregistered_unix=p['prereg']['created_unix'],parent_actual_unstarted_verified=True,child_actual_unstarted_verified=True,actual_stage=status['stage'],scope='Actual current source/manifests/allocation and strict no trainer claim/log/Adam ledger/update/checkpoint check; no model/GPU/healthy admission'))
content=(json.dumps(method,indent=2,allow_nan=False)+'\n').encode()
paths=[base/'preparation/policy-calibration-v3-selection-method.json' for base in (parent,b)]
assert all(not path.exists() for path in paths)
for path in paths:
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
 fd=os.open(str(path.parent),os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(method_bytes_base64=base64.b64encode(content).decode(),method_sha256=hashlib.sha256(content).hexdigest(),actual_current_zero_write_verified=True,checked_unix=time.time(),role='prospective_method_identity_freeze_not_admission')))
'''
r=subprocess.run(control.route('h100-8',shlex.quote(launch['base']+'/env/bin/python')+' -c '+shlex.quote(code)),input=json.dumps(payload),capture_output=True,text=True,timeout=60)
assert r.returncode==0,'Freeze observation/mutation failed; inspect same target method before any retry: '+r.stderr[-1800:]
receipt=json.loads(r.stdout);content=base64.b64decode(receipt.pop('method_bytes_base64'))
assert hashlib.sha256(content).hexdigest()==receipt['method_sha256']
with (prep/'policy-calibration-v3-selection-method.json').open('xb') as f:f.write(content)
(prep/'policy-calibration-v3-method-freeze-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
