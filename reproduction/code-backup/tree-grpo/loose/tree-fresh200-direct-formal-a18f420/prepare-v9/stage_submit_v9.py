"""One authorized h1009 code-only transfer, followed by the exact durable once helper.

No models/PT/data/index are read or transported. Never reuses a local operation
intent or issues a second submission after timeout/unknown.
"""
import datetime,hashlib,json,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent;OUT=HERE/'actual-direct-v9-dispatch'
sys.path.insert(0,str(HERE.parent));import remote
receipt=json.loads((HERE/'CURRENT_CHAIN_PACKET_BOUND_ARGV_V9.json').read_text())
packet=Path(receipt['archive']).read_bytes();assert hashlib.sha256(packet).hexdigest()==receipt['archive_sha256']
assert receipt['code_commit']=='e7236b7d8729482aad864419fe986066643f3be5' and receipt['packet_revision']==9
pins={key:receipt[key] for key in ('archive_sha256','archive_bytes','remote_package','remote_extract_destination','manifest_sha256','authority_sha256','release_sha256','dispatch_sha256','once_intent','code_commit')}
root_names=[Path(receipt[key]).name for key in ('authority_path','release_path','dispatch_path')]
code='PINS='+repr(pins)+'\nROOT_NAMES='+repr(root_names)+'\n'+'''import sys,hashlib,io,tarfile,json,socket,os
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
assert not any(key.startswith('SLURM_') for key in os.environ)
parent=Path(PINS['remote_extract_destination']);candidate=Path(PINS['remote_package'])
assert candidate.parent==parent/'candidates' and not candidate.exists() and not Path(PINS['once_intent']).exists()
assert all(not (parent/name).exists() for name in ROOT_NAMES)
raw=sys.stdin.buffer.read();assert len(raw)==PINS['archive_bytes'] and hashlib.sha256(raw).hexdigest()==PINS['archive_sha256']
archive_path=parent/'staged-code'/('current-fresh200-chain-mask-v9-e7236b7-'+PINS['archive_sha256']+'.tar.gz');archive_path.parent.mkdir(exist_ok=True)
with archive_path.open('xb') as stream:stream.write(raw)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 members=archive.getmembers();assert all(member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts for member in members)
 prefix='candidates/'+candidate.name+'/'
 allowed_prefix=lambda name:name.startswith(prefix) or name in ROOT_NAMES
 assert all(allowed_prefix(member.name) for member in members)
 candidate.mkdir()
 for member in members:
  target=parent/member.name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(archive.extractfile(member).read())
raw=(candidate/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==PINS['manifest_sha256'];manifest=json.loads(raw)
assert manifest['code_commit']==PINS['code_commit'] and manifest['packet_revision']==9 and manifest['scope']=='CURRENT_FRESH200_CHAIN_CONTINUATION'
assert manifest['current_chain_resume']['parent_completed_outer']==4 and manifest['current_chain_resume']['next_outer_update']==5
assert manifest['configured_prospective_thresholds']==dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=.0)
for relative,pin in manifest['files'].items():assert hashlib.sha256((candidate/relative).read_bytes()).hexdigest()==pin
for name,key in zip(ROOT_NAMES,('authority_sha256','release_sha256','dispatch_sha256')):assert hashlib.sha256((parent/name).read_bytes()).hexdigest()==PINS[key]
print(json.dumps(dict(status='CODE_ONLY_PACKET_STAGED_NOT_SUBMITTED',code_commit=manifest['code_commit'],manifest_sha256=PINS['manifest_sha256'],remote_package=str(candidate),archive_sha256=PINS['archive_sha256'],archive_bytes=len(raw),staged_regular_files=len(members),contains_model_data_index_checkpoint_assets=False)))
'''
(OUT/'STAGE_V9_NATIVE_CODE.py').write_text(code)
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONOPTIMIZE=0','/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python','-B','-c',code])
with (OUT/'STAGE_V9_INTENT.json').open('x') as f:json.dump(dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='h100-9',pins=pins),f,indent=2)
r=subprocess.run(remote.route(command,node='h100-9'),input=packet,capture_output=True,timeout=120)
with (OUT/'STAGE_V9.stdout').open('xb') as f:f.write(r.stdout)
with (OUT/'STAGE_V9.stderr').open('xb') as f:f.write(r.stderr)
with (OUT/'STAGE_V9_TRANSPORT_RECEIPT.json').open('x') as f:json.dump(dict(returncode=r.returncode,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest(),stderr_bytes=len(r.stderr)),f,indent=2)
assert r.returncode==0,r.stderr.decode(errors='replace');json.loads(r.stdout)
print('V9_CODE_ONLY_STAGE_COMPLETE',flush=True)
with (OUT/'SUBMISSION_V9_LOCAL_INTENT.json').open('x') as f:json.dump(dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='h100-9',status='EXACT_ONCE_HELPER_DISPATCH_STARTED',argv=receipt['once_submission_argv'],once_intent=receipt['once_intent']),f,indent=2)
result=subprocess.run(remote.route(receipt['once_submission_shell_command'],node='h100-9'),capture_output=True,timeout=120)
with (OUT/'SUBMISSION_V9.stdout').open('xb') as f:f.write(result.stdout)
with (OUT/'SUBMISSION_V9.stderr').open('xb') as f:f.write(result.stderr)
summary=dict(returncode=result.returncode,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),stderr_bytes=len(result.stderr),stdout_bytes=len(result.stdout),once_intent=receipt['once_intent'])
if result.returncode==0:
 value=json.loads(result.stdout);assert value['status']=='SUBMITTED';summary.update(status=value['status'],job_id=value['job_id'],run_id=value['run_id'],manifest_sha256=value['manifest_sha256'],cluster_name=value['cluster_name'])
with (OUT/'SUBMISSION_V9_TRANSPORT_RECEIPT.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
assert result.returncode==0,result.stderr.decode(errors='replace')
