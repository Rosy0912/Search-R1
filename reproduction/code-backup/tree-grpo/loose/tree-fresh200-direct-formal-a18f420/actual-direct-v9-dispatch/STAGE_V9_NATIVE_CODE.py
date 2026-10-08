PINS={'archive_sha256': '72fe50d3445b2d57d7abf8e9d51b2016f26976988c5b5af32980a313902d9c4e', 'archive_bytes': 578484, 'remote_package': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v9-e7236b7', 'remote_extract_destination': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300', 'manifest_sha256': '995819deb2d8ba85f4da92a4ab6ed7abc7eb4887e566cda730297f1e86305760', 'authority_sha256': 'c078fe36fd18cd1ec0b79172ea20629cc857f59c20f916825b66801b10bc21f6', 'release_sha256': '7e57ba49fd2b29daa7aaf3df2426db15938418b230e3964efa0dfb1756e79ee3', 'dispatch_sha256': '489c56b616f8c0fff5ba03f6f69c2aaeb7caf72b2bfc784c26574b895468e13a', 'once_intent': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/SUBMISSION_ONCE-current-fresh200-chain-mask-v9-e7236b7.json', 'code_commit': 'e7236b7d8729482aad864419fe986066643f3be5'}
ROOT_NAMES=['CURRENT_FRESH200_CHAIN_HUMAN_AUTHORITY-V9-e7236b7.json', 'CURRENT_FRESH200_CHAIN_RELEASE-V9-e7236b7.json', 'CURRENT_FRESH200_CHAIN_DISPATCH-V9-e7236b7.sbatch']
import sys,hashlib,io,tarfile,json,socket,os
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
