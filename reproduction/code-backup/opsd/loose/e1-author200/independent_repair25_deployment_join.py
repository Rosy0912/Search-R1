"""Read-only local join of actual remote deployment/start originals."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

B = Path(__file__).resolve().parent
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
MANIFEST = 'f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd'
originals = {}

def read(rel, expected=None):
    raw = (B / rel).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, (rel, digest)
    originals[rel] = {'sha256': digest, 'bytes': len(raw)}
    return json.loads(raw)

manifest = read('final-acceptance-f5c72c942641/manifest.json', MANIFEST)
deployment_dir = 'final-acceptance-deployment-f5c72c942641-originals'
deploy = read(deployment_dir + '/publish-once-original.stdout.json',
    '4df18a7c112e91893e6e5324433b0c771b9128fdc18490569cc465496088bfce')
intent = read(deployment_dir + '/publish-once-intent.json')
deploy_exit = read(deployment_dir + '/publish-once-exit-original.json')
assert type(deploy_exit['returncode']) is int and deploy_exit['returncode'] == 0
assert deploy['manifest_sha256'] == MANIFEST and deploy['manifest_last'] is True
assert deploy['directory'] == ROOT + '/tools/final-acceptance-f5c72c942641'
assert deploy['node'] == 'h100-10' and type(deploy['uid']) is int and deploy['uid'] == 0
assert deploy['new_repair_marker_present'] is False
assert intent['peer_review_sha256'] == '6b8b31a4476521d3c69f6fabf9895baa48626a8298b42be26d8fa4d436b9b52e'
assert intent['files'] == [{k: row[k] for k in ('name', 'sha256')} for row in deploy['published']]
expected = dict(manifest['tools']) | dict(manifest['review_records']) | {'manifest.json': MANIFEST}
rows = deploy['published']
assert len(rows) == len(expected) == 21 and rows[-1]['name'] == 'manifest.json'
assert len({r['name'] for r in rows}) == 21
for row in rows:
    assert row['exclusive_published'] is True and row['sha256'] == expected[row['name']]
    raw = (B / 'final-acceptance-f5c72c942641' / row['name']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
assert deploy['verified_dependencies'] == manifest['dependencies']
assert len(deploy['verified_dependencies']) == 268
allocation = deploy['allocation']
driver = allocation['driver']
assert driver['present'] is True and driver['pid'] == 1533312
assert driver['start_ticks'] == '1829085316' and driver['uids'] == [0] * 4
assert '/job_3261/step_0\n' in driver['cgroup_original']
sc = allocation['scontrol_original']
for field in ('JobId=3261 ', 'JobState=RUNNING ', 'Restarts=0 ',
              'NodeList=h100-10 ', 'NumNodes=1 ', 'TresPerNode=gres:gpu:h100:8 '):
    assert field in sc, field

start_dir = 'actual3261-checkpoint25-cpu-repair-start-once-v2'
start_intent = read(start_dir + '/intent.json')
start = read(start_dir + '/launcher-original.stdout.json',
    'd41af3d84525e3ce291f3cf76ea0e9c932be020c8298726b17b8f4e2e8c7d1cc')
start_exit = read(start_dir + '/launcher-exit-original.json')
assert type(start_exit['returncode']) is int and start_exit['returncode'] == 0
assert start['started'] is True and type(start['checkpoint_step']) is int and start['checkpoint_step'] == 25
assert start['scope'] == 'CPU25_FAILED_CHECKER_REPAIR_STORED_CONTENT_ONLY'
assert start['request'] == ROOT + '/runs/formal-3261/evidence/checkpoint-cpu-25-repair-v2-launch.json'
assert start['output_directory'] == ROOT + '/runs/formal-3261/evidence/checkpoint-25-cpu-inspection-repair-v2'
assert type(start['supervisor_pid']) is int and start['supervisor_pid'] == 1870258
assert start['request_sha256'] == 'ffb7a19aceef85a7ae27756d238dde46f45dc2a13726141a26fae9bd625d94b7'
assert start_intent['previous_failed_request_sha256'] == 'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f'
assert start_intent['deployment_original_sha256'] == originals[deployment_dir + '/publish-once-original.stdout.json']['sha256']
assert start_intent['manifest_sha256'] == MANIFEST
expected_command = ('env PYTHONDONTWRITEBYTECODE=1 /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python -u '
    + ROOT + '/tools/final-acceptance-f5c72c942641/checkpoint_repair_cpu.py start --tools-manifest-sha256 ' + MANIFEST)
assert start_intent['command'] == expected_command
for directory, filename in ((deployment_dir, 'publish-once-original.stderr'), (start_dir, 'launcher-original.stderr')):
    raw = (B / directory / filename).read_bytes()
    assert raw == b''
    originals[directory + '/' + filename] = {'bytes': 0, 'sha256': hashlib.sha256(raw).hexdigest()}

print(json.dumps({
    'version': 1,
    'status': 'PASS_COPIED_ACTUAL_REPAIR25_DEPLOYMENT_AND_ONCE_START_NOT_CONTENT_ACCEPTANCE',
    'reviewer': 'author200_review', 'utc': datetime.now(timezone.utc).isoformat(),
    'source_commit': SOURCE, 'training_job_id': '3261', 'checkpoint_step': 25,
    'manifest_sha256': MANIFEST, 'verified_remote_dependency_records': 268,
    'verified_manifest_last_exclusive_member_records': 21,
    'driver_pid': driver['pid'], 'driver_start_ticks': int(driver['start_ticks']),
    'supervisor_pid': start['supervisor_pid'], 'request_sha256': start['request_sha256'],
    'deployment_exit_observed': 0, 'launcher_exit_observed': 0,
    'originals': originals,
    'limitations': [
        'Reviewer performed read-only local joins of root-copied remote originals; no SSH/GPU/Slurm/signature action.',
        'Remote deployment records, not a second reviewer remote read, establish exact268 live dependency SHA equality.',
        'Launcher exit0 proves detached start only; native/srun/controller completion is still pending.',
        'Request body and actual CPU process identity require subsequent original collection.',
        'Old8441 native/srun/controller FAILED exit1 is preserved; no checkpoint25 content or scientific effect PASS.'
    ], 'whole_checkpoint25_content_pass': False, 'blockers': []
}, indent=2))
