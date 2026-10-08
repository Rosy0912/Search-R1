"""Replay copied original deployment/launch records; no remote action or CPU tensor load."""
from pathlib import Path
import datetime
import hashlib
import json

BASE = Path(__file__).resolve().parent
DEP = BASE / 'final-acceptance-deployment-8441e10fa456-originals'
START = BASE / 'actual3261-checkpoint25-cpu-start-once-v1'
FROZEN = BASE / 'final-acceptance-8441e10fa456'
MANIFEST = '8441e10fa4565bf53f7baf24a57a60ddbe9844dc979a2c9a0c97250219df7019'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
REMOTE = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006'
RUN = REMOTE + '/runs/formal-3261'
ORIGINALS = {}

def capture(path):
    raw = path.read_bytes()
    ORIGINALS[str(path.relative_to(BASE))] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw

def read(directory, name):
    return json.loads(capture(directory / name))

def slurm(text):
    pairs = dict(item.split('=', 1) for item in text.split() if '=' in item)
    for k, value in {'JobId': '3261', 'JobState': 'RUNNING', 'NodeList': 'h100-10', 'BatchHost': 'h100-10', 'NumNodes': '1', 'NumCPUs': '192', 'Restarts': '0', 'TresPerNode': 'gres:gpu:h100:8'}.items():
        assert pairs[k] == value, (k, pairs[k])
    return pairs

def main():
    raw = capture(FROZEN / 'manifest.json')
    assert hashlib.sha256(raw).hexdigest() == MANIFEST
    manifest = json.loads(raw)
    expected = manifest['tools'] | manifest['review_records'] | {'manifest.json': MANIFEST}
    assert len(expected) == 14
    for name, digest in expected.items():
        assert hashlib.sha256(capture(FROZEN / name)).hexdigest() == digest
    intent = read(DEP, 'intent.json')
    assert intent['manifest_sha256'] == MANIFEST and intent['tool_count'] == 7 and intent['dependency_count'] == 254
    assert intent['review_records'] == manifest['review_records']
    before = read(DEP, 'before-dependency-check-original.json')
    assert before['source_clean'] is True and before['source_commit'] == SOURCE
    assert len(before['dependencies']) == 254
    assert {x['path']: x['sha256'] for x in before['dependencies']} == manifest['dependencies']
    assert len({x['path'] for x in before['dependencies']}) == 254
    for item in before['dependencies']:
        assert type(item['bytes']) is int and item['bytes'] >= 0
    slurm(before['scontrol_original'])
    initial = read(DEP, 'after-unknown-hf-publish-original.json')
    slurm(initial['scontrol_original'])
    assert len(initial['files']) == 4
    for item in initial['files']:
        assert item['physical'] is True and item['regular'] is True
        assert item['sha256'] == expected[item['name']]
        assert item['bytes'] == (FROZEN / item['name']).stat().st_size
    resume_intent = read(DEP, 'resume-after-readonly-verification-intent.json')
    resumed = read(DEP, 'resume-original.stdout.json')
    assert read(DEP, 'resume-exit-original.json')['returncode'] == 0
    assert resume_intent['previous_readonly_original'] == 'after-unknown-hf-publish-original.json'
    assert set(resume_intent['file_names']) == set(expected) and resume_intent['file_names'][-1] == 'manifest.json'
    assert resumed['manifest_published_last'] is True
    assert resumed['scope'] == 'EXTERNAL_TOOL_DEPLOYMENT_ONLY_NOT_CHECKPOINT_PASS'
    assert len(resumed['files']) == 14 and resumed['files'][-1]['name'] == 'manifest.json'
    assert {item['name']: item['sha256'] for item in resumed['files']} == expected
    for index, item in enumerate(resumed['files']):
        assert item['existing_verified'] is True if index < 4 else item['exclusive_published'] is True
    assert before['utc'] < initial['utc'] < resume_intent['utc'] < resumed['utc']
    for path in sorted(DEP.glob('*.stderr')):
        assert capture(path) == b'', path
    post = read(DEP, 'post-deployment-and-prefixes-original.json')
    assert post['tool_manifest_sha256'] == MANIFEST
    slurm(post['live']['scontrol_original'])
    driver = post['live']['driver']
    assert driver['pid'] == 1533312 and driver['present'] is True and driver['start_ticks'] == '1829085316'
    assert driver['uids'] == [0] * 4 and '/job_3261/step_0' in driver['cgroup_original']
    for item in post['observed_logs']:
        pinned = post['log_prefixes'][item['role']]
        assert type(pinned['bytes']) is int and pinned['bytes'] == item['complete_prefix_bytes']
        assert item['bytes_at_open'] == item['complete_prefix_bytes'] + item['partial_tail_bytes_at_open']
        assert item['uid'] == 0 and len(pinned['sha256']) == 64
    start = read(START, 'intent-original.json')
    launch = read(START, 'launcher-original.stdout.json')
    assert read(START, 'launcher-exit-original.json')['returncode'] == 0
    assert capture(START / 'launcher-original.stderr') == b''
    assert start['tools_manifest_sha256'] == MANIFEST
    assert start['software_frozen_review_sha256'] == hashlib.sha256(capture(BASE / 'FROZEN_CHECKPOINT_TOOLS_INDEPENDENT_REVIEW.json')).hexdigest()
    argv = start['argv']
    assert argv[:6] == ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python', REMOTE + '/tools/final-acceptance-8441e10fa456/checkpoint_overlap.py', 'start']
    args = dict(zip(argv[6::2], argv[7::2]))
    assert set(args) == {'--checkpoint-step', '--tools-manifest-sha256', '--output-directory', '--log-prefixes-json'}
    assert args['--checkpoint-step'] == '25' and args['--tools-manifest-sha256'] == MANIFEST
    assert json.loads(args['--log-prefixes-json']) == post['log_prefixes']
    assert args['--output-directory'] == RUN + '/evidence/checkpoint-25-cpu-inspection-once-v1'
    assert launch['started'] is True and type(launch['checkpoint_step']) is int and launch['checkpoint_step'] == 25
    assert launch['supervisor_pid'] == 1631822
    assert launch['request'] == RUN + '/evidence/checkpoint-cpu-25-launch.json'
    assert launch['request_sha256'] == 'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f'
    assert launch['output_directory'] == args['--output-directory']
    result = {
        'version': 1, 'status': 'PASS_COPIED_ACTUAL_DEPLOYMENT_AND_ONCE_LAUNCH_RECORDS_NOT_CHECKPOINT_CONTENT',
        'reviewer': 'author200_review', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit': SOURCE, 'job_id': '3261', 'checkpoint_step': 25,
        'manifest_sha256': MANIFEST, 'actual_remote_dependency_records_verified': 254,
        'actual_remote_tool_and_review_publication_records_verified': 14,
        'manifest_last_record_and_order_verified': True,
        'unknown_publication_recovery': 'Copied read-only original established only4 files present; exact existing4 reverified, missing10 exclusively published, manifest final. No outcome guessed.',
        'driver_pid': 1533312, 'driver_start_ticks': 1829085316,
        'launcher_exit_observed': 0, 'supervisor_pid': 1631822,
        'request_sha256': launch['request_sha256'], 'log_prefixes': post['log_prefixes'],
        'limitations': ['Reviewer performed no remote action and did not reproduce remote file reads', 'Deployment originals record full remote SHA equality; local reviewed bytes independently rehashed', 'Copied launcher exit0 means detached CPU pipeline started, not pipeline or native inspection completed', 'No actual25 tensor content/HF slice/native-exit/srun-exit/controller-exit acceptance yet', 'Exactly one start intent is recorded here; later exclusive marker bytes and controller execution require actual collection'],
        'originals': ORIGINALS, 'blockers': [],
    }
    out = BASE / 'ACTUAL25_DEPLOYMENT_AND_LAUNCH_INDEPENDENT_REVIEW.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'review_path': str(out), 'review_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'status': result['status']}))

if __name__ == '__main__':
    main()
