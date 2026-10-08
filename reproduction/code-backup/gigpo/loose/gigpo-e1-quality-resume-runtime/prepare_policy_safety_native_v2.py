"""Build immutable native-v2 inputs; never authorize or submit a job."""
import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent.parent
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
HELPERS = ROOT + '/tools/policy-safety-native-52befe9-v2'
EVIDENCE_NAMES = [
    'GUARD-OWNERSHIP-CANDIDATE-V2-RECEIPT-v2.json',
    'GUARD-V2-INDEPENDENT-REVIEW-v2.json',
    'GUARD-OWNERSHIP-LINUX-CPU-PROBE-RAW-v1.json',
    'GUARD-OWNERSHIP-CONTROLLED-RACE-RAW-v1.json',
    'GUARD-OWNERSHIP-CANDIDATE-V2-LINUX-RAW-v2.json',
    'GUARD-OWNERSHIP-LIVE-SCHEDULING-LINUX-RAW-v2.json',
    'GUARD-OWNERSHIP-V2-CPU-26-v2.log',
    'GUARD-OWNERSHIP-V2-TARGET-CPU-TRANSPORT-v2.log',
    'GUARD-OWNERSHIP-3259-RAW-OBSERVATIONS-v1.json',
    'GUARD-OWNERSHIP-REVIEW-v2.md',
    'NATIVE3259-FAILURE-INDEPENDENT-REVIEW-v1.json',
    'NATIVE3259-TINY-COMPONENT-INDEPENDENT-REVIEW-v1.json',
    'GUARD-OWNERSHIP-SAFETY-NATIVE-V2-INTEGRATION-MANIFEST-v1.json',
    'GUARD-OWNERSHIP-SAFETY-NATIVE-V2-DIFF-v1.patch',
    'GUARD-OWNERSHIP-SAFETY-NATIVE-V2-INTEGRATION-v1.md',
]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old_path = OUT / 'NATIVE-CANDIDATE-52BEFE9-v1.json'
    assert digest(old_path) == '3276b298ee324e0d87719d0cf835af2ae2a93b22e4818dc65229decb3053c0d5'
    candidate = json.loads(old_path.read_text())
    old_helpers = candidate['helpers']
    candidate['helpers'] = HELPERS
    candidate['preparation_scope'] = 'New guard-v2 immutable integration; physical and independent deployment review pending'
    candidate['files'] = [x for x in candidate['files'] if not x['path'].startswith(old_helpers + '/')]
    directory = HERE / 'safety-native-v2'
    for p in sorted(directory.iterdir()):
        if p.is_file():
            candidate['files'].append(dict(path=HELPERS + '/' + p.name, sha256=digest(p)))
    assert len(candidate['files']) == len({x['path'] for x in candidate['files']})
    guard = {x['path'].split('/')[-1]: x for x in candidate['files'] if x['path'].startswith(HELPERS + '/')}
    assert guard['resource_guard_v2.py']['sha256'] == '9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'
    assert guard['resource_ownership_v2.py']['sha256'] == '7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'
    candidate['resource_guard'] = guard['resource_guard_v2.py']
    candidate['resource_ownership'] = guard['resource_ownership_v2.py']
    candidate['superseded_failed_native_job'] = '3259'
    candidate['inherited_candidate_sha256'] = digest(old_path)
    candidate['inherited_physical'] = dict(path=ROOT + '/logs/policy-safety-native-52befe9-physical-v1.json', sha256='c65db5f2df3be25b62d17945efe2a6091410c3bbcb935e036e86bbe64fd88099')
    for name in EVIDENCE_NAMES:
        candidate['evidence'].append(dict(path=ROOT + '/logs/' + name, sha256=digest(OUT / name)))
    assert len(candidate['evidence']) == len({x['path'] for x in candidate['evidence']})
    target = OUT / 'NATIVE-CANDIDATE-52BEFE9-v2.json'
    with target.open('x') as f:
        f.write(json.dumps(candidate, indent=2) + '\n')
    archive = HERE / 'policy-safety-native-52befe9-v2.tar.gz'
    with tarfile.open(archive, 'x:gz') as f:
        for p in sorted(directory.iterdir()):
            if p.is_file():
                f.add(p, arcname=p.name)
    print(json.dumps(dict(candidate=str(target), candidate_sha256=digest(target), files=len(candidate['files']), evidence=len(candidate['evidence']), archive=str(archive), archive_sha256=digest(archive), training_submitted=False)))


if __name__ == '__main__':
    main()
