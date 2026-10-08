"""One reviewed zero-write diagnostic attempt; preserve unknown outcomes."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from remote import route

WORK = Path(__file__).resolve().parent
FOLDER = WORK/'tree-vllm-window-boundaries3295'
REVIEW = WORK/'tree-vllm-window-final-review/boundaries3295'
ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    build_bytes = (FOLDER/'PACKAGE_BUILD.json').read_bytes()
    build = json.loads(build_bytes)
    source_bytes = (FOLDER/'SOURCE_UPLOAD_REMOTE_RESULT.json').read_bytes()
    source = json.loads(source_bytes)
    cpu_bytes = (REVIEW/'SOURCE_CPU_RECEIPT.json').read_bytes()
    cpu = json.loads(cpu_bytes)
    release_bytes = (REVIEW/'RELEASE.json').read_bytes()
    release = json.loads(release_bytes)
    assert sha(build_bytes) == '0c3015292238bbdb93155aa35831f66b0b5f507147b78e0aef6add9f56cc7870'
    assert sha(source_bytes) == '8073a5ab034e88dc0e4d0dc20a3188af3172f6feecff2a9a9ff199fe29befdd4'
    assert sha(release_bytes) == '0bde62135e38569ea90a84c4c865ffb832f67109dbc5f49986d40b0bba166d9f'
    assert build['runner_commit'] == '0f2aafd1ef38221de0a6bb93d0a78c6b0bfedc84'
    assert source['manifest_sha256'] == build['manifest_sha256'] and source['gpu_submitted'] is False
    assert sha(cpu_bytes) == '1bb602bf9c1f655c053a7933e93da21c5ec93465d9e6fb1dc31a979b5356ca93'
    assert cpu['affected_tests_passed'] == 29
    assert release['runner_commit'] == build['runner_commit']
    assert release['package_manifest_sha256'] == build['manifest_sha256']
    assert release['bundle_payload_sha256'] == build['bundle_payload_sha256']
    assert release['dispatch_intent_path'] == build['fixed_dispatch_intent']
    assert release['status'] == 'ZERO_UPDATE_GPU_PROBE_SOURCE_AND_DISPATCH_INDEPENDENTLY_RELEASED'
    assert release['probability_accepted'] is False and release['training_accepted'] is False
    remote_release = str(ROOT/'instruct300/SINGLE_TP1_WINDOW_BOUNDARIES3295_RELEASE.json')
    packet = dict(release_bytes=base64.b64encode(release_bytes).decode(), release_sha256=sha(release_bytes),
                  release_path=remote_release, package=build['remote_package'],
                  manifest_sha256=build['manifest_sha256'], intent=build['fixed_dispatch_intent'])
    prepared = dict(status='INDEPENDENT_RELEASE_BOUND_SINGLE_ATTEMPT_PREPARED_NO_EXECUTION_ACCEPTANCE',
                    epoch=time.time(), build_sha256=sha(build_bytes), source_readback_sha256=sha(source_bytes),
                    independent_cpu_receipt_sha256=sha(cpu_bytes), release_sha256=sha(release_bytes),
                    package=build['remote_package'], manifest_sha256=build['manifest_sha256'],
                    intent=build['fixed_dispatch_intent'], automatic_retry_allowed=False,
                    tensor_assets_transferred=False, actual_optimizer_writes=None)
    with (FOLDER/'DISPATCH_PREPARED.json').open('x') as stream:
        stream.write(json.dumps(prepared, indent=2)+'\n')
    remote = '''from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,sys
assert socket.gethostname().split('.')[0]=='h100-9'
p=json.loads(base64.b64decode(PACKET));release=Path(p['release_path'])
assert release.parent==Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
assert not Path(p['intent']).exists(), 'Spent/unknown intent: reconcile, never retry'
b=base64.b64decode(p['release_bytes']);assert hashlib.sha256(b).hexdigest()==p['release_sha256']
with release.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(release.read_bytes()).hexdigest()==p['release_sha256']
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
argv=['/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python','-B',str(Path(p['package'])/'dispatch_once.py'),'--package',p['package'],'--manifest-sha256',p['manifest_sha256'],'--release',p['release_path'],'--release-sha256',p['release_sha256'],'--intent',p['intent']]
r=subprocess.run(argv,env=env,capture_output=True);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
'''.replace('PACKET', repr(base64.b64encode(json.dumps(packet).encode()).decode()))
    (FOLDER/'DISPATCH_REMOTE_COMMAND.py').write_text(remote)
    command = 'env PYTHONDONTWRITEBYTECODE=1 /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -B -c '+shlex.quote(remote)
    result = subprocess.run(route(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (FOLDER/'DISPATCH_REMOTE_STDOUT.txt').write_bytes(result.stdout)
    (FOLDER/'DISPATCH_REMOTE_STDERR.txt').write_bytes(result.stderr)
    if result.stdout:
        # Keep exact bytes even when nonzero; never infer execution from submission.
        (FOLDER/'DISPATCH_REMOTE_RESULT.json').write_bytes(result.stdout)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == '__main__': main()
