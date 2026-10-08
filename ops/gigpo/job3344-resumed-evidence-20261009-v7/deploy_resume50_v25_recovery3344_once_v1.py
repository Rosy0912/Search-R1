"""One durable submission of resume50 with human-authorized video coexistence."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs/gigpo-e1-instruct200'
RUNTIME = BASE / 'work/gigpo-e1-quality-resume-runtime/safety-resume50-v25-ledgerpid1'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
NAMESPACE = ROOT + '/logs/resume50-v25-recovery3344-once-v1'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'

def receiver(payload):
    import os
    import datetime
    root = Path(ROOT)
    namespace = Path(NAMESPACE)
    prior = root / 'runs/gigpo-e1-fresh200-live-job3344-1791482568900917579/run-result.json'
    terminal = json.loads(prior.read_text())
    if terminal['status'] != 'FAILED' or terminal['cleanup_residual_states'] != {}:
        raise RuntimeError('Prior3344 is not actually terminated and clean')
    queue = subprocess.run(['squeue','--noheader','--states=RUNNING,COMPLETING','--format=%i'],capture_output=True,text=True,check=True)
    live = '\n'.join(line for line in queue.stdout.splitlines() if line.strip() == '3344')
    if live.strip():raise RuntimeError('Prior3344 is still active')
    if terminal.get('error') != 'Owned calibration executor interrupted by signal 15':
        raise RuntimeError('Prior failure differs; targeted diagnosis required')
    active = subprocess.run(['squeue','--noheader','--format=%i|%j|%T'],capture_output=True,text=True,check=True)
    if any('gigpo' in line.lower() for line in active.stdout.splitlines()):
        raise RuntimeError('Existing GiGPO scheduler job; reconcile rather than submit')
    namespace.mkdir(exist_ok=False)
    def write_once(path, data):
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    def record(path):
        raw = path.read_bytes()
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for row in payload['files']:
        raw = base64.b64decode(row['data'])
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise RuntimeError('Payload SHA differs')
    write_once(namespace / 'DEPLOY-INTENT.json', json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        publication=payload['publication'], submission_replay_forbidden=True)).encode())
    helper = root / 'tools/resume50-v25-ledgerpid1'
    if not helper.is_dir():raise RuntimeError('Published runtime missing')
    for row in payload['files']:
        rr=record(helper/row['name'])
        if rr['sha256'] != row['sha256']:
            raise RuntimeError('Frozen runtime differs')
    parent_path = root / 'logs/fresh200-xformers-repair3338-once-v1/BINDING.json'
    parent_raw = parent_path.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != '5f65aa458f3607f546608f38368da48321874f33b526fcc15e3764271d48199c':
        raise RuntimeError('Parent binding changed')
    binding = json.loads(parent_raw)
    parent_helpers = binding['helpers']
    old_guard = {binding['resource_guard']['path'], binding['resource_ownership']['path']}
    files = [row for row in binding['files'] if not row['path'].startswith(parent_helpers + '/') and row['path'] not in old_guard]
    files += [record(helper / row['name']) for row in payload['files']]
    binding.update(helpers=str(helper), files=files,
        resource_guard=record(helper / 'resource_guard_v2.py'),
        resource_ownership=record(helper / 'resource_ownership_v2.py'),
        runtime_variant='RESUME50_COHOST_V25_LEDGER_PID', owned_submit_namespace=NAMESPACE,
        resume50=dict(checkpoint=ROOT+'/runs/gigpo-e1-fresh200-live-job3340-1791441919057393097/checkpoints/actor/global_step_50',
            manifest_sha256='09b682b1661172760ca5d042372d0cca7aadb929e25214efeb2e484948ec8bc0',
            retained_outer=50, retained_adam_per_rank=1000, remaining_counted_outer=150),
        cohosting=dict(authorization='HUMAN_ALLOW_SAME_GPU_WITH_VIDEO_20261009', preserve_foreign=True,
            gpu_memory_utilization=.15, foreign_memory_fraction_limit=.65,
            foreign_sm_limit=None, minimum_total_free_bytes=1073741824),
        actual_runtime_static_review=None, actual_single_submit_review=None,
        publication=payload['publication'])
    binding_path = namespace / 'BINDING.json'
    write_once(binding_path, (json.dumps(binding, sort_keys=True, indent=2)+'\n').encode())
    binding_record = record(binding_path)
    # Exact physical closure, without executing any GPU test or baseline.
    for row in [record(helper / row['name']) for row in payload['files']]:
        if record(Path(row['path'])) != {key: row[key] for key in ('path','bytes','sha256')}:
            raise RuntimeError('Bound physical file differs: '+row['path'])
    checkpoint = Path(binding['resume50']['checkpoint'])
    if record(checkpoint/'training_state/manifest.json')['sha256'] != binding['resume50']['manifest_sha256']:
        raise RuntimeError('Checkpoint manifest changed')
    result = json.loads((checkpoint.parents[2]/'restore50-authorized20261009-verification-v1/RESULT.json').read_text())
    if result['status'] != 'PASS' or not result['all_complete_files_sha_verified']:
        raise RuntimeError('Missing completed full state SHA acceptance')
    command = ['sbatch','--parsable','--job-name=gigpo-resume50-cohost','--time=3-00:00:00',
        str(helper/'calibration_v1.sbatch'),str(binding_path),binding_record['sha256'],binding['repo'],str(helper),binding['environment_root']]
    write_once(namespace/'SUBMIT-INTENT.json',json.dumps(dict(command=command,binding=binding_record,at=datetime.datetime.now(datetime.timezone.utc).isoformat())).encode())
    submitted = subprocess.run(command,capture_output=True,text=True)
    receipt = dict(returncode=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr,binding=binding_record,
        namespace=NAMESPACE,submitted_exactly_once=True)
    write_once(namespace/'SUBMIT-RECEIPT.json',json.dumps(receipt,indent=2).encode())
    print(json.dumps(receipt),flush=True)
    if submitted.returncode:
        raise RuntimeError('Actual submission failed; inspect receipt, never replay')

def main():
    import inspect
    publication = json.loads((OUT/'RESUME50-COHOST-V25-PUBLICATION-v1.json').read_text())
    if publication['status'] != 'PUSHED_AND_INDEPENDENT_GIT_DOWNLOAD_VERIFIED':
        raise RuntimeError('Unpublished runtime')
    files = []
    for p in sorted(RUNTIME.iterdir()):
        if p.is_file() and p.suffix in ('.py','.sbatch'):
            raw=p.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
            if not any(row['original_source_path']==str(p) and row['sha256']==sha and row['downloaded_bytes_match'] for row in publication['manifest']):
                raise RuntimeError('Unpublished/changed file: '+str(p))
            files.append(dict(name=p.name,sha256=sha,data=base64.b64encode(raw).decode()))
    raw = zlib.compress(json.dumps(dict(files=files,publication=publication)).encode())
    receiver_source='import base64,hashlib,json,os,shlex,subprocess,sys,zlib\nfrom pathlib import Path\nROOT='+repr(ROOT)+'\nNAMESPACE='+repr(NAMESPACE)+'\n'+inspect.getsource(receiver)+'\nreceiver(json.loads(zlib.decompress(sys.stdin.buffer.read())))\n'
    worker=shlex.join(['env','-u','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',PYTHON,'-B','-c',receiver_source])
    management='''import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8']
args=prefix+['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',cfg['user']+'@'+cfg['gateway_ip']])]
keys=cfg.get('trusted_target_host_keys',{}).get('h100-8')
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[cfg['user']+'@'+cfg['targets']['h100-8'],WORKER],input=sys.stdin.buffer.read(),capture_output=True,timeout=220)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
'''.replace('WORKER',repr(worker))
    local_intent=OUT/'RESUME50-V25-RECOVERY3344-LOCAL-INTENT-v1.json'
    with local_intent.open('x') as f:json.dump(dict(namespace=NAMESPACE,payload_bytes=len(raw)),f)
    args=['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host','env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(management)]
    r=subprocess.run(args,input=raw,capture_output=True,timeout=240)
    value=dict(returncode=r.returncode,stdout=r.stdout.decode(errors='replace'),stderr=r.stderr.decode(errors='replace'),never_replay=True)
    with (OUT/'RESUME50-V25-RECOVERY3344-TRANSPORT-v1.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value))

if __name__=='__main__':main()
