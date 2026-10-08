"""Known preclaim transport recovery; same single remote claim, no training reset."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:
    raise RuntimeError('Use -E -B -S with active assertions')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

BASE = Path.cwd()
OUT = BASE / 'outputs/gigpo-e1-instruct200'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
STATE = ROOT + '/logs/fresh200-startuprepair3318-once-v1'
ENV = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1')
OPTIONS = ['--parsable', '--partition=gpu', '--nodelist=h100-8', '--nodes=1', '--ntasks=1',
    '--gres=gpu:h100:8', '--cpus-per-task=128', '--mem=512G', '--exclusive', '--no-requeue',
    '--time=1-00:00:00', '--export=ALL']


def checked(path, record):
    raw = path.read_bytes()
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise RuntimeError('Frozen bytes differ: ' + str(path))
    return raw


def main():
    derivation_path = OUT / 'FRESH200-STARTUP3318-REPAIR-DERIVATION-v1.json'
    derivation = json.loads(derivation_path.read_bytes())
    binding = derivation['binding_candidate']
    binding_raw = checked(Path(binding['path']), binding)
    manifest_raw = checked(Path(derivation['runtime_manifest']['path']), derivation['runtime_manifest'])
    manifest = json.loads(manifest_raw)
    reconciliation_path = OUT / 'FRESH200-STARTUP3318-REPAIR-UNCERTAIN-SUBMIT-RECONCILIATION-v1.json'
    reconciliation = json.loads(reconciliation_path.read_bytes())
    diagnostic_path = OUT / 'FRESH200-STARTUP3318-REPAIR-PRECLAIM-INPUT-ERROR-DIAGNOSIS-v2.json'
    diagnostic = json.loads(diagnostic_path.read_bytes())
    if not (reconciliation['hostname'] == diagnostic['hostname'] == 'h100-8'
            and reconciliation['state_exists'] is False and reconciliation['state_entries'] == []
            and reconciliation['queue']['returncode'] == reconciliation['gpu']['returncode'] == 0
            and reconciliation['queue']['stdout'] == reconciliation['gpu']['stdout'] == ''
            and diagnostic['missing'] == diagnostic['different'] == diagnostic['candidate_management_processes'] == []):
        raise RuntimeError('Unknown or claimed prior outcome: NEVER REPLAY')
    review_path = OUT / 'FRESH200-STARTUP3318-REPAIR-SINGLE-SUBMIT-INDEPENDENT-REVIEW-v3-framed.json'
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    if not (review['status'] == 'PASS' and review['independent_review'] is True
            and review['scope'] == 'FRESH200_STARTUP3318_REPAIR_SINGLE_SUBMIT_ONLY'
            and review['blocking_findings'] == [] and review['binding_sha256'] == binding['sha256']
            and review['derivation_sha256'] == hashlib.sha256(derivation_path.read_bytes()).hexdigest()
            and review['submission_adapter_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            and review['parent_job'] == '3318' and review['parent_completed_outer'] == 0
            and review['parent_actual_Adam_writes_per_rank'] == 0
            and review['known_NOCLAIM_transport_recovery'] is True
            and review['reconciliation_sha256'] == hashlib.sha256(reconciliation_path.read_bytes()).hexdigest()
            and review['preclaim_input_diagnostic_sha256'] == hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
            and review['repeat_CPU_suite_allowed'] is False and review['extra_validation_job_allowed'] is False
            and review['first5_count_toward200'] is True and review['same_process_continuation_required'] is True):
        raise RuntimeError('Missing exact genuine startup-repair submission review')
    staging = []
    for row in manifest['files']:
        local = Path(manifest['runtime_root']) / row['relative_path']
        staging.append(dict(path=manifest['canonical_future_runtime_root'] + '/' + row['relative_path'],
            bytes=row['bytes'], sha256=row['sha256'], raw=checked(local, row)))
    for row in derivation['extra_metadata_staging_local_to_remote']:
        staging.append(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'],
            raw=checked(Path(row['local_path']), row)))
    staging.append(dict(path=ROOT + '/logs/' + review_path.name, bytes=len(review_raw),
        sha256=hashlib.sha256(review_raw).hexdigest(), raw=review_raw))
    for row in staging:
        row['b64'] = base64.b64encode(zlib.compress(row.pop('raw'))).decode()
    abort_path = OUT / 'FRESH200-STARTUP3318-OWNED-PRECLAIM-ABORT-RECEIPT-v1.json'
    aborted = json.loads(abort_path.read_bytes())
    if not (aborted['status'] == 'EXACT_OWNED_PRECLAIM_RECEIVER_TERMINATED' and aborted['before']['pid'] == 3883617 and aborted['before']['code_sha256'] == '1f11e5e4fe48af2791c3020a344bad99209fa4493dcf482ee3189c62d0a85252' and aborted['receiver_gone'] is True and aborted['state_exists'] is False and review['owned_preclaim_abort_adapter_sha256'] == hashlib.sha256((BASE / 'work/abort_owned_startup3318_preclaim_receiver_v1.py').read_bytes()).hexdigest()):
        raise RuntimeError('Living or unknown prior receiver: NEVER REPLAY')
    payload = dict(root=ROOT, state=STATE, environment=ENV, options=OPTIONS,
        binding=dict(path=STATE + '/BINDING.json', bytes=len(binding_raw), sha256=binding['sha256']),
        binding_b64=base64.b64encode(zlib.compress(binding_raw)).decode(), staging=staging)
    code = r'''import base64,hashlib,json,os,socket,subprocess,sys,zlib
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Isolated startup required')
length=int(sys.argv[1])
if not 0 < length < 20000000:raise RuntimeError('Invalid exact input length')
raw=sys.stdin.buffer.read(length)
if len(raw)!=length:raise RuntimeError('Incomplete framed input')
p=json.loads(raw);root=Path(p['root']);state=Path(p['state'])
if socket.gethostname().split('.')[0]!='h100-8':raise RuntimeError('Wrong target node')
if root!=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006') or state!=root/'logs/fresh200-startuprepair3318-once-v1':raise RuntimeError('Different repair namespace')
if state.exists() or state.is_symlink():raise RuntimeError('Existing repair intent/namespace: NEVER REPLAY')
error_log=root/'logs/startuprepair3318-transport-v3-error.log'
if error_log.is_symlink():raise RuntimeError('Error-log symlink')
sys.stderr=error_log.open('x',buffering=1)
sys.stderr.write('OWNED_STARTUP_TRANSPORT_V3 pid='+str(os.getpid())+'\n')
old=root/'logs/fresh200-live200-submit-v8'
oldraw=(old/'SUBMIT-RECEIPT.json').read_bytes()
if hashlib.sha256(oldraw).hexdigest()!='0e6b861e547c26dd210af702387cdcac4da838e9309d227f4908d1c26448723b':raise RuntimeError('Original submit receipt changed')
receipt=json.loads(oldraw)
if receipt['job_id']!='3318' or receipt['status']!='SUBMITTED_EXACTLY_ONCE':raise RuntimeError('Wrong original job')
def bytes_checked(row, compressed):
 raw=zlib.decompress(base64.b64decode(compressed))
 if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Payload bytes differ')
 return raw
for row in p['staging']:
 path=Path(row['path']);raw=bytes_checked(row,row['b64'])
 if not path.is_relative_to(root/'tools/fresh200-continuation-v12-slurmcompat1') and path.parent!=root/'logs':raise RuntimeError('Staging escaped repair scope')
 if path.is_symlink():raise RuntimeError('Symlink target')
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.parent.resolve()!=path.parent:raise RuntimeError('Noncanonical staging directory')
 if path.exists():
  if path.read_bytes()!=raw:raise RuntimeError('Existing original differs')
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
bindingraw=bytes_checked(p['binding'],p['binding_b64']);binding=json.loads(bindingraw)
if binding['helpers']!=str(root/'tools/fresh200-continuation-v12-slurmcompat1') or binding['runtime_variant']!='slurmcompat1' or binding['startup_repair_parent_job']!='3318':raise RuntimeError('Wrong repair binding')
fact=json.loads(Path(binding['startup_repair_terminal_factual_record']['path']).read_bytes())
if fact['status']!='STARTUP_FAILED_EXIT1_BEFORE_TRAINING_CONFIRMED' or fact['job_id']!='3318' or fact['exit_code']!=1 or fact['completed_outer']!=0 or fact['actual_Adam_writes_per_rank']!=0:raise RuntimeError('Missing original zero-training terminal facts')
for row in binding['runtime_files']+binding['evidence']:
 raw=Path(row['path']).read_bytes()
 if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Repair input/evidence changed')
observations={}
for name,cmd in [('queue',['squeue','-h','-w','h100-8','-o','%i %j %T']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),('old_cgroup',['find','/sys/fs/cgroup','-maxdepth','6','-iname','*3318*'])]:
 r=subprocess.run(cmd,capture_output=True,text=True,env=p['environment'],timeout=30)
 observations[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode or r.stdout.strip():raise RuntimeError('Resource/old-job check failed; no submission: '+name)
if p['environment']!={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}:raise RuntimeError('Different submission environment')
expected=['--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--gres=gpu:h100:8','--cpus-per-task=128','--mem=512G','--exclusive','--no-requeue','--time=1-00:00:00','--export=ALL']
if p['options']!=expected:raise RuntimeError('Different exclusive submission options')
state.mkdir()
def write_once(path,raw):
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
write_once(state/'BINDING.json',bindingraw)
intent=dict(status='CLAIMED_NO_RESUBMISSION',parent_job='3318',parent_outer=0,parent_Adam_writes=0,binding=p['binding'],resource_observations=observations,first5_count_toward200=True,repeat_CPU_suite=False,extra_validation_job=False)
write_once(state/'SUBMIT-INTENT.json',(json.dumps(intent,sort_keys=True,indent=2)+'\n').encode())
command=['sbatch']+expected+[binding['helpers']+'/calibration_v1.sbatch',str(state/'BINDING.json'),p['binding']['sha256'],binding['repo'],binding['helpers'],binding['environment_root']]
try:
 r=subprocess.run(command,capture_output=True,text=True,env=p['environment'],timeout=60)
 job=r.stdout.strip()
 result=dict(status='SUBMITTED_EXACTLY_ONCE' if r.returncode==0 and job.isdecimal() else 'UNCERTAIN_NO_RESUBMISSION',command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,job_id=job if r.returncode==0 and job.isdecimal() else None,binding=p['binding'],parent_job='3318',parent_completed_outer=0,parent_Adam_writes_per_rank=0,training_updates_not_yet_observed=True)
except BaseException as e:
 result=dict(status='UNCERTAIN_NO_RESUBMISSION',error_type=type(e).__name__,error=str(e),binding=p['binding'],parent_job='3318')
write_once(state/'SUBMIT-RECEIPT.json',(json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps(result));raise SystemExit(0 if result['status']=='SUBMITTED_EXACTLY_ONCE' else 1)
'''
    serialized_payload = json.dumps(payload).encode()
    inner = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c ' + shlex.quote(code) + ' ' + str(len(serialized_payload))
    mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
    management_route = [arg.replace('ServerAliveInterval=15', 'ServerAliveInterval=120')
        .replace('ServerAliveCountMax=2', 'ServerAliveCountMax=6') for arg in control.ssh('mgmt-host')]
    route = management_route + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 '
        '-o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes '
        'h100-16 ' + shlex.quote(mid)]
    claim = OUT / 'FRESH200-STARTUP3318-REPAIR-LOCAL-INVOKE-CLAIM-v3-framed.json'
    with claim.open('x') as f:
        json.dump(dict(status='CLAIMED_LOCAL_ONE_INVOCATION_NEVER_REPLAY', binding_sha256=binding['sha256'],
            review_sha256=hashlib.sha256(review_raw).hexdigest()), f)
    done = subprocess.run(route, input=serialized_payload, capture_output=True, timeout=240)
    output = OUT / 'FRESH200-STARTUP3318-REPAIR-SINGLE-SUBMIT-INVOKE-v3-framed.json'
    with output.open('xb') as f:
        f.write(done.stdout or json.dumps(dict(route_returncode=done.returncode,
            stderr=done.stderr.decode(errors='replace'))).encode())
        f.write(b'\n')
    if done.returncode:
        raise RuntimeError('Inspect durable intent/receipt/queue; NEVER REPLAY: ' + str(output))
    print(done.stdout.decode())


if __name__ == '__main__':
    main()
