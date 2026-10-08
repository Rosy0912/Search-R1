"""Collect bounded own-run probe originals; never infer acceptance or terminal state."""
from pathlib import Path
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
from transport import remote

HERE = Path(__file__).resolve().parent
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
BINDING_NAME = 'runtime-tools-bound-fcecbea5d939-v5.json'
BINDING_SHA = '8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
PUBLIC_SHA = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
LIMIT = 20 * 1024 * 1024
TOTAL_LIMIT = 128 * 1024 * 1024
ROLES = ('allocation', 'data_manifest', 'environment', 'import_closure',
         'model_manifest', 'source', 'tool_manifest')
REQUIRED = ('approval-ready.json', 'bindings/manifest.json', 'bindings/public.json') + tuple(
    'bindings/' + name + '.json' for name in ROLES)
OPTIONAL = ('bindings/gpu-init-approval.json', 'formal-wrapper.log', 'retriever.log',
            'evidence/hardware-exit.json', 'evidence/hardware-torchrun.log',
            'evidence/service.json', 'evidence/http-top3-24.json') + tuple(
    'evidence/hardware-rank-' + str(rank) + '.json' for rank in range(8)) + (
    'evidence/capacity-micro4/exit.json', 'evidence/capacity-micro4/torchrun.log',
    'evidence/capacity-micro4/micro-4.json') + tuple(
    'evidence/capacity-micro4/' + prefix + str(rank) + '.json'
    for prefix in ('micro-4-rank-', 'old-logprob-rank-', 'teacher-logits-recompute-rank-')
    for rank in range(8))
ALLOWED = frozenset(REQUIRED + OPTIONAL)
SCOPE = 'BOUNDED_OWN_RUN_PROBE_BYTES_SNAPSHOT_NOT_ACCEPTANCE_OR_TERMINAL_PROOF'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def local_bytes(path):
    need(path.is_absolute() and path.resolve(strict=True) == path,
         'Local input must be physical')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_size < LIMIT,
         'Local input must be an owned regular file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        need((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
             'Local input replaced before open')
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b''.join(chunks)
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_uid, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        need(identity(opened) == identity(os.fstat(fd)) == identity(path.lstat())
             and len(raw) == opened.st_size, 'Local input changed during read')
        return raw
    finally:
        os.close(fd)


def local_request(binding):
    need(binding.name == BINDING_NAME and binding.parent == HERE
         and binding.resolve(strict=True) == binding, 'Current physical V5 binding required')
    raw = local_bytes(binding)
    need(sha(raw) == BINDING_SHA, 'Current V5 binding SHA changed')
    spec = unique(raw)
    need(spec['source_commit'] == SOURCE and spec['remote_root'] == ROOT
         and spec['remote_repo'] == ROOT + '/repo-' + SOURCE[:12], 'Wrong source/run root')
    intent_raw = local_bytes(HERE / 'submit-intent.json')
    submit_raw = local_bytes(HERE / 'submit-original.json')
    job_raw = local_bytes(HERE / 'actual-job-id.txt')
    intent, submitted = unique(intent_raw), unique(submit_raw)
    need(type(submitted.get('returncode')) is int and submitted['returncode'] == 0,
         'Positive original submission required')
    text = submitted.get('stdout')
    need(type(text) is str and re.fullmatch(r'[1-9][0-9]*(?:;[^\n]+)?\n?', text),
         'Actual parsable scheduler submission required')
    job = text.strip().split(';', 1)[0]
    need(job_raw in (job.encode(), (job + '\n').encode()), 'Actual job file differs from submission')
    need(intent['source_commit'] == SOURCE and type(intent.get('actual_outer')) is int
         and intent['actual_outer'] == 0, 'Fresh own-source submission intent required')
    args, env = intent['args'], intent['env']
    need(type(args) is list and args[:2] == ['sbatch', '--parsable']
         and '--nodelist=h100-10' in args and '--gres=gpu:h100:8' in args
         and '--exclusive' in args and '--chdir=' + spec['remote_repo'] in args,
         'Wrong actual submission node/repo/budget')
    need(env['SEARCH_R1_REPO'] == spec['remote_repo']
         and env['E1_FORMAL_TOOLS'] == spec['formal']['remote_directory']
         and env['E1_FORMAL_MANIFEST_SHA256'] == spec['formal']['manifest_sha256'],
         'Submission formal source/tool binding differs')
    return dict(job_id=job, source_commit=SOURCE, run_root=ROOT + '/runs/formal-' + job,
                repo=spec['remote_repo'], formal_sha=spec['formal']['manifest_sha256'],
                source_files=spec['source_files'], runtime_binding_sha256=BINDING_SHA,
                local_submission_pins=dict(intent=sha(intent_raw), original=sha(submit_raw),
                                           actual_job_id=sha(job_raw)))


REMOTE_SOURCE = r'''from pathlib import Path
import os,stat,json,hashlib,base64,datetime,socket
LIMIT=20*1024*1024
TOTAL_LIMIT=128*1024*1024
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(raw):
 def pairs(items):
  out={}
  for k,v in items:
   need(k not in out,'Duplicate JSON key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sig(s):return (s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read_exact(fd,size):
 out=[];left=size
 while left:
  b=os.read(fd,min(left,1024*1024));need(bool(b),'File shortened during read');out.append(b);left-=len(b)
 return b''.join(out)
def member(p,uid=0):
 need(p.is_absolute() and p.resolve(strict=True)==p,'Nonphysical member')
 first=p.lstat();need(stat.S_ISREG(first.st_mode) and first.st_uid==uid and first.st_size<LIMIT,'Wrong file type/UID/size')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);need(sig(opened)==sig(first),'Member replaced before open')
  raw=read_exact(fd,opened.st_size)
  after=os.fstat(fd);last=p.lstat()
  need((opened.st_dev,opened.st_ino,opened.st_uid)==(after.st_dev,after.st_ino,after.st_uid)==(last.st_dev,last.st_ino,last.st_uid),'Member identity changed')
  need(stat.S_ISREG(last.st_mode) and after.st_size<LIMIT and last.st_size<LIMIT,'Member exceeded bound/type')
  if p.suffix=='.log':
   need(after.st_size>=opened.st_size and last.st_size>=opened.st_size,'Log was truncated')
   os.lseek(fd,0,os.SEEK_SET);again=read_exact(fd,opened.st_size);need(again==raw,'Log prefix changed')
   final=os.fstat(fd);path_final=p.lstat()
   need((opened.st_dev,opened.st_ino,opened.st_uid)==(final.st_dev,final.st_ino,final.st_uid)==(path_final.st_dev,path_final.st_ino,path_final.st_uid),'Log identity changed after prefix check')
   need(final.st_size>=opened.st_size and path_final.st_size>=opened.st_size and final.st_size<LIMIT and path_final.st_size<LIMIT,'Log bound/truncation changed')
   last=path_final;scope='VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED'
  else:
   need(sig(opened)==sig(after)==sig(last),'JSON/raw member changed during read');scope='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF'
  return dict(base64=base64.b64encode(raw).decode(),sha256=sha(raw),bytes=len(raw),uid=opened.st_uid,physical=str(p),device=opened.st_dev,inode=opened.st_ino,stat_before=list(sig(opened)),stat_after=list(sig(last)),reading_scope=scope)
 finally:os.close(fd)
def collect(request,required,optional):
 need(os.getuid()==0 and socket.gethostname()=='h100-10','Own native node UID0 reader required')
 root=Path(request['run_root']);need(root.resolve(strict=True)==root and root.stat().st_uid==0,'Own physical run required')
 rows={}
 for name in required:
  rows[name]=member(root/name)
 ready=unique(base64.b64decode(rows['approval-ready.json']['base64']))
 bound=unique(base64.b64decode(rows['bindings/manifest.json']['base64']))
 need(type(ready['version']) is int and ready['version']==1 and type(ready['uid']) is int and ready['uid']==0,'Wrong ready identity types')
 need(ready['job_id']==request['job_id'] and ready['source_commit']==request['source_commit'] and ready['run_root']==str(root),'Wrong actual ready source/job/run')
 need(rows['bindings/manifest.json']['sha256']==ready['binding_sha256'],'Ready binding SHA differs')
 need(bound['job_id']==request['job_id'] if 'job_id' in bound else True,'Foreign binding job')
 need(bound['source_commit']==request['source_commit'] and bound['run_root']==str(root),'Wrong run binding')
 roles={'allocation','data_manifest','environment','import_closure','model_manifest','source','tool_manifest'}
 need(set(bound['files'])==roles,'Wrong binding roles')
 for role,pin in bound['files'].items():
  name='bindings/'+role+'.json';need(pin['path']==str(root/name) and pin['sha256']==rows[name]['sha256'],'Manifest role SHA/path differs')
 need(rows['bindings/public.json']['sha256']==ready['public_key_sha256']==PUBLIC_SHA,'Public key binding differs')
 src=unique(base64.b64decode(rows['bindings/source.json']['base64']))
 alloc=unique(base64.b64decode(rows['bindings/allocation.json']['base64']))
 need(src['clean'] is True and src['source_commit']==request['source_commit'] and src['repo']==request['repo'] and src['source_files']==request['source_files'] and src['formal_manifest_sha256']==request['formal_sha'],'Wrong actual source binding')
 need(type(alloc['uid']) is int and alloc['uid']==0 and alloc['job_id']==request['job_id'] and alloc['hostname']=='h100-10','Wrong actual allocation binding')
 missing=[]
 for name in optional:
  p=root/name
  if os.path.lexists(p):rows[name]=member(p)
  else:missing.append(name)
 need(sum(r['bytes'] for r in rows.values())<TOTAL_LIMIT,'Snapshot exceeds total bound')
 return dict(version=1,kind=SCOPE,scope=SCOPE,job_id=request['job_id'],source_commit=request['source_commit'],run_root=str(root),hostname=socket.gethostname(),uid=os.getuid(),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,missing_optional=missing)
'''


def remote_program(request):
    return REMOTE_SOURCE + '\nPUBLIC_SHA=' + repr(PUBLIC_SHA) + '\nSCOPE=' + repr(SCOPE) + (
        '\nprint(json.dumps(collect(' + repr(request) + ',' + repr(REQUIRED) + ',' + repr(OPTIONAL)
        + '),sort_keys=True))\n')


def decode_snapshot(raw, request):
    record = unique(raw)
    need(record.get('kind') == SCOPE and record.get('scope') == SCOPE,
         'Snapshot byte scope differs')
    for key in ('job_id', 'source_commit', 'run_root'):
        need(record.get(key) == request[key], 'Snapshot actual identity differs: ' + key)
    need(type(record.get('uid')) is int and record['uid'] == 0
         and record.get('hostname') == 'h100-10', 'Snapshot node/UID differs')
    rows = record['files']
    need(type(rows) is dict and set(REQUIRED) <= set(rows) <= ALLOWED, 'Unpermitted/missing member')
    missing = record['missing_optional']
    need(type(missing) is list and len(missing) == len(set(missing))
         and set(missing) == set(OPTIONAL) - set(rows), 'Missing optional list differs')
    decoded = {}
    for name, row in rows.items():
        need(type(name) is str and name in ALLOWED and not Path(name).is_absolute()
             and '..' not in Path(name).parts, 'Unsafe snapshot member name')
        need(type(row['bytes']) is int and 0 <= row['bytes'] < LIMIT
             and type(row['uid']) is int and row['uid'] == 0, 'Member bytes/UID invalid')
        need(row['physical'] == request['run_root'] + '/' + name, 'Member physical source differs')
        need(type(row.get('device')) is int and row['device'] >= 0
             and type(row.get('inode')) is int and row['inode'] > 0, 'Invalid physical member metadata')
        before, after = row.get('stat_before'), row.get('stat_after')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 6
             and all(type(value) is int for value in before + after), 'Invalid stat metadata types')
        need(before[:3] == after[:3] == [row['device'], row['inode'], 0]
             and before[3] == row['bytes'] and row['bytes'] <= after[3] < LIMIT,
             'Physical stat/length identity differs')
        if not name.endswith('.log'):
            need(before == after, 'Immutable JSON stat differs')
        need(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']),
             'Invalid member SHA')
        expected = ('VERIFIED_INITIAL_LENGTH_PREFIX_POSSIBLY_LIVE_NOT_CLOSED' if name.endswith('.log')
                    else 'STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF')
        need(row['reading_scope'] == expected, 'Wrong member reading scope')
        body = base64.b64decode(row['base64'], validate=True)
        need(len(body) == row['bytes'] and sha(body) == row['sha256'], 'Member SHA/length differs')
        decoded[name] = body
    need(sum(map(len, decoded.values())) < TOTAL_LIMIT, 'Decoded total bound exceeded')
    ready = unique(decoded['approval-ready.json'])
    manifest = unique(decoded['bindings/manifest.json'])
    need(type(ready['version']) is int and ready['version'] == 1
         and type(ready['uid']) is int and ready['uid'] == 0, 'Ready identity types differ')
    for key in ('job_id', 'source_commit', 'run_root'):
        need(ready[key] == request[key], 'Ready source/job/run differs')
    need(sha(decoded['bindings/manifest.json']) == ready['binding_sha256'], 'Ready manifest pin differs')
    need(manifest['source_commit'] == request['source_commit']
         and manifest['run_root'] == request['run_root'] and set(manifest['files']) == set(ROLES),
         'Run manifest scope/roles differs')
    for role, pin in manifest['files'].items():
        name = 'bindings/' + role + '.json'
        need(pin['path'] == request['run_root'] + '/' + name
             and pin['sha256'] == sha(decoded[name]), 'Bound role pin differs')
    source = unique(decoded['bindings/source.json'])
    allocation = unique(decoded['bindings/allocation.json'])
    need(source['clean'] is True and source['source_commit'] == request['source_commit']
         and source['repo'] == request['repo'] and source['source_files'] == request['source_files']
         and source['formal_manifest_sha256'] == request['formal_sha'], 'Source original differs')
    need(type(allocation['uid']) is int and allocation['uid'] == 0
         and allocation['job_id'] == request['job_id'] and allocation['hostname'] == 'h100-10',
         'Allocation original differs')
    need(sha(decoded['bindings/public.json']) == ready['public_key_sha256'] == PUBLIC_SHA,
         'Public key original differs')
    return record, decoded


def write_once(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = local_request(args.binding)
    out = args.output
    need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent
         and out.parent.stat().st_uid == os.getuid() and not os.path.lexists(out),
         'Fresh local physical owned output required')
    out.mkdir(mode=0o700)
    program = remote_program(request)
    compile(program, 'owned_read_only_probe_reader', 'exec')
    command = 'python3 -c ' + shlex.quote(program)
    write_once(out / 'reader-source.py', program.encode())
    write_once(out / 'command.original.json', (json.dumps(dict(command=command, request=request,
        reader_source_sha256=sha(program.encode()), scope=SCOPE), indent=2) + '\n').encode())
    try:
        result = remote(command, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        write_once(out / 'transport.stdout.original', stdout)
        write_once(out / 'transport.stderr.original', stderr)
        write_once(out / 'transport-exit.original.json', (json.dumps(dict(
            error_type=type(error).__name__, returncode=getattr(error, 'returncode', None),
            timed_out=isinstance(error, subprocess.TimeoutExpired), scope=SCOPE,
            retry='NOT_AUTOMATIC; original transport outcome retained'), indent=2) + '\n').encode())
        raise
    write_once(out / 'transport.stdout.original', result.stdout)
    write_once(out / 'transport.stderr.original', result.stderr)
    write_once(out / 'transport-exit.original.json', (json.dumps(dict(returncode=result.returncode,
        scope=SCOPE, stdout_sha256=sha(result.stdout), stderr_sha256=sha(result.stderr)), indent=2) + '\n').encode())
    need(type(result.returncode) is int and result.returncode == 0, 'Nonzero actual transport')
    record, bodies = decode_snapshot(result.stdout, request)
    for name, body in bodies.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_once(target, body)
    inventory = {key: value for key, value in record.items() if key != 'files'}
    inventory['files'] = {name: {key: value for key, value in row.items() if key != 'base64'}
                          for name, row in record['files'].items()}
    inventory['local_request'] = request
    inventory['collector_source_sha256'] = sha(Path(__file__).read_bytes())
    write_once(out / 'inventory.json', (json.dumps(inventory, indent=2) + '\n').encode())
    print(json.dumps(dict(job_id=request['job_id'], source_commit=SOURCE, count=len(bodies),
                         missing_optional=len(record['missing_optional']), scope=SCOPE,
                         inventory_sha256=sha((out / 'inventory.json').read_bytes()))))


if __name__ == '__main__':
    main()
