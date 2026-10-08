"""Persist one fresh A800 deployment intent, execute it once, then observe it.

Create: python -m reproduction.a800_common.deploy --source-commit FULL_SHA
Launch: python -m reproduction.a800_common.deploy --handle INTENT --execute
Read:   python -m reproduction.a800_common.deploy --handle INTENT --observe

Only the existing controller's baai-a800 SSH route is used. Credentials and SSH
configuration are never copied or changed. An uncertain launch is observed, not
retried. The optional bundle is an exact published Git snapshot fallback before
any pipeline launch; canonical data and official model metadata stay byte exact.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


PROJECT = Path('/Users/shaoshitong/Documents/Search-G1')
CONTROL = Path('/Users/shaoshitong/Documents/remote_codex_control/control.py')
OFFICIAL = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-300/official-model-manifest-original.json')
OFFICIAL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
SHARED_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
DATA_NAMES = ('manifest.json', 'train.parquet', 'validation.parquet', 'stage.parquet', 'final.parquet', 'exclusions-audit.json')
VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
RUNTIME_SUFFIXES = ('.py', '.sh', '.yaml', '.yml', '.json')


REMOTE_EXECUTE = r'''
import hashlib,io,json,os,re,shutil,socket,subprocess,sys,tarfile,time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def write_once(path,value):
    data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def state(stage,**extra):
    value=dict(deployment_id=intent['deployment_id'],base=str(base),source_commit=intent['source_commit'],
               stage=stage,observed_unix=time.time(),**extra)
    temp=base/'logs'/('deployment-status-'+intent['deployment_id']+'.tmp')
    with temp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(base/'logs/deployment-status.json')
    fd=os.open(base/'logs',os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print('DEPLOYMENT_JSON:'+json.dumps(value),flush=True)

def command(args,label,timeout=180,env=None):
    global last_command
    last_command=label
    with (base/'preparation'/(label+'.log')).open('xb') as out:
        result=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                              env=env,timeout=timeout)
        out.flush();os.fsync(out.fileno())
    return result.returncode

base=None;intent=None;created=False;last_command=None
try:
    archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
    first=archive.next();assert first is not None and first.name=='deployment-intent.json' and first.isfile()
    intent=json.load(archive.extractfile(first))
    assert intent['role']=='fresh_a800_common_deployment_intent' and intent['host']=='baai-a800'
    assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
    base=Path(intent['base'])
    assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
    assert base.parent.is_dir() and base.parent.resolve()==base.parent
    base.mkdir(mode=0o700,exist_ok=False)
    created=True
    for name in ('logs','preparation','data','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts','checkpoints'):
        (base/name).mkdir(parents=True,exist_ok=True)
    write_once(base/'preparation/deployment-intent.json',intent)
    state('RECEIVING_EXACT_LOCAL_INPUTS')
    expected={row['target']:row for row in intent['inputs']};seen=set()
    while True:
        item=archive.next()
        if item is None:break
        assert item.isfile() and item.name in expected and item.name not in seen
        row=expected[item.name];assert item.size==row['bytes']
        target=base/item.name;assert target.parent.resolve()==target.parent and not target.exists()
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        h=hashlib.sha256();count=0
        with os.fdopen(fd,'wb') as out:
            source=archive.extractfile(item)
            for part in iter(lambda:source.read(1024*1024),b''):out.write(part);h.update(part);count+=len(part)
            out.flush();os.fsync(out.fileno())
        assert count==row['bytes'] and h.hexdigest()==row['sha256'];seen.add(item.name)
    assert seen==set(expected)
    write_once(base/'logs/deployment-input-receipt.json',dict(deployment_id=intent['deployment_id'],
        inputs=[{k:row[k] for k in ('target','bytes','sha256')} for row in intent['inputs']],
        exact_original_local_bytes=True,training_data_partitioned=False))
    state('CLONING_PINNED_PUBLISHED_GITHUB_SOURCE')
    github='https://github.com/shaoshitong/Search-R1.git'
    existing=Path('/data/search-g1/repo')
    if (existing/'.git').exists():
        result=subprocess.run(['git','-C',str(existing),'remote','get-url','origin'],capture_output=True,text=True,timeout=10)
        candidate=result.stdout.strip()
        if result.returncode==0 and candidate in ('https://github.com/shaoshitong/Search-R1.git','git@github.com:shaoshitong/Search-R1.git'):
            github=candidate
    git_env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    repo=base/'repo';direct=False
    try:
        direct=command(['git','clone','--branch','search-g1','--single-branch',github,str(repo)],'github-clone',90,git_env)==0
        if direct:direct=command(['git','-C',str(repo),'checkout','--detach',intent['source_commit']],'github-checkout',30,git_env)==0
    except subprocess.TimeoutExpired:direct=False
    transport='github_direct'
    if not direct:
        bundle=base/'preparation/published-source.bundle'
        assert bundle.is_file(), 'Direct GitHub clone unavailable; no explicit published-source bundle was provided'
        assert not (base/'logs/pipeline-launch-claim.json').exists()
        if repo.exists():assert repo.resolve()==repo;shutil.rmtree(repo)
        assert command(['git','clone',str(bundle),str(repo)],'published-bundle-clone',90,git_env)==0
        assert command(['git','-C',str(repo),'checkout','--detach',intent['source_commit']],'published-bundle-checkout',30,git_env)==0
        transport='published_git_bundle_relay'
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()
    assert actual==intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
    sys.path.insert(0,str(repo))
    from reproduction.common8.source_files import runtime_files
    source={str(path.relative_to(repo)):digest(path) for path in runtime_files(repo)}
    assert source==intent['source_sha256'], 'Pinned runtime source bytes differ from the published Git snapshot'
    write_once(base/'logs/deployment-source-receipt.json',dict(deployment_id=intent['deployment_id'],
        source_commit=actual,source_sha256=source,source_transport=transport,
        source_is_exact_published_commit=True,uncommitted_source_copied=False))
    state('PREPARING_FRESH_OFFICIAL_RUN',source_transport=transport)
    # Preserve the original approved method, rebinding only fresh source/time.
    from reproduction.common8.formal_inline_safety import verify_method
    from reproduction.common8.policy_calibration_plan import document_sha256
    method=json.loads((base/'preparation/formal-inline-method-template.json').read_text())
    method.update(source_commit=actual,source_map_sha256=document_sha256(source),created_unix=time.time())
    write_once(base/'preparation/formal-inline-method.json',method)
    environment=dict(os.environ)
    for key in tuple(environment):
        if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            environment.pop(key,None)
    environment.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
        CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+environment.get('PATH','/usr/bin:/bin'),
        C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include',
        TMPDIR=str(base/'tmp'),RAY_TMPDIR=str(base/'ray'),HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
        NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost',RAY_DEDUP_LOGS='0')
    os.chdir(repo)
    python='/data/search-g1/env/bin/python'
    assert command([python,'-m','reproduction.a800_common.prepare_run','--base',str(base),
        '--model','/data/search-g1/assets/qwen2.5-3b-instruct',
        '--official-manifest',str(base/'preparation/official-model-manifest-original.json'),
        '--source-commit',actual],'prepare-run',600,environment)==0
    run=json.loads((base/'logs/run-manifest.json').read_text());verify_method(method,run)
    assert run['source_sha256']==source and run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None
    assert not (base/'logs/pipeline-launch-claim.json').exists() and not (base/'logs/process-ledger.json').exists()
    state('STARTING_ONE_ACTUAL_LOCAL_PIPELINE',run_id=run['run_id'],execution_id=run['execution_id'])
    args=[python,'-m','reproduction.a800_common.run_pipeline','--base',str(base)]
    log=base/'logs/deployment-pipeline.log'
    with log.open('xb') as out:
        process=subprocess.Popen(args,cwd=repo,env=environment,stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        text=(Path('/proc')/str(process.pid)/'stat').read_text()
        fields=text[text.rfind(')')+2:].split();ticks=int(fields[19]);sid=int(fields[3])
        assert ticks>0 and sid==process.pid
        receipt=dict(deployment_id=intent['deployment_id'],run_id=run['run_id'],execution_backend='ssh_local',
            execution_id=run['execution_id'],slurm_job_id=None,hostname=socket.gethostname(),
            pid=process.pid,start_ticks=ticks,sid=sid,stdout_log=str(log),source_commit=actual,
            source_transport=transport,launched_unix=time.time(),training_world_size=4,
            training_physical_indices=[0,1,2,3],same_process_after_five=True,no_automatic_retry=True)
        write_once(base/'logs/deployment-launch-receipt.json',receipt)
    state('LAUNCHED',**{k:receipt[k] for k in ('run_id','execution_id','pid','start_ticks','stdout_log','source_transport')})
except BaseException as failure:
    if created and base is not None and (base/'logs').is_dir() and intent is not None:
        # Never turn an uncertain Popen/receipt result into a second launch.
        state('DEPLOYMENT_FAILED_OR_UNCERTAIN',failure_type=type(failure).__name__,
              last_command=last_command,no_automatic_retry=True)
    else:
        print('DEPLOYMENT_JSON:'+json.dumps(dict(stage='REMOTE_INTENT_REJECTED',failure_type=type(failure).__name__)),flush=True)
    sys.exit(1)
'''


REMOTE_OBSERVE = r'''
import json,os,re,time
from pathlib import Path
intent=json.loads(__import__('sys').stdin.read());base=Path(intent['base'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
def read(name):
    path=base/name
    if not path.is_file():return None
    assert not path.is_symlink() and path.resolve()==path and path.stat().st_uid==os.getuid()
    return json.loads(path.read_text())
def diagnostic(path):
    if not path.is_file():return []
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
    selected=[]
    for line in lines:
        text=line.strip()
        # Retain traceback locations and bounded exception messages only.
        # No clone/auth output, model examples, JSON records, or QA log rows.
        if not (text.startswith('File "') or re.match(r'^(AssertionError|ValueError|RuntimeError|OSError|FileNotFoundError|PermissionError|ModuleNotFoundError|ImportError|KeyError|TypeError)(:|$)',text)):
            continue
        if any(word in text.lower() for word in ('password','authorization','bearer','github_pat','ghp_','prompt','question','answer','response')):
            text=text.split(':',1)[0] if not text.startswith('File "') else text.split(', in ',1)[0]
        if not text.startswith('File "'):
            text=re.split(r'[\{\[]',text,1)[0]
        selected.append(text[:240])
    return selected[-6:]
result=dict(deployment_id=intent['deployment_id'],base=str(base),base_exists=base.is_dir(),observed_unix=time.time())
owned=read('preparation/deployment-intent.json')
if owned is not None:
    assert owned==intent
    status=read('logs/deployment-status.json')
    if status:
        result['deployment_stage']=status['stage'];result['failure_type']=status.get('failure_type')
        result['last_command']=status.get('last_command')
        if status['stage']=='DEPLOYMENT_FAILED_OR_UNCERTAIN' and status.get('last_command')=='prepare-run':
            result['failure_diagnostic']=diagnostic(base/'preparation/prepare-run.log')
    receipt=read('logs/deployment-launch-receipt.json')
    if receipt:
        result.update({k:receipt[k] for k in ('run_id','execution_id','pid','start_ticks','stdout_log','source_commit','source_transport')})
        proc=Path('/proc')/str(receipt['pid'])/'stat';matches=False;process_state=None
        if proc.is_file():
            text=proc.read_text();fields=text[text.rfind(')')+2:].split()
            matches=int(fields[19])==receipt['start_ticks'];process_state=fields[0] if matches else None
        result.update(pipeline_birth_matches=matches,pipeline_process_state=process_state,
                      pipeline_alive=matches and process_state not in ('Z','X'))
    for name,key in (('logs/pipeline-launch-claim.json','pipeline_claimed'),('logs/training-launch-claim.json','trainer_claimed')):
        result[key]=(base/name).is_file()
    pipeline=read('logs/pipeline-status.json')
    if pipeline:
        result['pipeline_stage']=pipeline['stage']
        if pipeline['stage'] in ('FAILED','OWNED_CLEANUP_FAILED'):
            result['failure_diagnostic']=diagnostic(base/'logs/deployment-pipeline.log')
    records=base/'logs/actual-updates.jsonl'
    if records.is_file():
        steps=[json.loads(line)['step'] for line in records.read_text().splitlines()]
        result.update(recorded_completed_outers=len(steps),last_recorded_outer=steps[-1] if steps else 0)
    log=base/'logs/deployment-pipeline.log'
    if log.is_file():result.update(stdout_log_bytes=log.stat().st_size,stdout_log_modified_unix=log.stat().st_mtime)
print('DEPLOYMENT_JSON:'+json.dumps(result),flush=True)
'''


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'w') as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write('\n'); target.flush(); os.fsync(target.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def published_source(repo, commit):
    """Hash Git blobs at the pinned commit, never mutable working-tree bytes."""
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', commit + '^{commit}'], text=True).strip()
    assert actual == commit
    raw = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', commit])
    selected = []
    for item in raw.split(b'\0'):
        if not item:
            continue
        metadata, name = item.split(b'\t', 1)
        mode, kind, identity = metadata.decode().split()
        name = name.decode(); relative = Path(name)
        if (relative.suffix in RUNTIME_SUFFIXES or name in VERSION_FILES) and not {'.git', '__pycache__', 'evidence'}.intersection(relative.parts):
            assert kind == 'blob' and mode in ('100644', '100755')
            selected.append((name, identity))
    assert set(VERSION_FILES).issubset(name for name, _ in selected)
    payload = ('\n'.join(identity for _, identity in selected) + '\n').encode()
    process = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=payload, capture_output=True, check=True)
    blobs = io.BytesIO(process.stdout); result = {}
    for name, expected in selected:
        identity, kind, size = blobs.readline().decode().strip().split()
        assert identity == expected and kind == 'blob'
        result[name] = hashlib.sha256(blobs.read(int(size))).hexdigest()
        assert blobs.read(1) == b'\n'
    assert not blobs.read()
    return result


def create_intent(args):
    assert args.source_commit is not None and re.fullmatch('[0-9a-f]{40}', args.source_commit)
    repo = Path(__file__).resolve().parents[2]
    shared = args.shared_data.resolve(); official = args.official_manifest.resolve(); method = args.method.resolve()
    assert digest(shared / 'manifest.json') == SHARED_SHA and digest(official) == OFFICIAL_SHA
    manifest = json.loads((shared / 'manifest.json').read_text())
    inputs = []
    for name in DATA_NAMES:
        path = shared / name
        if name != 'manifest.json':
            expected = (manifest['exclusion_audit'] if name == 'exclusions-audit.json' else manifest['splits'][name.removesuffix('.parquet')])
            assert expected['path'] == name and digest(path) == expected['sha256']
        inputs.append(dict(source=str(path), target='data/' + name, bytes=path.stat().st_size, sha256=digest(path)))
    for source, target in ((official, 'preparation/official-model-manifest-original.json'),
                           (method, 'preparation/formal-inline-method-template.json')):
        inputs.append(dict(source=str(source), target=target, bytes=source.stat().st_size, sha256=digest(source)))
    if args.bundle is not None:
        bundle = args.bundle.resolve()
        subprocess.run(['git', '-C', str(repo), 'bundle', 'verify', str(bundle)], check=True, capture_output=True)
        inputs.append(dict(source=str(bundle), target='preparation/published-source.bundle', bytes=bundle.stat().st_size, sha256=digest(bundle)))
    identifier = uuid.uuid4().hex
    value = dict(schema_version=1, role='fresh_a800_common_deployment_intent', deployment_id=identifier,
        created_unix=time.time(), host='baai-a800', base='/data/shaoshitong/Search-G1-a800-fresh200-' + identifier,
        source_commit=args.source_commit, source_sha256=published_source(repo, args.source_commit),
        inputs=inputs, training_world_size=4, training_physical_indices=[0, 1, 2, 3],
        auxiliary_physical_indices=[5, 6], target_updates=200, same_process_after_five=True,
        execution_policy='one explicit execution; inspect the same handle after failure or uncertainty; no automatic retry')
    handle = args.handle.resolve() if args.handle is not None else PROJECT / 'preparation' / ('a800-common-deployment-' + identifier + '.json')
    write_once(handle, value)
    print(json.dumps(dict(handle=str(handle), base=value['base'], source_commit=value['source_commit'], intent_created=True)))


def ssh_command(code):
    spec = importlib.util.spec_from_file_location('a800_existing_control', CONTROL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.ssh('baai-a800') + ['python3 -c ' + shlex.quote(code)]


def perform(args):
    assert args.handle is not None, '--execute/--observe require the original persisted handle'
    handle = args.handle.resolve(); intent = json.loads(handle.read_text())
    assert intent['role'] == 'fresh_a800_common_deployment_intent' and intent['host'] == 'baai-a800'
    assert re.fullmatch('[0-9a-f]{32}', intent['deployment_id']) and re.fullmatch('[0-9a-f]{40}', intent['source_commit'])
    assert intent['base'] == '/data/shaoshitong/Search-G1-a800-fresh200-' + intent['deployment_id']
    if args.execute:
        # An exclusive claim precedes even the SSH attempt. A timeout or lost
        # receipt cannot make a later invocation launch a duplicate process.
        claim = handle.with_name(handle.stem + '.execute-once.json')
        write_once(claim, dict(deployment_id=intent['deployment_id'], claimed_unix=time.time(),
                              local_pid=os.getpid(), no_automatic_retry=True))
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            raw = json.dumps(intent, sort_keys=True, allow_nan=False).encode()
            member = tarfile.TarInfo('deployment-intent.json'); member.size = len(raw); archive.addfile(member, io.BytesIO(raw))
            for row in intent['inputs']:
                source = Path(row['source'])
                assert source.is_file() and source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
                with source.open('rb') as original:
                    member = tarfile.TarInfo(row['target']); member.size = row['bytes']; member.mode = 0o600
                    archive.addfile(member, original)
        payload = data.getvalue(); code = REMOTE_EXECUTE; timeout = 1000
    else:
        payload = json.dumps(intent, sort_keys=True, allow_nan=False).encode(); code = REMOTE_OBSERVE; timeout = 60
    action = 'execute' if args.execute else 'observe'
    stderr_path = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.stderr')
    output = b''; exit_code = None; failure_type = None
    with stderr_path.open('xb') as errors:
        try:
            result = subprocess.run(ssh_command(code), input=payload, stdout=subprocess.PIPE, stderr=errors, timeout=timeout)
            output, exit_code = result.stdout, result.returncode
        except (subprocess.TimeoutExpired, OSError) as failure:
            failure_type = type(failure).__name__
            output = getattr(failure, 'stdout', None) or b''
        errors.flush(); os.fsync(errors.fileno())
    rows = [json.loads(line[len('DEPLOYMENT_JSON:'):]) for line in output.decode(errors='replace').splitlines()
            if line.startswith('DEPLOYMENT_JSON:')]
    receipt_path = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.json')
    write_once(receipt_path, dict(deployment_id=intent['deployment_id'], action=action, exit_code=exit_code, failure_type=failure_type,
                                 observations=rows, transport_stderr=str(stderr_path), observed_unix=time.time()))
    print(json.dumps(dict(handle=str(handle), exit_code=exit_code, failure_type=failure_type, receipt=str(receipt_path),
                          observation=rows[-1] if rows else None, next_action='observe the same handle')))
    return exit_code if exit_code is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle', type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true'); actions.add_argument('--observe', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--shared-data', type=Path, default=PROJECT / 'shared-data')
    parser.add_argument('--official-manifest', type=Path, default=OFFICIAL)
    parser.add_argument('--method', type=Path, default=PROJECT / 'preparation/formal-inline-method.json')
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    if args.execute or args.observe:
        assert args.source_commit is None and args.bundle is None, 'Use the persisted intent without replacement inputs'
        return perform(args)
    create_intent(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
