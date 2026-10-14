"""Read one bounded GPU/process ownership snapshot of the existing A9 run.

This observes metadata only. It cannot launch, reserve, evaluate, repair, or
signal a recorded process. It imports the existing SSH route from the immutable
actor-probes-ten checkout, and writes only a fresh local metadata receipt.
"""
import argparse
import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path


DEPLOYMENT_ID = "cb042832b1814fad86a9734bbdd4e94c"
SOURCE_COMMIT = "c4d156de124de7a94652d75858bcbabc8c56083c"
RUN_ID = "a9be0ee91ff942349a26caf372e22a27"
EXECUTION_ID = "4872adb6212e478fa23fdbc71b812cb9"
HANDLE_NAME = "actor-probes-ten-production-fresh-resource-retry-deployment.json"
OBSERVATION_NAME = (
    "actor-probes-ten-production-fresh-resource-retry-deployment."
    "observe-1791941917586750000.json"
)
FILE_LIMIT = 2 * 1024 * 1024
RECEIPT_LIMIT = 512 * 1024
SSH_TIMEOUT = 120


class MetadataUnknown(Exception):
    """An observation is incomplete; its reason never includes raw output."""


def bounded_command(args, payload, timeout, stdout_limit, stderr_limit):
    """Drain bounded pipes once; never retain or display helper stderr.

    Timeout/overflow termination applies only to the helper created here (the
    local SSH client, or the remote nvidia-smi query), never a ledger/GPU PID.
    """
    process = subprocess.Popen(
        args, stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    output = bytearray()
    stderr_bytes = 0
    end = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    try:
        if payload is not None:
            if len(payload) > 4096:
                raise MetadataUnknown("query_payload_limit")
            process.stdin.write(payload)
            process.stdin.close()
        for pipe, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, label)
        while selector.get_map():
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise MetadataUnknown("query_timeout")
            for key, _ in selector.select(min(remaining, 0.25)):
                part = os.read(key.fd, 8192)
                if not part:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    if len(output) + len(part) > stdout_limit:
                        raise MetadataUnknown("query_stdout_limit")
                    output.extend(part)
                else:
                    stderr_bytes += len(part)
                    if stderr_bytes > stderr_limit:
                        raise MetadataUnknown("query_stderr_limit")
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise MetadataUnknown("query_timeout")
        returncode = process.wait(timeout=remaining)
        return bytes(output), returncode, stderr_bytes
    except BaseException:
        # Only this invocation's helper is eligible for timeout cleanup.
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


# The remote program is self-contained: no project, CUDA, or model imports.
# Include the identical bounded pipe reader without importing local source on
# the node or creating any remote files.
REMOTE_PREFIX = r'''
import csv,hashlib,io,json,os,re,selectors,socket,stat,subprocess,sys,time
class MetadataUnknown(Exception):pass
'''

REMOTE_BODY = r'''
SELECTED=(0,1,2,3,5,6)
FILE_LIMIT=2*1024*1024
OUTPUT_LIMIT=512*1024
PROC_LIMIT=32768
LEDGER_PROCESS_LIMIT=4096
deadline=time.monotonic()+105
stage='input'
out=dict(role='existing_A9_gpu_ownership_metadata_readonly',schema_version=1,
         status='unknown',launch_allowed=False,no_automatic_retry=True,
         no_remote_mutation=True,selected_physical_indices=list(SELECTED),
         gpu_rows=[],compute_apps=[],jobs=[],errors=[])

def need(value,reason):
    if not value:raise MetadataUnknown(reason)

def remaining():
    value=deadline-time.monotonic()
    need(value>0,'remote_deadline')
    return value

def stable_stat(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)

def open_directory(path):
    need(path.startswith('/') and '//' not in path,'directory_path')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for component in path.split('/')[1:]:
            need(component not in ('','.','..'),'directory_component')
            next_fd=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=next_fd
        return fd
    except BaseException:
        os.close(fd);raise

def small_json(logs_fd,name):
    remaining()
    need(name in ('run-manifest.json','process-ledger.json'),'metadata_path')
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=logs_fd)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid(),'metadata_owner_or_type')
        need(before.st_size<=FILE_LIMIT,'metadata_file_limit')
        chunks=[];count=0
        while True:
            part=os.read(fd,min(65536,FILE_LIMIT+1-count))
            if not part:break
            chunks.append(part);count+=len(part)
            need(count<=FILE_LIMIT,'metadata_file_limit')
        after=os.fstat(fd)
        need(stable_stat(before)==stable_stat(after) and count==before.st_size,'metadata_changed_while_reading')
    finally:os.close(fd)
    raw=b''.join(chunks)
    value=json.loads(raw)
    need(type(value) is dict,'metadata_object')
    return value,dict(relative='logs/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def boot_identity():
    remaining()
    fd=os.open('/proc/sys/kernel/random/boot_id',os.O_RDONLY|os.O_NOFOLLOW)
    try:raw=os.read(fd,129)
    finally:os.close(fd)
    need(len(raw)<=128,'boot_id_limit')
    boot=raw.decode('ascii').strip()
    need(re.fullmatch('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',boot) is not None,'boot_id_format')
    host=socket.gethostname()
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,253}',host) is not None,'hostname_format')
    return dict(hostname=host,boot_id=boot,uid=os.getuid())

def birth(pid):
    remaining()
    value=dict(pid=pid,read_status='unknown',start_ticks=None,state=None,uid=None,sid=None)
    proc_fd=None;stat_fd=None
    try:
        proc_fd=os.open('/proc/'+str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        owner=os.fstat(proc_fd).st_uid
        stat_fd=os.open('stat',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=proc_fd)
        raw=os.read(stat_fd,8193)
        need(len(raw)<=8192,'proc_stat_limit')
        text=raw.decode('utf-8',errors='strict')
        need(text.startswith(str(pid)+' ('),'proc_stat_pid')
        suffix=text[text.rfind(')')+2:].split()
        need(len(suffix)>=22 and suffix[0] in ('R','S','D','Z','T','t','X','x','K','W','P','I'),'proc_stat_fields')
        ticks=int(suffix[19]);sid=int(suffix[3])
        need(ticks>0 and sid>0 and owner==os.fstat(proc_fd).st_uid,'proc_stat_identity')
        value.update(read_status='read',start_ticks=ticks,state=suffix[0],uid=owner,sid=sid)
    except (FileNotFoundError,ProcessLookupError):value['read_status']='absent_or_exited'
    except PermissionError:value['read_status']='permission_denied'
    except (OSError,ValueError,UnicodeError,MetadataUnknown):value['read_status']='read_failed'
    finally:
        if stat_fd is not None:os.close(stat_fd)
        if proc_fd is not None:os.close(proc_fd)
    return value

def birth_key(row):return tuple(row.get(k) for k in ('pid','start_ticks','sid','uid'))
def live(row):return row.get('read_status')=='read' and row.get('state') not in ('Z','X','x')
def stable(before,after):return live(before) and live(after) and birth_key(before)==birth_key(after)

def pre_births():
    # This supplies BEFORE evidence even for compute-app PIDs absent from the
    # owned ledger, without an extra NVML query or any cmdline/status payload.
    pids=[]
    with os.scandir('/proc') as entries:
        for entry in entries:
            remaining()
            if entry.name.isdecimal():
                pids.append(int(entry.name))
                need(len(pids)<=PROC_LIMIT,'proc_pid_count_limit')
    return {pid:birth(pid) for pid in sorted(set(pids))}

def number(value):
    value=value.strip()
    if value in ('N/A','[N/A]','[Not Supported]','Not Supported'):return None
    need(re.fullmatch('[0-9]+(?:\.[0-9]+)?',value) is not None,'nvml_numeric_field')
    result=float(value) if '.' in value else int(value)
    need(0<=result<=2**63-1,'nvml_numeric_range')
    return result

def query(argument):
    raw,code,stderr_bytes=bounded_command(
        ['nvidia-smi',argument,'--id=0,1,2,3,5,6','--format=csv,noheader,nounits'],
        None,min(20,remaining()),128*1024,8192)
    need(code==0,'nvml_query_failed')
    # Discard stderr rather than emitting possible node/environment text.
    return list(csv.reader(io.StringIO(raw.decode('utf-8',errors='strict')))),stderr_bytes

try:
    p=json.loads(sys.stdin.buffer.read(4097))
    need(type(p) is dict,'payload_object')
    expected=dict(deployment_id='cb042832b1814fad86a9734bbdd4e94c',
        source_commit='c4d156de124de7a94652d75858bcbabc8c56083c',
        run_id='a9be0ee91ff942349a26caf372e22a27',execution_id='4872adb6212e478fa23fdbc71b812cb9',
        base='/data/shaoshitong/Search-G1-a800-fresh200-cb042832b1814fad86a9734bbdd4e94c')
    need(p==expected,'pinned_run_identity')
    out.update(expected)
    stage='node_identity_before';before_node=boot_identity();out['node_before']=before_node
    stage='manifest_and_ledger'
    logs_fd=open_directory(p['base']+'/logs')
    try:
        run,run_meta=small_json(logs_fd,'run-manifest.json')
        ledger,ledger_meta=small_json(logs_fd,'process-ledger.json')
    finally:os.close(logs_fd)
    need(all(run.get(k)==p[k] for k in ('run_id','execution_id','source_commit')),'run_manifest_identity')
    need(run.get('execution_backend')=='ssh_local' and run.get('slurm_job_id') is None,'run_manifest_backend')
    need(run.get('execution_host')==before_node['hostname'],'run_manifest_hostname')
    need(run.get('training_world_size')==4 and run.get('training_physical_indices')==[0,1,2,3]
         and run.get('auxiliary_physical_indices')==[5,6],'run_manifest_gpu_scope')
    need(ledger.get('run_id')==p['run_id'],'process_ledger_run_identity')
    for key in ('execution_id','source_commit'):
        if key in ledger:need(ledger[key]==p[key],'process_ledger_optional_identity')
    jobs=ledger.get('jobs');need(type(jobs) is dict and 0<len(jobs)<=64,'ledger_jobs_limit')
    records={};record_count=0;owned_pids=set()
    for name,job in sorted(jobs.items()):
        need(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,64}',name) is not None,'ledger_job_name')
        need(type(job) is dict and type(job.get('ledger')) is dict,'ledger_job_object')
        state=job.get('state')
        need(state in ('STARTED','STOPPED','CLEANUP_PENDING','CLEANUP_FAILED'),'ledger_job_state')
        sanitized=[]
        for saved_pid,record in job['ledger'].items():
            remaining();record_count+=1
            need(record_count<=LEDGER_PROCESS_LIMIT,'ledger_process_count_limit')
            need(type(record) is dict and all(type(record.get(k)) is int for k in ('pid','start_ticks','sid','uid')),'ledger_identity_fields')
            row={k:record[k] for k in ('pid','start_ticks','sid','uid')}
            need(row['pid']>0 and row['start_ticks']>0 and row['sid']>0 and row['uid']==before_node['uid'],'ledger_identity_range_or_uid')
            need(saved_pid==str(row['pid']),'ledger_pid_key')
            records.setdefault(birth_key(row),set()).add(name);owned_pids.add(row['pid']);sanitized.append(row)
        leader=job.get('leader')
        need(type(leader) is dict and all(type(leader.get(k)) is int for k in ('pid','start_ticks','sid','uid')),'ledger_leader_fields')
        leader={k:leader[k] for k in ('pid','start_ticks','sid','uid')}
        need(any(birth_key(row)==birth_key(leader) for row in sanitized),'ledger_leader_not_registered')
        out['jobs'].append(dict(name=name,ledger_state=state,leader=leader,
            registered_count=len(sanitized),registered_identities=sorted(sanitized,key=birth_key)))
    out.update(run_manifest=run_meta,process_ledger=ledger_meta,ledger_identity_set_complete=True,
               ledger_registered_record_count=record_count,
               ownership_semantics='owned means matching this run ledger and stable live PID/birth/session/UID before and after NVML; foreign_to_run_ledger means absent from this sampled ledger, not proven to belong to another project')
    stage='proc_before_nvml';before=pre_births()
    # Include stale ledger PIDs that were absent from the bounded /proc listing.
    for pid in owned_pids:
        if pid not in before:before[pid]=birth(pid)
    out['before_proc_pid_count']=len(before)
    out['nvml_started_unix']=time.time();stage='gpu_query'
    inventory,gpu_stderr=query('--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu')
    mapping={}
    for fields in inventory:
        need(len(fields)==5,'gpu_query_field_count')
        index=int(fields[0].strip());gpu_uuid=fields[1].strip()
        # Explicit filtering remains required even if --id is ignored.
        if index not in SELECTED:continue
        need(index not in mapping and re.fullmatch('GPU-[A-Za-z0-9-]{1,80}',gpu_uuid) is not None,'gpu_query_identity')
        need(gpu_uuid not in {row['uuid'] for row in mapping.values()},'gpu_uuid_duplicate')
        row=dict(index=index,uuid=gpu_uuid,used_memory_mib=number(fields[2]),
                 total_memory_mib=number(fields[3]),gpu_utilization_percent=number(fields[4]))
        need(row['used_memory_mib'] is not None and row['total_memory_mib'] is not None
             and row['gpu_utilization_percent'] is not None,'gpu_metric_unavailable')
        need(row['used_memory_mib']<=row['total_memory_mib'] and row['gpu_utilization_percent']<=100,'gpu_metric_range')
        mapping[index]=row
    need(set(mapping)==set(SELECTED),'selected_gpu_inventory_incomplete')
    stage='compute_apps_query'
    applications,apps_stderr=query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory')
    out['nvml_finished_unix']=time.time()
    uuid_to_index={row['uuid']:index for index,row in mapping.items()}
    selected_apps=[];app_keys=set()
    for fields in applications:
        need(len(fields)==3,'compute_query_field_count')
        gpu_uuid=fields[0].strip()
        if gpu_uuid not in uuid_to_index:continue
        pid=int(fields[1].strip());need(pid>0,'compute_pid')
        need((gpu_uuid,pid) not in app_keys,'compute_app_duplicate');app_keys.add((gpu_uuid,pid))
        selected_apps.append(dict(gpu_index=uuid_to_index[gpu_uuid],gpu_uuid=gpu_uuid,pid=pid,
                                  used_memory_mib=number(fields[2])))
    need(len(selected_apps)<=4096,'compute_app_count_limit')
    stage='proc_after_nvml'
    after={pid:birth(pid) for pid in sorted(owned_pids|{row['pid'] for row in selected_apps})}
    after_node=boot_identity();out['node_after']=after_node
    same_node=before_node==after_node;out['same_boot_and_hostname']=same_node
    if not same_node:out['errors'].append(dict(stage=stage,reason='node_identity_changed'))
    for job in out['jobs']:
        for row in job['registered_identities']:
            prior=before.get(row['pid'],dict(pid=row['pid'],read_status='not_observed_before'))
            current=after[row['pid']]
            row.update(before=prior,after=current,
                stable_matching_birth=same_node and stable(prior,current) and birth_key(row)==birth_key(prior))
    for app in sorted(selected_apps,key=lambda row:(row['gpu_index'],row['pid'])):
        pid=app['pid'];prior=before.get(pid,dict(pid=pid,read_status='not_observed_before'));current=after[pid]
        classification='unknown';reason='birth_not_stable_before_and_after_nvml';matched_jobs=[]
        if same_node and stable(prior,current):
            matched_jobs=sorted(records.get(birth_key(prior),set()))
            if matched_jobs:
                classification='owned';reason='stable_live_birth_in_full_run_ledger'
            elif pid in owned_pids:
                reason='pid_reused_or_identity_differs_from_ledger'
            else:
                classification='foreign_to_run_ledger';reason='stable_live_birth_absent_from_full_sampled_run_ledger'
        if app['used_memory_mib'] is None:
            classification='unknown';reason='compute_memory_unavailable';matched_jobs=[]
        app.update(classification=classification,reason=reason,matching_jobs=matched_jobs,before=prior,after=current)
        out['compute_apps'].append(app)
    for index in SELECTED:
        row=mapping[index];apps=[app for app in out['compute_apps'] if app['gpu_index']==index]
        sums={kind:sum(app['used_memory_mib'] for app in apps if app['classification']==kind and app['used_memory_mib'] is not None)
              for kind in ('owned','foreign_to_run_ledger','unknown')}
        app_total=sum(sums.values())
        row.update(owned_compute_memory_mib=sums['owned'],foreign_to_run_ledger_compute_memory_mib=sums['foreign_to_run_ledger'],
            unknown_compute_memory_mib=sums['unknown'],compute_memory_total_mib=app_total,
            device_minus_compute_memory_mib=row['used_memory_mib']-app_total,
            compute_app_count=len(apps),unknown_compute_app_count=sum(app['classification']=='unknown' for app in apps),
            memory_accounting='NVML device/app values are successive samples; their residual is not proof of foreign occupancy')
        out['gpu_rows'].append(row)
    out['status']='observed' if same_node and all(row['classification']!='unknown' for row in out['compute_apps']) else 'unknown'
    out.update(query_stderr_bytes=gpu_stderr+apps_stderr,observed_unix=time.time(),read_once=True)
except BaseException as error:
    # Never print source JSON, command arguments, process cmdlines, or stderr.
    out['status']='unknown';out['gpu_rows']=[];out['compute_apps']=[];out['jobs']=[]
    out['errors'].append(dict(stage=stage,reason=str(error) if type(error) is MetadataUnknown else 'metadata_read_failed',failure_type=type(error).__name__))
    out['observed_unix']=time.time()
encoded=json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
if len(encoded)>OUTPUT_LIMIT:
    out=dict(role='existing_A9_gpu_ownership_metadata_readonly',schema_version=1,status='unknown',
        launch_allowed=False,no_automatic_retry=True,no_remote_mutation=True,read_once=True,
        selected_physical_indices=list(SELECTED),errors=[dict(reason='receipt_output_limit')])
    encoded=json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sys.stdout.buffer.write(encoded+b'\n');sys.stdout.buffer.flush()
'''


def local_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise MetadataUnknown("local_metadata_owner_or_type")
        if before.st_size > FILE_LIMIT:
            raise MetadataUnknown("local_metadata_file_limit")
        with os.fdopen(fd, "rb", closefd=False) as source:
            raw = source.read(FILE_LIMIT + 1)
        after = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) > FILE_LIMIT or len(raw) != before.st_size or any(
            getattr(before, key) != getattr(after, key) for key in keys
        ):
            raise MetadataUnknown("local_metadata_changed_or_limit")
    finally:
        os.close(fd)
    value = json.loads(raw)
    if type(value) is not dict:
        raise MetadataUnknown("local_metadata_object")
    return value, hashlib.sha256(raw).hexdigest()


def write_receipt(root, value):
    preparation = root / "preparation"
    if preparation.resolve() != preparation or not preparation.is_dir():
        raise MetadataUnknown("local_receipt_directory")
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > RECEIPT_LIMIT + 16384:
        raise MetadataUnknown("local_receipt_limit")
    path = preparation / (
        "actor-probes-ten-A9-gpu-ownership-%d-%s.json" % (time.time_ns(), uuid.uuid4().hex)
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    directory_fd = os.open(preparation, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--handle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    receipt = dict(role="existing_A9_gpu_ownership_single_read_receipt", schema_version=1,
        status="unknown", launch_allowed=False, no_automatic_retry=True, no_remote_mutation=True,
        selected_physical_indices=[0, 1, 2, 3, 5, 6], created_unix=time.time())
    exit_code = 1
    try:
        if root.resolve() != root:
            raise MetadataUnknown("local_root_symlink")
        handle_path = args.handle.absolute()
        observation_path = args.observation.absolute()
        if handle_path != root / "preparation" / HANDLE_NAME or observation_path != root / "preparation" / OBSERVATION_NAME:
            raise MetadataUnknown("original_handle_or_snapshot_required")
        intent, handle_sha = local_json(handle_path)
        snapshot, observation_sha = local_json(observation_path)
        expected = dict(deployment_id=DEPLOYMENT_ID, source_commit=SOURCE_COMMIT,
            run_id=RUN_ID, execution_id=EXECUTION_ID,
            base="/data/shaoshitong/Search-G1-a800-fresh200-" + DEPLOYMENT_ID)
        if intent.get("role") != "fresh_a800_common_deployment_intent" or intent.get("host") != "baai-a800":
            raise MetadataUnknown("original_handle_role_or_route")
        if any(intent.get(key) != expected[key] for key in ("base", "deployment_id", "source_commit")):
            raise MetadataUnknown("original_handle_identity")
        if snapshot.get("exit_code") != 0 or snapshot.get("failure_type") is not None or snapshot.get("deployment_id") != DEPLOYMENT_ID:
            raise MetadataUnknown("snapshot_not_successful")
        observations = snapshot.get("observations")
        if type(observations) is not list or len(observations) != 1 or type(observations[0]) is not dict:
            raise MetadataUnknown("snapshot_single_observation_required")
        observed = observations[0]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise MetadataUnknown("snapshot_identity_not_aligned")
        for key in ("run_id", "execution_id"):
            if key in intent and intent[key] != observed[key]:
                raise MetadataUnknown("optional_handle_identity_not_aligned")
        receipt.update(expected, handle=str(handle_path), handle_sha256=handle_sha,
            observation=str(observation_path), observation_sha256=observation_sha,
            run_identity_origin="original observation aligned to immutable handle; rechecked remotely")
        immutable = root / "actor-probes-ten"
        route_source = immutable / "reproduction" / "a800_common" / "deploy.py"
        if immutable.resolve() != immutable or route_source.resolve() != route_source:
            raise MetadataUnknown("immutable_route_symlink")
        sys.path.insert(0, str(immutable))
        from reproduction.a800_common import deploy
        if Path(deploy.__file__).resolve() != route_source:
            raise MetadataUnknown("immutable_route_import_mismatch")
        import inspect
        remote = REMOTE_PREFIX + inspect.getsource(bounded_command) + "\n" + REMOTE_BODY
        receipt.update(route_checkout=str(immutable), route_source_sha256=hashlib.sha256(route_source.read_bytes()).hexdigest(),
            remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(), transport_attempts=1,
            transport_timeout_seconds=SSH_TIMEOUT)
        raw, returncode, stderr_bytes = bounded_command(
            deploy.ssh_command(remote), json.dumps(expected, sort_keys=True).encode(),
            SSH_TIMEOUT, RECEIPT_LIMIT, 16384,
        )
        receipt.update(transport_exit_code=returncode, transport_stderr_bytes=stderr_bytes)
        if returncode != 0:
            raise MetadataUnknown("metadata_transport_failed")
        value = json.loads(raw)
        if type(value) is not dict or value.get("role") != "existing_A9_gpu_ownership_metadata_readonly":
            raise MetadataUnknown("remote_metadata_schema")
        if value.get("launch_allowed") is not False or value.get("no_remote_mutation") is not True:
            raise MetadataUnknown("remote_metadata_scope")
        if value.get("status") not in ("observed", "unknown") or value.get("selected_physical_indices") != [0, 1, 2, 3, 5, 6]:
            raise MetadataUnknown("remote_metadata_status_or_gpu_scope")
        if value.get("status") == "observed" and any(value.get(key) != item for key, item in expected.items()):
            raise MetadataUnknown("remote_metadata_identity")
        receipt.update(status=value["status"], metadata=value)
        exit_code = 0 if value["status"] == "observed" else 1
    except Exception as error:
        receipt.update(status="unknown", error=dict(
            reason=str(error) if type(error) is MetadataUnknown else "metadata_read_failed",
            failure_type=type(error).__name__), completed_unix=time.time())
    receipt["completed_unix"] = time.time()
    try:
        path, digest = write_receipt(root, receipt)
        print(json.dumps(dict(receipt=str(path), sha256=digest, status=receipt["status"],
            launch_allowed=False, no_remote_mutation=True, no_automatic_retry=True), sort_keys=True))
    except Exception as error:
        print(json.dumps(dict(status="unknown", receipt=None, launch_allowed=False,
            no_remote_mutation=True, no_automatic_retry=True,
            failure_type=type(error).__name__, reason="local_receipt_write_failed"), sort_keys=True))
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
