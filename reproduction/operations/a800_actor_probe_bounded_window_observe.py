"""One scheduled, read-only observation plus existing outer-window GPU samples.

Only small metadata is returned.  --not-before waits on the local machine and
never queries early.  This entry cannot launch, resume, repair, sample NVML,
evaluate, verify model state, or change training/acceptance/stop thresholds.
It does not replace an existing timer.  Use only at an already authorized check.

The original immutable deploy.REMOTE_OBSERVE runs unchanged in the same SSH
invocation.  Its existing per-file limits remain its own limits; only the new
window reader has the enforced 64 MiB aggregate read budget.  Both share the
105 second remote deadline and the 120 second local transport deadline.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import selectors
import stat
import subprocess
import sys
import time
from pathlib import Path


LOCAL_FILE_LIMIT = 2 * 1024 * 1024
OUTPUT_LIMIT = 512 * 1024
TRANSPORT_TIMEOUT = 120
ROLE = "existing_actor_probe_bounded_window_observation"


class MetadataUnknown(Exception):
    """Sanitized metadata failure; raw stderr and source rows are never emitted."""


def need(condition, reason):
    if not condition:
        raise MetadataUnknown(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def local_json(path):
    need(path.resolve() == path, "local_metadata_symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(),
             "local_metadata_owner_or_type")
        need(before.st_size <= LOCAL_FILE_LIMIT, "local_metadata_limit")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            raw = source.read(LOCAL_FILE_LIMIT + 1)
        after = os.fstat(descriptor)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        need(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key)
             for key in keys), "local_metadata_changed")
    finally:
        os.close(descriptor)
    value = json.loads(raw)
    need(type(value) is dict, "local_metadata_object")
    return value, hashlib.sha256(raw).hexdigest()


def binding(intent, snapshot, handle):
    """Select current_authoritative only, never an older top-level run identity."""
    current = snapshot.get("current_authoritative", snapshot)
    need(type(current) is dict, "current_identity_object")
    expected = {key: intent[key] for key in ("base", "deployment_id", "source_commit")}
    expected.update({key: current.get(key) for key in ("run_id", "execution_id")})
    expected["source_map_sha256"] = hashlib.sha256(canonical(intent["source_sha256"])).hexdigest()
    need(current.get("handle", current.get("handle_path")) == str(handle), "current_handle_binding")
    for key in ("base", "source_commit", "source_map_sha256"):
        need(current.get(key) == expected[key], "current_" + key + "_binding")
    for key in ("run_id", "execution_id"):
        need(type(expected[key]) is str and re.fullmatch("[0-9a-f]{32}", expected[key]),
             "current_" + key + "_format")
        if key in intent:
            need(intent[key] == expected[key], "optional_handle_" + key + "_binding")
    need(current.get("training_world_size") == 4 and current.get("execution_backend") == "ssh_local",
         "current_world_or_backend")
    need(current.get("training_physical_indices") == [0, 1, 2, 3]
         and current.get("auxiliary_physical_indices") == [5, 6], "current_physical_scope")
    return expected


def validate_handle(intent):
    need(intent.get("role") == "fresh_a800_common_deployment_intent"
         and intent.get("host") == "baai-a800", "handle_role_or_route")
    need(type(intent.get("deployment_id")) is str
         and re.fullmatch("[0-9a-f]{32}", intent["deployment_id"]), "handle_deployment_id")
    need(intent.get("base") == "/data/shaoshitong/Search-G1-a800-fresh200-" + intent["deployment_id"],
         "handle_remote_base")
    need(type(intent.get("source_commit")) is str
         and re.fullmatch("[0-9a-f]{40}", intent["source_commit"]), "handle_source_commit")
    source_map = intent.get("source_sha256")
    need(type(source_map) is dict and 0 < len(source_map) <= 4096, "handle_source_map")
    for name, digest in source_map.items():
        need(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
             and type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "handle_source_entry")
    need(intent.get("preparation_mode") == "remote_prepared_actor_probe_v1"
         and intent.get("reward_schedule") == "g1_through100_then_binary"
         and intent.get("training_world_size") == 4 and intent.get("target_updates") == 200
         and intent.get("same_process_after_five") is True
         and intent.get("training_physical_indices") == [0, 1, 2, 3]
         and intent.get("auxiliary_physical_indices") == [5, 6], "handle_actor_probe_world4_recipe")


def bounded_command(command, payload):
    """One transport, bounded pipes, no retry; cleanup only our own SSH client."""
    need(len(payload) <= LOCAL_FILE_LIMIT, "transport_payload_limit")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    output = bytearray()
    stderr_bytes = 0
    written = 0
    deadline = time.monotonic() + TRANSPORT_TIMEOUT
    try:
        for pipe, label, event in ((process.stdin, "stdin", selectors.EVENT_WRITE),
                                   (process.stdout, "stdout", selectors.EVENT_READ),
                                   (process.stderr, "stderr", selectors.EVENT_READ)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, event, label)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            need(remaining > 0, "transport_timeout")
            for key, _ in selector.select(min(remaining, 0.25)):
                if key.data == "stdin":
                    count = os.write(key.fd, payload[written:written + 4096])
                    written += count
                    if written == len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "stdout":
                    need(len(output) + len(chunk) <= OUTPUT_LIMIT, "transport_stdout_limit")
                    output.extend(chunk)
                else:
                    stderr_bytes += len(chunk)
                    need(stderr_bytes <= 16384, "transport_stderr_limit")
        remaining = deadline - time.monotonic()
        need(remaining > 0, "transport_timeout")
        return bytes(output), process.wait(timeout=remaining), stderr_bytes
    except BaseException:
        if process.poll() is None:
            process.kill()  # The newly created local transport, never a trainer PID.
        process.wait()
        raise
    finally:
        selector.close()
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()


# Remote extension imports only Python stdlib.  It does not import a project
# module, inspect a live PID, execute nvidia-smi, or touch a remote output file.
REMOTE_EXTENSION = r'''
import contextlib,hashlib,io,json,math,os,re,signal,socket,stat,sys,time
from pathlib import Path
TOTAL_LIMIT=64*1024*1024
SMALL_LIMIT=2*1024*1024
LINE_LIMIT=256*1024
ROW_LIMIT=100000
PHYSICAL=[0,1,2,3,5,6]
deadline=time.monotonic()+105
spent=0
files=[]
class RemoteUnknown(Exception):pass
def need(value,reason):
    if not value:raise RemoteUnknown(reason)
def check():need(time.monotonic()<deadline,'remote_deadline')
def alarm(signum,frame):raise RemoteUnknown('remote_deadline')
signal.signal(signal.SIGALRM,alarm);signal.alarm(105)
def number(value):return type(value) in (int,float) and math.isfinite(value)
def integer(value,minimum=0):return type(value) is int and value>=minimum
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def open_directory(path):
    need(path.startswith('/') and '//' not in path,'directory_path')
    descriptor=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for name in path.split('/')[1:]:
            check();need(name not in ('','.','..'),'directory_component')
            newer=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor);descriptor=newer
        need(os.fstat(descriptor).st_uid==os.getuid(),'directory_owner')
        return descriptor
    except BaseException:os.close(descriptor);raise
class Snapshot:
    def __init__(self,directory,name,limit,mode='immutable_json'):
        check();self.directory=directory;self.name=name;self.mode=mode
        need(mode in ('immutable_json','append_jsonl','mutable_json'),'snapshot_mode')
        tail=mode=='append_jsonl'
        self.fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        try:
            self.before=os.fstat(self.fd)
            need(stat.S_ISREG(self.before.st_mode) and self.before.st_uid==os.getuid(),'metadata_owner_or_type')
            need(tail or self.before.st_size<=limit,'small_metadata_limit')
            self.end=self.before.st_size;self.start=max(0,self.end-limit) if tail else 0
            self.meta=dict(relative='logs/'+name,device=self.before.st_dev,inode=self.before.st_ino,
                initial_file_bytes=self.end,scan_start_byte=self.start,scan_end_byte=self.end,
                scan_from_start=self.start==0,snapshot_mode=mode,append_permitted=tail,
                path_replacement_permitted=mode=='mutable_json',
                verification_scope='separate before/after semantic identity' if mode=='mutable_json'
                    else 'fixed initial inode and byte range')
            files.append(self);self.raw=None
        except BaseException:os.close(self.fd);raise
    def bound_stat(self,current,reason):
        need(stat.S_ISREG(current.st_mode) and current.st_uid==os.getuid()
             and (current.st_dev,current.st_ino)==(self.before.st_dev,self.before.st_ino)
             and (current.st_size>=self.end if self.mode=='append_jsonl' else current.st_size==self.end),reason)
    def read_range(self):
        global spent
        check();remaining=self.end-self.start
        need(spent+remaining<=TOTAL_LIMIT,'extension_aggregate_read_limit')
        data=[];offset=self.start
        while offset<self.end:
            check();chunk=os.pread(self.fd,min(65536,self.end-offset),offset)
            need(chunk,'metadata_truncated');spent+=len(chunk);offset+=len(chunk);data.append(chunk)
        return b''.join(data)
    def read(self):
        self.bound_stat(os.fstat(self.fd),'snapshot_fd_changed_before_read')
        self.raw=self.read_range();self.meta.update(scan_bytes=len(self.raw),scanned_range_sha256=digest(self.raw))
        self.bound_stat(os.fstat(self.fd),'snapshot_fd_changed_after_read')
        return self.raw
    def verify(self):
        need(self.mode!='mutable_json','mutable_ledger_requires_semantic_recheck')
        check();current=os.stat(self.name,dir_fd=self.directory,follow_symlinks=False)
        self.bound_stat(current,'metadata_path_changed_before_prefix_recheck')
        self.bound_stat(os.fstat(self.fd),'metadata_fd_changed_before_prefix_recheck')
        need(digest(self.read_range())==self.meta['scanned_range_sha256'],'scanned_prefix_rewritten')
        self.bound_stat(os.fstat(self.fd),'metadata_fd_changed_after_prefix_recheck')
        current=os.stat(self.name,dir_fd=self.directory,follow_symlinks=False)
        self.bound_stat(current,'metadata_path_changed_after_prefix_recheck')
        self.meta.update(verified_initial_range_unchanged=True,observed_file_bytes_after=current.st_size)
    def verify_mutable_fd(self):
        # Atomic publication may replace the path; the already-held document FD
        # must still contain the exact bounded bytes used for its identity.
        need(self.mode=='mutable_json','held_fd_recheck_requires_mutable_document')
        check();self.bound_stat(os.fstat(self.fd),'mutable_fd_changed_before_hash_recheck')
        need(digest(self.read_range())==self.meta['scanned_range_sha256'],
             'mutable_held_document_bytes_rewritten')
        current=os.fstat(self.fd)
        self.bound_stat(current,'mutable_fd_changed_after_hash_recheck')
        self.meta.update(verified_held_fd_range_unchanged=True,
            held_fd_file_bytes_after=current.st_size,path_inode_not_required_for_held_fd_recheck=True)
    def rows(self):
        need(self.mode=='append_jsonl','jsonl_snapshot_mode')
        raw=self.read();leading=0
        if self.start:
            # Conservatively discard the first tail record even at a boundary.
            found=raw.find(b'\n');leading=len(raw) if found<0 else found+1
        final=raw.rfind(b'\n')+1;final=max(leading,final)
        prefix=raw[leading:final]
        self.meta.update(discarded_leading_fragment_bytes=leading,discarded_final_partial_bytes=len(raw)-final,
            complete_start_byte=self.start+leading,complete_end_byte=self.start+final,
            complete_rows_prefix_sha256=digest(prefix),complete_prefix_scope='all newline-complete scanned bytes in stated range',
            complete_line_count=0,malformed_complete_lines=0,oversized_complete_lines=0)
        values=[]
        for line in prefix.split(b'\n')[:-1]:
            check();self.meta['complete_line_count']+=1
            need(self.meta['complete_line_count']<=ROW_LIMIT,'metadata_row_limit')
            if len(line)+1>LINE_LIMIT:self.meta['oversized_complete_lines']+=1;continue
            try:
                value=json.loads(line)
                if type(value) is not dict:raise ValueError()
            except (ValueError,UnicodeError):self.meta['malformed_complete_lines']+=1;continue
            values.append(value)
        need(not self.meta['malformed_complete_lines'] and not self.meta['oversized_complete_lines'],
             'malformed_or_oversized_complete_rows')
        return values
    def document(self):
        need(self.mode in ('immutable_json','mutable_json'),'json_document_snapshot_mode')
        raw=self.read();value=json.loads(raw);need(type(value) is dict,'metadata_object')
        self.meta.update(complete_start_byte=0,complete_end_byte=self.end,
            complete_rows_prefix_sha256=digest(raw),complete_prefix_scope='complete fixed initial JSON document')
        return value
def same_execution(row,expected):
    return (row.get('run_id')==expected['run_id'] and row.get('execution_id')==expected['execution_id']
        and row.get('execution_backend')=='ssh_local' and row.get('slurm_job_id') is None)
def identity(value):
    need(type(value) is dict and all(integer(value.get(k),1 if k!='uid' else 0)
         for k in ('pid','start_ticks','sid','uid')) and value['uid']==os.getuid(),'recorded_pid_birth_identity')
    return {k:value[k] for k in ('pid','start_ticks','sid','uid')}
def trainer_snapshot(ledger,expected,claim):
    need(ledger.get('run_id')==expected['run_id'],'process_ledger_run')
    for key in ('execution_id','source_commit'):
        if key in ledger:need(ledger[key]==expected[key],'process_ledger_optional_identity')
    trainer=ledger.get('jobs',{}).get('trainer');need(type(trainer) is dict,'recorded_trainer_job')
    leader=identity(trainer.get('leader'))
    need((leader['pid'],leader['start_ticks'])==(claim['shell_pid'],claim['shell_start_ticks']),'trainer_claim_birth')
    births={};records=trainer.get('ledger')
    need(type(records) is dict and len(records)<=8192,'trainer_ledger_record_limit')
    for saved_pid,record in records.items():
        check();record=identity(record);birth=(record['pid'],record['start_ticks'])
        need(saved_pid==str(record['pid']),'trainer_record_pid_key')
        if record['sid']==leader['sid'] and record['uid']==leader['uid']:
            need(birth not in births,'trainer_record_duplicate_birth');births[birth]=record
    need(births.get((leader['pid'],leader['start_ticks']))==leader,'trainer_leader_not_registered')
    run_identity={key:ledger.get(key,expected[key]) for key in ('run_id','execution_id','source_commit')}
    return run_identity,leader,births
def unavailable(step,attempt,reason):
    return dict(step=step,attempt_id=attempt,status='unavailable',reason=reason,
        begin_unix=None,complete_unix=None,sample_count=0,physical_sampled_peak_mib=None,
        exclusive_owned_model_peak_mib=None,maximum_sample_gap_seconds=None,
        maximum_query_duration_seconds=None,begin_uncovered_seconds=None,complete_uncovered_seconds=None)
def extension(payload):
    global spent
    expected=payload['expected'];intent=payload['intent'];directory=None
    out=dict(role='existing_completed_outer_gpu_window_samples_readonly',schema_version=1,status='unavailable',
        **expected,last_requested_windows=payload['last_windows'],windows=[],files=[],
        existing_samples_only=True,remote_mutation=False,new_nvml_queries=0,
        exclusive_owned_model_peak_mib=None,training_acceptance=False,changes_stop_or_acceptance_thresholds=False,
        window_scope='physical-device sampled occupancy in min four-rank BEGIN to max four-rank COMPLETE; excludes pre-BEGIN preload; not an individual optimizer.step or exclusive model peak',
        limits=dict(aggregate_read_bytes=TOTAL_LIMIT,gpu_tail_bytes=16*1024*1024,ledger_tail_bytes=2*1024*1024,
            json_document_bytes=SMALL_LIMIT,line_bytes=LINE_LIMIT,rows_per_file=ROW_LIMIT,
            remote_deadline_seconds=105,aggregate_includes_prefix_recheck=True,
            aggregate_includes_mutable_ledger_before_after=True,
            aggregate_includes_mutable_held_fd_rechecks=True))
    try:
        need(intent['base']==expected['base'],'payload_base_identity')
        directory=open_directory(expected['base']+'/logs')
        docs={name:Snapshot(directory,name,SMALL_LIMIT) for name in (
            'run-manifest.json','native-allocation-admission.json','pipeline-launch-claim.json',
            'training-launch-claim.json')}
        process_before=Snapshot(directory,'process-ledger.json',SMALL_LIMIT,'mutable_json')
        process_before.meta['snapshot_phase']='before_window_scan'
        ledgers={name:Snapshot(directory,name,2*1024*1024,'append_jsonl') for name in (
            'actual-updates.jsonl','actual-attempts.jsonl',*[('scheduler-rank-%d.jsonl'%rank) for rank in range(4)])}
        gpu_file=Snapshot(directory,'gpu-samples-'+expected['run_id']+'.jsonl',16*1024*1024,'append_jsonl')
        # All initial inode/byte boundaries are captured before any ledger scan.
        run=docs['run-manifest.json'].document();allocation=docs['native-allocation-admission.json'].document()
        pipeline=docs['pipeline-launch-claim.json'].document();claim=docs['training-launch-claim.json'].document()
        ledger=process_before.document()
        need(all(run.get(k)==expected[k] for k in ('run_id','execution_id','source_commit')),'run_identity')
        need(run.get('source_sha256')==intent['source_sha256'] and digest(canonical(run['source_sha256']))==expected['source_map_sha256'],
             'run_source_map_identity')
        need(same_execution(run,expected) and run.get('training_world_size')==4
             and run.get('training_physical_indices')==[0,1,2,3] and run.get('auxiliary_physical_indices')==[5,6]
             and run.get('reward_schedule')=='g1_through100_then_binary'
             and run.get('probe_representation_source')=='actor_detached_binary_heads_v1'
             and run.get('probe_refresh_interval')==10
             and run.get('probe_parameter_ownership')=='independent_binary_heads_only','run_world4_recipe')
        host=run.get('execution_host');need(type(host) is str and host==socket.gethostname(),'run_hostname')
        need(same_execution(allocation,expected) and allocation.get('accepted') is True
             and allocation.get('node')==host and allocation.get('training_world_size')==4
             and allocation.get('actual_allocated_indices')==PHYSICAL
             and allocation.get('training_physical_indices')==[0,1,2,3]
             and allocation.get('auxiliary_physical_indices')==[5,6],'allocation_identity')
        producer=identity(run.get('pipeline_identity'))
        need(identity(allocation.get('pipeline_identity'))==producer,'allocation_sampler_birth')
        need(same_execution(pipeline,expected) and pipeline.get('node')==host
             and identity(pipeline.get('pipeline_identity'))==producer
             and pipeline.get('allocation_sha256')==docs['native-allocation-admission.json'].meta['scanned_range_sha256'],
             'pipeline_sampler_binding')
        need(same_execution(claim,expected) and claim.get('hostname')==host
             and integer(claim.get('shell_pid'),1) and integer(claim.get('shell_start_ticks'),1)
             and number(claim.get('claimed_unix')),'training_claim_identity')
        uuids=allocation.get('assigned_gpu_uuids');need(type(uuids) is dict and set(uuids)=={str(i) for i in PHYSICAL},'allocation_device_uuids')
        inventory={}
        for row in allocation.get('physical_gpu_inventory',[]):
            need(type(row) is list and len(row)>=3,'allocation_inventory_row')
            index=int(row[0])
            if index not in PHYSICAL:continue
            need(index not in inventory,'allocation_inventory_duplicate')
            inventory[index]=(str(row[1]).strip(),int(row[2]))
        need(set(inventory)==set(PHYSICAL) and len({v[0] for v in inventory.values()})==6
             and all(uuids[str(i)]==inventory[i][0] and inventory[i][1]>0 for i in PHYSICAL)
             and pipeline.get('assigned_gpu_uuids')==uuids
             and set(allocation.get('actual_cuda_runtime_uuids',[]))=={uuids[str(i)] for i in range(4)},'device_uuid_binding')
        ledger_run,leader,births=trainer_snapshot(ledger,expected,claim)
        updates=ledgers['actual-updates.jsonl'].rows();attempts=ledgers['actual-attempts.jsonl'].rows()
        previous=0;seen_updates=set()
        for update in updates:
            step=update.get('step');attempt=update.get('attempt_id');probe=update.get('probe_actor_binding',{})
            need(update.get('run_id')==expected['run_id'] and integer(step,1) and previous<step<=200
                 and type(attempt) is str and re.fullmatch('[0-9a-f]{32}',attempt) and attempt not in seen_updates,
                 'actual_update_identity_or_order')
            need(probe.get('run_id')==expected['run_id'] and probe.get('execution_id')==expected['execution_id']
                 and probe.get('logical_update')==step and probe.get('completed_outer')==step-1
                 and probe.get('attempt_id')==attempt and probe.get('attempt_index')==update.get('attempt_index'),
                 'returned_update_actor_identity')
            previous=step;seen_updates.add(attempt)
        by_attempt={};previous_attempt=0
        for attempt in attempts:
            aid=attempt.get('attempt_id')
            need(same_execution(attempt,expected) and type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid)
                 and aid not in by_attempt and integer(attempt.get('attempt_index'),1)
                 and attempt['attempt_index']>previous_attempt,'attempt_identity_or_order')
            previous_attempt=attempt['attempt_index'];by_attempt[aid]=attempt
        schedules={};rank_births=set();rank_records={};budget_sha=digest(canonical(run['budget']))
        for rank in range(4):
            rows=ledgers['scheduler-rank-%d.jsonl'%rank].rows();birth=None;previous_time=0
            for row in rows:
                current=(row.get('process_pid'),row.get('process_start_ticks'))
                need(same_execution(row,expected) and row.get('source_commit')==expected['source_commit']
                     and row.get('source_map_sha256')==expected['source_map_sha256']
                     and row.get('scheduler_contract_sha256')==budget_sha and row.get('hostname')==host
                     and row.get('schema_version')==1 and row.get('role')=='fresh200_outer_scheduler'
                     and type(row.get('rank')) is int and row['rank']==rank and row.get('world_size')==4
                     and all(integer(v,1) for v in current) and current in births
                     and (birth is None or birth==current) and number(row.get('observed_unix'))
                     and row['observed_unix']>=previous_time,'scheduler_rank_birth_or_identity')
                birth=current;previous_time=row['observed_unix']
            need(birth is not None and birth not in rank_births,'scheduler_missing_or_duplicate_rank_birth')
            rank_births.add(birth);rank_records[rank]=births[birth];schedules[rank]=rows
        samples=gpu_file.rows();previous_end=None
        for sample in samples:
            start=sample.get('query_started_unix');end=sample.get('unix');devices=sample.get('gpus')
            need(same_execution(sample,expected) and sample.get('schema_version')==1
                 and sample.get('role')=='actual_common8_gpu_occupancy'
                 and sample.get('source_map_sha256')==expected['source_map_sha256']
                 and sample.get('sampler_identity')==producer and sample.get('hostname')==host
                 and sample.get('training_physical_indices')==[0,1,2,3]
                 and sample.get('auxiliary_physical_indices')==[5,6]
                 and number(start) and number(end) and 0<start<=end
                 and (previous_end is None or previous_end<=start),'sample_identity_or_order')
            need(type(devices) is list and len(devices)==6 and all(type(d) is dict for d in devices)
                 and all(type(d.get('index')) is int for d in devices)
                 and sorted(d['index'] for d in devices)==PHYSICAL,'sample_physical_indices')
            for device in devices:
                index=device['index']
                need((device.get('uuid'),device.get('total_mib'))==inventory[index]
                     and all(integer(device.get(k)) for k in ('used_mib','total_mib','util_percent'))
                     and device['used_mib']<=device['total_mib'] and device['util_percent']<=100,'sample_device_uuid_or_scalar')
            previous_end=end
        for update in updates[-payload['last_windows']:]:
            step=update['step'];aid=update['attempt_id'];window=unavailable(step,aid,'completed_attempt_missing')
            try:
                attempt=by_attempt.get(aid);need(type(attempt) is dict,'completed_attempt_not_in_scan')
                before=update.get('optimizer_writes_before');after=update.get('optimizer_writes_after')
                writes=update.get('actual_optimizer_writes');metrics=update.get('metrics',{})
                need(integer(before) and integer(after) and integer(writes,1) and writes<=20 and after-before==writes,
                     'returned_update_write_counters')
                need(attempt.get('logical_update')==step and attempt.get('completed_outer')==step
                     and attempt.get('status')=='complete' and attempt.get('complete_outer') is True
                     and attempt.get('all_ranks_agreed') is True and attempt.get('real_rank_count')==4
                     and attempt.get('scheduler_advances')==1 and attempt.get('opportunities_processed')==20
                     and update.get('opportunities_processed')==20
                     and attempt.get('attempt_index')==update.get('attempt_index')
                     and attempt.get('optimizer_writes_before')==before and attempt.get('optimizer_writes_after')==after
                     and attempt.get('optimizer_writes_this_attempt')==writes
                     and metrics.get('actor/optimizer_steps_this_update')==writes
                     and metrics.get('actor/update_opportunities_this_attempt')==20
                     and metrics.get('actor/skipped_update_opportunities')==20-writes,'driver_returned_completed_attempt')
                begins=[];completes=[];rank_windows=[]
                for rank in range(4):
                    selected=[row for row in schedules[rank] if row.get('attempt_id')==aid]
                    need([row.get('kind') for row in selected]==['BEGIN','COMMIT_PREPARED','COMPLETE'],
                         'scheduler_attempt_begin_not_covered_or_incomplete')
                    need(all(row.get('logical_update')==step and row.get('attempt_index')==attempt['attempt_index']
                         and row.get('writes_before')==before and row.get('completed_outer_before')==step-1
                         for row in selected),'four_rank_same_attempt')
                    first,last=selected[0],selected[-1]
                    need(last.get('completed_outer_after')==step and last.get('scheduler_advances')==1
                         and last.get('opportunities_processed')==20 and last.get('writes_after')==after
                         and last.get('actual_optimizer_writes')==writes
                         and last['observed_unix']>=first['observed_unix'],'four_rank_completion_counters')
                    begins.append(first['observed_unix']);completes.append(last['observed_unix'])
                    rank_windows.append(dict(rank=rank,process_pid=first['process_pid'],
                        process_start_ticks=first['process_start_ticks'],begin_unix=begins[-1],complete_unix=completes[-1]))
                begin=min(begins);complete=max(completes)
                window.update(attempt_index=attempt['attempt_index'],begin_unix=begin,complete_unix=complete,
                    rank_windows=rank_windows,actual_optimizer_writes=writes,opportunities_processed=20,
                    completed_attempt_evidence='returned actual-updates + complete/all_ranks_agreed actual-attempts + same four-rank scheduler attempt')
                need(claim['claimed_unix']<=begin<=complete,'window_before_training_claim')
                need(samples and (gpu_file.start==0 or samples[0]['query_started_unix']<=begin),
                     'gpu_tail_does_not_cover_window_begin')
                selected=[row for row in samples if row['query_started_unix']>=begin and row['unix']<=complete]
                window['sample_count']=len(selected);need(selected,'no_existing_sample_entirely_inside_window')
                peaks={index:max(d['used_mib'] for row in selected for d in row['gpus'] if d['index']==index) for index in PHYSICAL}
                gaps=[selected[i]['unix']-selected[i-1]['unix'] for i in range(1,len(selected))]
                window.update(status='observed',reason=None,
                    physical_sampled_peak_mib=[dict(physical_index=index,uuid=uuids[str(index)],
                        assignment='training' if index<4 else 'auxiliary',sampled_peak_mib=peaks[index]) for index in PHYSICAL],
                    maximum_sample_gap_seconds=max(gaps) if gaps else None,
                    maximum_query_duration_seconds=max(row['unix']-row['query_started_unix'] for row in selected),
                    begin_uncovered_seconds=selected[0]['query_started_unix']-begin,
                    complete_uncovered_seconds=complete-selected[-1]['unix'],
                    first_included_query_started_unix=selected[0]['query_started_unix'],
                    last_included_sample_unix=selected[-1]['unix'])
            except RemoteUnknown as error:window['reason']=str(error)
            out['windows'].append(window)
        # Immutable JSON must retain exact size/inode/bytes. JSONL may append
        # only beyond its fixed boundary; both recheck path and FD around hashing.
        for file in files:
            if file.mode!='mutable_json':file.verify()
        # OwnedProcesses publishes this JSON through normal atomic replacement.
        # Capture a fresh bounded document instead of insisting on its old inode
        # or on unrelated child registrations/job-state fields staying unchanged.
        process_after=Snapshot(directory,'process-ledger.json',SMALL_LIMIT,'mutable_json')
        process_after.meta['snapshot_phase']='after_window_scan'
        ledger_after=process_after.document()
        after_run,after_leader,after_births=trainer_snapshot(ledger_after,expected,claim)
        need(after_run==ledger_run and after_leader==leader,'mutable_ledger_run_or_trainer_leader_changed')
        for rank,record in rank_records.items():
            need(after_births.get((record['pid'],record['start_ticks']))==record,
                 'mutable_ledger_selected_rank_no_longer_registered_or_identity_changed')
        process_before.verify_mutable_fd()
        process_after.verify_mutable_fd()
        process_before.meta['semantic_identity_verified']=True
        process_after.meta['semantic_identity_verified']=True
        out['process_ledger_semantic_check']=dict(status='verified',run_identity=ledger_run,
            trainer_leader=leader,selected_rank_identities=[dict(rank=rank,**rank_records[rank]) for rank in range(4)],
            before_snapshot=process_before.meta,after_snapshot=process_after.meta,
            path_inode_changed=(process_before.before.st_dev,process_before.before.st_ino)
                !=(process_after.before.st_dev,process_after.before.st_ino),
            byte_size_changed=process_before.end!=process_after.end,
            raw_document_hash_changed=process_before.meta['scanned_range_sha256']
                !=process_after.meta['scanned_range_sha256'],
            normal_path_replacement_and_unrelated_job_state_or_child_additions_permitted=True,
            all_four_rank_birth_session_uid_records_still_registered=True,process_liveness_verified=False)
        out.update(hostname=host,sampler_identity=producer,assigned_gpu_uuids=uuids,
            status='observed' if out['windows'] and all(w['status']=='observed' for w in out['windows'])
                else 'partial' if any(w['status']=='observed' for w in out['windows']) else 'unavailable',
            reason=None if out['windows'] else 'no_completed_update_in_scanned_range')
    except Exception as error:
        out.update(status='unavailable',reason=str(error) if type(error) is RemoteUnknown else 'metadata_read_failed',
            failure_type=type(error).__name__)
        for window in out['windows']:
            window.update(status='unavailable',reason='extension_snapshot_not_verified',physical_sampled_peak_mib=None)
    finally:
        out.update(files=[file.meta for file in files],aggregate_bytes_read=spent,snapshot_finished_unix=time.time())
        for file in files:os.close(file.fd)
        if directory is not None:os.close(directory)
    return out
class LimitedCapture(io.StringIO):
    def __init__(self):super().__init__();self.bytes_written=0
    def write(self,value):
        self.bytes_written+=len(value.encode());need(self.bytes_written<=512*1024,'legacy_output_limit')
        return super().write(value)
def run_observation(payload,legacy_code):
    observed=None;observation_error=None
    original_stdin=sys.stdin
    try:
        check();capture=LimitedCapture();sys.stdin=io.StringIO(json.dumps(payload['intent'],allow_nan=False))
        with contextlib.redirect_stdout(capture):exec(legacy_code,{'__name__':'__main__'})
        values=[json.loads(line.split(':',1)[1]) for line in capture.getvalue().splitlines()
                if line.startswith('DEPLOYMENT_JSON:')]
        need(len(values)==1 and type(values[0]) is dict,'legacy_single_observation')
        observed=values[0]
        need(all(observed.get(k)==payload['expected'][k] for k in ('base','deployment_id','source_commit','run_id','execution_id')),
             'legacy_current_identity')
    except Exception as error:
        observation_error=dict(reason=str(error) if type(error) is RemoteUnknown else 'existing_observation_failed',
            failure_type=type(error).__name__)
    finally:sys.stdin=original_stdin
    # Preserve the established observation first; telemetry exhaustion never
    # overwrites its independent pipeline/trainer state with a terminal claim.
    windows=extension(payload);signal.alarm(0)
    out=dict(role='existing_actor_probe_bounded_window_observation',schema_version=1,
        status='observed' if observation_error is None else 'unknown',observation=observed,
        observation_error=observation_error,outer_gpu_windows=windows,checked_unix=time.time(),
        read_once=True,transport_attempts=1,no_automatic_retry=True,no_remote_mutation=True,
        launch_allowed=False,new_nvml_queries=0,training_acceptance=False,
        legacy_scope='unchanged existing deploy REMOTE_OBSERVE; existing per-file metadata/ledger limits, no aggregate-byte instrumentation',
        shared_remote_deadline_seconds=105)
    raw=json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)+1>512*1024:
        raw=json.dumps(dict(role=out['role'],schema_version=1,status='unknown',reason='remote_output_limit',
            read_once=True,transport_attempts=1,no_automatic_retry=True,no_remote_mutation=True,
            launch_allowed=False,new_nvml_queries=0,training_acceptance=False)).encode()
    sys.stdout.buffer.write(raw+b'\n');sys.stdout.buffer.flush()
'''


def load_deploy(root, intent):
    source = root / "actor-probes-ten" / "reproduction" / "a800_common" / "deploy.py"
    need(source.resolve() == source, "immutable_route_symlink")
    raw = source.read_bytes()
    need(len(raw) <= LOCAL_FILE_LIMIT and hashlib.sha256(raw).hexdigest()
         == intent["source_sha256"].get("reproduction/a800_common/deploy.py"), "immutable_route_source_hash")
    # Direct file import avoids the mutable ROOT repo, package verifiers, and
    # any git operation (including partial-clone lazy fetch). deploy is stdlib.
    spec = importlib.util.spec_from_file_location("a800_window_immutable_deploy", source)
    module = importlib.util.module_from_spec(spec)
    original = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    return module, hashlib.sha256(raw).hexdigest()


def write_receipt(path, value):
    raw = canonical(value) + b"\n"
    need(len(raw) <= OUTPUT_LIMIT + 32768, "local_receipt_limit")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--handle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--identity-snapshot", type=Path,
                        help="Small current identity document; defaults to preparation/actor-probes-ten-handoff.json")
    parser.add_argument("--last-windows", type=int, default=5)
    parser.add_argument("--not-before", type=float,
                        help="Local Unix-time wait only; never an early query or a trainer continuation")
    args = parser.parse_args()
    root = args.root.absolute()
    handle = args.handle.absolute()
    target = args.receipt.absolute()
    identity_path = (args.identity_snapshot or root / "preparation" / "actor-probes-ten-handoff.json").absolute()
    receipt = dict(role="single_existing_actor_probe_window_receipt", schema_version=1, status="unknown",
                   created_unix=time.time(), remote_query_performed=False, transport_attempts=0,
                   no_automatic_retry=True, no_remote_mutation=True, launch_allowed=False,
                   training_acceptance=False, not_before_unix=args.not_before)
    exit_code = 1
    try:
        need(root.resolve() == root and root.is_dir(), "local_root_directory")
        handle.relative_to(root / "preparation")
        identity_path.relative_to(root / "preparation")
        need(target.parent in (root / "preparation", root / "progress-checks")
             and target.parent.resolve() == target.parent and target.parent.is_dir()
             and not target.exists() and not target.is_symlink(), "fresh_small_receipt_path")
        need(1 <= args.last_windows <= 10, "last_windows_range")
        need(args.not_before is None or math.isfinite(args.not_before), "not_before_finite")
        intent, handle_sha = local_json(handle)
        validate_handle(intent)
        snapshot, snapshot_sha = local_json(identity_path)
        expected = binding(intent, snapshot, handle)
        receipt.update(expected, handle=str(handle), handle_sha256=handle_sha,
                       identity_snapshot=str(identity_path), identity_snapshot_sha256=snapshot_sha,
                       last_requested_windows=args.last_windows)
        if args.not_before is not None:
            print(json.dumps(dict(role="local_single_window_observer_waiting", local_pid=os.getpid(),
                not_before_unix=args.not_before, remote_query_performed=False, **expected)), flush=True)
            while time.time() < args.not_before:
                time.sleep(min(30, max(0, args.not_before - time.time())))
        current_intent, current_handle_sha = local_json(handle)
        current_snapshot, current_snapshot_sha = local_json(identity_path)
        need(current_intent == intent and current_handle_sha == handle_sha, "handle_changed_during_wait")
        need(binding(intent, current_snapshot, handle) == expected, "current_identity_changed_during_wait")
        receipt["identity_snapshot_rechecked_sha256"] = current_snapshot_sha
        deploy, route_sha = load_deploy(root, intent)
        remote = REMOTE_EXTENSION + "\nlegacy_code=" + repr(deploy.REMOTE_OBSERVE) + \
            "\npayload=json.loads(sys.stdin.buffer.read(2*1024*1024+1))\nrun_observation(payload,legacy_code)\n"
        payload = canonical(dict(intent=intent, expected=expected, last_windows=args.last_windows))
        original_bytecode_policy = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True  # The existing control.py file import also stays read-only.
            command = deploy.ssh_command(remote)
        finally:
            sys.dont_write_bytecode = original_bytecode_policy
        if args.not_before is not None:
            need(time.time() >= args.not_before, "query_not_before")
        receipt.update(route_source_sha256=route_sha, remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),
                       transport_timeout_seconds=TRANSPORT_TIMEOUT, transport_attempts=1,
                       remote_query_performed=True, transport_started_unix=time.time())
        raw, returncode, stderr_bytes = bounded_command(command, payload)
        receipt.update(transport_exit_code=returncode, transport_stderr_bytes=stderr_bytes)
        need(returncode == 0, "existing_metadata_transport_failed")
        value = json.loads(raw)
        need(type(value) is dict and value.get("role") == ROLE and value.get("read_once") is True
             and value.get("transport_attempts") == 1 and value.get("no_remote_mutation") is True
             and value.get("launch_allowed") is False and value.get("new_nvml_queries") == 0,
             "remote_observation_schema_or_scope")
        need(value.get("status") in ("observed", "unknown"), "remote_observation_status")
        receipt.update(status=value["status"], result=value)
        # Window unavailability is telemetry, never a new trainer stop/acceptance
        # criterion. Preserve the original observation's independently stated state.
        exit_code = 0 if value["status"] == "observed" else 1
    except Exception as error:
        receipt.update(status="unknown", error=dict(reason=str(error) if type(error) is MetadataUnknown
            else "metadata_observation_failed", failure_type=type(error).__name__))
    receipt["completed_unix"] = time.time()
    try:
        need(target.parent in (root / "preparation", root / "progress-checks")
             and target.parent.resolve() == target.parent, "local_receipt_directory")
        digest = write_receipt(target, receipt)
        result = receipt.get("result", {})
        print(json.dumps(dict(receipt=str(target), sha256=digest, status=receipt["status"],
            window_status=result.get("outer_gpu_windows", {}).get("status"),
            remote_query_performed=receipt["remote_query_performed"],
            transport_attempts=receipt["transport_attempts"], no_automatic_retry=True,
            no_remote_mutation=True, launch_allowed=False, training_acceptance=False)), flush=True)
    except Exception as error:
        print(json.dumps(dict(status="unknown", receipt=None, reason="local_receipt_write_failed",
            failure_type=type(error).__name__, remote_query_performed=receipt["remote_query_performed"],
            transport_attempts=receipt["transport_attempts"], no_automatic_retry=True,
            no_remote_mutation=True, launch_allowed=False)), flush=True)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
