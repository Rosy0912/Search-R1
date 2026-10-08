"""Prepare a closed, read-only v5 scalar packet; never choose or admit a policy.

Default CLI only verifies explicit original local inputs and compiles the RPC.
``--execute`` is a separate future human/root action, requiring a SHA-bound
actual COMPLETED/0:0 terminal. Its single RPC reads small original metadata on
the recorded node, streams the native segment into a closed scalar projection,
and never sends model/Adam tensors, training text, or a private token journal.
Independent native checkpoint/Adam/raw2560 auditing remains pending.
"""
import argparse
import ast
import base64
import codecs
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CONSTRUCTOR_SHA256 = '1b0cfaa85d88c1b355fdf7aec5a1735a9de7acfb11888dbada395774f56a5093'
READER_SHA256 = 'f10c3371a446a1816840e89ebb60aa3716a8e61157936733cb2b6a9f514297fd'
MAX_FILE_BYTES = 4 * 1024**2
MAX_PACKET_BYTES = 64 * 1024**2
MAX_SEGMENT_BYTES = 64 * 1024**2
MAX_STRING = 16384
SKIPPED_SEGMENT_FIELDS = frozenset(('actual_optimizer_states', 'actual_worker_births_at_segment_close',
    'optimizer_state_proof_sha256', 'actual_training_inputs', 'actual_training_groups',
    'training_input_proof_sha256', 'checked_unix', 'semantics'))
FORBIDDEN_KEYS = frozenset(('question', 'raw_question', 'raw_answer', 'answer', 'prompt',
    'input_ids', 'token_ids', 'raw_generated_token_ids', 'actual_request_token_ids',
    'selected_logprobs', 'raw_trace', 'actor_rows', 'responses', 'state_dict',
    'optimizer_state', 'optimizer_states', 'exp_avg', 'exp_avg_sq', 'raw_training_text'))
MANIFEST_FIELDS = frozenset(('policy_safety_required', 'profile', 'run_id', 'started_unix',
    'completed_updates', 'initial_model', 'reference_model', 'initial_model_sha256',
    'model_revision', 'budget', 'source_commit', 'source_sha256', 'source_snapshot_semantics',
    'input_sha256', 'calibration_question_ids', 'target_updates', 'save_interval',
    'acceptance_update', 'initial_probe_version', 'checkpoint_dir', 'train_log',
    'optimizer_semantics', 'retrieval_placement', 'probe_visible_device', 'probe_logical_ordinal',
    'policy_calibration_discovery_base', 'policy_calibration_confirmation_base',
    'parent_production_run_id', 'policy_calibration_role', 'policy_calibration_parent_manifest_sha256',
    'policy_calibration_slurm_job_id', 'policy_calibration_plan_sha256'))
PLAN_FIELDS = frozenset(('schema_version','role','run_id','parent_production_run_id',
    'slurm_job_id','resolved_config_sha256','initial_reward_fingerprint','identity',
    'initialization','mathematics','phase_budget','initial_probability_gate',
    'candidate_thresholds','threshold_policy','health_rule','discovery'))
EVALUATION_FIELDS = frozenset(('role','question_count','source_counts','question_ids','data_sources','rows',
    'weighted_em','weighted_f1','executed_search_question_rate','executed_search_calls_per_question',
    'executed_search_question_count','executed_search_call_count','valid_action_rate',
    'valid_action_count','action_opportunity_count','valid_action_rate_definition',
    'executed_search_definition','production_admission','semantics','run_id',
    'parent_production_run_id','slurm_job_id','plan_sha256','step','phase_role','validation_sha256','checked_unix'))
CLAIM_FIELDS = frozenset(('run_id','slurm_job_id','uid','root','pid','start_ticks','phase',
    'created_unix','files','root_device','root_inode','cleanup_file_sizes','parent_device',
    'parent_inode','creator','capacity_admission','corpus_extraction','removed_file_sizes'))


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def metadata_tree(value, depth=0):
    """Bounded scalar/hash trees only; never arbitrary question/token/tensor data."""
    require(depth <= 32, 'Metadata nesting exceeds closed scalar bound')
    if type(value) is dict:
        require(len(value) <= 4096, 'Oversized metadata mapping')
        for key, item in value.items():
            hash_only = key in ('input_ids', 'attention_mask', 'position_ids', 'responses') and isinstance(item,str) and re.fullmatch('[0-9a-f]{64}',item)
            require(isinstance(key, str) and len(key) <= 512 and (key not in FORBIDDEN_KEYS or hash_only),
                    'Forbidden raw/text/tensor metadata key: ' + str(key))
            metadata_tree(item, depth + 1)
    elif type(value) is list:
        require(len(value) <= 4096, 'Oversized metadata sequence')
        for item in value: metadata_tree(item, depth + 1)
    elif type(value) is str: require(len(value) <= MAX_STRING, 'Oversized metadata string')
    else: require(value is None or type(value) is bool or
                  (type(value) in (int, float) and math.isfinite(value)), 'Not finite scalar metadata')


def stable_identity(st):
    return (st.st_dev, st.st_ino, st.st_uid, st.st_mode, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def open_owned(path, maximum):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            'Metadata path is not canonical or follows symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
                before.st_nlink == 1 and 0 < before.st_size <= maximum,
                'Not owned regular bounded metadata')
        require(stable_identity(before) == stable_identity(os.lstat(path)), 'Opened metadata inode changed')
        return fd, before
    except BaseException:
        os.close(fd); raise


def finish_read(path, fd, before, count, digest, *, started):
    require(count == before.st_size and stable_identity(before) == stable_identity(os.fstat(fd)) ==
            stable_identity(os.lstat(path)) and Path(path).resolve() == Path(path),
            'Original metadata bytes/inode changed during readonly capture')
    return dict(path=str(path), size_bytes=count, sha256=digest, uid=before.st_uid,
        device=before.st_dev, inode=before.st_ino, mtime_unix=before.st_mtime,
        mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns,
        read_started_unix=started, read_finished_unix=time.time())


def read_owned(path, maximum=MAX_FILE_BYTES):
    started=time.time(); fd, before=open_owned(path, maximum)
    try:
        with os.fdopen(os.dup(fd), 'rb') as stream: raw=stream.read(maximum + 1)
        receipt=finish_read(path,fd,before,len(raw),sha(raw),started=started)
        return raw,receipt
    finally: os.close(fd)


class SegmentProjection:
    """Streaming strict JSON, discarding excluded values without materializing them."""
    def __init__(self, stream, fields):
        self.stream=stream; self.fields=fields; self.buffer=''; self.index=0; self.ended=False
        self.decoder=codecs.getincrementaldecoder('utf-8')('strict'); self.digest=hashlib.sha256()
        self.count=0; self.characters=0; self.retained=0

    def peek(self):
        if self.index >= len(self.buffer) and not self.ended:
            raw=self.stream.read(65536)
            self.digest.update(raw); self.count+=len(raw)
            require(self.count <= MAX_SEGMENT_BYTES, 'Segment exceeds native projection bound')
            self.buffer=self.decoder.decode(raw,final=not raw); self.index=0; self.ended=not raw
            if not self.buffer and not self.ended: return self.peek()
        return self.buffer[self.index] if self.index < len(self.buffer) else ''

    def take(self):
        character=self.peek(); require(bool(character),'Truncated original segment JSON')
        self.index+=1; self.characters+=1; return character

    def space(self):
        while self.peek() and self.peek() in ' \t\r\n': self.take()

    def string(self, keep):
        require(self.take() == '"','Expected JSON string'); chunks=['"'] if keep else None
        escaped=False; digits=0; length=0
        while True:
            character=self.take(); length+=1
            if keep:
                chunks.append(character); require(length <= MAX_FILE_BYTES,'Oversized retained segment string')
            if digits:
                require(character in '0123456789abcdefABCDEF','Invalid JSON Unicode escape'); digits-=1; continue
            if escaped:
                require(character in '"\\/bfnrtu','Invalid JSON escape')
                if character == 'u': digits=4
                escaped=False; continue
            if character == '"': break
            require(ord(character)>=32,'JSON string contains control byte')
            escaped=character=='\\'
        return strict_json(''.join(chunks)) if keep else None

    def value(self, keep, depth=0):
        require(depth<=64,'Original segment nesting exceeds bounded parser'); self.space(); first=self.peek()
        require(bool(first),'Missing segment JSON value')
        if first=='"': return self.string(keep)
        if first in '{[':
            opener=self.take(); closing='}' if opener=='{' else ']'; result={} if opener=='{' else []
            keys=set(); count=0; self.space()
            if self.peek()==closing: self.take(); return result if keep else None
            while True:
                if opener=='{':
                    self.space(); key=self.string(True)
                    require(len(key)<=512 and key not in keys,'Duplicate/oversized segment nested key'); keys.add(key)
                    self.space(); require(self.take()==':','Missing JSON colon')
                item=self.value(keep,depth+1); count+=1
                require(count<=100000,'Oversized excluded/retained segment collection')
                if keep:
                    if opener=='{': result[key]=item
                    else: result.append(item)
                self.space(); separator=self.take()
                if separator==closing: break
                require(separator==',','Invalid JSON collection separator')
            return result if keep else None
        token=[]
        while self.peek() and self.peek() not in ' \t\r\n,]}':
            token.append(self.take()); require(len(token)<=128,'Oversized JSON scalar token')
        value=strict_json(''.join(token))
        require(value is None or type(value) in (bool,int,float),'Invalid segment scalar')
        return value if keep else None

    def project(self):
        self.space(); require(self.take()=='{','Native segment must be an object'); output={}; seen=set()
        allowed=self.fields|SKIPPED_SEGMENT_FIELDS; self.space()
        require(self.peek()!='}','Empty original segment')
        while True:
            self.space(); key=self.string(True)
            require(key in allowed and key not in seen,'Unknown/duplicate original segment field: '+key); seen.add(key)
            self.space(); require(self.take()==':','Missing original field colon')
            start=self.characters; value=self.value(key in self.fields)
            if key in self.fields:
                require(self.characters-start<=MAX_FILE_BYTES,'Retained segment field exceeds scalar bound')
                output[key]=value
            self.space(); separator=self.take()
            if separator=='}': break
            require(separator==',','Invalid original segment separator')
        self.space(); require(not self.peek(),'Trailing content after original segment JSON')
        require(seen==allowed and set(output)==self.fields,'Missing original segment fields')
        metadata_tree(output)
        return output


def project_segment(path, fields):
    started=time.time(); fd,before=open_owned(path,MAX_SEGMENT_BYTES)
    try:
        with os.fdopen(os.dup(fd),'rb') as stream:
            parser=SegmentProjection(stream,fields); value=parser.project()
        receipt=finish_read(path,fd,before,parser.count,parser.digest.hexdigest(),started=started)
        return value,receipt
    finally: os.close(fd)


def whitelist_metadata(name, raw, small):
    """File names and top-level payload schema are both closed before transport."""
    require(name in small.METADATA_FILES or re.fullmatch(
        r'discovery/repo/outputs/\d{4}-\d\d-\d\d/\d\d-\d\d-\d\d/\.hydra/(config|hydra|overrides)\.yaml',name),
        'Not a whitelisted original metadata path')
    if name.endswith('.yaml'):
        require(len(raw)<=131072,'Generated YAML exceeds original small bound'); return
    rows=[strict_json(line) for line in raw.splitlines()] if name.endswith('.jsonl') else [strict_json(raw)]
    for value in rows: metadata_tree(value)
    if '/optimizer-rank-' in name:
        require(len(rows)==61,'Need exactly61 original scalar worker lines'); return
    value=rows[0]; require(type(value) is dict,'Original metadata must be a dictionary')
    base=Path(name).name
    if base in ('run-manifest.json','parent-run-manifest.json'):
        require(set(value)<=MANIFEST_FIELDS,'Unknown manifest payload field')
    elif base=='policy-safety-calibration-plan.json': require(set(value)==PLAN_FIELDS,'Unknown plan field')
    elif base.startswith('policy-calibration-evaluation-'):
        require(set(value)==EVALUATION_FIELDS,'Unknown evaluation/raw text field')
    elif base=='process-ledger.json': require(set(value)=={'run_id','jobs'},'Unknown process ledger field')
    elif base=='pipeline-status.json': require(set(value)=={'run_id','stage','unix'},'Unknown terminal pipeline payload')
    elif base=='cleanup-status.json': require(set(value)=={'run_id','cleaned','checked_unix'},'Unknown cleanup payload')
    elif base in ('retrieval-staging-claim.json','corpus-cache-claim.json'):
        require(set(value)<=CLAIM_FIELDS,'Unknown resource claim field')
    elif base=='training-launch-claim.json':
        require(set(value)=={'run_id','slurm_job_id','shell_pid','shell_start_ticks','claimed_unix','semantics'},'Unknown trainer claim field')
    elif base=='resolved-training-config.json':
        require(set(value)=={'run_id','slurm_job_id','checked_unix','recipe_sha256','resolved_config_sha256',
            'resolved_config','actual_argv_overrides','semantics'},'Unknown resolved config proof field')
    elif base=='checkpoint-complete.json':
        require(set(value)=={'run_id','step','validated_tensor_count','finite_all_saved_tensors','checkpoint','files_sha256'},
                'Unknown checkpoint scalar receipt field')
    elif base=='actual-updates.jsonl':
        require(len(rows)==1 and set(value)=={'run_id','step','probe_version','metrics',
            'behavior_probability_gate_sha256','behavior_trace_sha256'},'Unknown update scalar field')
    elif base=='behavior-probability-outer-1.json': pass  # exact gate schema checked by unchanged small helper
    else: raise ValueError('Unhandled closed metadata schema: '+name)


def load_modules():
    path=HERE/'prepare_policy_confirmation_v5_selection.py'
    raw,_=read_owned(path)
    require(sha(raw)==CONSTRUCTOR_SHA256,'Frozen scalar constructor bytes changed')
    spec=importlib.util.spec_from_file_location('_collector_v5_small',path)
    small=importlib.util.module_from_spec(spec); spec.loader.exec_module(small)
    reader=small.confirmation_reader(HERE/'submit_policy_confirmation_v5_native.py',READER_SHA256,small.METHOD_SHA256)
    return small,reader


def input_contract(artifacts, expected_cpu_tests):
    small,reader=load_modules()
    require(type(artifacts) is dict and set(artifacts)=={'method','preregistration','cpu','launch','birth','terminal'},
            'All six explicit original input paths/SHA are required')
    originals={}; receipts={}
    fixed=dict(method=small.METHOD_SHA256,preregistration=small.PREREGISTRATION_SHA256,
               cpu=small.CPU_PROOF_SHA256,launch=small.LAUNCH_SHA256,birth=small.BIRTH_SHA256)
    for label,item in artifacts.items():
        require(type(item) is dict and set(item)=={'path','sha256'} and small.digest(item['sha256']),
                'Missing explicit original path/SHA: '+label)
        raw,receipt=read_owned(Path(item['path']),32*1024**2 if label=='cpu' else MAX_FILE_BYTES)
        require(sha(raw)==item['sha256'] and (label not in fixed or sha(raw)==fixed[label]),
                'Original fixed input SHA differs: '+label)
        originals[label]=raw; receipts[label]=receipt
    require(type(expected_cpu_tests) is int and expected_cpu_tests==small.ACTUAL_CPU_TESTS,
            'Explicit actual239 target CPU count is required')
    method=small.strict_json(originals['method']); launch=reader.prospective_binding(method,originals['launch'],originals['birth'])
    require(launch['source_commit']==small.SOURCE_COMMIT and sha(canonical(launch['source_sha256']))==small.SOURCE_MAP_SHA256,
            'Approved actual runtime353 contract differs')
    reader.preregistration_binding(originals['preregistration'],method,launch)
    reader.cpu_contract(small.strict_json(originals['cpu']),launch)
    terminal=reader.read_terminal_observation(originals['terminal'],expected_sha256=artifacts['terminal']['sha256'],
        launch_bytes=originals['launch'],expected_launch_sha256=small.LAUNCH_SHA256,
        expected_run_id=method['discovery_run_id'],expected_parent_run_id=method['parent_production_run_id'],
        expected_state='COMPLETED',expected_exit_code='0:0',birth_observation_bytes=originals['birth'],expected_birth_sha256=small.BIRTH_SHA256)
    return small,reader,originals,receipts,method,launch,terminal


def source_map_before_import(parent, discovery, mapping):
    """Authenticates exact source353 and discovers only three native YAML extras."""
    require(type(mapping) is dict and len(mapping)==353,'Missing original source353')
    extras=[]
    for base in (parent,discovery):
        repo=base/'repo'; require(repo.resolve()==repo and repo.is_dir(),'Native source namespace symlink')
        actual={}
        for path in repo.rglob('*'):
            if {'.git','__pycache__','evidence'}.intersection(path.relative_to(repo).parts): continue
            require(not path.is_symlink(),'Native source tree contains symlink')
            relative=str(path.relative_to(repo))
            if path.is_file() and (path.suffix in ('.py','.sh','.yaml','.yml','.json') or
                relative in ('verl/version/version','verl/single_controller/version/version')): actual[relative]=path
        unknown=set(actual)-set(mapping)
        require(set(mapping)<=set(actual),'Missing original source file')
        if base==parent: require(not unknown,'Fresh production source has generated/foreign file')
        else:
            require(len(unknown)==3 and len({str(Path(x).parent) for x in unknown})==1 and
                {Path(x).name for x in unknown}=={'config.yaml','hydra.yaml','overrides.yaml'} and
                all(re.fullmatch(r'outputs/\d{4}-\d\d-\d\d/\d\d-\d\d-\d\d/\.hydra/(config|hydra|overrides)\.yaml',x) for x in unknown),
                'Only three actual same-run Hydra extras are allowed')
            extras=sorted(unknown)
        for name,digest in mapping.items():
            raw,_=read_owned(actual[name]); require(sha(raw)==digest,'Frozen native source changed: '+name)
    return extras


def namespace_empty(discovery, run, checked_after):
    marker=('SEARCH_G1_BASE='+str(discovery)).encode(); active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid==os.getuid() and marker in (proc/'environ').read_bytes().split(b'\0'):
                if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z': active.append(proc.name)
        except (FileNotFoundError,ProcessLookupError): pass
    root=Path('/dev/shm')/('search-g1-common8-'+run['run_id']+'-'+run['policy_calibration_slurm_job_id'])
    present=[str(root)] if os.path.lexists(root) else []
    require(not active and not present,'Actual owned namespace/temporary root remains')
    checked=time.time(); require(checked>=checked_after,'Actual metadata capture clock precedes terminal')
    return dict(run_id=run['run_id'],base=str(discovery),slurm_job_id=run['policy_calibration_slurm_job_id'],
        checked_unix=checked,active_owned_processes=[],temporary_roots_present=[])


def native_validation(discovery, run):
    """Only fixed512 two columns, native CPU; no prompt text is exported."""
    import pyarrow.parquet as pq
    from reproduction.common8.verify_data import MANIFEST_SHA, probe_question_id
    manifest_raw,manifest_receipt=read_owned(discovery/'data/manifest.json')
    require(sha(manifest_raw)==MANIFEST_SHA==run['input_sha256']['manifest.json'],'Canonical shared manifest bytes differ')
    record=strict_json(manifest_raw)['splits']['validation']
    require(record['path']=='validation.parquet' and type(record['rows']) is int and record['rows']==512 and
        record['sha256']==run['input_sha256']['validation.parquet'],'Canonical actual512 product differs')
    path=discovery/'data/validation.parquet'; started=time.time(); fd,before=open_owned(path,64*1024**2)
    try:
        h=hashlib.sha256(); count=0
        with os.fdopen(os.dup(fd),'rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): h.update(block); count+=len(block)
        require(h.hexdigest()==record['sha256'],'Original validation512 bytes differ')
        os.lseek(fd,0,os.SEEK_SET)
        with os.fdopen(os.dup(fd),'rb') as stream:
            table=pq.read_table(stream,columns=['prompt','data_source'])
        require(table.num_rows==512,'Not actual canonical512 rows')
        ids=[probe_question_id(row) for row in table['prompt'].to_pylist()]
        sources=table['data_source'].to_pylist(); require(Counter(sources)=={'nq':256,'hotpotqa':256},'Canonical512 source counts differ')
        receipt=finish_read(path,fd,before,count,h.hexdigest(),started=started)
        return dict(validation_sha256=record['sha256'],question_ids=ids,data_sources=sources),[manifest_receipt,receipt]
    finally: os.close(fd)


def evaluation_arithmetic(raw, segment, run, validation, aggregate):
    """Verify actual512 arithmetic; leave health drops entirely to candidate tool."""
    for label,step in (('before',0),('after',1)):
        content=raw['discovery/logs/policy-calibration-evaluation-'+label+'.json'];value=strict_json(content)
        require(sha(content)==segment['evaluation_sha256'][label] and value==segment['evaluation_'+label],
                'Original paired evaluation/projection SHA differs')
        require(value['run_id']==run['run_id'] and value['parent_production_run_id']==run['parent_production_run_id'] and
            value['slurm_job_id']==run['policy_calibration_slurm_job_id'] and value['plan_sha256']==segment['plan_sha256'] and
            type(value['step']) is int and value['step']==step and value['validation_sha256']==validation['validation_sha256'] and
            value['phase_role']==segment['role'].removesuffix('_evidence'),'Original paired512 identity differs')
        calculated=aggregate(value['rows'],validation['question_ids'],validation['data_sources'])
        require(all(canonical(value[key])==canonical(item) for key,item in calculated.items()),'Actual512 aggregate differs')


def collect_native(payload, small, reader):
    """One actual readonly target pass; no launches, writes, kernels or tensor reads."""
    begun=time.time();launch=payload['launch'];method=payload['method'];terminal=payload['terminal']
    require(begun>=terminal['observation_finished_unix'],'Capture clock precedes original completed observation')
    route=reader.current_completed_terminal(terminal)
    parent,discovery=Path(launch['parent_base']),Path(launch['base'])
    require(parent.resolve()==parent and discovery.resolve()==discovery,'Canonical namespaces required')
    extras=source_map_before_import(parent,discovery,launch['source_sha256'])
    global REPO
    sys.path.insert(0,str(parent/'repo'));small.REPO=parent/'repo'; REPO=parent/'repo'
    from reproduction.common8.freeze_calibration_plan import _unstarted,_mathematics
    raw={}; receipts={}
    # Read manifest first; checkpoint path comes only from that actual manifest.
    for name in ('parent/logs/run-manifest.json','discovery/logs/run-manifest.json'):
        base=parent if name.startswith('parent/') else discovery
        content,receipt=read_owned(base/Path(name).relative_to(name.split('/')[0]));whitelist_metadata(name,content,small)
        raw[name]=content;receipts[name]=receipt
    parent_run=strict_json(raw['parent/logs/run-manifest.json']);run=strict_json(raw['discovery/logs/run-manifest.json'])
    require(parent_run['run_id']==method['parent_production_run_id'] and run['run_id']==method['discovery_run_id'] and
        run['parent_production_run_id']==parent_run['run_id'] and run['policy_calibration_slurm_job_id']==launch['job_id'] and
        run['policy_calibration_role']==reader.DISCOVERY_ROLE and parent_run.get('policy_calibration_role','production')=='production' and
        parent_run['source_commit']==run['source_commit']==launch['source_commit'] and
        parent_run['source_sha256']==run['source_sha256']==launch['source_sha256'],'Original runtime/parent/child identity differs')
    _unstarted(parent,parent_run)
    names=small.METADATA_FILES|{'discovery/repo/'+name for name in extras}
    checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_1/complete.json'
    require(discovery in checkpoint.parents and checkpoint.resolve()==checkpoint,'Original checkpoint scalar path escapes namespace')
    for name in sorted(names-set(raw)):
        path=checkpoint if name=='discovery/checkpoint-complete.json' else discovery/Path(name).relative_to('discovery')
        content,receipt=read_owned(path);whitelist_metadata(name,content,small);raw[name]=content;receipts[name]=receipt
    segment,segment_receipt=project_segment(discovery/'logs/policy-calibration-segment.json',small.SCALAR_SEGMENT_FIELDS)
    original_parent=strict_json(raw['discovery/preparation/parent-run-manifest.json'])
    require(sha(raw['discovery/preparation/parent-run-manifest.json'])==run['policy_calibration_parent_manifest_sha256'],
            'Frozen original parent bytes SHA differs')
    readers=small.pure_readers(launch['source_sha256']);identity=readers.plan._identity(parent_run)
    require(canonical(identity)==canonical(readers.plan._identity(run))==canonical(readers.plan._identity(original_parent)),
            'Source/model/data scientific identity differs')
    plan=strict_json(raw['discovery/logs/policy-safety-calibration-plan.json']);plan_sha=sha(canonical(plan))
    readers.plan.verify_plan(plan,parent_run,expected_plan_sha256=run['policy_calibration_plan_sha256'])
    require(segment['run_id']==run['run_id'] and segment['parent_production_run_id']==parent_run['run_id'] and
        segment['slurm_job_id']==launch['job_id'] and segment['plan_sha256']==plan_sha and
        segment['role']==reader.DISCOVERY_ROLE+'_evidence' and segment['origin']=='actual_native_policy_segment' and
        type(segment['completed_outer_updates']) is int and segment['completed_outer_updates']==1 and
        type(segment['optimizer_writes_per_rank']) is int and segment['optimizer_writes_per_rank']==20 and
        segment['ranks']==list(range(8)) and all(type(rank) is int for rank in segment['ranks']) and
        segment['production_admission'] is False and segment['healthy_checkpoint_accepted'] is False,
        'Incomplete actual first20/40 scalar segment')
    for key in ('identity','initial_reward_fingerprint','resolved_config_sha256'):
        require(canonical(segment[key])==canonical(plan[key]),'Actual segment plan identity differs: '+key)
    cfg=strict_json(raw['discovery/logs/resolved-training-config.json'])
    require(cfg['resolved_config_sha256']==plan['resolved_config_sha256'],'Actual resolved config differs')
    _mathematics(cfg['resolved_config'],discovery,run)
    descriptor=small.hydra_descriptor(raw,{'discovery/repo/'+name for name in extras},run,launch)
    minimal=dict(discovery_run_id=run['run_id'],discovery_slurm_job_id=launch['job_id'],
                 discovery_generated_hydra_snapshots=descriptor)
    snapshot=reader.verify_source_snapshot_sets(parent,discovery,launch,minimal)
    small.check_snapshot_receipt(snapshot,descriptor)
    small.check_ledgers(raw,segment,readers)
    validation,data_receipts=native_validation(discovery,run)
    evaluation_arithmetic(raw,segment,run,validation,readers.aggregate)
    st=receipts['discovery/logs/actual-updates.jsonl']
    updates_stat={key:st[key] for key in ('size_bytes','sha256','mtime_unix')}
    small.check_gate_and_update(raw,segment,run,method,updates_stat,readers.verify_sampling)
    small.check_checkpoint(raw,segment,run,str(discovery))
    packet=dict(schema_version=1,role='policy_confirmation_v5_local_scalar_input',
        segment_sha256=segment_receipt['sha256'],segment=segment,
        files={name:dict(base64=base64.b64encode(content).decode(),sha256=sha(content)) for name,content in raw.items()},
        source_snapshot_verification=snapshot,canonical_validation=validation,
        namespace_cleanup_snapshot=namespace_empty(discovery,run,terminal['checked_unix']),actual_updates_stat=updates_stat)
    small.check_cleanup(raw,packet,run,terminal);small.original_files(packet)
    # Recheck every original FD identity/hash after CPU arithmetic; no mixed-time packet.
    for name,receipt in receipts.items():
        current,current_receipt=read_owned(Path(receipt['path']))
        require(sha(current)==receipt['sha256'] and all(current_receipt[key]==receipt[key] for key in
            ('device','inode','size_bytes','mtime_ns','ctime_ns','uid')),'Metadata changed before capture finished: '+name)
    repeated,repeated_receipt=project_segment(Path(segment_receipt['path']),small.SCALAR_SEGMENT_FIELDS)
    require(canonical(repeated)==canonical(segment) and repeated_receipt['sha256']==segment_receipt['sha256'],
            'Original segment changed before capture finished')
    require(len(canonical(packet))<=MAX_PACKET_BYTES,'Closed scalar packet exceeds bounded output')
    return dict(packet=packet,receipt=dict(role='policy_confirmation_v5_readonly_scalar_capture',
        read_started_unix=begun,read_finished_unix=time.time(),run_id=run['run_id'],slurm_job_id=launch['job_id'],
        source_commit=launch['source_commit'],source_map_sha256=sha(canonical(launch['source_sha256'])),
        original_terminal_sha256=terminal['observation_sha256'],current_terminal_route=route,
        files=receipts,original_segment=segment_receipt,canonical_validation_originals=data_receipts,
        packet_sha256=sha(canonical(packet)),full_checkpoint_verification_pending=True,
        optimizer_tensor_verification_pending=True,raw2560_verification_pending=True,
        original_full_segment_independent_reread_pending=True,production_admission=False,
        healthy_checkpoint_accepted=False,physical_execution_authenticated=False,
        scope='Actual readonly small metadata and arithmetic capture; not healthy or physical tensor/raw authentication'))


def remote_code(small, reader):
    """Embed exact reviewed pure helpers, without importing a remote external tool."""
    reader_names=('canonical','sha','require','strict_json','_digest','_number','_job_fields','_scheduler_time',
        'verify_terminal_current','current_completed_terminal','hydra_descriptor_schema','owned_small_bytes',
        'verify_source_snapshot_sets')
    small_names=('pure_readers','original_files','hydra_descriptor','check_snapshot_receipt','check_ledgers',
        'check_gate_and_update','check_checkpoint','check_cleanup')
    own_names=('require','sha','canonical','strict_json','metadata_tree','stable_identity','open_owned','finish_read',
        'read_owned','SegmentProjection','project_segment','whitelist_metadata','source_map_before_import',
        'namespace_empty','native_validation','evaluation_arithmetic','collect_native')
    prefix='import ast,base64,codecs,datetime,hashlib,json,math,os,re,stat,subprocess,sys,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom collections import Counter\nfrom collections.abc import Mapping,Sequence\n'
    reader_body='\n'.join(inspect.getsource(getattr(reader,name)) for name in reader_names)
    reader_body+='\ncommon=SimpleNamespace(require=require,sha=sha,canonical=canonical,strict_json=strict_json)\n'
    for name in ('INVALID_JOB','BIRTH_FIELDS'): reader_body+=name+'='+repr(getattr(reader,name))+'\n'
    # Keep reviewed helper globals isolated; no accidental overwriting by collector.
    prefix+='reader_ns={};exec('+repr(prefix+reader_body)+',reader_ns);reader=SimpleNamespace(**reader_ns)\n'
    small_body='\n'.join(inspect.getsource(getattr(small,name)) for name in small_names)
    small_body+='\nnumber=lambda value:type(value) in (int,float) and math.isfinite(value)\ndigest=lambda value:isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None\n'
    small_body+='\ndef same(left,right,message):require(canonical(left)==canonical(right),message)\n'
    for name in ('SCALAR_SEGMENT_FIELDS','METADATA_FILES'): small_body+=name+'='+repr(getattr(small,name))+'\n'
    own_source=Path(__file__).read_text();tree=ast.parse(own_source)
    own_body='\n'.join(ast.get_source_segment(own_source,next(node for node in tree.body
        if isinstance(node,ast.ClassDef) and node.name==name)) if inspect.isclass(globals()[name])
        else inspect.getsource(globals()[name]) for name in own_names)
    constants='\n'.join(name+'='+repr(globals()[name]) for name in
        ('MAX_FILE_BYTES','MAX_PACKET_BYTES','MAX_SEGMENT_BYTES','MAX_STRING','SKIPPED_SEGMENT_FIELDS',
         'FORBIDDEN_KEYS','MANIFEST_FIELDS','PLAN_FIELDS','EVALUATION_FIELDS','CLAIM_FIELDS'))
    code=prefix+constants+'\n'+own_body+'\n'+small_body+'\nsmall=SimpleNamespace(**globals())\n'
    code+='p=json.load(sys.stdin);reader.DISCOVERY_ROLE="policy_safety_calibration_discovery"\n'
    code+='result=collect_native(p,small,reader);print(json.dumps(result,ensure_ascii=False,allow_nan=False))\n'
    compile(code,'<readonly-v5-scalar-rpc>','exec');return code


def execute_once(artifacts, expected_cpu_tests, *, output, execute=False, rpc=None):
    require(execute is True,'Readonly collection requires explicit future execute authorization')
    small,reader,originals,input_receipts,method,launch,terminal=input_contract(artifacts,expected_cpu_tests)
    output=Path(output);require(output.is_absolute() and output.parent.resolve()==output.parent and
        not output.exists() and not output.is_symlink(),'Output must be a new local canonical exclusive file')
    code=remote_code(small,reader)
    payload=dict(launch=launch,method=method,terminal=terminal)
    if rpc is None:
        sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
        def rpc(code,payload):
            result=subprocess.run(control.route('h100-8',shlex.quote(str(Path(launch['base'])/'env/bin/python'))+' -B -c '+shlex.quote(code)),
                input=json.dumps(payload,allow_nan=False),capture_output=True,text=True,timeout=55)
            require(result.returncode==0,'Readonly capture failed; original evidence unchanged: '+result.stderr[-2000:])
            require(len(result.stdout.encode())<=MAX_PACKET_BYTES*2,'Remote scalar capture exceeded output bound')
            return strict_json(result.stdout)
    captured=rpc(code,payload)
    require(type(captured) is dict and set(captured)=={'packet','receipt'},'Unknown readonly capture response')
    packet=captured['packet'];receipt=captured['receipt']
    require(set(packet)==small.INPUT_FIELDS and receipt['packet_sha256']==sha(canonical(packet)) and
        receipt['original_terminal_sha256']==terminal['observation_sha256'] and
        receipt['run_id']==method['discovery_run_id'] and receipt['slurm_job_id']==launch['job_id'] and
        all(receipt[key] is False for key in ('production_admission','healthy_checkpoint_accepted','physical_execution_authenticated')) and
        all(receipt[key] is True for key in ('full_checkpoint_verification_pending','optimizer_tensor_verification_pending',
            'raw2560_verification_pending','original_full_segment_independent_reread_pending')),'Capture identity or pending boundaries differ')
    raw,_=small.original_files(packet)
    for name,content in raw.items(): whitelist_metadata(name,content,small)
    metadata_tree(packet['segment'])
    receipt['original_local_inputs']=input_receipts
    # One O_EXCL artifact keeps exact packet + readonly receipt together. The
    # candidate receives canonical(envelope['packet']), never the wrapper SHA.
    return write_capture(output,captured)


def write_capture(output, captured):
    """Only exclusive local archival; this byte writer does not authenticate native facts."""
    output=Path(output);require(output.is_absolute() and output.parent.resolve()==output.parent and
        not output.exists() and not output.is_symlink(),'Output must be a new local canonical exclusive file')
    content=canonical(captured);require(len(content)<=MAX_PACKET_BYTES*2,'Local readonly envelope too large')
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
    finally: pass
    directory=os.open(output.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(output=str(output),envelope_sha256=sha(content),packet_sha256=sha(canonical(captured['packet'])),
        production_admission=False,healthy_checkpoint_accepted=False,physical_execution_authenticated=False,
        full_checkpoint_verification_pending=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('method','preregistration','cpu','launch','birth','terminal'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--expected-cpu-tests',type=int,required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args(); artifacts={name:dict(path=str(getattr(args,name)),sha256=getattr(args,name+'_sha256'))
        for name in ('method','preregistration','cpu','launch','birth','terminal')}
    if args.execute:
        require(args.output is not None,'Explicit future execute also requires new local output path')
        result=execute_once(artifacts,args.expected_cpu_tests,output=args.output,execute=True)
    else:
        small,reader,_,_,_,_,_=input_contract(artifacts,args.expected_cpu_tests)
        remote_code(small,reader)
        result=dict(role='v5_readonly_collector_local_structure_check',production_admission=False,
            healthy_checkpoint_accepted=False,physical_execution_authenticated=False,remote_executed=False)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
