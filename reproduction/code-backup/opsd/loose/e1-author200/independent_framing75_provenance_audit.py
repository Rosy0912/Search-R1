"""Local-only independently join additional physical warning source and stderr."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import runpy

B = Path(__file__).resolve().parent
RAW = B/'raw-log-framing75-20261005T235349402397Z'
SRC = B/'actual-fsdp-warning-source-20261005T235824197037Z'
CLOSURE = B/'actual-snapshots/3261-20261005T232501896059Z/original-files/bindings/import_closure.json'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def unique(items):
    out = {}
    for k,v in items:
        assert k not in out
        out[k] = v
    return out

def load(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

def physical_raw(p):
    assert p.resolve(strict=True) == p and p.is_file()
    before = p.stat()
    with p.open('rb') as f:
        import os
        opened = os.fstat(f.fileno())
        raw = f.read()
        after_open = os.fstat(f.fileno())
    after = p.stat()
    fields = ('st_dev','st_ino','st_uid','st_size','st_mtime_ns','st_ctime_ns')
    assert [getattr(before,k) for k in fields] == [getattr(opened,k) for k in fields] == [getattr(after_open,k) for k in fields] == [getattr(after,k) for k in fields]
    return raw

joined = load(physical_raw(B/'independent-framing75-raw-cpu-v2-original.json'))
assert joined['status'] == 'RAW_EDGE_JSON_DRIVER_JOIN_CONFIRMED_FSDP_SOURCE_AND_STDERR_PROVENANCE_PENDING'
pins = {}
transports = {}
for directory in [RAW, SRC]:
    transport_bytes = physical_raw(directory/'transport-original.json')
    transport = load(transport_bytes)
    summary_bytes = physical_raw(directory/'summary.json')
    summary = load(summary_bytes)
    reduced = dict(transport)
    reduced['files'] = {name:{k:v for k,v in record.items() if k != 'base64'} for name,record in transport['files'].items()}
    assert reduced == summary
    for name, record in transport['files'].items():
        raw = physical_raw(directory/name)
        assert raw == base64.b64decode(record['base64'], validate=True)
        assert len(raw) == record['bytes'] and sha(raw) == record['sha256']
        assert record['same_opened_inode'] is True and record['complete_prefix_at_start'] is True
        pins[str(directory.name+'/'+name)] = dict(bytes=len(raw),sha256=sha(raw),physical_path=record['physical_path'])
    transports[directory.name] = sha(transport_bytes)

closure_bytes = physical_raw(CLOSURE)
assert sha(closure_bytes) == '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
closure = load(closure_bytes)
source_summary = load(physical_raw(SRC/'summary.json'))
helper = closure['modules']['torch.utils._contextlib']
assert source_summary['pinned_native210_torch_helper'] == helper
assert helper == {'path':'/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/_contextlib.py', 'sha256':'cf7aa5b08f44974ba8c1d08cd71ef70ffd13d1c48a4931576eb235306bfa46b5'}
torch_root = Path(helper['path']).parent.parent
source_path = torch_root/'distributed/fsdp/fully_sharded_data_parallel.py'
assert source_summary['derived_installed_torch_root'] == str(torch_root)
assert source_summary['files']['fully_sharded_data_parallel.py']['physical_path'] == str(source_path)
assert all(v.get('path') != str(source_path) for v in closure['modules'].values())
source_record = source_summary['files']['fully_sharded_data_parallel.py']
assert source_record['before'] == source_record['after']
assert source_record['before']['uid'] == 0 and source_record['sha256'] == 'ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
source_bytes = physical_raw(SRC/'fully_sharded_data_parallel.py')
tree = ast.parse(source_bytes)
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.lineno == 689]
assert len(calls) == 1
call = calls[0]
assert isinstance(call.func, ast.Attribute) and call.func.attr == 'warn' and isinstance(call.func.value, ast.Name) and call.func.value.id == 'warnings'
assert len(call.args) == 2 and not call.keywords and isinstance(call.args[1], ast.Name) and call.args[1].id == 'FutureWarning'
message = ast.literal_eval(call.args[0])
suffix = joined['rejected'][0]['suffix']
prefix = '\x1b[36m(WorkerDict pid=1533910)\x1b[0m '
expected_warning = str(source_path)+':689: FutureWarning: '+message
assert suffix == prefix+expected_warning+'\n'
stderr_bytes = physical_raw(SRC/'rank0-original.err')
stderr_lines = stderr_bytes.decode('utf-8').split('\n')
matching = [n for n,line in enumerate(stderr_lines,1) if line == expected_warning]
assert matching and all(stderr_lines[n] == '  warnings.warn(' for n in matching)
rank = source_summary['rank0_process']
assert rank['present'] is True and type(rank['pid']) is int and rank['pid'] == 1533910
assert rank['start_ticks'] == '1829086072' and rank['uids'] == [0]*4
assert rank['cgroup_original'] == '1:freezer:/slurm_h100-10/uid_0/job_3261/step_0\n0::/system.slice/slurmd.service\n'
assert '\nPid:\t1533910\n' in rank['status_original'] and '\nName:\tray::WorkerDict\n' in '\n'+rank['status_original']
stderr_path = '/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs/worker-29c1f19308dadf572ed2b91a1792df098eeda628db2a448cfbe0b11c-02000000-1533910.err'
assert source_summary['rank0_stderr_actual_fd2'] == source_summary['files']['rank0-original.err']['physical_path'] == stderr_path
for directory in [RAW,SRC]:
    s = load(physical_raw(directory/'summary.json'))
    driver = s['allocation']['driver']
    assert driver['pid'] == 1533312 and driver['start_ticks'] == '1829085316' and driver['present'] is True
    slurm = s['allocation']['scontrol_original']
    for exact in ['JobId=3261 ', 'JobState=RUNNING ', 'Restarts=0 ', 'NodeList=h100-10 ', 'TresPerNode=gres:gpu:h100:8 ']:
        assert exact in slurm
result = dict(version=1,
    status='PASS_ACTUAL_RAW_FSDP_FRAME_PROVENANCE_NOT_NEW_PARSER_OR_CHECKPOINT_ACCEPTANCE',
    scope='LOCAL_ORIGINAL_BYTES_JSON_DRIVER_JOIN_OWNED_STDERR_AND_ADDITIONAL_INSTALLED_SOURCE_ONLY',
    source_commit=joined['source_commit'], config_sha256=joined['config_sha256'], job_id='3261',
    raw_join_sha256=sha(physical_raw(B/'independent-framing75-raw-cpu-v2-original.json')),
    actual_original_pins=pins, transport_original_sha256=transports,
    gate_sha256=joined['frozen_gate_sha256'], original_gate_rejected_one=joined['rejected'],
    complete_line_driver_count=76, complete_line_aggregate_count=75, aggregate_unclosed_tail_bytes=18035,
    rejected_complete_json_equals_driver74=True,
    warning=dict(kind='FutureWarning', source_path=str(source_path), source_sha256=source_record['sha256'],
                 source_call_line=689, message=message, actual_stderr_path=stderr_path,
                 actual_stderr_sha256=source_summary['files']['rank0-original.err']['sha256'],
                 matching_stderr_physical_lines=matching, owned_pid=1533910, start_ticks=1829086072),
    native210_closure_sha256=sha(closure_bytes), source_was_already_in_native210=False,
    derived_installation_root_anchor=helper,
    root_reader_failure_is_training_failure=False,
    limits=['No stderr timestamps establish the exact individual warning instance for outer74.',
            'This independently pins an added physical source, without changing native210.',
            'The exact warning content/PID/path is proven; arbitrary suffixes remain unrecognized.',
            'New consumer bytes/tests/freeze must receive separate independent review.',
            'No actual75 checkpoint CPU content, GPU effect or final200 acceptance is asserted.'])
print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
