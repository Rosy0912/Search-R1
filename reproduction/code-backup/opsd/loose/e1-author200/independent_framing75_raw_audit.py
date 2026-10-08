"""Local-only raw log framing audit; no remote execution or parsing relaxation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import types

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw-log-framing75-20261005T235349402397Z'
GATE = HERE.parent / 'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
GATE_SHA = 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
WORKERS = [1533910, 1534229, 1534230, 1534231, 1534232, 1534233, 1534234, 1534235]
KW = dict(driver_pid=1533312, job_id='3261', worker_pids=WORKERS,
          checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def unique(items):
    out = {}
    for k, v in items:
        assert k not in out
        out[k] = v
    return out

def reject(value):
    raise ValueError(value)

decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=reject)
gate_bytes = GATE.read_bytes()
assert sha(gate_bytes) == GATE_SHA
g = types.ModuleType('independent_exact_gate')
g.__file__ = str(GATE)
exec(compile(gate_bytes, str(GATE), 'exec'), g.__dict__)
summary_bytes = (RAW/'summary.json').read_bytes()
summary = json.loads(summary_bytes)
originals = {}
records = {}
failures = []
edge = None
for name in ['driver.out', 'train.log']:
    raw = (RAW/name).read_bytes()
    pin = summary['files'][name]
    assert len(raw) == pin['bytes'] and sha(raw) == pin['sha256']
    assert pin['source_offset'] == 0 and pin['same_opened_inode'] is True
    closed = raw[:raw.rfind(b'\n')+1]
    offset = 0
    found = []
    originals[name] = dict(bytes=len(raw), sha256=sha(raw), physical_path=pin['physical_path'],
                           complete_line_prefix_bytes=len(closed), complete_line_prefix_sha256=sha(closed),
                           unclosed_tail_bytes=len(raw)-len(closed))
    for n, line in enumerate(closed.splitlines(keepends=True), 1):
        if b'OPSD_E1_OUTER_JSON=' in line:
            text = line.decode('utf-8')
            try:
                row = g.parse_log_marker(text, 'OPSD_E1_OUTER_JSON=', **KW)
                accepted = True
            except ValueError as exc:
                assert name == 'train.log' and str(exc) == 'Unrecognized/unowned interleaved warning suffix'
                prefix, body = text.split('OPSD_E1_OUTER_JSON=', 1)
                assert prefix == '\x1b[36m(main_task pid=1533312)\x1b[0m '
                body = body.lstrip()
                row, end = decoder.raw_decode(body)
                suffix = body[end:]
                assert suffix == '\x1b[36m(WorkerDict pid=1533910)\x1b[0m /mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .\n'
                clean = prefix + 'OPSD_E1_OUTER_JSON=' + body[:end]
                assert g.parse_log_marker(clean, 'OPSD_E1_OUTER_JSON=', **KW) == row
                failures.append(dict(file=name, line=n, offset=offset, bytes=len(line), sha256=sha(line), step=row['global_step'], suffix=suffix, error=str(exc)))
                edge = dict(raw_line_sha256=sha(line), raw_json_sha256=sha(body[:end].encode()), row=row)
                accepted = False
            g.validate_outer(row['metrics'], profile='author200')
            assert row['source_commit'] == SOURCE and row['config_sha256'] == CONFIG
            assert row['driver_pid'] == 1533312 and row['job_id'] == '3261'
            found.append((row, accepted))
        offset += len(line)
    assert [v[0]['global_step'] for v in found] == list(range(1, len(found)+1))
    records[name] = found
assert len(failures) == 1 and failures[0]['step'] == 74
assert failures[0]['sha256'] == '2653535246fd9c4d4fe193fbc58c4f5502822c1cf7b189f0d40021f6cafb92a2'
assert len(records['driver.out']) == 76 and len(records['train.log']) == 75
assert all(r[1] for r in records['driver.out'])
assert sum(r[1] for r in records['train.log']) == 74
assert [v[0] for v in records['train.log']] == [v[0] for v in records['driver.out'][:75]]
assert edge['row'] == records['driver.out'][73][0]
reported = json.loads((RAW/'source-parser-framing-failures.json').read_bytes())
assert len(reported) == 1 and reported[0]['record_sha256'] == failures[0]['sha256']
assert reported[0]['suffix'] == failures[0]['suffix']
assert reported[0]['offset'] == failures[0]['offset'] and reported[0]['line'] == failures[0]['line']
result = dict(scope='INDEPENDENT_LOCAL_RAW_FRAMING_EDGE_JOIN_NOT_NEW_PARSER_OR_CHECKPOINT_ACCEPTANCE',
              status='RAW_EDGE_JSON_DRIVER_JOIN_CONFIRMED_FSDP_SOURCE_AND_STDERR_PROVENANCE_PENDING',
              raw_summary_sha256=sha(summary_bytes), source_commit=SOURCE, config_sha256=CONFIG,
              frozen_gate_sha256=GATE_SHA, originals=originals,
              driver_complete_records=76, aggregate_complete_records=75,
              aggregate_original_gate_accepted=74, aggregate_original_gate_rejected=1,
              all_aggregate_json_equal_same_driver_rows=True,
              rejected=failures, edge_json_sha256=edge['raw_json_sha256'],
              scope_limits=['No broad suffix fallback was implemented.',
                            'No actual75 CPU inspection has been launched by this reviewer.',
                            'Torch FSDP source is not covered by the existing210 closure.',
                            'FutureWarning content is not evidence of a failed training update.',
                            'This record does not upgrade the failed source parser.'])
print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
