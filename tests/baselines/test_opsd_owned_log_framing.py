"""Replay original old-run framing only; these are not new training results."""
import hashlib
import json
from pathlib import Path

import pytest

from search_r1.baselines import checkpoint_gate as gate

WORK = Path(__file__).resolve().parents[3]
OLD = WORK / 'e1-author200'
SOURCE = OLD / 'actual-fsdp-warning-source-20261005T235824197037Z'
WARNING = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'


def actual_frame():
    raw = (OLD / 'raw-log-framing75-20261005T235349402397Z/train.log').read_bytes().splitlines(keepends=True)[983]
    assert hashlib.sha256(raw).hexdigest() == '2653535246fd9c4d4fe193fbc58c4f5502822c1cf7b189f0d40021f6cafb92a2'
    source = (SOURCE / 'fully_sharded_data_parallel.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == gate.FSDP_WARNING_SHA256
    return raw.decode(), source


def parse(raw, source):
    return gate.parse_log_marker(raw, 'OPSD_E1_OUTER_JSON=', driver_pid=1533312, job_id='3261',
                                worker_pids=(1533910,), checkpoint_warning_path=WARNING,
                                fsdp_warning_source_bytes=source)


def test_actual_outer74_complete_json_equals_original_driver_record():
    raw, source = actual_frame()
    result = parse(raw, source)
    driver = OLD / 'raw-log-framing75-20261005T235349402397Z/driver.out'
    values = [json.loads(line.split('OPSD_E1_OUTER_JSON=', 1)[1]) for line in driver.read_text().splitlines()
              if 'OPSD_E1_OUTER_JSON=' in line]
    assert result == [row for row in values if row['actual_outer_updates'] == 74][0]
    assert result['actual_outer_updates'] == 74


@pytest.mark.parametrize('old,new', [
    ('WorkerDict pid=1533910', 'WorkerDict pid=9999999'),
    ('fully_sharded_data_parallel.py:689', 'fully_sharded_data_parallel.py:688'),
    ('fully_sharded_data_parallel.py:689', 'other_file.py:689'),
    ('FSDP1, FSDP2, DDP.', 'FSDP2, FSDP1, DDP.'),
    ('\x1b[36m', '\x1b[31m'),
])
def test_unknown_worker_path_line_literal_colour_rejected(old, new):
    raw, source = actual_frame()
    with pytest.raises(ValueError):
        parse(raw.replace(old, new), source)


@pytest.mark.parametrize('suffix', [' {}', ' arbitrary', ' OPSD_E1_OUTER_JSON={}', ' SEARCH_R1_FINAL_METRICS_JSON={}'])
def test_extra_suffix_second_json_or_marker_rejected(suffix):
    raw, source = actual_frame()
    with pytest.raises(ValueError):
        parse(raw.rstrip('\n') + suffix, source)


def test_wrong_or_nonbytes_source_rejected():
    raw, source = actual_frame()
    for wrong in (source + b' ', b'', source.decode()):
        with pytest.raises(ValueError):
            parse(raw, wrong)


def test_actual_native_dedup_then_owned_amp_warning_preserves_json():
    line = (OLD / 'actual-snapshots/3261-20261005T210622178545Z/original-files/train.log').read_text().splitlines()[444]
    marker = 'FULL_STATE_RANK_JSON='
    expected, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
    assert expected['rank'] == 0 and expected['pid'] == 1533910
    assert gate.parse_log_marker(line, marker, driver_pid=1533312, job_id='3261',
                                worker_pids=(1533910,), checkpoint_warning_path=WARNING) == expected
    for wrong in (line+' {}', line.replace('[repeated 7x', '[repeated 9x'), line.replace('pid=1533910', 'pid=9999999'),
                  line.replace('checkpoint.py:1399', 'checkpoint.py:1400')):
        with pytest.raises(ValueError):
            gate.parse_log_marker(wrong, marker, driver_pid=1533312, job_id='3261',
                                  worker_pids=(1533910,), checkpoint_warning_path=WARNING)
