"""Local CPU replay of actual raw originals, never a remote checkpoint acceptance."""
import hashlib
import json
from pathlib import Path

import author200_observation as m
from test_fsdp_framing import CONTEXT, RAW, SOURCE, SOURCE_REC, SOURCE_RAW, FAILURE


def prefix(path):
    raw = path.read_bytes()
    n = raw.rfind(b'\n') + 1
    return raw[:n], dict(path=str(path.resolve()), whole_bytes=len(raw),
        whole_sha256=hashlib.sha256(raw).hexdigest(), complete_prefix_bytes=n,
        complete_prefix_sha256=hashlib.sha256(raw[:n]).hexdigest(), excluded_tail_bytes=len(raw) - n)


draw, drec = prefix(RAW / 'driver.out')
araw, arec = prefix(RAW / 'train.log')
driver = m.parse_records(draw, CONTEXT)
aggregate = m.parse_records(araw, CONTEXT)
join = m.join_log_prefixes(driver, aggregate, CONTEXT, 75)
evaluation = m.check_evaluation_sequence(driver, join['authoritative_last_outer'])
assert (join['authoritative_last_outer'], join['aggregate_last_outer']) == (76, 75)
assert len(aggregate['interleaved_frames']) == 1
frame = aggregate['interleaved_frames'][0]
assert frame['source_offset'] == FAILURE['offset'] and frame['record_sha256'] == FAILURE['record_sha256']
assert driver['outers'][73] == aggregate['outers'][73]
assert frame['source_binding']['source_record']['sha256'] == SOURCE_REC['sha256']
value = dict(version=1, scope='LOCAL_CPU_ACTUAL_RAW_PREFIX_REPLAY_NOT_NATIVE75_OR_RUNTIME_MANIFEST_ACCEPTANCE',
    source_commit=m.SOURCE, unchanged_gate_sha256=m.GATE_SHA,
    candidate_common_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
    driver_prefix=drec, aggregate_prefix=arec, log_join=join,
    original_gate_still_rejects_interleaved_fsdp=True,
    original_outer74_payload_exact_to_driver=True,
    owned_fsdp_frame=frame, native_console_record_count=len(aggregate['native_console']),
    fsdp_frame_did_not_increment_native_console=True,
    actual_extra_fsdp_source=dict(path=str((SOURCE / 'fully_sharded_data_parallel.py').resolve()),
        physical_origin=SOURCE_REC['physical_path'], sha256=hashlib.sha256(SOURCE_RAW).hexdigest(),
        bytes=len(SOURCE_RAW), native210_still_unchanged=True),
    evaluation_coverage=evaluation,
    limitations=['Source pin in CPU replay comes from actual independently captured source packet; new immutable runtime manifest is not yet finalized.',
                 'Only complete LF-terminated prefixes, not end-of-lifetime logs.',
                 'No remote execution, GPU, checkpoint75 PT load, scientific efficacy or terminal acceptance.',
                 'Old formal/c738 final supervisor still rejects preserved raw outer74 frame; its real future exit must not be upgraded.'])
print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
