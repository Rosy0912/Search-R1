#!/usr/bin/env python3
"""LOCAL PREPARE ONLY. Explicit bounded-disk budget; no asset or host actions.

Bounds below are imposed write ceilings / observed training references, not a
promise about future native sizes, compression ratios or global disk growth.
"""
import os

PROFILE = 'own3341-bounded-disk-v1'
MIB = 1024**2
GIB = 1024**3
NATIVE_ALLOCATED_REFERENCE = 49376825344
HF_ALLOCATED_REFERENCE = 13604483072
SERIALIZATION_MARGIN = 16*MIB
# One next trajectory: conservative ceiling above largest own observed sample.
# Unknown global growth remains unbounded; 1GiB is only an explicit allowance.
NEXT_TRAJECTORY_ALLOWANCE = 2*GIB
UNKNOWN_GROWTH_ALLOWANCE = GIB
METADATA_ALLOWANCE = 128*MIB


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rounded(value, block=4096):
    require(type(value) is int and value >= 0 and type(block) is int and block > 0, 'Invalid byte budget')
    return ((value+block-1)//block)*block


def compression_cap(original_bytes, block=4096):
    # Hard imposed bound, NOT a compression ratio forecast. Exceeding it aborts.
    return rounded((original_bytes*101+99)//100 + MIB, block)


def training_reserve(latest_complete_outer, block=4096):
    require(type(latest_complete_outer) is int and 1 <= latest_complete_outer <= 200, 'Latest native outer outside scope')
    # Until native151/201 proves prior milestone save path progressed, preserve
    # both native+HF peak at149..150/199..200. Other outers reserve one native.
    milestone_peak = latest_complete_outer in (149,150,199,200)
    peak = NATIVE_ALLOCATED_REFERENCE + (HF_ALLOCATED_REFERENCE if milestone_peak else 0)
    return rounded(peak+SERIALIZATION_MARGIN+NEXT_TRAJECTORY_ALLOWANCE+UNKNOWN_GROWTH_ALLOWANCE, block)


def estimate(available_bytes, original_bytes, latest_complete_outer, block=4096):
    require(type(available_bytes) is int and available_bytes >= 0 and type(original_bytes) is int and original_bytes > 0, 'Positive bounded source/available required')
    cap = compression_cap(original_bytes,block)
    temp = rounded(original_bytes,block)
    reserve = training_reserve(latest_complete_outer,block)
    total = reserve+cap+temp+METADATA_ALLOWANCE
    return dict(profile=PROFILE, available_bytes=available_bytes, source_already_resident_incremental_bytes=0,
                compressed_hard_cap_bytes=cap, one_cpu_decode_temporary_bytes=temp,
                training_reference_and_allowances_bytes=reserve, metadata_allowance_bytes=METADATA_ALLOWANCE,
                required_available_before_new_archive_bytes=total, deficit_bytes=max(0,total-available_bytes),
                may_begin_new_archive=available_bytes >= total,
                release_after_existing_archive_and_real_cpu_receipt_required_metadata_bytes=METADATA_ALLOWANCE,
                future_native_size_and_global_growth_guaranteed=False)


def available(paths):
    s = os.fstatvfs(paths.run_fd)
    return s.f_bavail*s.f_frsize, s.f_frsize


def guard(paths, latest, additional, stage):
    free, block = available(paths)
    required = training_reserve(latest,block)+rounded(additional,block)+METADATA_ALLOWANCE
    require(free >= required, '%s: own candidate stops; available=%d required=%d; training is never controlled' % (stage,free,required))
    return dict(stage=stage, available_bytes=free, required_bytes=required, latest_complete_outer=latest, profile=PROFILE)


class CappedArchiveWriter:
    """Unbuffered destination, bounded encoded writes and remaining-stage guard."""
    def __init__(self, stream, paths, latest, original_bytes):
        self.stream,self.paths,self.latest = stream,paths,latest
        _, block = available(paths)
        self.cap,self.temp = compression_cap(original_bytes,block),rounded(original_bytes,block)
        self.written = 0
    def write(self, data):
        require(self.written+len(data) <= self.cap, 'Encoded gzip exceeds explicit hard cap; original retained')
        guard(self.paths,self.latest,self.cap-self.written+self.temp,'compress')
        count = self.stream.write(data)
        require(count == len(data), 'Short own archive write; source retained')
        self.written += count
        return count
    def flush(self):
        self.stream.flush()
