"""Target stat/hash/git metadata only. No new source, torch, tokenizer or CUDA import."""
import csv
import io
import json
import socket
import subprocess
import sys
from pathlib import Path
from deployment_contract_v6 import (ROOT, COMMON, SHARED, CONTROLS, CANDIDATE, PHYSICAL,
    require, read_bound, record, core, sha, validate_candidate, validate_physical, verify_installed_controls, write_once)


def metadata(item, newly_hash):
    path = Path(item['path']); stat = path.stat()
    require(stat.st_size == item['bytes'], 'Changed file length: ' + str(path))
    actual = dict(path=str(path), resolved_path=str(path.resolve()), bytes=stat.st_size,
                  device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
                  sha256=item['sha256'], newly_hashed=newly_hash)
    if newly_hash:
        require(sha(path) == item['sha256'], 'Changed actual bytes: ' + str(path))
    return actual


def observe(candidate_record, output=PHYSICAL, run=subprocess.check_output, hostname=socket.gethostname):
    require(hostname().split('.')[0] == 'h100-8', 'Physical observation on wrong node')
    require(Path(__file__).resolve() == CONTROLS / 'verify_physical_v6.py', 'Physical helper installed path differs')
    require(Path(candidate_record['path']) == CANDIDATE and Path(output) == PHYSICAL, 'Wrong unique physical/candidate path')
    c = read_bound(candidate_record); proofs = validate_candidate(c)
    verify_installed_controls(c)
    repo = Path(c['repo'])
    require(repo.resolve() == repo and repo.is_dir(), 'Actual final repo is not a canonical independent directory')
    commit = run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    clean = not run(['git', '-C', str(repo), 'status', '--porcelain'], text=True, timeout=15).strip()
    names = run(['git', '-C', str(repo), 'ls-files', '-z'], timeout=15).decode().split('\0')[:-1]
    require(commit == c['source_commit'] and clean, 'Actual source not the final clean commit')
    require({str(repo / name) for name in names} == {r['path'] for r in c['source_files']}, 'Actual tracked source coverage differs')
    prior = {r['path']: r for r in proofs['asset_metadata']['metadata']}
    assets = {r['path'] for r in c['asset_files']}; rows = []
    for item in c['files']:
        is_asset = item['path'] in assets
        row = metadata(item, not is_asset)
        if is_asset:
            old = prior[item['path']]
            require(all(row[key] == old[key] for key in ('bytes', 'device', 'inode', 'mtime_ns')),
                    'Inherited immutable asset identity changed: ' + item['path'])
            row['sha_origin'] = 'BOUND_PRIOR_IMMUTABLE_SHA_AND_CURRENT_STAT_NOT_NEW_FULL_REHASH'
        else:
            row['sha_origin'] = 'ACTUAL_NEW_HASH'
        rows.append(row)
    evidence = []
    for item in c['evidence']:
        require(record(item['path']) == core(item), 'Physical evidence changed')
        evidence.append(core(item))
    raw = run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True, timeout=15)
    inventory = list(csv.reader(io.StringIO(raw)))
    require(len(inventory) == 8 and all(len(row) == 3 and 'H100' in row[2] for row in inventory), 'Canonical node is not full8 H100')
    uuids = [row[1].strip() for row in inventory]
    expected = read_bound(c['canonical_uuid_inventory_record'])
    require([row[0].strip() for row in inventory] == [str(index) for index in range(8)]
            and isinstance(expected, list) and len(expected) == len(set(uuids)) == 8
            and all(isinstance(value, str) for value in expected)
            and expected == uuids == c['expected_gpu_uuid_inventory'], 'Canonical hardware UUID differs')
    require((COMMON / 'env').resolve() == SHARED / 'env'
            and (COMMON / 'env/bin/python').resolve() == Path('/usr/bin/python3.10'), 'Actual environment identity differs')
    value = dict(status='PHYSICAL_NEW_CODE_SHA_INHERITED_ASSET_IDENTITY_ONLY_NOT_GPU',
        candidate_sha256=candidate_record['sha256'], source_commit=commit, source_clean=clean,
        source_tracked_file_count=len(names), hostname='h100-8', metadata=rows, evidence=evidence,
        raw_canonical_gpu_inventory_csv=raw, expected_gpu_uuids=uuids,
        canonical_uuid_inventory_record=c['canonical_uuid_inventory_record'],
        canonical_uuid_plain_array_read_only=True,
        old_canonical_inventory_is_not_allocation_or_lease=True,
        future_actual_exclusive_slurm_uuid_ofd_admission_required=True,
        asset_metadata_sha256=c['proofs']['asset_metadata']['sha256'], runtime_all_file_sha_before_gpu_required=True,
        environment_root_resolved=str((COMMON / 'env').resolve()),
        environment_python_resolved=str((COMMON / 'env/bin/python').resolve()),
        source_imported=False, cuda_initialized=False, actual_gpu_acceptance=False,
        actual_probability_acceptance=False, submitted=False)
    validate_physical(value, c, candidate_record['sha256'])
    write_once(output, value)
    return dict(physical=record(output), READY=False, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 4, 'Need candidate path, exact SHA and bytes')
    print(json.dumps(observe(dict(path=sys.argv[1], sha256=sys.argv[2], bytes=int(sys.argv[3])))))
