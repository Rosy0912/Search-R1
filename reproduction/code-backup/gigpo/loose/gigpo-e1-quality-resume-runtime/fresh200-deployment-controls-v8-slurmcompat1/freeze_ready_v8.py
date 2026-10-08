"""One owned live first5-to200 launch after exact independent dependency reviews."""
import copy
import json
import socket
import sys
from pathlib import Path
from deployment_contract_v8 import (CONTROLS, CANDIDATE, PHYSICAL, READY, SCOPE,
    require, read_bound, record, closure, validate_candidate, validate_physical, validate_ready,
    validate_live_deployment_review, verify_installed_controls, write_once)


def freeze(candidate_record, physical_record, review_record, output=READY):
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Only target READY freeze')
    require(Path(__file__).resolve() == CONTROLS / 'freeze_ready_v8.py', 'Different installed freeze helper')
    require(Path(candidate_record['path']) == CANDIDATE and Path(physical_record['path']) == PHYSICAL
            and Path(output) == READY, 'Wrong unique freeze paths')
    c = read_bound(candidate_record); validate_candidate(c)
    verify_installed_controls(c)
    physical = read_bound(physical_record); validate_physical(physical, c, candidate_record['sha256'])
    review = read_bound(review_record)
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['independent_review'] is True
            and review['candidate_sha256'] == candidate_record['sha256']
            and review['physical_sha256'] == physical_record['sha256'], 'Wrong independent deployment review')
    validate_live_deployment_review(review, c)
    # Rehash fresh code and all original proof bytes. Assets retain separately
    # bound immutable metadata; the runtime hashes every bound input before GPU.
    assets = {r['path'] for r in c['asset_files']}
    for item in c['files'] + c['evidence']:
        if item['path'] not in assets:
            require(record(item['path']) == {k: item[k] for k in ('path', 'bytes', 'sha256')}, 'Fresh input changed before READY')
    ready = copy.deepcopy(c)
    ready.update(status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY', READY=True,
        readiness='READY_FOR_FRESH200_SAME_PROCESS_FIRST5_THEN200_GATED_LAUNCH',
        candidate_record=candidate_record, physical_record=physical_record, deployment_review=review_record)
    added = list(c['evidence']) + [candidate_record, physical_record, review_record]
    ready['evidence'] = list(closure(added, 'READY originals').values())
    validate_ready(ready)
    write_once(output, ready)
    return dict(ready=record(output), submitted=False, probability_acceptance=False, formal_iterations_contributed=0)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Need exact JSON spec containing candidate/physical/deployment_review records')
    spec_record = json.loads(sys.argv[1])
    spec = read_bound(spec_record)
    require(set(spec) == {'candidate', 'physical', 'deployment_review'}, 'Wrong READY input roles')
    print(json.dumps(freeze(spec['candidate'], spec['physical'], spec['deployment_review'])))
