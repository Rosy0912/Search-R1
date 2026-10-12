"""Freeze native-only READY after real target physical and independent review."""
import copy
import json
import sys
from pathlib import Path
from deployment_contract_v1 import (SOURCE, SCOPE, CONTROLS, READY, require, sha,
    read_bound, write_once, validate_candidate, validate_physical, validate_ready)


def freeze(candidate_path, candidate_sha, physical_path, physical_sha, review_path, review_sha, output):
    require(Path(output) == READY, 'Wrong immutable native READY destination')
    candidate_record = dict(path=str(candidate_path), bytes=Path(candidate_path).stat().st_size, sha256=candidate_sha)
    physical_record = dict(path=str(physical_path), bytes=Path(physical_path).stat().st_size, sha256=physical_sha)
    review_record = dict(path=str(review_path), bytes=Path(review_path).stat().st_size, sha256=review_sha)
    candidate = read_bound(candidate_record)
    validate_candidate(candidate)
    physical = read_bound(physical_record)
    validate_physical(physical, candidate, candidate_sha)
    review = read_bound(review_record)
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['code_commit'] == SOURCE
            and review['candidate_sha256'] == candidate_sha and review['physical_sha256'] == physical_sha
            and review['freeze_helper_sha256'] == sha(CONTROLS / 'freeze_ready_v1.py')
            and review['submit_helper_sha256'] == sha(CONTROLS / 'submit_once_v1.py'), 'Independent actual deployment review differs')
    value = copy.deepcopy(candidate)
    value.update(status='REVIEWED_FOR_NATIVE_ONLY', readiness='READY_FOR_OFFICIAL3B_RAW_PREFIX_NATIVE_ONLY',
        candidate_record=candidate_record, physical_record=physical_record, deployment_review=review_record,
        actual_gpu_acceptance=False)
    validate_ready(value)
    write_once(output, value)
    return dict(ready=str(output), ready_sha256=sha(output), gpu_accepted=False, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 8, 'Need candidate/SHA physical/SHA review/SHA immutable READY output')
    print(json.dumps(freeze(*sys.argv[1:])))
