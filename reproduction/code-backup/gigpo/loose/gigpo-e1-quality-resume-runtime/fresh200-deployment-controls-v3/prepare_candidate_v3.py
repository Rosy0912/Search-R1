"""Validate already complete final inputs and create one immutable candidate."""
import json
import socket
import sys
from pathlib import Path
from deployment_contract_v3 import (CANDIDATE, CONTROLS, require, read_bound, record,
                                    validate_candidate, verify_installed_controls, write_once)


def prepare(blueprint_record, output=CANDIDATE):
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Only target h100-8 preparation')
    require(Path(__file__).resolve() == CONTROLS / 'prepare_candidate_v3.py', 'Different installed prepare bytes/path')
    require(Path(output) == CANDIDATE, 'Candidate outside unique namespace')
    candidate = read_bound(blueprint_record)
    validate_candidate(candidate)
    verify_installed_controls(candidate)
    # No source/model/runtime import, Hydra compose, submission or asset mutation.
    write_once(output, candidate)
    return dict(candidate=record(output), READY=False, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 4, 'Need blueprint path, exact SHA and bytes')
    print(json.dumps(prepare(dict(path=sys.argv[1], sha256=sys.argv[2], bytes=int(sys.argv[3])))))
