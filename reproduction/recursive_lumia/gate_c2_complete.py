"""Verify actual C2 completion before the study may advance to parameters.

C1 completion remains a separate prerequisite; this does not submit any job.
"""
import argparse
import json
from pathlib import Path

from .gate_c1_complete import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.run, args.log, args.job, group='C2-random-final')
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'group', 'job', 'run_id', 'rounds')}), flush=True)


if __name__ == '__main__':
    main()
