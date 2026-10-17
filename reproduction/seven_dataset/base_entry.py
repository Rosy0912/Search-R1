"""Official initialization model on the unchanged full-seven search protocol."""
import argparse
import json
from pathlib import Path
from .base_contracts import inspect_base
from .entry import run_frozen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    cfg,expected,root=inspect_base(plan)
    if a.validate_only:
        print(json.dumps(dict(status='BASE_INPUTS_VERIFIED_NO_INFERENCE',count=len(expected),
            checkpoint_identity=plan['checkpoint_identity'],repeat_index=plan['repeat_index'])))
        return
    run_frozen(plan,a.plan,cfg,expected,root)


if __name__=='__main__':main()
