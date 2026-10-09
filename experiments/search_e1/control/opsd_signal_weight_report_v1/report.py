"""Immutable final-only CPU report renderer; explicit synthetic visual fixture."""
import argparse
import json
import os
from pathlib import Path
import sys
try:
    from .common import STUDY_ROOT, require
    from .validate import validate
    from .render import render
except ImportError:
    from common import STUDY_ROOT, require
    from validate import validate
    from render import render


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request');parser.add_argument('--request-sha256')
    parser.add_argument('--output-dir',required=True);parser.add_argument('--synthetic-fixture',action='store_true')
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if args.synthetic_fixture:
        require(args.request is None and args.request_sha256 is None,'Synthetic fixture must not bind production evidence')
        try:from .fixture import example_data
        except ImportError:from fixture import example_data
        data=example_data()
    else:
        require(args.request and args.request_sha256,'Formal pinned request/SHA required')
        require(os.getuid()==1001 and output.is_relative_to(Path(STUDY_ROOT)/'analysis') and
                os.environ.get('CUDA_VISIBLE_DEVICES')=='','Formal output must stay authorized HKUST4 analysis; CVD empty')
        data=validate(dict(path=str(Path(args.request).resolve()),sha256=args.request_sha256),__file__)
    require('torch' not in sys.modules,'Report renderer imported Torch')
    ref=render(data,output)
    require('torch' not in sys.modules,'Plot rendering imported Torch')
    print(json.dumps(dict(accepted=not args.synthetic_fixture,formal_report=not args.synthetic_fixture,
        synthetic_fixture=args.synthetic_fixture,output=ref),sort_keys=True))
    return 0


if __name__=='__main__':main()
