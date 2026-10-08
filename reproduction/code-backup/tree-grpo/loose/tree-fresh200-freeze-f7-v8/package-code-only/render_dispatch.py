"""Render one reviewed zero-stage dispatch script; does not invoke sbatch.

An accepted new receipt/manifest is mandatory. This helper does not create an
intent or submit journal; root owns same-node queue refresh and once dispatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def render(package,manifest_sha,receipt_sha):
    package=Path(package)
    if not str(package).startswith('/dataset/shaoshitong/search-r1-rl/tree-grpo/'):
        raise ValueError('Package must be a new same-h100-9 task path')
    for digest in (manifest_sha,receipt_sha):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Exact reviewed SHA pins required')
    return '\n'.join(('#!/usr/bin/env bash','#SBATCH --job-name=tree-fresh200-mu-zero',
        '#SBATCH --nodes=1','#SBATCH --nodelist=h100-9','#SBATCH --gres=gpu:8',
        '#SBATCH --cpus-per-task=32','#SBATCH --mem=256G','#SBATCH --exclusive',
        '#SBATCH --time=03:00:00','#SBATCH --no-requeue','set -euo pipefail',
        'export TREE_INSTRUCT_TOOLS='+shlex.quote(str(package)),
        'export TREE_INSTRUCT_PACKAGE_SHA256='+shlex.quote(manifest_sha),
        'export TREE_INSTRUCT_REVIEW_SHA256='+shlex.quote(receipt_sha),
        'exec bash '+shlex.quote(str(package/'launch.sbatch')),''))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True);parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--receipt-sha',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); text=render(args.package,args.manifest_sha,args.receipt_sha)
    path=Path(args.output)
    if path.exists(): raise ValueError('Dispatch text is no-clobber')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: stream.write(text)
    print(json.dumps(dict(status='DISPATCH_TEXT_RENDERED_NOT_SUBMITTED',path=str(path.resolve()),
                         sha256=hashlib.sha256(text.encode()).hexdigest(),slurm_submitted=False)))


if __name__=='__main__': main()
