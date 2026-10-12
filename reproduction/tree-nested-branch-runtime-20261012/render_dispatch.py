"""Render pinned production SBATCH text only; never submits or creates intent."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
from production_contract import read_bound,require,ROOT


def render(package,manifest_sha,release_path,release_sha,wall_hours):
    package=Path(package);release_path=Path(release_path)
    require(package.is_absolute() and '..' not in package.parts and package.is_relative_to(ROOT/'instruct300/candidates'),'Own same-node candidate path required')
    require(release_path.is_absolute() and '..' not in release_path.parts and release_path.is_relative_to(ROOT),'Own same-node external release path required')
    require(all(type(d) is str and re.fullmatch('[0-9a-f]{64}',d) for d in (manifest_sha,release_sha)),'Exact immutable SHA pins required')
    require(type(wall_hours) is int and wall_hours>0,'Explicit positive finite formal wall hours required; no default')
    return '\n'.join(('#!/usr/bin/env bash','#SBATCH --job-name=tree-nested-branch-fresh200',
        '#SBATCH --nodes=1','#SBATCH --nodelist=h100-10','#SBATCH --gres=gpu:8',
        '#SBATCH --cpus-per-task=32','#SBATCH --mem=256G','#SBATCH --exclusive',
        '#SBATCH --output='+str(ROOT/'instruct300/node-move-logs/%j.out'),
        '#SBATCH --error='+str(ROOT/'instruct300/node-move-logs/%j.err'),
        f'#SBATCH --time={wall_hours}:00:00','#SBATCH --no-requeue','set -euo pipefail',
        'export TREE_INSTRUCT_TOOLS='+shlex.quote(str(package)),
        'export TREE_INSTRUCT_PACKAGE_SHA256='+shlex.quote(manifest_sha),
        'export TREE_PRODUCTION_RELEASE_PATH='+shlex.quote(str(release_path)),
        'export TREE_INSTRUCT_REVIEW_SHA256='+shlex.quote(release_sha),
        'exec bash '+shlex.quote(str(package/'launch.sbatch')),''))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','release-path','release-sha','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--wall-hours',required=True,type=int);a=parser.parse_args()
    text=render(a.package,a.manifest_sha,a.release_path,a.release_sha,a.wall_hours)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(text)
    print(json.dumps(dict(status='PRODUCTION_DISPATCH_TEXT_RENDERED_NOT_SUBMITTED',path=str(path.resolve()),sha256=hashlib.sha256(text.encode()).hexdigest(),slurm_submitted=False,requires_same_node_queue_refresh_and_once_journal=True)))


if __name__=='__main__':main()
