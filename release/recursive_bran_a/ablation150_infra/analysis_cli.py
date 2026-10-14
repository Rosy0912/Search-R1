"""Invoke isolated published CPU helpers; never alter a training owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')


def process(pid):
    try:
        path = Path('/proc', str(pid))
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, uid=path.stat().st_uid, tick=int(fields[19]), state=fields[0])
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--mode', choices=['once', 'metadata', 'start', 'watch', 'status'], required=True)
    args = parser.parse_args()
    publication = json.loads(args.publication.read_text())
    source = Path(publication['source'])
    assert source.parent == BASE and source.name == 'source-analysis-'+publication['source_commit'][:8]
    assert Path(__file__).resolve().is_relative_to(source)
    for name, expected in publication['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    output = BASE/'analysis'/('aggregate-watch-'+publication['source_commit'][:8])
    output.mkdir(exist_ok=True)
    command = [sys.executable, '-B', '-u', '-m',
        'release.recursive_bran_a.ablation150_aggregate_watch_v1.watch',
        '--base', str(BASE), '--output', str(output), '--interval', '180']
    if args.mode == 'once':
        return subprocess.run(command+['--once'], env=env, cwd=source).returncode
    if args.mode == 'metadata':
        return subprocess.run([sys.executable, '-B', '-m',
            'release.recursive_bran_a.ablation150_storage_v1.inspect', '--source-commit',
            publication['scientific_source_commit']], env=env, cwd=source).returncode
    owner_path, exit_path = output/'watch-owner.json', output/'watch-actual-exit.json'
    if args.mode == 'status':
        owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
        print(json.dumps(dict(time=time.time(), owner=owner,
            observed=process(owner['pid']) if owner else None,
            state=json.loads((output/'watch-state.json').read_text()) if (output/'watch-state.json').exists() else None,
            actual_exit=json.loads(exit_path.read_text()) if exit_path.exists() else None)), flush=True)
        return 0
    if args.mode == 'watch':
        rc = subprocess.run(command, env=env, cwd=source).returncode
        exit_path.write_text(json.dumps(dict(time=time.time(), rc=rc, owner=process(os.getpid()),
                                            source_commit=publication['source_commit'])))
        return rc
    if owner_path.exists():
        owner = json.loads(owner_path.read_text())
        live = process(owner['pid'])
        assert live and live['uid'] == owner['uid'] and live['tick'] == owner['tick'], \
            'Prior CPU watcher is terminal; inspect it before a new independent recovery'
        print(json.dumps(dict(already_running=True, owner=owner)), flush=True)
        return 0
    with (output/'watch.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--publication', str(args.publication), '--mode', 'watch'],
            env=env, cwd=source, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    owner = dict(process(child.pid), time=time.time(), source_commit=publication['source_commit'],
                 publication=str(args.publication), output=str(output), training_action='NONE')
    owner_path.write_text(json.dumps(owner, indent=2))
    print(json.dumps(dict(started_not_progress_accepted=True, owner=owner)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
