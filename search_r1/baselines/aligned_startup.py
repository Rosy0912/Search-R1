"""Exclusive new-run claim, permitting only resource-supervisor bootstrap logs."""
import argparse
import json
import os
from pathlib import Path

BOOTSTRAP_FILES = {'resource-watch.jsonl', 'resource_watch.log', 'resource-watch.log', 'watch.jsonl'}


def claim_new_run(directory, source, partial_resume=None):
    root = Path(directory)
    if not root.is_absolute() or root.resolve() != root or root.parent.name != 'runs':
        raise ValueError('Require a canonical independent runs/<new-id> directory')
    root.mkdir(parents=True, exist_ok=True)
    for member in root.rglob('*'):
        if member.is_symlink() or (member.is_file() and member.name not in BOOTSTRAP_FILES):
            raise ValueError('Refusing prior experiment or unrecognized run artifact: ' + str(member))
    if partial_resume is not None:
        from .aligned_partial_resume import validate_manifest,require
        require(validate_manifest(partial_resume)==root and partial_resume['target_source_commit']==source, 'pending transaction claim binding differs')
    fd = os.open(root / '.aligned-startup-claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(dict(source_commit=source, launcher_pid=os.getpid(), scope='new-run-only-no-resume' if partial_resume is None else 'same-experiment-pending20-stage-only-resume'), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    claim_new_run(a.run, a.source)


if __name__ == '__main__':
    main()
