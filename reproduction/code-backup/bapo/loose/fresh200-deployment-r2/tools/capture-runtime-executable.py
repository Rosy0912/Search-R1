"""Capture the actual target interpreter; this record is not launch approval."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006')


def capture(source):
    assert re.fullmatch(r'[0-9a-f]{40}', source)
    assert sys.platform == 'linux' and os.uname().nodename == 'h100-5'
    assert ROOT.resolve() == ROOT and not ROOT.is_symlink()
    marker = ROOT / 'source-expected.txt'
    assert marker.is_file() and not marker.is_symlink()
    assert marker.read_text().strip() == source
    assert subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'rev-parse', 'HEAD'], text=True).strip() == source
    assert not subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'status', '--porcelain'], text=True).strip()
    assert sys.prefix == str(ROOT / 'env')
    executable = Path(sys.executable).resolve(strict=True)
    native = Path('/proc/self/exe').resolve(strict=True)
    expected = (ROOT / 'env/bin/python').resolve(strict=True)
    assert executable == native == expected
    stat = executable.stat()
    native_stat = native.stat()
    assert (stat.st_dev, stat.st_ino) == (native_stat.st_dev, native_stat.st_ino)
    record = dict(source_commit=source, root=str(ROOT), sys_prefix=sys.prefix,
                  executable=dict(resolved_path=str(executable), device=stat.st_dev, inode=stat.st_ino),
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  host=os.uname().nodename,
                  scope='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE')
    raw = (json.dumps(record, sort_keys=True, indent=2) + '\n').encode()
    output = ROOT / 'logs/runtime-executable.json'
    assert output.parent.resolve() == ROOT / 'logs' and not output.parent.is_symlink()
    with output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(path=str(output), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), record=record)))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    capture(sys.argv[1])
