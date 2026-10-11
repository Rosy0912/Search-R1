"""Short, private Ray socket root bound to one native run; retain remote logs."""
import os
from pathlib import Path
import stat


def root_for(run_id):
    assert len(run_id) == 32 and all(c in '0123456789abcdef' for c in run_id)
    return Path('/tmp') / ('g1m-' + run_id[:12])


def identity(path):
    value = path.lstat()
    assert stat.S_ISDIR(value.st_mode) and not path.is_symlink()
    assert path.resolve() == path and value.st_uid == os.getuid()
    assert stat.S_IMODE(value.st_mode) == 0o700
    return dict(device=value.st_dev, inode=value.st_ino, uid=value.st_uid,
                mode=stat.S_IMODE(value.st_mode))


def prepare(run):
    path = root_for(run.get('native_ray_execution_id',run['run_id']))
    path.mkdir(mode=0o700, exist_ok=False)
    run['ray_tmp_dir'] = str(path)
    run['ray_tmp_dir_identity'] = identity(path)
    run['ray_tmp_retention'] = 'remote original Ray logs retained; owned processes cleaned by pipeline'
    verify(run, str(path))


def verify(run, actual):
    path = root_for(run.get('native_ray_execution_id',run['run_id']))
    assert actual == run['ray_tmp_dir'] == str(path)
    assert identity(path) == run['ray_tmp_dir_identity'], 'Private native Ray root identity changed'
    longest = path / 'session_0000-00-00_00-00-00_000000_2147483647' / 'sockets/plasma_store'
    assert len(os.fsencode(str(longest))) < 107, 'Ray AF_UNIX socket path exceeds platform limit'
    return str(path)
