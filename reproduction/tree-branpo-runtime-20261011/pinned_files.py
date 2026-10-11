import os,stat,hashlib,re
from pathlib import Path
from production_contract import require
def read_regular_bound(path, expected):
    """Read pinned ordinary bytes on one FD; no symlink or guessed digest."""
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected), 'Exact source SHA required')
    path=Path(path)
    observed=path.lstat()
    require(stat.S_ISREG(observed.st_mode), 'Reviewed source must be a regular file, not symlink: '+str(path))
    flags=os.O_RDONLY|os.O_NOFOLLOW
    with os.fdopen(os.open(path,flags),'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'Reviewed source is not a regular file: '+str(path))
        require((observed.st_dev,observed.st_ino)==(before.st_dev,before.st_ino),
                'Reviewed source path changed before opening: '+str(path))
        raw=stream.read(); after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Reviewed source changed during read: '+str(path))
    require(hashlib.sha256(raw).hexdigest()==expected,'Reviewed source bytes differ: '+str(path))
    return raw

