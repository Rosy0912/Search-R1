"""Real unprivileged filesystem availability; never alters reserved blocks/assets."""
import json,os,socket,time
from pathlib import Path

def observe_storage(path,contract):
    if socket.gethostname().split('.')[0]!='h100-10':raise ValueError('Storage guard only on retained18')
    path=Path(path).resolve(strict=True)
    root=Path(contract['real_storage_root']).resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError('Output/cache outside reviewed retained18 mount')
    stat=os.statvfs(path);available=stat.f_bavail*stat.f_frsize
    if available<contract['minimum_available_bytes']:raise ValueError(f'Insufficient real f_bavail: {available}')
    # Probe write capacity rather than relying on free/reserved blocks alone.
    probe=path/('.nested-writecheck-'+str(os.getpid()))
    try:
        with probe.open('xb') as stream:stream.write(b'fresh-nested-storage-probe\n');stream.flush();os.fsync(stream.fileno())
    finally:probe.unlink(missing_ok=True)
    return dict(epoch=time.time(),realpath=str(path),f_bavail_bytes=available,f_bfree_bytes=stat.f_bfree*stat.f_frsize,
        minimum_available_bytes=contract['minimum_available_bytes'],reserved_configuration_modified=False)
