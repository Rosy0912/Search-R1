"""One bounded pipe transfer with no local archive and no automatic retry."""
import subprocess
import time


def transfer(source_command, target_command, timeout=2400):
    deadline=time.monotonic()+timeout
    source=target=None
    try:
        source=subprocess.Popen(source_command,stdout=subprocess.PIPE)
        target=subprocess.Popen(target_command,stdin=source.stdout)
        source.stdout.close()
        target_rc=target.wait(timeout=max(.001,deadline-time.monotonic()))
        source_rc=source.wait(timeout=max(.001,deadline-time.monotonic()))
        if source_rc or target_rc:
            raise RuntimeError('Stream transfer failed: source=%s target=%s; preserve target partial, no replay'%(source_rc,target_rc))
        return dict(source_returncode=source_rc,target_returncode=target_rc)
    finally:
        for process in (target,source):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
