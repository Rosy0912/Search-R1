"""Independent CPU child for Linux SO_PEERCRED native fixtures; never CUDA."""
import json
import os
from pathlib import Path
import sys
import traceback

from test_ipc import build_fixture
import protocol,server

root=Path(sys.argv[1]).resolve()
control=json.loads((root/'control.json').read_text())
sess,tensors,manifest,_,runtime=build_fixture(root)
actual=protocol.process_identity(os.getpid())
manifest.update(driver=control['driver'],parent_cgroup=control['parent_cgroup'])
manifest['service']=dict(actual,uid=os.getuid(),socket=str(root/'s.sock'))
sess.manifest_sha=protocol.digest(protocol.canonical(manifest));sess.peer_identity=None;sess.actual_service_identity=None
service=server.SharedUnixServer(sess)
status='FAILED'
try:
    service.start()
    # Capability is a private control artifact, never stdout or public evidence.
    fd=os.open(root/'ready-private.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(manifest,stream)
    service.serve();status='COMPLETE_CPU_FIXTURE'
except Exception:
    traceback.print_exc()
finally:
    service.close()
    protocol.write_private_json(root,'child-evidence.json',dict(scope='CPU_ONLY_SO_PEERCRED_NOT_HARDWARE',status=status,
        actual_process=actual,stages=runtime.stages,releases=runtime.releases,cuda_initialized=__import__('torch').cuda.is_initialized(),
        completed_outer_steps=sess.next_step-1))
if status!='COMPLETE_CPU_FIXTURE':sys.exit(1)
