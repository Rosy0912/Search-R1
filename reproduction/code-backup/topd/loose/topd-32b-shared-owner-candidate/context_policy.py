"""Explicitly reuse pinned native caller CPU/context permission policy."""
import ast,hashlib,json,os,time
from pathlib import Path
CALLER_SHA='aca296084c788a38011f7329ac6285d4a077c6feea6dda815e7fc06c149ec1a8'


def policies(path,gate):
    path=Path(path).resolve()
    gate.require(hashlib.sha256(path.read_bytes()).hexdigest()==CALLER_SHA,'Native caller context policy source drift')
    # Execute exact original function bodies with the explicitly verified canonical
    # gate symbols; no score_gate import shadowing or alternate provenance factory.
    tree=ast.parse(path.read_text());names={'same_cpu_identity','context_receipt','authorization'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    gate.require({n.name for n in selected}==names,'Incomplete native caller context policy')
    namespace=dict(require=gate.require,PARAMETERS=gate.PARAMETERS,json=json,Path=Path)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def wait_permission(path,policy,phase,identity,provenance_sha,job,timeout=300):
    path=Path(path);began=time.monotonic()
    while not path.exists():
        policy['require'](time.monotonic()-began<timeout,'Parent context permission timeout')
        time.sleep(.1)
    policy['require'](path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid()
                      and path.stat().st_mode & 0o077==0,'Unsafe parent context permission file')
    receipt=json.loads(path.read_text())
    policy['authorization'](receipt,phase,identity,provenance_sha,job)
    return receipt


def empty_contexts(runtime,torch,uuids,identity,policy):
    runtime.flags()
    for device in range(8):
        token=torch.empty((),device='cuda:%d'%device,dtype=torch.float32)
        torch.cuda.synchronize(device);del token
    allocators=runtime.release_allocators()
    receipt=dict(pid=identity['pid'],gpu_pid_uuid=runtime.gpu_identity(),allocators=allocators)
    policy['context_receipt'](receipt,identity,uuids)
    return receipt
