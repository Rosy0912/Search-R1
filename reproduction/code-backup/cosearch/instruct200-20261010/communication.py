"""Keep native role RPCs awaitable and per-instance rollout sockets distinct."""
from pathlib import Path
import hashlib,json

def patch(root):
    root=Path(root)
    base=root/'verl/verl/single_controller/ray/base.py'
    rollout=root/'verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    before=base.read_text()
    old='        def _fuw_execute(self, method_name: str, *args, **kwargs):'
    assert before.count(old)==1
    after=before.replace(old,'        async def _fuw_execute(self, method_name: str, *args, **kwargs):')
    old='            return udc_method(*args, **kwargs)'
    assert after.count(old)==1
    after=after.replace(old,'            result = udc_method(*args, **kwargs)\n            if inspect.isawaitable(result):\n                return await result\n            return result')
    old_roll=rollout.read_text()
    old='address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"'
    assert old_roll.count(old)==1
    new_roll=old_roll.replace(old,'address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{id(self):x}_{getpass.getuser()}.ipc"')
    records={str(base.relative_to(root)):{'before':hashlib.sha256(before.encode()).hexdigest(),'after':hashlib.sha256(after.encode()).hexdigest()},str(rollout.relative_to(root)):{'before':hashlib.sha256(old_roll.encode()).hexdigest(),'after':hashlib.sha256(new_roll.encode()).hexdigest()}}
    base.write_text(after);rollout.write_text(new_roll)
    (root/'communication-patch.json').write_text(json.dumps(records,indent=2)+'\n')
    return records
if __name__=='__main__':
    import sys
    print(json.dumps(patch(sys.argv[1])))
