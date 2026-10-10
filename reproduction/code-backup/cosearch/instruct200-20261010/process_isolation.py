"""Give each trainable role its own process on each shared GPU."""
import ast
import hashlib
import json
from pathlib import Path


def patch(root):
    root = Path(root)
    pool = 'verl/verl/trainer/ppo/ray_trainer.py'
    trainer = 'verl/verl/trainer/ppo/search_r1_reranker_ray_trainer.py'
    changes = {}
    for name in (pool, trainer):
        before = (root / name).read_text()
        if name == pool:
            marker = 'process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name'
            assert before.count(marker) == 1
            after = before.replace(marker, marker.replace('max_colocate_count=1', 'max_colocate_count=2'))
        else:
            marker = '        for resource_pool, class_dict in self.resource_pool_to_cls.items():\n'
            assert before.count(marker) == 1
            replacement = '''        isolated_role_groups = []
        for resource_pool, roles in self.resource_pool_to_cls.items():
            expected = {str(Role.ActorRollout), str(Role.RerankerActorRollout)}
            if set(roles) != expected:
                raise RuntimeError("Process isolation requires exactly two integrated actor/rollout/ref roles")
            if resource_pool.max_colocate_count != 2 or resource_pool.world_size != 4:
                raise RuntimeError("Two role processes must share exactly four physical GPUs")
            for role, role_cls in roles.items():
                isolated_role_groups.append((resource_pool, {role: role_cls}))
        for resource_pool, class_dict in isolated_role_groups:
'''
            after = before.replace(marker, replacement)
        ast.parse(after)
        (root / name).write_text(after)
        changes[name] = {'before': hashlib.sha256(before.encode()).hexdigest(),
                         'after': hashlib.sha256(after.encode()).hexdigest()}
    (root / 'process-isolation-patch.json').write_text(json.dumps(changes, indent=2) + '\n')
    return changes
