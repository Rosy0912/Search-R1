"""Independent v2 lazy metadata and exact source-delta CPU checks."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'work/tree-probability-probe-runner'
NEW=ROOT/'work/tree-probability-probe-runner-v2'
sys.path.insert(0,str(NEW))
from search_r1.baselines import probability_probe_runner as R


def state():
    model=torch.nn.Linear(2,2,bias=False)
    return model,R.module_state(model)['parameters']


class LazyDeltaCases(unittest.TestCase):
    def test_only_exact_public_master_equal_private_creation(self):
        model,before=state();model.weight._local_shard=model.weight.data.clone()
        after=R.module_state(model)['parameters'];changes=R.verify_lazy_private_shards(before,after)
        self.assertEqual(set(changes),{'weight'});self.assertEqual(changes['weight']['before'],None)
        self.assertEqual(changes['weight']['after'],{k:v for k,v in after['weight'].items() if k!='private_local_shard'})

    def test_new_private_dtype_shape_bytes_device_individually_bound(self):
        model,before=state();model.weight._local_shard=model.weight.data.clone()
        after=R.module_state(model)['parameters']
        for field,new in [('dtype','torch.bfloat16'),('shape',[4]),('device','cuda:0'),('full_shape_dtype_bytes_sha256','0'*64)]:
            corrupted=copy.deepcopy(after);corrupted['weight']['private_local_shard'][field]=new
            with self.assertRaisesRegex(ValueError,'private shard differs'):R.verify_lazy_private_shards(before,corrupted)

    def test_public_master_identity_fields_individually_bound(self):
        _,before=state()
        for field,new in [('dtype','torch.bfloat16'),('shape',[4]),('device','cuda:0'),('full_shape_dtype_bytes_sha256','0'*64)]:
            corrupted=copy.deepcopy(before);corrupted['weight'][field]=new
            with self.assertRaisesRegex(ValueError,'public master bytes'):R.verify_lazy_private_shards(before,corrupted)

    def test_existing_private_must_stay_and_new_keys_refused(self):
        model,_=state();model.weight._local_shard=model.weight.data.clone();before=R.module_state(model)['parameters']
        self.assertEqual(R.verify_lazy_private_shards(before,copy.deepcopy(before)),{})
        corrupted=copy.deepcopy(before);corrupted['weight']['private_local_shard']=None
        with self.assertRaisesRegex(ValueError,'already-existing'):R.verify_lazy_private_shards(before,corrupted)
        corrupted=copy.deepcopy(before);corrupted['another']=copy.deepcopy(before['weight'])
        with self.assertRaisesRegex(ValueError,'logical master parameter keys'):R.verify_lazy_private_shards(before,corrupted)

    def test_postwarm_parameters_remain_in_full_baseline(self):
        source=(NEW/'search_r1/baselines/probability_probe_runner.py').read_text()
        self.assertIn("end_state==baseline and rng_state(device)==postwarm_rng",source)
        self.assertIn("module_state(fsdp,include_parameters=False)=={key:value for key,value in baseline.items() if key!='parameters'}",source)
        self.assertIn('lazy_private_local_shard_initialization=lazy_private',source)

    def test_native_core_original_constructor_and_monitor_unchanged_bytes(self):
        for rel in ['search_r1/baselines/probability_probe_capture.py','search_r1/baselines/probability_probe_monitor.py',
                    'search_r1/baselines/probability_probe_contracts.py','verl/workers/actor/dp_actor.py',
                    'verl/workers/fsdp_workers.py','verl/utils/fsdp_utils.py','verl/utils/torch_functional.py']:
            self.assertEqual((OLD/rel).read_bytes(),(NEW/rel).read_bytes(),rel)

    def test_all_runner_functions_except_specific_delta_ast_unchanged(self):
        def defs(repo):
            t=ast.parse((repo/'search_r1/baselines/probability_probe_runner.py').read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=defs(OLD),defs(NEW)
        self.assertEqual(set(after)-set(before),{'verify_lazy_private_shards'})
        for name in set(before)-{'run_probe'}:self.assertEqual(before[name],after[name],name)

    def test_loader_constructor_functions_exact_ast_unchanged(self):
        def defs(repo):
            t=ast.parse((repo/'search_r1/baselines/probability_probe_loader.py').read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=defs(OLD),defs(NEW)
        self.assertEqual(set(before),set(after))
        for name in set(before)-{'exhaustive_hf_cast_inventory'}:self.assertEqual(before[name],after[name],name)


if __name__=='__main__':unittest.main(verbosity=2)
