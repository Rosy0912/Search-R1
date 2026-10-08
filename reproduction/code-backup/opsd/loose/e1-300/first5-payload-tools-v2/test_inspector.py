"""CPU-only core/schema tests; actual3190 tiny stored DTensor is a layout fixture.
No fixture is represented as an accepted production first-five checkpoint.
"""
import copy, importlib.util, json, os, pathlib, struct, tempfile, unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
P=pathlib.Path
HERE=P(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('inspector',HERE/'inspect_payloads.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=HERE.parent
FIXTURE=P(os.environ.get('FIRST5_TEST_FIXTURE',str(ROOT/'job3190-actual-evidence/fixture/baseline-fp32/roundtrip-step5/rank_0.pt')))
fs=importlib.util.spec_from_file_location('full_state',P(os.environ.get('FIRST5_TEST_REPO',str(ROOT.parent/'opsd-e1-300')))/'search_r1/baselines/full_state.py')
f=importlib.util.module_from_spec(fs);fs.loader.exec_module(f)

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=torch.load(FIXTURE,map_location='cpu',weights_only=False)
        assert not torch.cuda.is_initialized()
    def optimizer(self):
        v=copy.deepcopy(self.raw['optimizer'])
        for row in v['state'].values():row['step'].fill_(100)
        return v
    def test_native_dtensor_preserves_real_cuda_mesh_cpu_local(self):
        out=m.layout(self.raw['model']['0.weight'],0)
        self.assertEqual(out['mesh_device_type'],'cuda');self.assertEqual(out['mesh_ranks'],list(range(8)))
        self.assertEqual(out['global_shape'],[8,8]);self.assertEqual(out['local_shape'],[1,8])
        self.assertFalse(torch.cuda.is_initialized())
    def test_wrong_rank_rejected(self):
        with self.assertRaises(ValueError):m.layout(self.raw['model']['0.weight'],1)
    def test_wrong_world_rejected(self):
        with self.assertRaises(ValueError):m.layout(self.raw['model']['0.weight'],0,4)
    def test_flattened_model_rejected(self):
        with self.assertRaises(ValueError):m.layout(self.raw['model']['0.weight'].to_local(),0)
    def test_actual_adam_schema_and_100_clock(self):
        out=m.adam(self.optimizer(),12);self.assertEqual(out['steps'],[100]);self.assertEqual(out['local_parameter_numel'],12)
    def test_old_adam_20_not_accepted_as_100(self):
        with self.assertRaises(ValueError):m.adam(self.raw['optimizer'],12)
    def test_adam_missing_state_binding(self):
        v=self.optimizer();v['param_groups'][0]['params']=[]
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_adam_duplicate_binding(self):
        v=self.optimizer();v['param_groups'][0]['params']*=2
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_adam_missing_moment(self):
        v=self.optimizer();del next(iter(v['state'].values()))['exp_avg_sq']
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_adam_nan(self):
        v=self.optimizer();next(iter(v['state'].values()))['exp_avg'][0]=float('nan')
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_adam_negative_second_moment(self):
        v=self.optimizer();next(iter(v['state'].values()))['exp_avg_sq'][0]=-1
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_adam_incomplete_local_count(self):
        with self.assertRaises(ValueError):m.adam(self.optimizer(),13)
    def test_adam_wrong_lr(self):
        v=self.optimizer();v['param_groups'][0]['lr']=1e-6
        with self.assertRaises(ValueError):m.adam(v,12)
    def test_scheduler_native(self):
        self.assertEqual(m.scheduler(self.raw['scheduler'])['last_epoch'],5)
    def test_scheduler_adam_clock_is_not_outer_clock(self):
        v=copy.deepcopy(self.raw['scheduler']);v['last_epoch']=100
        with self.assertRaises(ValueError):m.scheduler(v)
    def test_rng_actual_cpu_and_philox(self):
        out=m.rng(self.raw['rng'],True);self.assertEqual(out['cuda'][0]['bytes'],16)
        self.assertFalse(torch.cuda.is_initialized())
    def test_rng_missing_cuda(self):
        v=copy.deepcopy(self.raw['rng']);v['cuda']=[]
        with self.assertRaises(ValueError):m.rng(v,True)
    def test_driver_rng_cannot_have_cuda(self):
        with self.assertRaises(ValueError):m.rng(self.raw['rng'],False)
    def test_rng_invalid_cpu_state(self):
        v=copy.deepcopy(self.raw['rng']);v['torch']=torch.zeros(12,dtype=torch.uint8)
        with self.assertRaises(ValueError):m.rng(v,True)
    def test_rng_wrong_philox_size(self):
        v=copy.deepcopy(self.raw['rng']);v['cuda'][0]['state']=torch.zeros(15,dtype=torch.uint8)
        with self.assertRaises(ValueError):m.rng(v,True)
    def test_rng_nonfinite_numpy_metadata(self):
        v=copy.deepcopy(self.raw['rng']);v['numpy']=(*v['numpy'][:4],float('nan'))
        with self.assertRaises(ValueError):m.rng(v,True)
    def test_rng_nonfinite_python_metadata(self):
        v=copy.deepcopy(self.raw['rng']);v['python']=(*v['python'][:2],float('inf'))
        with self.assertRaises(ValueError):m.rng(v,True)
    def test_nonfinite_tensor(self):
        with self.assertRaises(ValueError):m.tensor_hash(torch.tensor([float('nan')]))
    def test_duplicate_json(self):
        with self.assertRaises(ValueError):m.unique('{"a":1,"a":2}')
    def test_nonfinite_json(self):
        with self.assertRaises(ValueError):m.unique('{"a":NaN}')
    def test_physical_file_only_and_pinned_record(self):
        with tempfile.TemporaryDirectory() as d:
            root=P(d).resolve();p=root/'x';p.write_bytes(b'raw');pin=m.sha(p)
            self.assertEqual(m.record({'path':str(p),'sha256':pin},root,{}),p)
            with self.assertRaises(ValueError):m.record({'path':str(p),'sha256':'0'*64},root,{})
            alias=root/'alias';alias.symlink_to(p)
            with self.assertRaises(ValueError):m.sha(alias)
            with self.assertRaises(ValueError):m.record({'path':str(p),'sha256':pin},root/'child',{})
    def inventory_fixture(self,d):
        root=P(d).resolve();(root/'model.bin').write_bytes(b'unit inventory only')
        original=P(os.environ.get('FIRST5_TEST_BASE_MANIFEST',str(ROOT/'official-model-manifest-original.json')))
        manifest=root/'VERIFIED-MODEL-MANIFEST.json';manifest.write_bytes(original.read_bytes())
        return root,manifest
    def test_known_metadata_hashed_not_trusted_as_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d);(root/'VERIFIED').write_bytes(b'not evidence of PASS')
            pins={};files=m.base_asset_inventory(root,{'model.bin'},manifest,pins)
            self.assertEqual(set(files),{'model.bin','VERIFIED','VERIFIED-MODEL-MANIFEST.json'})
            self.assertEqual(files['VERIFIED'],m.sha(root/'VERIFIED'))
            self.assertEqual(files[manifest.name],m.MODEL_SHA);self.assertEqual(set(pins),{str(root/n) for n in files})
    def test_unknown_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d);(root/'OTHER-VERIFIED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unknown'):m.base_asset_inventory(root,{'model.bin'},manifest,{})
    def test_metadata_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d);(root/'VERIFIED').symlink_to(root/'model.bin')
            with self.assertRaisesRegex(ValueError,'symlink'):m.base_asset_inventory(root,{'model.bin'},manifest,{})
    def test_metadata_manifest_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d);manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError,'metadata manifest identity'):m.base_asset_inventory(root,{'model.bin'},manifest,{})
    def test_metadata_must_be_exact_name(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d);wrong=root/'different-manifest.json';manifest.rename(wrong)
            with self.assertRaisesRegex(ValueError,'metadata manifest identity'):m.base_asset_inventory(root,{'model.bin'},wrong,{})
    def test_manifest_outside_directory_does_not_allow_metadata_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root,manifest=self.inventory_fixture(d)
            with self.assertRaisesRegex(ValueError,'unknown'):m.base_asset_inventory(root,{'model.bin'},ROOT/'official-model-manifest-original.json',{})
    def test_streamed_base_digest_matches_frozen_native_format_and_local_slice(self):
        values={'a':torch.arange(16).reshape(8,2).to(torch.bfloat16),'b':torch.ones(8,dtype=torch.bfloat16)}
        with tempfile.TemporaryDirectory() as d:
            directory=P(d).resolve();headers={};raw=b''
            for key,t in values.items():
                b=t.view(torch.uint8).numpy().tobytes();headers[key]={'dtype':'BF16','shape':list(t.shape),'data_offsets':[len(raw),len(raw)+len(b)]};raw+=b
            header=json.dumps(headers).encode();(directory/'toy.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
            base=object.__new__(m.Base);base.directory=directory;base.headers=headers;base.index={key:'toy.safetensors' for key in values};base.starts={'toy.safetensors':8+len(header)}
            self.assertEqual(base.named_parameter_digest(),f.state_digest({key:t.float() for key,t in values.items()}))
            self.assertTrue(torch.equal(base.local('a',3),values['a'][3:4].float()))
    def test_externally_pinned_source_must_not_be_future_assumption(self):
        class Args:expected_source='not-a-commit'
        with self.assertRaisesRegex(ValueError,'external exact source'):m.inspect(Args())

if __name__=='__main__':unittest.main(verbosity=2)
