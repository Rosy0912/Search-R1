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
    def hf_fixture(self,d,values):
        root=P(d).resolve();headers={};raw=b''
        for key,t in values.items():
            b=t.contiguous().view(torch.uint8).numpy().tobytes();headers[key]={'dtype':'F32','shape':list(t.shape),'data_offsets':[len(raw),len(raw)+len(b)]};raw+=b
        header=json.dumps(headers).encode();shard=root/'model-00001.safetensors';shard.write_bytes(struct.pack('<Q',len(header))+header+raw)
        index=root/'model.safetensors.index.json';index.write_text(json.dumps({'weight_map':{k:shard.name for k in values}}))
        return {'actor/'+index.name:index,'actor/'+shard.name:shard}
    def test_hf_all_rank_slices_exact(self):
        values={'embed':torch.arange(16,dtype=torch.float32).reshape(8,2),'head':torch.arange(16,dtype=torch.float32).reshape(8,2),'norm':torch.arange(8,dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as d:
            hf=m.Export(self.hf_fixture(d,values),set(values))
            for rank in range(8):
                for key,t in values.items():self.assertTrue(hf.equal_slice(key,rank,t[rank:rank+1]))
    def test_hf_changed_slice_even_with_new_hash_rejected(self):
        original=torch.arange(16,dtype=torch.float32).reshape(8,2);changed=original.clone();changed[3,0]+=1
        with tempfile.TemporaryDirectory() as d:
            paths=self.hf_fixture(d,{'embed':changed});pins={str(p):m.sha(p) for p in paths.values()}
            self.assertTrue(pins);hf=m.Export(paths,{'embed'})
            with self.assertRaisesRegex(ValueError,'exact slice'):hf.equal_slice('embed',3,original[3:4])
    def test_hf_norm_no_tolerance(self):
        original=torch.ones(8,dtype=torch.float32);changed=original.clone();changed[4]+=torch.finfo(torch.float32).eps
        with tempfile.TemporaryDirectory() as d:
            hf=m.Export(self.hf_fixture(d,{'norm':changed}),{'norm'})
            with self.assertRaisesRegex(ValueError,'exact slice'):hf.equal_slice('norm',4,original[4:5])
    def test_hf_missing_index_key(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.hf_fixture(d,{'embed':torch.ones(8)})
            with self.assertRaisesRegex(ValueError,'key inventory'):m.Export(paths,{'embed','head'})
    def test_hf_duplicate_index_key(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.hf_fixture(d,{'embed':torch.ones(8)});paths['actor/model.safetensors.index.json'].write_text('{"weight_map":{"embed":"model-00001.safetensors","embed":"model-00001.safetensors"}}')
            with self.assertRaisesRegex(ValueError,'duplicate JSON'):m.Export(paths,{'embed'})
    def test_hf_tied_head_alias_difference(self):
        embed=torch.ones(8);head=embed.clone();head[1]+=1
        with tempfile.TemporaryDirectory() as d:
            hf=m.Export(self.hf_fixture(d,{'embed':embed,'head':head}),{'embed','head'})
            self.assertTrue(hf.equal_slice('embed',1,embed[1:2]))
            with self.assertRaisesRegex(ValueError,'exact slice'):hf.equal_slice('head',1,embed[1:2])
    def test_hf_header_unindexed_extra_key(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.hf_fixture(d,{'embed':torch.ones(8),'extra':torch.ones(8)})
            paths['actor/model.safetensors.index.json'].write_text(json.dumps({'weight_map':{'embed':'model-00001.safetensors'}}))
            with self.assertRaisesRegex(ValueError,'wrongly assigned'):m.Export(paths,{'embed'})
    def test_hf_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.hf_fixture(d,{'embed':torch.ones(8)})
            paths['actor/model.safetensors.index.json'].write_text(json.dumps({'weight_map':{'embed':'../model-00001.safetensors'}}))
            with self.assertRaisesRegex(ValueError,'unsafe'):m.Export(paths,{'embed'})
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


class TeacherProjection(unittest.TestCase):
    def test_actual_config_integer_keys_and_tuple_json_semantics(self):
        raw={'model_config':{'id2label':{0:'LABEL_0',1:'LABEL_1'}},'counters':{'cuda_update_entries':5},'shape':(2,3)}
        report=json.loads(json.dumps(raw))
        self.assertNotEqual(raw,report)
        self.assertEqual(m.teacher_report_projection(raw),report)
        self.assertEqual(set(m.teacher_report_projection(raw)),set(raw))
    def test_changed_nested_values_not_discarded(self):
        raw={'model_config':{'id2label':{0:'LABEL_0'}},'counters':{'cuda_update_entries':5}}
        changed=json.loads(json.dumps(raw));changed['counters']['cuda_update_entries']=4
        self.assertNotEqual(m.teacher_report_projection(raw),changed)
        changed=json.loads(json.dumps(raw));changed['model_config']['id2label']['0']='CHANGED'
        self.assertNotEqual(m.teacher_report_projection(raw),changed)
    def test_integer_string_key_collision_rejected(self):
        for bad in [{0:'a','0':'b'},{'nested':{1:'a','1':'a'}}]:
            with self.assertRaises(ValueError):m.teacher_report_projection(bad)
    def test_nonfinite_nested_and_unsupported_types_rejected(self):
        for bad in [{'x':[float('nan')]},{'x':float('inf')},{'x':float('-inf')},{'x':torch.tensor(1)},{False:'ambiguous'},{None:'ambiguous'},{1.0:'ambiguous'},{'x':set([1])}]:
            with self.assertRaises(ValueError):m.teacher_report_projection(bad)
