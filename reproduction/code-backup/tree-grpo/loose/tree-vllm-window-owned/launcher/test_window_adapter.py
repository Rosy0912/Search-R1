"""Owner metadata-only adapter fixtures; no real files, models or kernels."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_package as package
import resource_adapter_window as adapter
from test_dispatch_window_once import synthetic_envelopes


class WindowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.bundle,_,self.release=synthetic_envelopes()
        self.bundle['identity']={'config_sha256':package.ORIGINAL_CONFIG_SHA}
        self.launch=dict(diagnostic_profile=package.WINDOW_PROFILE,
            window_contract=copy.deepcopy(package.WINDOW_CONTRACT),
            window_inputs_sha256=self.bundle['window_inputs_sha256'])
        self.bundle['inventory']={str(r['rank']):dict(kind='raw',rank=r['rank'],generation_call=5,
            row_count=320,**r['source_snapshot']) for r in self.bundle['window_inputs']['ranks']}

    def test_runtime_profile_requires_actual_bound_launch(self):
        with patch.object(adapter,'ACTIVE',None):
            self.assertFalse(adapter.supports_runtime_profile(package.WINDOW_PROFILE))
        with patch.object(adapter,'ACTIVE',self.launch):
            self.assertTrue(adapter.supports_runtime_profile(package.WINDOW_PROFILE))
            self.assertFalse(adapter.supports_runtime_profile('native_only'))

    def test_old_profile_or_wrong_budget_refuses_before_any_source_reads(self):
        for kind in ('bundle','launch','release','count','inputsha','intent','protocol'):
            b,l,r=copy.deepcopy((self.bundle,self.launch,self.release))
            if kind=='bundle':b['window_profile']='native_only'
            elif kind=='launch':l['diagnostic_profile']='native_only'
            elif kind=='release':r['diagnostic_profile']='native_only'
            elif kind=='count':l['window_contract']['generation_invocations_per_rank']=5
            elif kind=='inputsha':r['window_inputs_sha256']='0'*64
            elif kind=='intent':r['dispatch_intent_path']=str(package.ROOT/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json')
            else:r['protocol']['gpu_count']=4
            with patch.object(adapter,'bound') as source_read:
                with self.assertRaises(ValueError):adapter.bind_window_profile(b,l,r)
                source_read.assert_not_called()

    def position_fixture(self):
        return {'inspections':[dict(kind='raw_same_turn',generation_call=5,rank=row['rank'],
            inspection=dict(full_input_tensor_identity=row['source_full_input_identity']))
            for row in self.bundle['window_inputs']['ranks']]}

    def test_source_snapshot_identity_and_full_row_count_checked(self):
        for key,value in [('rank',1),('generation_call',4),('row_count',319),('sha256','0'*64)]:
            b=copy.deepcopy(self.bundle);b['inventory']['0'][key]=value
            with patch.object(adapter,'bound',return_value=self.position_fixture()):
                with self.assertRaises(ValueError):adapter.bind_window_profile(b,self.launch,self.release)

    def test_original_json_record_not_unknown_default(self):
        for record in ({},{'rank':0,'world_size':8,'generation_call':5,'saved_rows':319,
                           'local_rows':320,'config_sha256':package.ORIGINAL_CONFIG_SHA}):
            with patch.object(adapter,'bound',side_effect=lambda path,sha: self.position_fixture() if path.endswith('position.json') else record):
                with self.assertRaises(ValueError):adapter.bind_window_profile(self.bundle,self.launch,self.release)

    def test_full_synthetic_metadata_path_without_importing_engine_or_runner(self):
        repo=package.ROOT/'synthetic-only/repo'
        names=('single_window_runner','window_runtime','window_observer','window_contracts','vllm_window_weights')
        b=copy.deepcopy(self.bundle)
        b['probe_source_files']=[dict(path=str(repo/(name+'.py')),sha256='e'*64) for name in names]
        wc=SimpleNamespace(__file__=str(repo/'window_contracts.py'),bind_installed_sources=lambda entries:True)
        def bound(path,sha):
            if path.endswith('position.json'):return {'inspections':[dict(kind='raw_same_turn',generation_call=5,rank=row['rank'],
                inspection=dict(full_input_tensor_identity=row['source_full_input_identity'])) for row in b['window_inputs']['ranks']]}
            rank=int(path.split('rank-')[1][:2])
            return dict(rank=rank,world_size=8,generation_call=5,saved_rows=320,local_rows=320,
                        config_sha256=package.ORIGINAL_CONFIG_SHA)
        pins={x['path']:x for x in package.KERNEL_BINARY_PINS}
        def local(path):
            if str(path) in pins:return SimpleNamespace(stat=lambda:SimpleNamespace(st_size=pins[str(path)]['bytes']),__str__=lambda:str(path))
            return Path(path)
        def sha(path):
            if isinstance(path,SimpleNamespace):
                return next(x['sha256'] for x in package.KERNEL_BINARY_PINS if x['bytes']==path.stat().st_size)
            return 'e'*64
        imports=[]
        def module(name):
            imports.append(name)
            self.assertEqual(name,'search_r1.baselines.probability_probe_io')
            return SimpleNamespace(__file__=str(repo/'search_r1/baselines/probability_probe_io.py'))
        with patch.object(adapter,'bound',side_effect=bound),patch.object(adapter,'local',side_effect=local), \
             patch.object(adapter,'sha',side_effect=sha),patch.object(adapter.importlib,'import_module',side_effect=module), \
             patch.dict(sys.modules,{'window_contracts':wc}):
            self.assertIs(adapter.bind_window_profile(b,self.launch,self.release),b['window_inputs'])
        self.assertEqual(imports,['search_r1.baselines.probability_probe_io'])


if __name__=='__main__':unittest.main(verbosity=2)
