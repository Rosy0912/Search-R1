"""Narrow CPU/JSON compatibility tests; no actual material or GPU analysis."""
import ast
import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('readonly_backend_analyzer_v3', HERE/'probability_analyzer_v3.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def fixture():
    layers = [dict(name=f'model.layers.{i}.self_attn.attn',
        implementation_class='vllm.attention.backends.flash_attn.FlashAttentionImpl',
        head_size=128, num_heads=16, num_kv_heads=2, sliding_window=[-1,-1]) for i in range(36)]
    rows = [dict(scope='ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY',
        rank=rank,pid=3000+rank,source_commit=analyzer.SOURCE,binding_sha256='a'*64,
        requested_backend='FLASH_ATTN',runner_backend_name='flash-attn',
        runner_backend_class='vllm.attention.backends.flash_attn.FlashAttentionBackend',
        model_dtype='torch.bfloat16',cache_dtype='auto',block_size=16,model_sliding_window=None,
        attention_layers=copy.deepcopy(layers),observation_forward_calls=0,observation_tensor_value_reads=0,
        actual_vllm_runner_positions_observed=False,probability_accepted=False) for rank in range(8)]
    return dict(status='PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY',expected_backend='FLASH_ATTN',
        rank_count=8,layers_per_rank=36,actual_observations=rows,probability_accepted=False)


class BackendAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.value=fixture()

    def reject(self,row_key=None,value=None,layer_key=None,top_key=None,delete=None):
        x=copy.deepcopy(self.value)
        if row_key is not None:x['actual_observations'][2][row_key]=value
        if layer_key is not None:x['actual_observations'][3]['attention_layers'][17][layer_key]=value
        if top_key is not None:x[top_key]=value
        if delete is not None:del x[delete]
        with self.assertRaises((ValueError,KeyError,TypeError)):
            analyzer.validate_backend_original(x,'a'*64)

    def test_backend_all8_36_scope_only_no_tolerance(self):
        result=analyzer.validate_backend_original(self.value,'a'*64)
        self.assertEqual(len(result['actual_observations']),8)
        self.assertFalse(result['probability_accepted'])
        self.assertNotIn('tolerance',result)

    def test_wrong_source(self):self.reject('source_commit','base')
    def test_wrong_binding(self):self.reject('binding_sha256','b'*64)
    def test_bad_status(self):self.reject(top_key='status',value='OBSERVED_UNREVIEWED')
    def test_missing_rows(self):self.reject(delete='actual_observations')
    def test_missing_rank(self):self.reject(top_key='actual_observations',value=self.value['actual_observations'][:-1])
    def test_duplicate_rank(self):self.reject('rank',1)
    def test_bool_rank(self):self.reject('rank',True)
    def test_duplicate_pid(self):self.reject('pid',3001)
    def test_bool_pid(self):self.reject('pid',True)
    def test_invalid_pid(self):self.reject('pid',0)
    def test_env_xformers(self):self.reject('requested_backend','XFORMERS')
    def test_runner_xformers(self):self.reject('runner_backend_class','vllm.attention.backends.xformers.XFormersBackend')
    def test_wrong_backend_name(self):self.reject('runner_backend_name','FLASH_ATTN')
    def test_wrong_dtype(self):self.reject('model_dtype','torch.float32')
    def test_fp8(self):self.reject('cache_dtype','fp8')
    def test_bad_block(self):self.reject('block_size',8)
    def test_sliding(self):self.reject('model_sliding_window',4096)
    def test_empty_layers(self):self.reject('attention_layers',[])
    def test_duplicate_layer(self):self.reject(layer_key='name',value='model.layers.0.self_attn.attn')
    def test_layer_fallback(self):self.reject(layer_key='implementation_class',value='vllm.attention.backends.xformers.XFormersImpl')
    def test_bad_head(self):self.reject(layer_key='head_size',value=96)
    def test_bad_gqa(self):self.reject(layer_key='num_kv_heads',value=4)
    def test_layer_sliding(self):self.reject(layer_key='sliding_window',value=[4096,4096])
    def test_forward_forbidden(self):self.reject('observation_forward_calls',1)
    def test_tensor_read_forbidden(self):self.reject('observation_tensor_value_reads',1)
    def test_internal_positions_not_claimed(self):self.reject('actual_vllm_runner_positions_observed',True)
    def test_probability_not_claimed(self):self.reject('probability_accepted',True)
    def test_top_probability_not_claimed(self):self.reject(top_key='probability_accepted',value=True)

    def binding(self):
        helper=str(analyzer.ROOT/'tools/policy-safety-calibration-52befe9-v9')
        return dict(helpers=helper,backend_change_scope=
            'CALIBRATION_FLASH_ATTN_ENV_PROPAGATION_AND_READONLY_ACTUAL_BACKEND_GATE_ONLY',files=[
            dict(path=helper+'/'+name,sha256=digest) for name,digest in [
              ('calibration_driver_v1.py','31f191711844c0c875149502e9c2ddfc9569f40ef88163baf860aa46dbd762b9'),
              ('calibration_worker_v1.py','f2867517e79ecd87293c8ccf56c5600d61e597192fd49da233818d882f81df4d'),
              ('calibration_run_v1.py','b0b344d4f8cb13d215538110570ea4ae124ca3ebea88de6e86f59477adc0e7c5')]])

    def test_only_exact_v9_backend_helper_binding(self):analyzer.validate_backend_binding(self.binding())
    def test_old_helper_binding_refuses(self):
        x=self.binding();x['helpers']=x['helpers'].replace('-v9','-v8')
        with self.assertRaises(ValueError):analyzer.validate_backend_binding(x)
    def test_unreviewed_scope_refuses(self):
        x=self.binding();x['backend_change_scope']='WRAPPER_ONLY'
        with self.assertRaises(ValueError):analyzer.validate_backend_binding(x)
    def test_missing_wrapper_binding_refuses(self):
        x=self.binding();x['files'].pop()
        with self.assertRaises(ValueError):analyzer.validate_backend_binding(x)
    def test_changed_worker_binding_refuses(self):
        x=self.binding();x['files'][1]['sha256']='b'*64
        with self.assertRaises(ValueError):analyzer.validate_backend_binding(x)

    def test_backend_is_only_new_required_original(self):
        old=HERE.parent/'probability-analysis-v2/probability_analyzer_v2.py'
        tree=ast.parse(old.read_text())
        original=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='REQUIRED' for t in n.targets)))
        self.assertEqual(analyzer.REQUIRED,original+('actual-vllm-backend-identity.json',))

    def test_original_failure_contract_bytes_unchanged(self):
        self.assertEqual((HERE/'failure_contract_v2.py').read_bytes(),
                         (HERE.parent/'probability-analysis-v2/failure_contract_v2.py').read_bytes())

    def test_all_old_parsers_only_analyze_orchestration_changed(self):
        old=ast.parse((HERE.parent/'probability-analysis-v2/probability_analyzer_v2.py').read_text())
        new=ast.parse((HERE/'probability_analyzer_v3.py').read_text())
        functions=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)}
        a,b=functions(old),functions(new)
        self.assertEqual({n for n in a if a[n]!=b[n]}, {'analyze'})
        self.assertEqual(set(b)-set(a),{'validate_backend_binding','validate_backend_original'})


if __name__=='__main__':unittest.main(verbosity=2)
