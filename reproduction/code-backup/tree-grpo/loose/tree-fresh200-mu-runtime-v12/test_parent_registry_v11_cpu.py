"""Narrow parent registry probes only. Tiny CPU fixtures; no official asset/GPU."""
import ast
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as N
import unittest
from unittest.mock import patch
import calibration_contract as contract
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'tree-behavior-mu-utf8-v2'

class MuPrefixAccepted(Exception):pass

def parent_registry_child(queue):
    try:
        import search_r1.baselines.tree_raw_tokens as raw
        import search_r1.baselines.tree_behavior_mu as mu
        from test_tree_behavior_mu_data_cpu import BehaviorMuDataTests
        assert not raw._VALIDATION_TOKENIZERS
        fixture=BehaviorMuDataTests();fixture.setUp()
        try:
            os.environ['SEARCH_R1_BASELINE_RUN_ID']='tree-fresh200-0'
            fixture.engine.behavior_mu_provenance['SEARCH_R1_BASELINE_RUN_ID']='tree-fresh200-0'
            data=fixture.output();raw._VALIDATION_TOKENIZERS.clear();contract._PARENT_VALIDATION_TOKENIZERS.clear()
            try:fixture.validate(data)
            except ValueError as error:missing=str(error)
            else:raise AssertionError('missing parent registry accepted')
            path=Path(fixture.tmp.name).resolve()
            for name in contract.PARENT_TOKENIZER_FILES:
                if not (path/name).exists():(path/name).write_bytes(b'{"tiny_metadata":true}')
            package={'official_model_path':str(path),'code_commit':'a'*40,'common_data_identity':'c'*64,'model_identity':{'sha256':'d'*64,'files':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in contract.PARENT_TOKENIZER_FILES}}}
            run=path/'tree-fresh200-0';loader_calls=[];utils=ModuleType('verl.utils')
            def loader(value,**kw):loader_calls.append((value,kw));return fixture.tok
            utils.hf_tokenizer=loader
            payload={'rows':2560,'identity':{'outer_step':1},'tensors':{},'non_tensor_batch':{},'meta_info':{}}
            actual=mu.validate_behavior_mu_batch;actual_receipts=[]
            def probe(data,*args,**kwargs):
                value=actual(data,*args,**kwargs);actual_receipts.append(value);raise MuPrefixAccepted('actual tiny source mu prefix validated')
            proto=N(from_dict=lambda **kw:data)
            # Older coverage/worker contracts are deliberately injected: this
            # probe targets only actual parent registration-to-mu interface.
            with patch.dict(sys.modules,{'verl.utils':utils}),patch.object(fixture.verl,'DataProto',proto),patch.object(contract,'OFFICIAL_TOKENIZER_PATH',path),patch.object(contract,'OFFICIAL_TOKENIZER_SHA',package['model_identity']['files']['tokenizer.json']),patch.object(contract.socket,'gethostname',return_value='h100-9'),patch.object(contract,'validate_ready_boundary',return_value=None),patch.object(contract,'validate_fresh_start',return_value=None),patch.object(contract,'held_json',return_value=({'status':'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED','rows':2560,'payloads':{'actor-inputs.pt':'tensorpin','trajectories.jsonl.gz':'tracepin'}},'readypin')),patch.object(contract,'held_tensor',return_value=payload),patch.object(contract,'sha',return_value='tracepin'),patch.object(mu,'validate_behavior_mu_batch',side_effect=probe):
                try:contract.validate_calibration_ready(path/'ready.json',run,package,'b'*64)
                except MuPrefixAccepted:pass
                else:raise AssertionError('source mu callback not reached')
                assert len(loader_calls)==1
                assert loader_calls[0]==(str(path),{'local_files_only':True,'trust_remote_code':False})
                for change in ('vocab','normalizer','flag'):
                    if change=='vocab':mutation=patch.dict(fixture.tok.parts,{2:'WRONG'})
                    elif change=='normalizer':mutation=patch.object(fixture.tok,'backend_normalizer',{'type':'Lowercase'})
                    else:mutation=patch.object(fixture.tok,'split_special_tokens',True,create=True)
                    with mutation:
                        try:contract.validate_calibration_ready(path/'ready.json',run,package,'b'*64)
                        except ValueError as error:assert 'drifted' in str(error),str(error)
                        else:raise AssertionError('live cached tokenizer drift accepted')
                assert len(loader_calls)==1
            queue.put({'ok':True,'fresh_spawn':True,'missing_parent_registry':missing,'actual_tiny_mu_accepted':actual_receipts[0],'cached_actual_object_reused':True,'vocab_normalizer_flag_drift_refused':True,'loader_calls':loader_calls,'official_assets':False,'full_zero_boundary_mocked':True,'full_zero_accepted':False})
        finally:fixture.tearDown()
    except BaseException as error:queue.put({'ok':False,'error':repr(error)})

class V11ParentTests(unittest.TestCase):
    def test_actual_fresh_parent_entry_registration_tiny_mu_and_live_cache_drift(self):
        context=multiprocessing.get_context('spawn');queue=context.Queue();process=context.Process(target=parent_registry_child,args=(queue,));process.start();process.join(20)
        self.assertFalse(process.is_alive());self.assertEqual(process.exitcode,0);result=queue.get(timeout=3)
        self.assertTrue(result['ok'],result);self.assertIn('not registered',result['missing_parent_registry']);self.assertTrue(result['vocab_normalizer_flag_drift_refused'])
    def test_malformed_fixed_manifest_path_sha_or_host_refuses_before_io_and_mu(self):
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()
            for name in contract.PARENT_TOKENIZER_FILES:(path/name).write_bytes(b'{"fixture":true}')
            package={'official_model_path':str(path),'model_identity':{'files':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in contract.PARENT_TOKENIZER_FILES}}}
            with patch.object(contract.socket,'gethostname',return_value='h100-9'),patch.object(contract,'held_json',side_effect=lambda *a,**kw:calls.append('saved-input')):
                with self.assertRaises(ValueError):contract.validate_calibration_ready('unused','unused',package,'unused')
                with patch.object(contract,'OFFICIAL_TOKENIZER_PATH',path):
                    with self.assertRaises(ValueError):contract.validate_calibration_ready('unused','unused',package,'unused')
                    bad=copy.deepcopy(package);bad['model_identity']['files']['tokenizer.json']='0'*64
                    with self.assertRaises(ValueError):contract.validate_calibration_ready('unused','unused',bad,'unused')
                self.assertEqual(calls,[])
                with patch.object(contract.socket,'gethostname',return_value='h100-8'),self.assertRaises(ValueError):contract.initialize_parent_validation_tokenizer(package)
    def test_actual_entry_registers_before_mu_and_formal_has_no_parent_zero_callback(self):
        tree=ast.parse((HERE/'calibration_contract.py').read_text());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_calibration_ready')
        self.assertIsInstance(entry.body[0],ast.Expr);self.assertEqual(entry.body[0].value.func.id,'initialize_parent_validation_tokenizer')
        formal=HERE.parent/'tree-fresh200-train-runtime-v5';inventory=json.loads((formal/'FILE_HASHES_V5.json').read_text())
        callsites=[]
        for name in inventory['wrapper_files']:
            if not name.endswith('.py'):continue
            tree=ast.parse((formal/name).read_text())
            callsites += [(name,n.lineno) for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='validate_calibration_ready') or (isinstance(n.func,ast.Attribute) and n.func.attr=='validate_calibration_ready'))]
        self.assertEqual(callsites,[])

if __name__=='__main__':unittest.main()
