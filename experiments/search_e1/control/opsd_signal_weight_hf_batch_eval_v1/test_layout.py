import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import evaluate
from worker import chunks,enrich

class Layout(unittest.TestCase):
    def test_eight_logical_shards_two_batches_32_no_missing_or_reordered_slots(self):
        data=list(range(512));seen=[]
        for rank in range(8):
            batches=chunks(data,rank);self.assertEqual([len(q) for ids,q in batches],[32,32])
            self.assertEqual([i for ids,q in batches for i in ids],list(range(rank,512,8)))
            for ids,q in batches:self.assertEqual(ids,q);seen.extend(ids)
        self.assertEqual(sorted(seen),list(range(512)))
    def test_identity_preserves_full_raw_cursor_and_fp32_digest(self):
        identity={'expected_adapter_tensor_sha256':'c'*64,'validation':{'sha256':'d'*64},
            'raw_checkpoint':{'kind':'raw_end_of_outer','manifest':{'path':'cp','sha256':'a'*64},
                'cursor':{'next_pair_cursor':301},'optimizer_updates':149,'rank_count':4,'no_best_or_cache':True}}
        module=types.ModuleType('evaluation');seen=[]
        def reader(config,completed,checkpoint):
            self.assertEqual(config['_config_sha256'],'e'*64)
            seen.append((completed,checkpoint));return identity,[{}]*512
        module.input_identity=reader
        request={'config':{'path':'cfg','sha256':'e'*64},'completed_outer':15,'checkpoint':{'path':'cp','sha256':'a'*64},'data':{'sha256':'d'*64}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(evaluate,'read',return_value={}),patch.object(evaluate,'modules'),patch.dict(sys.modules,{'evaluation':module}):
            evaluate.prepare_endpoint(request,Path(tmp))
            saved=json.loads((Path(tmp)/'model-identity.json').read_text())
            self.assertEqual(saved['identity'],identity);self.assertTrue(saved['no_weight_merge'])
            self.assertEqual(saved['precision'],'BF16_base_unmerged_FP32_LoRA')
        self.assertEqual(seen,[(15,request['checkpoint'])])
    def test_strict_em_comes_from_frozen_final_action_not_termination(self):
        q={'question_id':'a','question':'q','answers':['yes']}
        r={'response':'<think><answer>yes</answer>','termination':'answer','policy_spans':[[0,32]],'search_calls':0}
        score={'protocol_em':0,'protocol_valid_final_action':False,'failure_class':'final_invalid'}
        self.assertEqual(enrich(q,r,score)['reward'],0)
    def test_cpu_controller_import_does_not_import_torch(self):
        import subprocess
        p=subprocess.run([sys.executable,'-B','-c',"import evaluate,worker,dependencies,sys;assert 'torch' not in sys.modules"],
            cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
