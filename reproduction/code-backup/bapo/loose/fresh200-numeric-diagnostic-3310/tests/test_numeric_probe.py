import importlib.util
from pathlib import Path
import unittest
import torch
import tempfile
spec=importlib.util.spec_from_file_location('numeric',Path(__file__).parents[1]/'numeric-probe.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)


class NumericTests(unittest.TestCase):
    def fixture(self):
        inputs=torch.tensor([[0,0,10,2,3,8,4,5]]).repeat(8,1)
        mask=torch.tensor([[0,0,1,1,1,1,1,1]]).repeat(8,1)
        tensors=dict(input_ids=inputs,attention_mask=mask,position_ids=(mask.cumsum(-1)-1)*mask,
            responses=inputs[:,3:].clone(),old_log_probs=torch.full((8,5),-1.),sampling_log_probs=torch.full((8,5),-1.1),
            sampling_mask=torch.tensor([[1,1,0,1,1]]).repeat(8,1))
        turns=[dict(actual_request_token_ids=[10],raw_generated_token_ids=[2,3],kept_token_count=2,environment_token_ids=[8]),
               dict(actual_request_token_ids=[10,2,3,8],raw_generated_token_ids=[4,5],kept_token_count=2,environment_token_ids=[])]
        return tensors,[turns for _ in range(8)]

    def test_exact_actual_prefix_position_label(self):
        tensors,journals=self.fixture();point=n.witness(tensors,journals,5,3)
        self.assertEqual(point['predictor_tokens'],[10,2,3,8])
        self.assertEqual((point['target_token'],point['query_position'],point['full_padded_query_index']),(4,3,5))
        self.assertEqual(point['predictor_positions'],[0,1,2,3])

    def test_wrong_native_request_fail_closed(self):
        tensors,journals=self.fixture();journals[5][1]['actual_request_token_ids']=[11,2,3,8]
        with self.assertRaisesRegex(ValueError,'prefixes differ'):n.witness(tensors,journals,5,3)

    def test_fixed_slot_same_shape_tokens_control(self):
        tensors,journals=self.fixture();point=n.witness(tensors,journals,5,3)
        cases=n.case_batches(tensors,journals,point,100)
        self.assertEqual(len(cases),6)
        adjacent=cases[3][1];control=cases[5][1]
        self.assertEqual(cases[3][2],1);self.assertEqual(cases[4][2],1);self.assertEqual(cases[5][2],1)
        for key in n.h.KEYS:self.assertTrue(torch.equal(adjacent[key][1],control[key][1]))
        self.assertTrue(torch.equal(adjacent['attention_mask'],control['attention_mask']))
        self.assertTrue(torch.equal(adjacent['position_ids'],control['position_ids']))
        self.assertFalse(torch.equal(adjacent['input_ids'][0],control['input_ids'][0]))
        self.assertTrue(torch.equal(control['input_ids'][:,-5:],control['responses']))

    def test_original_padded_and_flat_predictor_mapping(self):
        tensors,journals=self.fixture();point=n.witness(tensors,journals,5,3);batch=n.case_batches(tensors,journals,point,100)[3][1]
        self.assertEqual(n.predictor_index(batch,1,3,False),(1,5))
        self.assertEqual(n.predictor_index(batch,1,3,True),(0,9))

    def test_cpu_fp64_logprob_reference(self):
        vector=torch.tensor([2.,3.,4.],dtype=torch.bfloat16);summary=n.vector_summary(vector,1)
        self.assertAlmostEqual(summary['cpu_fp64_logprob'],float(torch.log_softmax(vector.double(),0)[1]))
        self.assertEqual(summary['top10_tokens'] if len(vector)>=10 else [],[])

    def sidecar(self):
        records=[]
        for precision in ('BF16_ORIGINAL','FP32_EAGER_CONFOUNDED'):
            for mode in ('FULL','EXACT_REQUEST','TRUNCATED','FIXED_SLOT_ADJACENT','FIXED_SLOT_CHANGED_LENGTHS','FIXED_SLOT_SAME_SHAPE_OTHER_TOKENS'):
                for row,pos in n.POINTS:
                    records.append(dict(precision=precision,mode=mode,row=row,position=pos,label=1,
                        original_dtype='torch.bfloat16'if precision=='BF16_ORIGINAL'else'torch.float32',
                        original_logits_float32=torch.tensor([2.,3.,4.])))
        return dict(schema_version=1,source_commit=n.SOURCE,raw_sha256=n.h.RAW_SHA,vocab_size=3,records=records)

    def test_actual_original_sidecar_independent_float64_replay(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'selected.pt';metadata=n.persist_sidecar(path,self.sidecar())
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(metadata['count'],36)
            self.assertEqual(metadata['sha256'],n.h.digest(path))
            for record,entry in zip(saved['records'],metadata['queries']):
                independently=float(torch.log_softmax(record['original_logits_float32'].double(),0)[record['label']])
                self.assertAlmostEqual(entry['cpu_fp64_logprob'],independently)
            with self.assertRaises(FileExistsError):n.persist_sidecar(path,self.sidecar())

    def test_sidecar_incomplete_duplicate_wrongdtype_and_nonfinite_rejected(self):
        sidecar=self.sidecar();sidecar['records'].pop()
        with self.assertRaisesRegex(ValueError,'exact36'):n.validate_sidecar(sidecar)
        sidecar=self.sidecar();sidecar['records'][1]=sidecar['records'][0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):n.validate_sidecar(sidecar)
        sidecar=self.sidecar();sidecar['records'][0]['original_logits_float32']=torch.tensor([2.,3.,4.],dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'Invalid original'):n.validate_sidecar(sidecar)
        sidecar=self.sidecar();sidecar['records'][0]['original_logits_float32'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid original'):n.validate_sidecar(sidecar)


if __name__=='__main__':unittest.main()
