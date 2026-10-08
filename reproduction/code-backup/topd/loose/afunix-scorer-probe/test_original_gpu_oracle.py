"""CPU fixtures for prospective original-function oracle; no hardware approval."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_gpu_oracle',HERE/'original_gpu_oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
spec=importlib.util.spec_from_file_location('probe_round_checks',HERE/'probe_round_checks.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
ROOT=Path(__file__).resolve().parents[1]
repo=Path(__import__('os').environ.get('AGENT_ROOT',str(ROOT)))/'repo'
if not repo.exists():repo=ROOT/'topd'
sys.path.insert(0,str(repo))


def inputs(step):
    p,r,a=checks.request_rows([2,3],0,step)
    return dict(prompts=torch.tensor(p),responses=torch.tensor(r),attention=torch.tensor(a),row_indices=torch.arange(16))


class OracleCPUFixtures(unittest.TestCase):
    def test_real_frozen_qwen_original_function_no_grad_and_mask(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from search_r1.baselines.distillation import FrozenTeacher
        torch.set_num_threads(1);torch.manual_seed(17)
        teacher=FrozenTeacher.__new__(FrozenTeacher)
        teacher.tokenizer=SimpleNamespace(pad_token_id=0);teacher.device=torch.device('cpu');teacher.scored_tokens=0
        teacher.model=Qwen2ForCausalLM(Qwen2Config(vocab_size=17,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=8192,attn_implementation='eager')).eval().requires_grad_(False)
        original={k:v.clone() for k,v in teacher.model.state_dict().items()}
        x=inputs(1);x['attention'][0,-3:]=0
        calls=[];real=teacher.log_probs
        def tracked(p,r,a,temperature,targets,chunk):
            calls.append((p.shape[0],temperature,targets,chunk));return real(p,r,a,temperature,targets,chunk)
        teacher.log_probs=tracked
        actual=oracle.original_score(torch,teacher,x)
        self.assertEqual(calls,[(1,1.,None,16)]*16)
        self.assertTrue(torch.isfinite(actual).all());self.assertTrue((actual[0,-3:]==0).all())
        self.assertEqual(teacher.scored_tokens,16*(256+32)-3)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.model.parameters()))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in teacher.model.state_dict().items()))
        self.assertTrue(torch.equal(actual[8:9],real(x['prompts'][8:9],x['responses'][8:9],x['attention'][8:9],1.,None,16)))

    def test_three_original_shapes_and_long_counts(self):
        self.assertEqual([oracle.validate_input(torch,inputs(i),i,0) for i in (1,2,3)],[4608,12096,13740])

    def test_reordered_rows_rejected(self):
        x=inputs(1);x['row_indices'][8]=9
        with self.assertRaisesRegex(RuntimeError,'row order'):oracle.validate_input(torch,x,1,0)

    def test_private_gold_field_rejected(self):
        x=inputs(1);x['gold']=torch.tensor([2])
        with self.assertRaisesRegex(RuntimeError,'token inputs only'):oracle.validate_input(torch,x,1,0)

    def test_masked_long_row_rejected(self):
        x=inputs(3);x['attention'][15,4095]=0
        with self.assertRaisesRegex(RuntimeError,'masked/truncated'):oracle.validate_input(torch,x,3,0)

    def test_invalid_mask_rejected(self):
        x=inputs(1);x['attention'][8,9]=2
        with self.assertRaisesRegex(RuntimeError,'binary'):oracle.validate_input(torch,x,1,0)

    def test_float_input_rejected(self):
        x=inputs(1);x['prompts']=x['prompts'].float()
        with self.assertRaisesRegex(RuntimeError,'dtype/device'):oracle.validate_input(torch,x,1,0)

    def test_truncated_short_input_rejected(self):
        x=inputs(1);x['responses']=x['responses'][:,:31]
        with self.assertRaisesRegex(RuntimeError,'shapes'):oracle.validate_input(torch,x,1,0)

    def test_atomic_record_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';oracle.atomic_json(p,dict(status='CPU_FIXTURE_NOT_GPU_PROOF'))
            self.assertEqual(json.loads(p.read_text())['status'],'CPU_FIXTURE_NOT_GPU_PROOF')
            self.assertFalse(p.with_name(p.name+'.tmp').exists())
            with self.assertRaisesRegex(RuntimeError,'existing'):oracle.atomic_json(p,{})


if __name__=='__main__':unittest.main()
