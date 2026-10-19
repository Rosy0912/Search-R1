"""Actual offline tiny HF reload and tensor corruption tests; never a 7B acceptance."""
import tempfile
from pathlib import Path
import unittest
import importlib.util
from .prepare import compare_loaded


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Requires the frozen PJ inference runtime')
class Reload(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen2Config,AutoModelForCausalLM
        self.torch=torch;self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        config=Qwen2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=2,vocab_size=128,
                           tie_word_embeddings=False)
        with torch.random.fork_rng():
            torch.manual_seed(717);model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16)
            model.save_pretrained(self.root,safe_serialization=True)
        self.model=AutoModelForCausalLM.from_pretrained(self.root,local_files_only=True,
                            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map='cpu')
        self.count=len(self.model.state_dict());self.elements=sum(v.numel() for v in self.model.state_dict().values())
        self.assertFalse(torch.cuda.is_initialized())
    def tearDown(self):self.directory.cleanup()
    def test_actual_offline_reload_all_parameters_exact(self):
        r=compare_loaded(self.model,self.root,self.count,self.elements)
        self.assertTrue(r['all_loaded_parameters_exact']);self.assertFalse(self.torch.cuda.is_initialized())
    def test_reject_changed_loaded_tensor(self):
        with self.torch.no_grad():next(self.model.parameters()).view(-1)[0]+=1
        with self.assertRaises(AssertionError):compare_loaded(self.model,self.root,self.count,self.elements)
    def test_reject_nonfinite_loaded_tensor(self):
        with self.torch.no_grad():next(self.model.parameters()).view(-1)[0]=float('inf')
        with self.assertRaises(AssertionError):compare_loaded(self.model,self.root,self.count,self.elements)
    def test_reject_missing_export_key(self):
        from safetensors.torch import load_file,save_file
        path=self.root/'model.safetensors';state=load_file(str(path));state.pop(next(iter(state)))
        save_file(state,str(path),metadata={'format':'pt'})
        with self.assertRaises(AssertionError):compare_loaded(self.model,self.root,self.count,self.elements)


if __name__=='__main__':unittest.main()
