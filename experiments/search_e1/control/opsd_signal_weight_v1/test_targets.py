import unittest
from contextlib import contextmanager
from types import SimpleNamespace
import torch
import targets
from precision import PrecisionContract

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(128,4);self.head=torch.nn.Linear(4,128,bias=False)
        self.lora_A=torch.nn.ModuleDict({'default':torch.nn.Linear(4,4,bias=False),'outer_old':torch.nn.Linear(4,4,bias=False)})
        self.active_adapter='default';self.disabled=False
        for p in self.parameters():p.requires_grad_(False)
        self.lora_A['default'].weight.requires_grad_(True)
    def set_adapter(self,name):self.active_adapter=name
    @contextmanager
    def disable_adapter(self):
        old=self.disabled;self.disabled=True
        try:yield
        finally:self.disabled=old
    def get_base_model(self):return self
    @property
    def base_model(self):return self
    def get_output_embeddings(self):return self.head
    def forward(self,input_ids,use_cache=False):
        x=self.embedding(input_ids)
        if not self.disabled:x=x+self.lora_A[self.active_adapter](x)
        return SimpleNamespace(last_hidden_state=x)

def old_identity(model):return {'adapter_tensor_sha256':targets.tensor_digest({'lora_A.weight':model.lora_A['outer_old'].weight}), 'outer_index':0,'checkpoint_sha256':'a'*64}

class TargetTests(unittest.TestCase):
    def test_causal_shift_excludes_last_D_hidden(self):
        h=torch.arange(6).reshape(6,1);self.assertEqual(targets.causal_hidden(h,3,3).flatten().tolist(),[2,3,4])
    def test_same_D_targets_frozen_and_old_stable_after_current_updates(self):
        m=Toy();t=targets.SharedTargets(m,PrecisionContract(device_type='cpu'),old_adapter='outer_old',old_identity=old_identity(m))
        encoded={'prefixes':[[1,2],[3,4,5]],'response':[6,7,8]}
        with t.current_graph():
            s,q,o=t.hidden(encoded);self.assertTrue(s.requires_grad);self.assertFalse(q.requires_grad);self.assertFalse(o.requires_grad)
            oq=o.clone();qq=q.clone();t.mark_backward();s.sum().backward()
        with torch.no_grad():m.lora_A['default'].weight.add_(1)
        with t.current_graph():
            s,q,o=t.hidden(encoded);self.assertTrue(torch.equal(o,oq));self.assertTrue(torch.equal(q,qq))
    def test_mutated_old_or_wrong_identity_refused(self):
        m=Toy();bad=old_identity(m);bad['adapter_tensor_sha256']='b'*64
        with self.assertRaises(ValueError):targets.SharedTargets(m,PrecisionContract(),old_adapter='outer_old',old_identity=bad)
        t=targets.SharedTargets(m,PrecisionContract(),old_adapter='outer_old',old_identity=old_identity(m))
        with torch.no_grad():m.lora_A['outer_old'].weight.add_(1)
        with self.assertRaises(ValueError):t.check_old()
    def test_resume_keeps_original_old_not_updated_current(self):
        m=Toy();identity=old_identity(m);saved={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():m.lora_A['default'].weight.add_(2)
        saved_current=m.lora_A['default'].weight.detach().clone()
        resumed=Toy();resumed.load_state_dict(saved)
        with torch.no_grad():resumed.lora_A['default'].weight.copy_(saved_current)
        t=targets.SharedTargets(resumed,PrecisionContract(),old_adapter='outer_old',old_identity=identity)
        self.assertTrue(torch.equal(resumed.lora_A['outer_old'].weight,saved['lora_A.outer_old.weight']))
        self.assertFalse(torch.equal(resumed.lora_A['outer_old'].weight,resumed.lora_A['default'].weight))
    def test_switch_during_current_graph_refused(self):
        m=Toy();t=targets.SharedTargets(m,PrecisionContract())
        with self.assertRaises(ValueError):
            with t.current_graph():m.set_adapter('outer_old')

if __name__=='__main__':unittest.main()
