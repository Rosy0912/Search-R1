"""Independent tiny CPU mathematics/implementation boundaries, never GPU acceptance."""
import copy, hashlib, importlib.util, json, os, random, sys, unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / 'opsd-unclipped-multitraj200'
sys.path.insert(0, str(ROOT))
from search_r1.baselines.distillation import generalized_jsd, opsd_loss
from transformers import Qwen2Config, Qwen2ForCausalLM

class DesignMathematics(unittest.TestCase):
    def test_full_kl_nonnegative_and_correct_logit_gradient(self):
        logits=torch.tensor([.1,.9],dtype=torch.float64).log().requires_grad_()
        q=torch.tensor([.9,.1],dtype=torch.float64)
        loss=generalized_jsd(logits,q.log(),component_clip=None)
        grad=torch.autograd.grad(loss.sum(),logits)[0]
        self.assertAlmostEqual(float(loss),.8*np.log(9),places=6)
        torch.testing.assert_close(grad,torch.tensor([-.8,.8],dtype=torch.float64),atol=1e-7,rtol=1e-7)

    def token_case(self):
        generator=torch.Generator().manual_seed(914)
        features=torch.randn(256,7,5,dtype=torch.float64,generator=generator)
        q=torch.randn(256,7,9,dtype=torch.float64,generator=generator).softmax(-1)
        lengths=torch.arange(256)%7+1
        mask=torch.arange(7)[None,:]<lengths[:,None]
        mask[:32]=False  # one completely empty rank, while global mini remains nonempty
        w=torch.randn(5,9,dtype=torch.float64,generator=generator).requires_grad_()
        return features,q,mask,w

    def numerator(self,features,q,mask,w):
        logp=(features@w).log_softmax(-1)
        return (q*(q.log()-logp)).sum(-1).mul(mask).sum()

    def test_global_token_ddp_average_and_local4_micro_sums(self):
        x,q,mask,w=self.token_case();total=mask.sum()
        reference=self.numerator(x,q,mask,w)/total
        reference_grad=torch.autograd.grad(reference,w)[0]
        ranks=[]
        for rank in range(8):
            terms=[]
            for offset in range(rank*32,(rank+1)*32,4):
                terms.append(self.numerator(x[offset:offset+4],q[offset:offset+4],mask[offset:offset+4],w)/(total/8))
            ranks.append(torch.autograd.grad(sum(terms),w)[0])
        torch.testing.assert_close(sum(ranks)/8,reference_grad,atol=1e-14,rtol=1e-12)

    def test_local_micro_means_are_not_global_token_objective(self):
        x,q,mask,w=self.token_case();total=mask.sum()
        expected=torch.autograd.grad(self.numerator(x,q,mask,w)/total,w)[0]
        wrong=sum(self.numerator(x[i:i+4],q[i:i+4],mask[i:i+4],w)/mask[i:i+4].sum().clamp_min(1) for i in range(0,256,4))/8
        wrong_grad=torch.autograd.grad(wrong,w)[0]
        self.assertGreater(float((wrong_grad-expected).abs().max()),.01)

    def test_sequence_mean_is_not_preserved_token_mean(self):
        x,q,mask,w=self.token_case();total=mask.sum()
        expected=torch.autograd.grad(self.numerator(x,q,mask,w)/total,w)[0]
        pertoken=(q*(q.log()-(x@w).log_softmax(-1))).sum(-1)*mask
        sequence_mean=(pertoken.sum(-1)/mask.sum(-1).clamp_min(1)).mean()
        other=torch.autograd.grad(sequence_mean,w)[0]
        self.assertGreater(float((other-expected).abs().max()),1e-3)

    def test_actual_zero_initialized_57_outer_clock(self):
        spec=importlib.util.spec_from_file_location('independent_immutable_schedule',ROOT/'verl/utils/torch_functional.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        p=torch.nn.Parameter(torch.ones(())); optimizer=torch.optim.SGD([p],lr=1e-6)
        scheduler=mod.get_constant_schedule_with_warmup(optimizer,57)
        used=[];post=[]
        for outer in range(1,61):
            used.append(optimizer.param_groups[0]['lr'])
            for _ in range(10):p.grad=torch.ones_like(p);optimizer.step()
            scheduler.step();post.append(optimizer.param_groups[0]['lr'])
        self.assertEqual(used[0],0)
        self.assertAlmostEqual(used[56],1e-6*56/57,places=18)
        self.assertEqual(post[56],1e-6)
        self.assertEqual(used[57],1e-6)
        self.assertEqual(scheduler.last_epoch,60)

    def test_torch_fork_rng_does_not_restore_python_numpy(self):
        python=random.getstate();numpy=np.random.get_state()
        try:
            random.seed(31);np.random.seed(31);torch.manual_seed(31)
            py0=random.getstate();np0=np.random.get_state();t0=torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                random.random();np.random.random();torch.rand(4)
            self.assertNotEqual(random.getstate(),py0)
            self.assertFalse(np.array_equal(np.random.get_state()[1],np0[1]) and np.random.get_state()[2]==np0[2])
            self.assertTrue(torch.equal(torch.get_rng_state(),t0))
        finally:random.setstate(python);np.random.set_state(numpy)

    def test_rng_restoration_does_not_restore_teacher_mutable_buffers(self):
        model=torch.nn.Module();model.register_buffer('history_counter',torch.zeros(()))
        t0=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):model.history_counter.add_(1);torch.rand(2)
        self.assertTrue(torch.equal(torch.get_rng_state(),t0))
        self.assertEqual(float(model.history_counter),1)

class TinyQwenResponseProjection(unittest.TestCase):
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,add_special_tokens=False):return [5,6]

    def case(self,checkpointing=False):
        torch.manual_seed(57)
        config=Qwen2Config(vocab_size=37,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,tie_word_embeddings=True,attention_dropout=0.)
        config._attn_implementation='eager'
        student=Qwen2ForCausalLM(config).eval();teacher_model=copy.deepcopy(student).eval().requires_grad_(False)
        with torch.no_grad():teacher_model.model.embed_tokens.weight.add_(torch.randn_like(teacher_model.model.embed_tokens.weight)*.1)
        prompts=torch.tensor([[0,1,5,6],[1,7,8,9]])
        responses=torch.tensor([[10,11,12,13,0],[14,15,16,0,0]])
        ids=torch.cat([prompts,responses],dim=1);attention=ids.ne(0).long()
        positions=(attention.cumsum(-1)-1).clamp_min(0)
        # Valid generated tokens include an observation gap and padding exclusions.
        mask=torch.tensor([[1,0,0,1,0],[1,0,1,0,0]])
        data=dict(input_ids=ids,attention_mask=attention,position_ids=positions,prompts=prompts,responses=responses,loss_mask=mask)
        teacher=SimpleNamespace(model=teacher_model,device=torch.device('cpu'),measure=lambda:nullcontext(),scored_tokens=0)
        settings=SimpleNamespace(beta=0,temperature=1,component_clip=None,chunk_size=2,teacher_prompt_mode='legacy',max_student_logits_gib=8,loss_checkpointing=checkpointing)
        return student,teacher,data,settings

    def compare(self,checkpointing=False):
        a,teacher,data,settings=self.case(checkpointing);b=copy.deepcopy(a)
        settings.response_only_logits=False
        old=opsd_loss(a,teacher,data,self.Tokenizer(),None,settings,data['loss_mask'].sum())
        old.backward()
        settings.response_only_logits=True
        new=opsd_loss(b,teacher,data,self.Tokenizer(),None,settings,data['loss_mask'].sum())
        new.backward()
        torch.testing.assert_close(old,new,rtol=1e-6,atol=1e-7)
        ga=dict(a.named_parameters());gb=dict(b.named_parameters())
        self.assertEqual(set(ga),set(gb))
        for key in ga:
            self.assertEqual(ga[key].grad is None,gb[key].grad is None,key)
            if ga[key].grad is not None:torch.testing.assert_close(ga[key].grad,gb[key].grad,rtol=3e-5,atol=3e-7,msg=key)

    def test_actual_opsd_all_parameter_gradients_padding_observations(self):self.compare(False)
    def test_actual_opsd_all_parameter_gradients_loss_checkpointing(self):self.compare(True)
    def test_native_keep_width_is_shifted_keep_width_plus_one_is_correct(self):
        model,_,data,_=self.case();width=data['responses'].shape[1]
        kwargs={k:data[k] for k in ('input_ids','attention_mask','position_ids')}
        with torch.no_grad():
            full=model(**kwargs,use_cache=False).logits[:, -width-1:-1]
            correct=model(**kwargs,use_cache=False,num_logits_to_keep=width+1).logits[:, :-1]
            wrong=model(**kwargs,use_cache=False,num_logits_to_keep=width).logits
        torch.testing.assert_close(full,correct,rtol=1e-6,atol=1e-7)
        self.assertGreater(float((full-wrong).abs().max()),1e-3)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
