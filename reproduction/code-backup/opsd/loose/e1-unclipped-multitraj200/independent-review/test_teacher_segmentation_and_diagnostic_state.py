"""Independent local actual-Qwen tests, not native GPU acceptance."""
from pathlib import Path
import copy,importlib.util,sys,unittest
from contextlib import nullcontext
import torch
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_design_cases',P/'test_independent_design_math.py')
case_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(case_module)
from search_r1.baselines.distillation import teacher_inputs,opsd_loss
from search_r1.baselines.trajectory_archive import isolated_rng,isolated_generation_state,short_teacher_records,generate_rank_teacher_diagnostics
from search_r1.baselines.full_state import rng_state,state_digest

class DiagnosticTokenizer(case_module.TinyQwenResponseProjection.Tokenizer):
    eos_token_id=2
    all_special_ids=[0,1,2]
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):return [1,7,8,9]
    def decode(self,ids,skip_special_tokens=False):return ','.join(map(str,ids))

class IndependentTeacherTests(unittest.TestCase):
    def case(self,checkpoint=False):
        student,teacher,data,settings=case_module.TinyQwenResponseProjection().case(checkpoint)
        data={k:v.repeat(2,1) for k,v in data.items()}
        return student,teacher,data,settings
    def compare_grad(self,teacher_micro,checkpoint=False):
        student,teacher,data,settings=self.case(checkpoint); other=copy.deepcopy(student)
        tok=DiagnosticTokenizer(); count=data['loss_mask'].sum()
        settings.response_only_logits=False;settings.teacher_forward_micro_batch=None
        old=opsd_loss(student,teacher,data,tok,None,settings,count);old.backward()
        settings.response_only_logits=True;settings.teacher_forward_micro_batch=teacher_micro
        new=opsd_loss(other,teacher,data,tok,None,settings,count);new.backward()
        torch.testing.assert_close(old,new,rtol=1e-6,atol=1e-7)
        ga=dict(student.named_parameters());gb=dict(other.named_parameters())
        for name in ga:torch.testing.assert_close(ga[name].grad,gb[name].grad,rtol=3e-5,atol=3e-7,msg=name)
        self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
    def test_local4_teacher_micro2_all_parameter_gradients(self):self.compare_grad(2)
    def test_local4_teacher_micro1_all_parameter_gradients(self):self.compare_grad(1)
    def test_local4_teacher_micro2_checkpointed_all_parameter_gradients(self):self.compare_grad(2,True)
    def test_teacher_hidden_full_vocabulary_q_and_alignment_same(self):
        _,teacher,data,settings=self.case(); tok=DiagnosticTokenizer(); width=data['responses'].shape[1]
        ids,mask,pos=teacher_inputs(tok,data['prompts'],data['responses'],data['attention_mask'][:,-width:],None,64,'legacy')
        with torch.no_grad():
            full=teacher.model.base_model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
            split=torch.cat([teacher.model.base_model(input_ids=ids[i:i+2],attention_mask=mask[i:i+2],use_cache=False).last_hidden_state for i in range(0,len(ids),2)])
            torch.testing.assert_close(full,split,rtol=1e-6,atol=1e-7)
            take=pos.clamp(0,full.shape[1]-1)[...,None].expand(-1,-1,full.shape[-1])
            q0=teacher.model.lm_head(full.gather(1,take)).softmax(-1)
            q1=teacher.model.lm_head(split.gather(1,take)).softmax(-1)
            torch.testing.assert_close(q0,q1,rtol=1e-6,atol=1e-7)
    def test_real_qwen_generate_preserves_next_q_loss_gradients_and_all_rng(self):
        s,t,d,settings=self.case(); other=copy.deepcopy(s);tok=DiagnosticTokenizer()
        settings.response_only_logits=True;settings.teacher_forward_micro_batch=2
        count=d['loss_mask'].sum()
        old=opsd_loss(s,t,d,tok,None,settings,count);old.backward()
        params=state_digest(t.model.state_dict()); rng=state_digest(rng_state())
        q=dict(question_uid='nq:train:10',question_position=0,data_source='nq',question='independent actual Qwen CPU generation',accepted_answers=['gold'],accepted_answers_provenance='explicit synthetic fixture')
        with isolated_rng(91) as rw,isolated_generation_state(t.model) as mw:
            rows=short_teacher_records(t.model,tok,[q],torch.device('cpu'),max_new_tokens=8)
        self.assertEqual(params,state_digest(t.model.state_dict()))
        self.assertEqual(rng,state_digest(rng_state()))
        self.assertTrue(rw['rng_restored'] and mw['model_state_restored'])
        self.assertTrue(rows and 1<=len(rows[0]['output_token_ids'])<=8)
        new=opsd_loss(other,t,d,tok,None,settings,count);new.backward()
        torch.testing.assert_close(old,new,atol=1e-7,rtol=1e-6)
        for name,p in s.named_parameters():torch.testing.assert_close(p.grad,dict(other.named_parameters())[name].grad,atol=3e-7,rtol=3e-5,msg=name)
    def test_old_no_meta_is_true_noop_and_no_model_rng_access(self):
        class Forbidden:
            def __getattribute__(self,name):raise AssertionError('No-op touched '+name)
        from types import SimpleNamespace
        state=state_digest(rng_state())
        self.assertEqual(generate_rank_teacher_diagnostics(Forbidden(),SimpleNamespace(meta_info={}),Forbidden()),{})
        self.assertEqual(state,state_digest(rng_state()))
    def test_teacher_micro_invalid_types_refused(self):
        student,teacher,data,settings=self.case();tok=DiagnosticTokenizer()
        for value in (True,False,0,-1,1.0,'2'):
            settings.teacher_forward_micro_batch=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'positive integer'):
                opsd_loss(student,teacher,data,tok,None,settings,data['loss_mask'].sum())

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main(verbosity=2)
