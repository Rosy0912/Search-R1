"""CPU tests of real scorer forward, state restore and weighted native gradient."""
import copy
import importlib.util
import os
import hashlib
from pathlib import Path
import random
import unittest
import numpy as np
import torch
import sibling_scores as scores
import objective
from precision import PrecisionContract
from targets import SharedTargets
from test_targets import Toy

class Tok:
    def __call__(self,text,**kwargs):
        return dict(input_ids=[ord(c) for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])
    def apply_chat_template(self,prompt,**kwargs):return [1,2] if prompt[1] is None else [1,3,4]
    def decode(self,ids):return ''.join(chr(i) for i in ids)
class Core:
    @staticmethod
    def prompt(question,reference):return question,reference
    @staticmethod
    def response_encoding(tokenizer,response,spans):
        return tokenizer(response)['input_ids'],[int(any(a<=i and i+1<=b for a,b in spans)) for i in range(len(response))]
def fixture(responses=None,selected=2):
    responses=responses or ['<think>Q</think><answer>'+x+'</answer>' for x in 'ABCDE']
    rows=[dict(question_id='q',question='Q?',answers=['A'],sibling=j,response=text,reward=int(j==0),
               policy_spans=[[0,len(text)]],search_calls=0,token_count=len(text)) for j,text in enumerate(responses)]
    shas=[objective.canonical_sha(row) for row in rows]
    pair=dict(question_id='q',question='Q?',answers=['A'],reference=rows[0]['response'],student=rows[selected]['response'],
        reference_reward=1,student_reward=rows[selected]['reward'],student_policy_spans=rows[selected]['policy_spans'],pair_weight=1.,
        mining=dict(student_sibling=selected,reference_sibling=0,sibling_row_sha256=shas,reference_row_sha256=shas[0],student_row_sha256=shas[selected]))
    enc,reason=objective.encoded_pair(Tok(),pair,Core);assert reason is None
    group=scores.encode_sibling_group(Tok(),pair,rows,Core,training_encoded=enc)
    return rows,pair,enc,group
def scored(m,group,variant='G_POS',c=None,t=None):
    c=c or PrecisionContract(device_type='cpu');t=t or SharedTargets(m,c)
    receipt=scores.score_sibling_group(m,group,variant,precision=c,targets=t,
        preupdate_identity={'adapter_tensor_sha256':'a'*64,'optimizer_update':7})
    return receipt,c,t

class ScorerTests(unittest.TestCase):
    def test_region_logic_exact_original_and_invalid_final_payload(self):
        path=Path(os.environ.get('OPSD_SIGNAL_REGION_FIXTURE') or str(Path(__file__).resolve().parent.parent/'opsd_first5_em_loss_analysis_v1/extract.py'))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'59491b6d5e93799e80a9e892b60196f28082256ba38831d84e6d964b9289124c')
        spec=importlib.util.spec_from_file_location('_original_regions',path);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        for text in ('<think>x<answer>A</answer>','<answer>A</answer><answer>B</answer>',
                     '<answer></answer>','<search>q</search><information>d</information><answer>A</answer>'):
            spans=[[0,len(text)]];segments=scores.character_regions(text,spans)
            self.assertEqual(segments,original.character_regions(text,spans))
            for offset in ((0,1),(0,len(text)),(len(text),len(text))):self.assertEqual(scores.category(offset,segments),original.category(offset,segments))
        text='<think>x<answer>A</answer>'
        self.assertIn('final_answer',[scores.category((i,i+1),scores.character_regions(text,[[0,len(text)]])) for i in range(len(text))])
    def test_original_identity_same_reference_selected_encoding_no_alias(self):
        rows,pair,enc,g=fixture();self.assertEqual(g['encoded'][2],enc)
        self.assertEqual([x['prefixes'][1] for x in g['encoded']],[[1,3,4]]*5)
        bad=copy.deepcopy(rows);bad[1]['response']+='X'
        with self.assertRaises(ValueError):scores.encode_sibling_group(Tok(),pair,bad,Core)
        enc['mask'][0]=0;self.assertNotEqual(g['encoded'][2]['mask'],enc['mask'])
        with self.assertRaises(ValueError):scores.encode_sibling_group(Tok(),pair,rows,Core,training_encoded=enc)
    def test_all5_full_entropy_answer_gap_and_detached_state(self):
        _,_,enc,g=fixture();m=Toy();r,c,t=scored(m,g);item=r['selected_score']
        self.assertEqual(len(r['siblings']),5);self.assertAlmostEqual(sum(r['normalization']['weights']),5.)
        with t.current_graph():
            s,q,_=t.hidden(enc);head=m.get_output_embeddings();i=item['answer_token_indices'][0]
            gap=(head(s[i:i+1]).float().log_softmax(-1)-head(q[i:i+1]).float().log_softmax(-1))[0,enc['response'][i]].item()
            pos=torch.tensor(enc['mask'],dtype=torch.bool).nonzero().flatten();lp=head(s[pos]).float().log_softmax(-1)
            ent=(-(lp.exp()*lp).sum(-1)).mean().item()
        self.assertAlmostEqual(item['answer_gap'],gap,places=5);self.assertAlmostEqual(item['student_entropy'],ent,places=5)
        self.assertTrue(r['state_restore']['state_restored']);self.assertFalse(r['state_restore']['rng_consumed'])
        self.assertEqual(r['state_restore']['rng_before_sha256'],r['state_restore']['restored_rng_sha256'])
        self.assertTrue(all(p.grad is None for p in m.parameters()))
    def test_no_payload_NA_not_zero_and_entropy_available(self):
        _,_,_,g=fixture(['<answer>A</answer>','<think>B</think>','<answer></answer>','<answer>C</answer>','<answer>D</answer>'])
        m=Toy();r,_,_=scored(m,g);h,_,_=scored(m,g,'H_POS')
        self.assertIsNone(r['siblings'][1]['answer_gap']);self.assertIsNone(r['siblings'][2]['answer_gap'])
        self.assertEqual(r['normalization']['weights'][1:3],[1.,1.]);self.assertTrue(all(x['student_entropy'] is not None for x in h['siblings']))
    def test_any_illegal_sibling_wholegroup_neutral_without_forward_or_D_change(self):
        rows,pair,enc,_=fixture(['<answer>A</answer>']+['<think>'+'X'*50+'</think><answer>B</answer>']*3+['<answer>E</answer>'],4)
        g=scores.encode_sibling_group(Tok(),pair,rows,Core,training_encoded=enc,max_context=40)
        self.assertEqual(g['neutral_reason'],'sibling_encoding_failure');self.assertEqual(g['encoded'][4],enc)
        m=Toy();calls=[0];hook=m.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        r,_,_=scored(m,g,'H_NEG');hook.remove();self.assertEqual(calls[0],0);self.assertEqual(r['normalization']['weights'],[1.]*5)
    def test_modes_flags_adapter_precision_and_all_CPU_rng_restored(self):
        _,_,_,g=fixture();m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c)
        m.train();m.head.eval();m.lora_A['default'].weight.requires_grad_(False);c.role='before';c.stage='before-stage'
        flags=[p.requires_grad for p in m.parameters()];modes=[x.training for x in m.modules()];rng=scores._rng_state(torch)
        r,_,_=scored(m,g,c=c,t=t)
        self.assertEqual(flags,[p.requires_grad for p in m.parameters()]);self.assertEqual(modes,[x.training for x in m.modules()])
        self.assertEqual((c.role,c.stage),('before','before-stage'));self.assertEqual(m.active_adapter,'default')
        self.assertTrue(scores._rng_equal(rng,scores._rng_state(torch),torch))
    def test_nonfinite_is_fault_and_error_restores_flags_modes(self):
        _,_,_,g=fixture();m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c)
        m.eval();m.embedding.train();m.lora_A['default'].weight.requires_grad_(False)
        flags=[p.requires_grad for p in m.parameters()];modes=[x.training for x in m.modules()]
        with torch.no_grad():m.head.weight[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):scored(m,g,c=c,t=t)
        self.assertEqual(flags,[p.requires_grad for p in m.parameters()]);self.assertEqual(modes,[x.training for x in m.modules()])
    def test_rng_consumption_fault_restores_exact_state(self):
        _,_,_,g=fixture();m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c);rng=scores._rng_state(torch)
        hook=m.register_forward_pre_hook(lambda *args:(random.random(),np.random.random(),torch.rand(1)) and None)
        with self.assertRaisesRegex(ValueError,'consumed production RNG'):scored(m,g,c=c,t=t)
        hook.remove();self.assertTrue(scores._rng_equal(rng,scores._rng_state(torch),torch))
    def test_overlap_and_gradient_mutation_rejected(self):
        _,_,_,g=fixture();m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c)
        with t.current_graph():
            with self.assertRaisesRegex(ValueError,'overlap'):scored(m,g,c=c,t=t)
        m.lora_A['default'].weight.grad=torch.ones_like(m.lora_A['default'].weight)
        def mutate(module,args):
            with torch.no_grad():m.lora_A['default'].weight.grad.add_(1)
        hook=m.register_forward_pre_hook(mutate)
        with self.assertRaisesRegex(ValueError,'mutated'):scored(m,g,c=c,t=t)
        hook.remove()
    def test_scorer_matches_native_train_and_auxiliary_records_ignored(self):
        _,_,enc,g=fixture();m=Toy();r,c,t=scored(m,g,'G_NEG');rows=[];w=r['selected_pair_weight']
        with t.current_graph():
            loss,stats=objective.pair_objective(m,enc,10,32,7,record=rows.append,precision=c,targets=t,pair_weight=w)
            t.mark_backward();(loss/7).backward()
        comparison=scores.compare_selected_training_metrics(r,rows+[{'token_index':0,'scope':'full_vocab'}])
        self.assertTrue(comparison['accepted']);self.assertAlmostEqual(loss.item(),w*stats['post_loss'],places=6)
        self.assertAlmostEqual(sum(x['global_batch_contribution'] for x in rows),loss.item()/7,places=6)
        rows[0]['student_entropy']+=1
        with self.assertRaises(ValueError):scores.compare_selected_training_metrics(r,rows)
    def test_detached_weight_gradient_and_clip_order(self):
        z=torch.tensor([[1.,-.2]],requires_grad=True);q=torch.tensor([[-100.,0.]])
        w=(1+.1*z[0,0]).detach();raw=objective.vocabulary_metrics(z,q,10.)['post'].sum()
        expected=torch.autograd.grad(raw,z,retain_graph=True)[0]*w;actual=torch.autograd.grad(w*raw,z,retain_graph=True)[0]
        self.assertTrue(torch.allclose(expected,actual));wrong=torch.autograd.grad((1+.1*z[0,0])*raw,z,retain_graph=True)[0]
        self.assertFalse(torch.allclose(actual,wrong));lp=z.log_softmax(-1);target=q.log_softmax(-1)
        self.assertFalse(torch.allclose(w*raw,(w*lp.exp()*(lp-target)).clamp(max=10).sum()))
    def test_objective_explicitly_detaches_tensor_weight(self):
        _,_,enc,_=fixture();m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c)
        w=torch.tensor(1.4,requires_grad=True)
        with t.current_graph():
            loss,stats=objective.pair_objective(m,enc,10,32,7,precision=c,targets=t,pair_weight=w)
            t.mark_backward();(loss/7).backward()
        self.assertIsNone(w.grad);self.assertEqual(stats['pair_weight'],float(w.detach()))
        for bad in (True,0.,float('nan'),torch.ones(2)):
            with self.assertRaises(ValueError):objective.detached_pair_weight(bad)

    def test_bf16_base_fp32_adapter_and_old_diagnostic_match_native(self):
        from test_targets import old_identity
        class BF16Toy(Toy):
            def forward(self,input_ids,use_cache=False):
                from types import SimpleNamespace
                x=self.embedding(input_ids)
                if not self.disabled:x=x+self.lora_A[self.active_adapter](x.float()).to(x.dtype)
                return SimpleNamespace(last_hidden_state=x)
        _,_,enc,g=fixture();m=BF16Toy();m.embedding.to(torch.bfloat16);m.head.to(torch.bfloat16)
        c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c,old_adapter='outer_old',old_identity=old_identity(m))
        r,_,_=scored(m,g,'H_NEG',c,t);rows=[]
        with t.current_graph():
            loss,stats=objective.pair_objective(m,enc,10,32,31,record=rows.append,precision=c,targets=t,pair_weight=r['selected_pair_weight'])
            t.mark_backward();(loss/31).backward()
        self.assertTrue(scores.compare_selected_training_metrics(r,rows)['accepted'])
        self.assertEqual(m.lora_A['default'].weight.grad.dtype,torch.float32)
        self.assertIsNone(m.lora_A['outer_old'].weight.grad)

    def test_actual_tail_denominator_four_rank_SUM_not_weight_sum(self):
        for n in (1,7,31,32):
            z=torch.tensor([[.2,-.3]]*n,requires_grad=True);q=torch.tensor([[.9,-.7]]*n);w=torch.linspace(.2,1.4,n)
            losses=objective.vocabulary_metrics(z,q,10.)['post'];expected=torch.autograd.grad((w*losses).sum()/n,z,retain_graph=True)[0]
            summed=torch.zeros_like(z)
            for rank in range(4):
                indices=list(range(rank,n,4))
                if indices:summed+=torch.autograd.grad((w[indices]*losses[indices]).sum()/n,z,retain_graph=True)[0]
            self.assertTrue(torch.allclose(expected,summed,atol=1e-7))
            wrong=torch.autograd.grad((w*losses).sum()/w.sum(),z)[0];self.assertFalse(torch.allclose(expected,wrong))
    def test_positive_negative_complement_same_current_model(self):
        _,_,_,g=fixture();m=Toy()
        for positive,negative in (('G_POS','G_NEG'),('H_POS','H_NEG')):
            a,_,_=scored(m,g,positive);b,_,_=scored(m,g,negative)
            self.assertEqual(a['normalization']['scores'],b['normalization']['scores'])
            self.assertTrue(all(abs(x+y-2)<1e-12 for x,y in zip(a['normalization']['weights'],b['normalization']['weights'])))
    def test_boundary_merged_token_not_counted_as_answer(self):
        class CrossTok(Tok):
            def __call__(self,text,**kwargs):
                value=super().__call__(text,**kwargs)
                if text.startswith('<answer>'):
                    value['input_ids']=value['input_ids'][:7]+[99]+value['input_ids'][10:]
                    value['offset_mapping']=value['offset_mapping'][:7]+[(7,10)]+value['offset_mapping'][10:]
                return value
        class OffsetCore(Core):
            @staticmethod
            def response_encoding(tokenizer,response,spans):
                result=tokenizer(response)
                return result['input_ids'],[int(any(a<=x and y<=b for a,b in spans)) for x,y in result['offset_mapping']]
        rows,pair,_,_=fixture(['<answer>'+letter+'</answer>' for letter in 'ABCDE'])
        enc,reason=objective.encoded_pair(CrossTok(),pair,OffsetCore);self.assertIsNone(reason)
        g=scores.encode_sibling_group(CrossTok(),pair,rows,OffsetCore,training_encoded=enc)
        self.assertEqual(g['answer_positions'],[[]]*5)
        self.assertIn('boundary',g['regions'][2]);m=Toy();r,c,t=scored(m,g);records=[]
        self.assertEqual(r['normalization']['weights'],[1.]*5)
        with t.current_graph():
            loss,_=objective.pair_objective(m,enc,10,32,1,record=records.append,precision=c,targets=t)
            t.mark_backward();loss.backward()
        self.assertTrue(scores.compare_selected_training_metrics(r,records)['checked'])

    def test_external_information_mask_excluded_from_entropy(self):
        texts=['<search>q</search><information>docs</information><answer>'+letter+'</answer>' for letter in 'ABCDE']
        rows,pair,_,_=fixture(texts)
        for row in rows:
            start=row['response'].index('<information>');end=row['response'].index('</information>')+len('</information>')
            row['policy_spans']=[[0,start],[end,len(row['response'])]]
        shas=[objective.canonical_sha(row) for row in rows]
        pair['student_policy_spans']=rows[2]['policy_spans']
        pair['mining'].update(sibling_row_sha256=shas,reference_row_sha256=shas[0],student_row_sha256=shas[2])
        enc,reason=objective.encoded_pair(Tok(),pair,Core);self.assertIsNone(reason)
        g=scores.encode_sibling_group(Tok(),pair,rows,Core,training_encoded=enc)
        r,_,_=scored(Toy(),g,'H_POS')
        self.assertEqual(r['selected_score']['valid_policy_token_indices'],[i for i,m in enumerate(enc['mask']) if m])
        self.assertEqual(r['selected_score']['valid_policy_tokens'],sum(enc['mask']))
        self.assertLess(r['selected_score']['valid_policy_tokens'],len(enc['response']))

    def test_repeat_scoring_budget_and_cold_identity(self):
        _,_,_,g=fixture();m=Toy();a,_,_=scored(m,g);b,_,_=scored(m,g)
        self.assertTrue(scores.compare_score_receipts(a,b)['accepted'])
        bad=copy.deepcopy(b);bad['normalization']['weights'][0]+=.002
        with self.assertRaisesRegex(ValueError,'weight budget'):scores.compare_score_receipts(a,bad)
        bad=copy.deepcopy(b);bad['siblings'][0]['answer_gap']+=.1
        with self.assertRaisesRegex(ValueError,'score budget'):scores.compare_score_receipts(a,bad)
        bad=copy.deepcopy(b);bad['preupdate_identity']['optimizer_update']+=1
        with self.assertRaisesRegex(ValueError,'identity'):scores.compare_score_receipts(a,bad)

    def test_missing_equal_floor_and_nonfinite_weight_math(self):
        for values in ([None]*5,[3.]*5,[None,1,None,None,None]):self.assertEqual(scores.weights(values,1)['weights'],[1.]*5)
        self.assertLess(max(abs(w-1) for w in scores.weights([i*1e-10 for i in range(5)],1)['weights']),.001)
        with self.assertRaises(ValueError):scores.weights([0,1,2,float('nan'),4],1,neutral_reason='failure')
        self.assertAlmostEqual(sum(scores.weights([None,-4,-2,0,None],1)['weights']),5.)

if __name__=='__main__':unittest.main()
