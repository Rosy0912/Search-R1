"""CPU behavioral/provenance fixtures; no Torch/CUDA/model or real experiment data."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
try:
    from . import training_metrics as train, evaluation as ev, result_analysis as stats, evaluation_worker as worker
    from .checkpoint import exclusive_json, fileproof, object_sha
except ImportError:
    import training_metrics as train, evaluation as ev, result_analysis as stats, evaluation_worker as worker
    from checkpoint import exclusive_json, fileproof, object_sha


class Core:
    @staticmethod
    def reward(response, answers, spans):
        return int(bool(spans) and response[spans[-1][0]:spans[-1][1]] == '<answer>'+answers[0]+'</answer>')


class QA:
    @staticmethod
    def extract_final_answer(spans):
        if not spans or not spans[-1].startswith('<answer>') or not spans[-1].endswith('</answer>'):
            return None
        return spans[-1][8:-9]


def cohort():
    return [dict(question_id='q'+str(i), question='question '+str(i), answers=['gold'],
                 data_source='nq' if i < 256 else 'hotpotqa') for i in range(512)]


def train_rows(questions):
    rows=[]
    for i,q in enumerate(questions):
        for sibling in range(5):
            correct=i%2==0 and sibling<3
            text='<answer>'+('gold' if correct else 'wrong')+'</answer>'
            rows.append(dict(q,sibling=sibling,response=text,policy_spans=[[0,len(text)]],reward=int(correct),search_calls=sibling%4,termination='answer'))
    return rows


def endpoint(request,data):
    return [dict(primary=True,status='complete',evaluation_id=request['evaluation_id'],run_id='r',
        version=request['version'],completed_outer=request['completed_outer'],validation_index=i,rank=i%8,slot=i//8,
        question_id=q['question_id'],dataset=q['data_source'],row_semantic_sha256=object_sha(q),
        protocol_em=int(i%2==0),protocol_valid_final_action=True,failure_class=None if i%2==0 else 'valid_gold_mismatch',
        search_calls=i%4,termination='answer',hook_control=dict(accepted=True,excluded_from_primary_denominator=True),
        zero_effect_control=dict(accepted=True)) for i,q in enumerate(data)]


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.questions=cohort();self.rows=train_rows(self.questions);self.model={'sha256':'a'*64}
    def calculate(self,rows=None):
        return train.calculate(self.rows if rows is None else rows,self.questions,'G_POS',0,self.model,Core)
    def test_allwrong_and_unselected_in_denominator(self):
        result=self.calculate();self.assertEqual(result['overall']['strict_correct'],768)
        self.assertEqual(result['overall']['denominator'],2560);self.assertEqual(result['overall']['em'],.3)
        self.assertEqual(result['all5_wrong'],256);self.assertFalse(result['post_training_em'])
        self.assertEqual(result['measurement_time'],'outer_start_sampling_before_any_this_outer_update')
    def test_equal_question_mean(self):
        r=self.calculate();self.assertAlmostEqual(r['overall']['em'],sum(q['s_over_5'] for q in r['per_question'])/512)
    def test_allwrong_complete_still_valid(self):
        rows=train_rows(self.questions)
        for r in rows:r.update(response='<answer>wrong</answer>',policy_spans=[[0,22]],reward=0)
        result=self.calculate(rows);self.assertEqual(result['overall']['em'],0);self.assertEqual(result['all5_wrong'],512)
    def test_partial_rejected(self):
        with self.assertRaises(ValueError):self.calculate(self.rows[:-1])
    def test_duplicate_rejected(self):
        rows=copy.deepcopy(self.rows);rows[-1]=rows[0]
        with self.assertRaises(ValueError):self.calculate(rows)
    def test_wrong_stored_reward_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['reward']=0
        with self.assertRaises(ValueError):self.calculate(rows)
    def test_bool_reward_not_integer(self):
        rows=copy.deepcopy(self.rows);rows[0]['reward']=True
        with self.assertRaises(ValueError):self.calculate(rows)
    def test_gold_changed(self):
        rows=copy.deepcopy(self.rows);rows[0]['answers']=['different']
        with self.assertRaises(ValueError):self.calculate(rows)
    def test_model_identity_required(self):
        with self.assertRaises(ValueError):train.calculate(self.rows,self.questions,'G_POS',0,{},Core)
    def test_25_outer_range(self):
        self.assertEqual(train.calculate(self.rows,self.questions,'H_NEG',24,self.model,Core)['displayed_iteration'],25)
        with self.assertRaises(ValueError):train.calculate(self.rows,self.questions,'H_NEG',25,self.model,Core)
    def test_complete_summary_counts_every_batch_and_tail(self):
        sampling=self.calculate()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);token=root/'token.jsonl';token.write_text('{}\n');token_ref=fileproof(token)
            refs=[]
            for update,indices in ((1,range(32)),(2,[32])):
                metrics=[dict(pair_index=i,token_log=token_ref,valid_tokens=2,raw_loss=1.,post_loss=2.,
                              weighted_post_loss=2.*(1.+i/100),pair_weight=1.+i/100) for i in indices]
                value=dict(schema='opsd-online-real-update-v1',accepted=True,outer_index=0,run_id='r',optimizer_updates=update,
                    global_actual_pairs=len(metrics),all_rank_actual_token_metrics=[dict(rank=r,pairs=[m for m in metrics if m['pair_index']%4==r]) for r in range(4)],
                    scaler={'grad_norm':.1})
                refs.append(exclusive_json(root/('update'+str(update)+'.json'),value))
            cp=exclusive_json(root/'cp.json',dict(outer_update_receipts=refs,cursor=dict(next_pair_cursor=33,usable_pair_count=33)))
            completion=dict(accepted=True,run_id='r',outer_index=0,P=40,K=33,optimizer_updates_delta=2,optimizer_updates=2,cumulative_pair_exposure=33,checkpoint=cp)
            result=train.complete_outer_summary(sampling,completion)
            self.assertEqual(result['global_valid_tokens'],66);self.assertEqual(result['K'],33)
            self.assertAlmostEqual(result['actual_outer_unweighted_post_loss'],2.)
            self.assertAlmostEqual(result['weight']['mean'],1.16)
    def test_complete_zero_K_has_no_fabricated_loss(self):
        sampling=self.calculate()
        with tempfile.TemporaryDirectory() as directory:
            cp=exclusive_json(Path(directory)/'cp.json',dict(outer_update_receipts=[],cursor=dict(next_pair_cursor=0,usable_pair_count=0)))
            completion=dict(accepted=True,run_id='r',outer_index=0,P=2,K=0,optimizer_updates_delta=0,optimizer_updates=0,cumulative_pair_exposure=0,checkpoint=cp)
            r=train.complete_outer_summary(sampling,completion)
            self.assertIsNone(r['actual_outer_raw_loss']);self.assertIsNone(r['weight']['ess']);self.assertTrue(r['empty_outer_no_adam'])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data=cohort();self.cfg={'run_id':'r','variant':'G_POS'}
        self.request={'evaluation_id':'e','version':'G_POS','completed_outer':15}
        self.rows=endpoint(self.request,self.data)
    def aggregate(self,rows=None):return ev.aggregate(self.rows if rows is None else rows,self.data,self.cfg,self.request,verify_traces=False)
    def test_native512_counts(self):
        result=self.aggregate();self.assertEqual(result['overall']['hits'],256)
        self.assertEqual(result['primary_slots'],512);self.assertEqual(result['datasets']['nq']['denominator'],256)
    def test_only_requested_times(self):
        with self.assertRaises(ValueError):ev.input_identity({},5)
    def test_duplicate_slot(self):
        rows=copy.deepcopy(self.rows);rows[-1]=rows[0]
        with self.assertRaises(ValueError):self.aggregate(rows)
    def test_wrong_logical_partition(self):
        rows=copy.deepcopy(self.rows);rows[9]['rank']=0
        with self.assertRaises(ValueError):self.aggregate(rows)
    def test_partial512(self):
        with self.assertRaises(ValueError):self.aggregate(self.rows[:-1])
    def test_hook_control_is_gate(self):
        rows=copy.deepcopy(self.rows);rows[0]['hook_control']['accepted']=False
        with self.assertRaises(ValueError):self.aggregate(rows)
    def test_zero_effect_control_is_gate(self):
        req={'evaluation_id':'e','version':'source100','completed_outer':0};rows=endpoint(req,self.data)
        rows[0]['zero_effect_control']['accepted']=False
        with self.assertRaises(ValueError):ev.aggregate(rows,self.data,self.cfg,req,False)
    def test_gold_mismatch_not_invalid(self):
        decoded=train.protocol(Core,QA,dict(response='<answer>wrong</answer>',policy_spans=[[0,22]]),['gold'])
        self.assertEqual(decoded['failure_class'],'valid_gold_mismatch')
    def test_unclosed_tag_not_valid(self):
        decoded=train.protocol(Core,QA,dict(response='<answer>gold',policy_spans=[[0,12]]),['gold'])
        self.assertEqual(decoded['failure_class'],'final_invalid')
    def test_invalid_offsets_rejected(self):
        with self.assertRaises(ValueError):train.protocol(Core,QA,dict(response='abc',policy_spans=[[-1,3]]),['gold'])
    def test_nullhook_preserves_return(self):
        class Model:
            def generate(self,*args,**kwargs):return self.result
        class Tokenizer:
            def __call__(self,*args,**kwargs):return types.SimpleNamespace(input_ids=[1,2])
        model=Model();model.result=object();agent=types.SimpleNamespace(model=model,tokenizer=Tokenizer())
        rollout=types.SimpleNamespace(requests=types.SimpleNamespace())
        original=agent.tokenizer
        with worker.capture_native(agent,rollout) as trace:
            self.assertIs(model.generate(foo=1),model.result)
        self.assertTrue(trace['hooks_restored']);self.assertIs(agent.tokenizer,original)
        self.assertFalse(trace['generation_options_modified']);self.assertFalse(trace['logging_errors'])
    def test_shared_zero_requires_parent_wait_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=cohort();datafile=root/'validation.jsonl'
            datafile.write_text(''.join(json.dumps(q)+'\n' for q in data));validation=fileproof(datafile)
            source=[]
            for i in range(12):
                path=root/('source'+str(i));path.write_text('cpu_fixture');source.append(fileproof(path))
            source_manifest=exclusive_json(root/'model-manifest.json',{'files':source})
            payload=[]
            for name in ('adapter_config.json','adapter_model.safetensors'):
                path=root/name;path.write_text('cpu_fixture');payload.append(fileproof(path))
            initial=exclusive_json(root/'initial.json',dict(adapter_tensor_sha256='a'*64,payload_files=payload))
            runtime={'path':str(root/'runtime'),'sha256':'b'*64}
            cfg=dict(study_id='study',source_model_manifest=source_manifest,runtime_manifest=runtime,validation_identity=validation,
                     common_initial_adapter_manifest=initial,common_initial_adapter_tensor_sha256='a'*64,owner={'uid':1001})
            request=dict(evaluation_id='e',version='source100',completed_outer=0);records=endpoint(request,data)
            resultfile=root/'results.jsonl';resultfile.write_text(''.join(json.dumps(r)+'\n' for r in records))
            summary=exclusive_json(root/'summary.json',dict(accepted=True,primary_slots=512,completed_outer=0,overall={'denominator':512}))
            child=exclusive_json(root/'child.json',dict(accepted=True,study_id='study',run_id='r',config={'sha256':'c'*64},
                completed_outer=0,version='source100',primary_slots=512,actual_child_exit_and_release=True,summary=summary,results=fileproof(resultfile),
                input_identity=dict(protocol=ev.PROTOCOL,source_model_manifest=source_manifest,runtime={'manifest':runtime},
                    validation=validation,common_initial=initial,expected_adapter_tensor_sha256='a'*64)))
            exits=[dict(actual_wait_returncode=0,identity={'uid':1001,'starttick':123})]
            terminal=exclusive_json(root/'terminal.json',dict(child_exit_evidence=exits))
            parentbody=dict(schema='opsd-signal-evaluation-parent-v1',accepted=True,run_id='r',config_sha256='c'*64,
                completed_outer=0,raw_checkpoint=None,child=child,isolated_child_no_training_rng_consumed=True,raw_state_unchanged=True,parent_torch_imported=False,
                exit_proof=dict(accepted=True,actual_terminal=terminal,child_exit_evidence=exits,gpu_release={'all_owned_release':{'accepted':True}}))
            parent=exclusive_json(root/'parent.json',parentbody)
            wrapper=dict(schema='opsd-signal-common-zero-reference-v1',accepted=True,child=child,parent=parent,study_id='study',
                source_model_identity=source_manifest['sha256'],runtime_identity=runtime['sha256'],validation_identity=validation,common_initial_adapter=initial)
            shared=exclusive_json(root/'shared.json',wrapper)
            with patch.object(ev.prepare,'verify_runtime',return_value={'manifest':runtime}):
                self.assertTrue(ev.verify_common_zero(cfg,shared)['accepted'])
                bad=copy.deepcopy(parentbody);bad['exit_proof']['child_exit_evidence'][0]['actual_wait_returncode']=2
                wrapper['parent']=exclusive_json(root/'badparent.json',bad)
                badref=exclusive_json(root/'badshared.json',wrapper)
                with self.assertRaises(ValueError):ev.verify_common_zero(cfg,badref)
    def test_adapter_alpha_and_forward_flags_not_just_tensor_digest(self):
        targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        cfg={'scientific':dict(lora_rank=16,lora_alpha=32,lora_dropout=0,target_modules=targets)}
        value=dict(r=16,lora_alpha=32,lora_dropout=0.,target_modules=targets,task_type='CAUSAL_LM',peft_type='LORA',
                   use_rslora=False,use_dora=False,bias='none',fan_in_fan_out=False,base_model_name_or_path='/remote/source')
        self.assertTrue(ev.validate_adapter_config(value,cfg)['accepted'])
        for key,wrong in [('lora_alpha',16),('r',8),('lora_dropout',.1),('use_rslora',True),('use_dora',True),
                          ('bias','all'),('fan_in_fan_out',True),('task_type','SEQ_CLS'),('target_modules',targets[:-1]),
                          ('rank_pattern',{'q_proj':8}),('alpha_pattern',{'q_proj':16})]:
            bad=dict(value);bad[key]=wrong
            with self.assertRaises(ValueError,msg=key):ev.validate_adapter_config(bad,cfg)
    def test_actual_layer_scaling_bad_rejected_despite_same_tensor_dtype(self):
        def layer():
            A=types.SimpleNamespace(weight=types.SimpleNamespace(dtype='torch.float32',shape=(16,8)))
            B=types.SimpleNamespace(weight=types.SimpleNamespace(dtype='torch.float32',shape=(8,16)))
            return types.SimpleNamespace(lora_A={'x':A},lora_B={'x':B},r={'x':16},lora_alpha={'x':32},
                scaling={'x':2.},lora_dropout={'x':types.SimpleNamespace(p=0.)})
        modules=[('l'+str(i),layer()) for i in range(252)]
        model=types.SimpleNamespace(named_modules=lambda:iter(modules))
        self.assertTrue(worker.validate_loaded_lora(model,'x')['accepted'])
        modules[41][1].scaling['x']=1.
        with self.assertRaises(ValueError):worker.validate_loaded_lora(model,'x')


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_extremes(self):
        self.assertEqual(stats.exact_mcnemar(0,0),1.)
        self.assertEqual(stats.exact_mcnemar(2,2),1.)
        self.assertEqual(stats.exact_mcnemar(0,5),.0625)
        self.assertEqual(stats.exact_mcnemar(0,6),.03125)
    def test_holm_two_family(self):
        result=stats.holm({'a':.01,'b':.04});self.assertEqual(result['a']['holm_p'],.02)
        self.assertEqual(result['b']['holm_p'],.04);self.assertTrue(result['b']['reject_at_family_alpha'])
        self.assertFalse(stats.holm({'a':.03,'b':.04})['a']['reject_at_family_alpha'])
    def test_bootstrap_stratified_and_paired(self):
        d={'a':[1,1,-1,-1],'b':[-1,-1,1,1]};s=['nq','nq','hotpotqa','hotpotqa']
        result=stats.paired_bootstrap(d,s,100,42)
        self.assertEqual(result['a']['low'],0);self.assertEqual(result['a']['high'],0)
        self.assertEqual(result,stats.paired_bootstrap(d,s,100,42))
    def test_constant_label_and_feature_are_NA(self):
        self.assertIsNone(stats.pearson([1,2],[1,1]));self.assertIsNone(stats.pearson([1,1],[0,1]))
        self.assertIsNone(stats.auc([1,2],[1,1]));self.assertIsNone(stats.auc([],[]))
    def test_auc_ties_and_direction(self):
        self.assertEqual(stats.auc([0,1],[0,1]),1);self.assertEqual(stats.auc([0,1],[1,0]),0)
        self.assertEqual(stats.auc([1,1],[0,1]),.5)
    def test_spearman_midranks(self):
        self.assertEqual(stats.ranks([3,1,1]),[3,1.5,1.5])
    def test_full4608_endpoint_budget(self):
        data=cohort();rows=[]
        for version,completed in [('source100',0)]+[(v,t) for v in train.VARIANTS for t in (15,25)]:
            rows.extend(endpoint(dict(evaluation_id='e',version=version,completed_outer=completed),data))
        result=stats.analyze(rows,repetitions=20)
        self.assertEqual(result['formal_predictions'],4608);self.assertEqual(result['primary_family_size'],2)
        for value in result['paired'].values():
            self.assertEqual(value['net_hits'],0)
            if value['primary_comparison']:self.assertEqual(value['exact_mcnemar_holm']['raw_p'],1.)
    def test_partial_endpoint_rejected(self):
        with self.assertRaises(ValueError):stats.analyze([],10)
    def test_actual_relations_single_label(self):
        rows=[dict(variant='G_POS',outer_index=0,student_reward=1,success_count=5,gap=1.,entropy=2.,weight=1.)]
        r=stats.relationships(rows)['buckets']['G_POS']['features']['gap']
        self.assertIsNone(r['em_pearson']);self.assertIsNone(r['original_increasing_direction_em_auc'])
        self.assertEqual(r['rows'],1)
    def test_numerical_fault_not_missing(self):
        with self.assertRaises(ValueError):stats.relationships([dict(variant='G_POS',outer_index=0,student_reward=1,success_count=5,gap=float('nan'))])


if __name__=='__main__':unittest.main()
