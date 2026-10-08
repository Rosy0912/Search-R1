"""Three CPU schema/committed-exposure fixtures; no Node, Torch, or QA data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).absolute().parent
c = types.ModuleType('collector_fixture'); c.__file__ = str(HERE/'collector.py')
exec(compile((HERE/'collector.py').read_bytes(),c.__file__,'exec'),c.__dict__)


def save(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();p.write_bytes(raw)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())


def receipt(step, pair_indices, cumulative):
    ranks=[]
    for rank in range(8):
        pairs=[dict(pair_index=i,valid_tokens=5,raw_loss=3.,post_loss=2.,pair_weight=1. if i==0 else 2.,
                    weighted_post_loss=2. if i==0 else 4.,student_to_teacher_raw_kl=3.,
                    student_to_outer_old_raw_kl=None,student_to_mixture_raw_kl=3.,clipped_vocabulary_count=3)
               for i in pair_indices[rank::8]]
        ranks.append(dict(rank=rank,pairs=pairs,grad_norm=.7))
    return dict(schema='opsd-online-real-update-v1',accepted=True,run_id='a'*32,outer_index=0,
        optimizer_updates=step,global_actual_pairs=len(pair_indices),global_valid_tokens=5*len(pair_indices),
        cumulative_pair_exposure=cumulative,contract={'fixture':'reverse'},all_rank_actual_token_metrics=ranks,
        scaler=dict(accepted=True,adam_advanced=True,overflow=False,grad_norm=.7,scaler_after={'scale':1.}),retries=0)


class Tests(unittest.TestCase):
    def test_committed_refs_only_dedup_tail_actual_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();r1=save(root/'attempt-success/update-1.json',receipt(1,list(range(32)),32))
            r2=save(root/'attempt-success/update-2.json',receipt(2,[32],33))
            save(root/'attempt-failed/unused-update.json',{'uncommitted':True})
            reader=c.Reader();facts,pairs,tokens,duplicates=c.updates([r1,r1,r2],reader,'a'*32,0,0,33,list(range(33)),{'fixture':'reverse'},set())
            self.assertEqual([r['global_actual_pairs'] for r in facts],[32,1])
            self.assertEqual((len(pairs),tokens,duplicates),(33,165,1))
            self.assertEqual(c.summarize_pairs(pairs)['weighted_pair_exposure'],65.)
            self.assertEqual(c.summarize_pairs(pairs)['equal_trajectory_means']['post_loss']['value'],2.)
            self.assertEqual(len(reader.finish()),2)
            with self.assertRaises(c.Rejected): c.updates([r1,r2],c.Reader(),'a'*32,0,0,33,list(reversed(range(33))),{'fixture':'reverse'},set())

    def test_zero_and_missing_metrics_are_na_nonfinite_counts_reject(self):
        facts,pairs,tokens,_=c.updates([],c.Reader(),'a'*32,0,0,0,[],{'fixture':'reverse'},set())
        self.assertEqual((facts,pairs,tokens),([],[],0))
        self.assertIsNone(c.summarize_pairs(pairs)['equal_trajectory_means']['post_loss']['value'])
        partial=c.summarize_pairs([{'pair_weight':1.,'raw_loss':3.},{'pair_weight':2.}])
        self.assertIsNone(partial['equal_trajectory_means']['raw_loss']['value'])
        self.assertEqual(partial['equal_trajectory_means']['raw_loss']['observed_trajectories'],1)
        self.assertIsNone(partial['clipped_vocabulary_count'])
        with self.assertRaises(c.Rejected):c.summarize_pairs([{'raw_loss':float('nan')}])
        for reward,P,classes,cc in [({'0':2560},0,dict(all_wrong=512,all_correct=0,mixed=0),0),
                                   ({'1':2560},512,dict(all_wrong=0,all_correct=512,mixed=0),512)]:
            stats=dict(questions=512,generated_trajectories=2560,P=P,classes=classes,
                       pair_classes=dict(CC=cc,CW=0),diagnostics={'strict_reward':reward})
            self.assertEqual(c.sampling_facts(stats)['strict_reward_mean'],float(bool(cc)))
            del stats['diagnostics']
            self.assertIsNone(c.sampling_facts(stats)['strict_reward_mean'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'receipt.json';v=receipt(1,[0],1);v['global_valid_tokens']=6;r=save(p,v)
            with self.assertRaises(c.Rejected):c.updates([r],c.Reader(),'a'*32,0,0,1,[0],{'fixture':'reverse'},set())

    def test_real_producer_shapes_partial_seven_scope_and_sha_reject(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();groups=[];configs=[]
            for n,variant in enumerate(c.VARIANTS):
                output=root/('group-'+variant)/'runs';run='a'*32 if n==0 else '%032x'%n
                cfg=dict(schema='opsd-ablation-config-v1',variant=variant,run_id=run,study_id='fixture',
                    scientific={'outer_iterations':10},paths={'output_root':str(output)},precision={'mode':'bf16'},
                    runtime_manifest={'sha256':'1'*64},source_model_manifest={'sha256':'2'*64})
                cr=save(output.parent/'receipts/config.json',cfg);configs.append((cfg,cr))
                groups.append(dict(variant=variant,run_id=run,config=cr,output_root=str(output)))
            cfg,cr=configs[0];output=Path(cfg['paths']['output_root']);outer=output/'outers/outer-00000'
            contract=dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',variant='C0',precision='bf16',mixture_alpha=0.)
            stats=dict(questions=512,generated_trajectories=2560,P=1,classes=dict(all_wrong=511,all_correct=0,mixed=1),
                pair_classes=dict(CC=0,CW=1),self_pair=0,response_text_equal=0,pair_weight_sum=1.,variant='C0',outer_index=0,
                diagnostics={'strict_reward':{'0':2559,'1':1}})
            pr=save(outer/'sampling/pairs.jsonl',{'opaque_fixture_only':True})
            om=save(outer/'outer-manifest.json',dict(run_id='a'*32,config_sha256=cr['sha256'],outer_index=0,
                contract=contract,pairs=pr,P=1,sampling_stats=stats,cumulative_optimizer_updates_before=0))
            enc=dict(P=1,K=1,skipped={},rows=['3'*64],sha256=c.canonical_sha(['3'*64]))
            er=save(outer/'training/encoding.json',enc);order=save(outer/'training/order.json',dict(outer_index=0,pair_indices=[0],encoded_sha256=enc['sha256']))
            v=receipt(1,[0],1);v['contract']=contract;ur=save(outer/'training/attempt-accepted/update-1.json',v)
            inputs=dict(outer_manifest=om,pairs=pr,encoding=er,order=order)
            cp=save(output/'checkpoints/update-00000001-outer-00000/manifest.json',dict(schema='opsd-online-checkpoint-v1',accepted=True,
                run_id='a'*32,config_sha256=cr['sha256'],optimizer_updates=1,cumulative_pair_exposure=1,protection_reason=['outer_end'],
                contract=contract,runtime_identity='1'*64,source_model_identity='2'*64,training_input=inputs,outer_update_receipts=[ur],
                cursor=dict(outer_index=0,encoded_sha256=enc['sha256'],order_sha256=order['sha256'],usable_pair_count=1,next_pair_cursor=1)))
            save(outer/'training/outer-complete.json',dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id='a'*32,outer_index=0,
                P=1,K=1,skipped={},optimizer_updates=1,optimizer_updates_delta=1,cumulative_pair_exposure=1,global_valid_tokens=5,
                checkpoint=cp,contract=contract,training_input=inputs))
            goal=save(root/'goal.json',dict(schema='opsd-ablation-outer-backup-goal-v1',accepted=True,cluster='HKUST',
                goal_scope='seven-group-outer-end10',study_id='fixture',groups=groups))
            result=c.collect(goal)
            self.assertEqual(result['coverage']['completed_outers'],1);self.assertFalse(result['coverage']['all70_complete'])
            self.assertEqual(result['totals'],dict(committed_unique_updates=1,raw_pair_exposure=1,valid_tokens=5))
            self.assertEqual(result['outers'][0]['sampling']['strict_reward_mean'],1/2560)
            self.assertEqual(len(result['outers']),70)
            Path(ur['path']).write_bytes(b'changed committed receipt')
            with self.assertRaises(c.Rejected):c.collect(goal)


if __name__=='__main__':unittest.main()
