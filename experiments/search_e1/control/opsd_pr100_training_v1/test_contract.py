"""Focused stdlib-only checks for the newly changed budget/boundary/cohort/cache."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from controller import Controller,validate_until_outer
from prepare import template,validate_config
from prepare_bundle import check_original_prefix
from checkpoint import exclusive_json,fileproof,prune_checkpoints
from miner_v1 import uniform_sibling,RANDOM_VERSION,RANDOM_VARIANTS,WEIGHTED_VARIANTS
from cohort_v1 import freeze_cohorts,object_sha


class ChangedContract(unittest.TestCase):
    def test_fixed100_and_ten_boundaries(self):
        config=template('/data/shaoshitong/search-e1-opsd-pr100-fixture','PR','fixture')
        validate_config(config)
        self.assertEqual(config['scientific']['outer_iterations'],100)
        self.assertEqual(config['precision']['mode'],'fp32_master_fp16')
        self.assertEqual(config['sampling']['dtype'],'float16')
        for n in range(10,101,10):self.assertEqual(validate_until_outer(n),n)
        for invalid in (None,0,1,9,11,101,200,True):
            with self.assertRaises(ValueError):validate_until_outer(invalid)
        changed=copy.deepcopy(config);changed['scientific']['outer_iterations']=10
        with self.assertRaises(ValueError):validate_config(changed)
        self.assertEqual(RANDOM_VARIANTS,('PR',));self.assertEqual(WEIGHTED_VARIANTS,())

    def test_uniform_original_stream_extended_to99(self):
        for outer in (0,9,10,99):
            selected,proof=uniform_sibling('fixture-question',outer)
            payload=[RANDOM_VERSION,42,outer,'fixture-question',0]
            digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).digest()
            self.assertEqual(selected,int.from_bytes(digest[:8],'big')%5)
            self.assertEqual(proof['accepted_sha256'],digest.hex())
            self.assertEqual(uniform_sibling('fixture-question',outer)[0],selected)
        with self.assertRaises(ValueError):uniform_sibling('fixture-question',100)

    def test_cohort_prefix_is_original_shuffle_not_resampled(self):
        # Reduced synthetic population still exercises the real100x512 freeze.
        population=51203
        train=[dict(question_id=str(i),question='q'+str(i),answers=['a']) for i in range(population)]
        val=[dict(question_id='v'+str(i),question='v'+str(i),answers=['a']) for i in range(512)]
        refs=[dict(path='/remote/'+name,sha256='1'*64) for name in ('train','val')]
        with patch('cohort_v1.TRAIN_ROWS',population):frozen=freeze_cohorts(train,val,*refs)
        permutation=list(range(population));random.Random(42).shuffle(permutation)
        self.assertEqual(frozen['cohorts'][0]['train_indices'],permutation[:512])
        self.assertEqual(frozen['cohorts'][99]['train_indices'],permutation[99*512:100*512])
        self.assertEqual(frozen['selected_unique_question_count'],51200)
        original=dict(outer_count=10,questions_per_outer=512,cohorts=frozen['cohorts'][:10],
            permutation_sha256=object_sha(permutation),train_identity=refs[0],validation_identity=refs[1])
        self.assertTrue(check_original_prefix(frozen,original)['identical_first_ten_outers'])
        original['cohorts']=copy.deepcopy(original['cohorts']);original['cohorts'][0]['train_indices'][0]=-1
        with self.assertRaises(ValueError):check_original_prefix(frozen,original)

    def test_retention_keeps_keys_outer_end_and_last2(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'search-e1-opsd-pr100-retention';root=base/'runs';root.mkdir(parents=True)
            policy=dict(mode='keep_key_and_resume',authorized=True,merged_cache_cleanup_authorized=True,
                checkpoint_keep_latest=2,protect_updates=[0,1,5],keep_all_outer_end=True)
            cfg=exclusive_json(base/'config.json',dict(schema='opsd-pr100-config-v1',variant='PR',retention=policy))
            exclusive_json(root/'run-identity.json',dict(run_id='r',config_sha256=cfg['sha256'],config=cfg,retention=policy))
            refs={}
            for update in (0,1,5,6,7,8,9):
                directory=root/'checkpoints'/f'update-{update:08d}-outer-00000';directory.mkdir(parents=True)
                adapter=directory/'adapter';adapter.mkdir()
                a=adapter/'adapter.fixture';a.write_bytes(b'adapter')
                state=directory/'rank.fixture';state.write_bytes(b'rng')
                optimizer=directory/'optimizer.fixture';optimizer.write_bytes(b'Adam')
                manifest=dict(accepted=True,run_id='r',config_sha256=cfg['sha256'],optimizer_updates=update,
                    cursor=dict(outer_index=0),protection_reason=['outer_end'] if update==7 else [],
                    adapter=dict(files=[fileproof(a)]),ranks=[dict(file=fileproof(state))],shared_optimizer=fileproof(optimizer))
                refs[update]=exclusive_json(directory/'manifest.json',manifest)
            result=prune_checkpoints(root,'r',cfg['sha256'],refs[9])
            self.assertEqual(len(result['removed']),1)
            for update in (0,1,5,7,8,9):
                self.assertTrue((Path(refs[update]['path']).parent/'optimizer.fixture').exists())
            six=Path(refs[6]['path']).parent
            self.assertTrue((six/'manifest.json').exists());self.assertTrue((six/'payload-pruned.json').exists())
            self.assertFalse((six/'optimizer.fixture').exists())
            self.assertEqual(prune_checkpoints(root,'r',cfg['sha256'],refs[9])['removed'],[])

    def test_cache_cleanup_exact_completed_scope_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'search-e1-opsd-pr100-fixture';outer=base/'runs'/'outers'/'outer-00000'
            cache=outer/'merged-model-cache';cache.mkdir(parents=True)
            payload=cache/'rebuildable.fixture';payload.write_bytes(b'cache')
            retained=outer/'rollouts.fixture';retained.write_bytes(b'debug')
            checkpoint=dict(path='/remote/retained-checkpoint',sha256='a'*64)
            identity=dict(kind='merged',path=str(cache),files=[fileproof(payload)])
            exclusive_json(outer/'merged-model.json',identity)
            child=exclusive_json(outer/'cpu-merge-child.json',dict(result=identity))
            exclusive_json(outer/'cpu-merge-parent.json',dict(accepted=True,run_id='fixture',child=child))
            completed=exclusive_json(outer/'training'/'outer-complete.json',dict(accepted=True,run_id='fixture',outer_index=0,checkpoint=checkpoint))
            terminal=dict(gpu_release=dict(all_owned_release=dict(accepted=True)),actual_terminal=dict(path='/remote/wait',sha256='b'*64))
            c=Controller.__new__(Controller);c.run_id='fixture';c.config_ref=dict(path='/remote/config',sha256='c'*64)
            c.config=dict(schema='opsd-pr100-config-v1',variant='PR',retention=dict(merged_cache_cleanup_authorized=True),
                paths=dict(base=str(base),output_root=str(base/'runs')))
            c.verify_exit=lambda *a:None
            receipt=c.reclaim_completed_cache(outer,0,completed,terminal,checkpoint)
            result=json.loads(Path(receipt['path']).read_text())
            self.assertFalse(cache.exists());self.assertTrue(retained.exists())
            self.assertTrue(Path(completed['path']).exists());self.assertEqual(result['bytes_reclaimed'],5)
            self.assertTrue(result['no_old_namespace_touched'])
            # No old namespace or nonmatching current/incomplete path is deletable.
            c.config['paths']['base']=str(Path(tmp)/'old-opsd')
            with self.assertRaises(ValueError):c.reclaim_completed_cache(outer,0,completed,terminal,checkpoint)


if __name__=='__main__':unittest.main()
