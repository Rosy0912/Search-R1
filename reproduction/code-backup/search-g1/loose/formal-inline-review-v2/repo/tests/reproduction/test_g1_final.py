import json,tempfile
from pathlib import Path
from unittest.mock import patch
from reproduction.a800 import verify_full as m
import unittest
from reproduction.a800.verify_full import actual_updates,final_validation,bind_probe_versions

class FinalRunTests(unittest.TestCase):
    def log(self):
        fields=' - '.join(k+':0.400' for k in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/ppo_kl','actor/grad_norm'))
        return '\n'.join('SEARCH_G1_PROBE_VERSION=r:'+str(((step-1)//50)*50)+':v\nstep:'+str(step)+' - '+fields for step in range(1,301))
    def test_real_updates_and_versions_parsed(self):
        rows=actual_updates(self.log());self.assertEqual(len(rows),300);self.assertEqual(rows[50]['probe_version'],'r:50:v')
    def test_missing_duplicate_and_nonfinite_updates_rejected(self):
        log=self.log()
        for bad in (log.replace('step:300 - ','step:299 - '),log.replace('step:300 - ','validation:300 - '),log.replace('actor/grad_norm:0.400','actor/grad_norm:nan',1)):
            with self.assertRaises(AssertionError):actual_updates(bad)
    def test_64_source_counts_and_bounded_em_required(self):
        expected={'nq':10,**{k:9 for k in ('triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle')}}
        metrics={**{'val/evaluated_count/'+k:v for k,v in expected.items()},**{'val/test_score/'+k:.5 for k in expected}}
        line='SEARCH_G1_FINAL_METRICS_JSON='+json.dumps(metrics)
        self.assertEqual(final_validation(line,expected)['weighted_em'],.5)
        metrics['val/evaluated_count/nq']=9
        with self.assertRaises(AssertionError):final_validation('SEARCH_G1_FINAL_METRICS_JSON='+json.dumps(metrics),expected)
        with self.assertRaises(AssertionError):final_validation(line+'\n'+line,expected)

    def test_exact_refit_versions_required(self):
        rows=actual_updates(self.log());versions=['r:'+str(step)+':v' for step in (50,100,150,200,250)]
        bind_probe_versions(rows,'r','r:0:v',versions)
        rows[50]['probe_version']='r:50:different-uuid'
        with self.assertRaises(AssertionError):bind_probe_versions(rows,'r','r:0:v',versions)

    def test_final_hash_change_never_rewrites_original_marker(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);logs=base/'logs';logs.mkdir();ck=base/'ckpts/actor/global_step_300';ck.mkdir(parents=True)
            metrics={**{'val/evaluated_count/'+k:v for k,v in {'nq':10,**{k:9 for k in ('a','b','c','d','e','f')}}.items()},**{'val/test_score/'+k:.5 for k in ('nq','a','b','c','d','e','f')}}
            text='SEARCH_G1_RUN r\n'+self.log()+'\n'+'\n'.join('SEARCH_G1_REFIT_ADOPTED=r:'+str(n)+':v' for n in (50,100,150,200,250))+'\nSEARCH_G1_FINAL_METRICS_JSON='+json.dumps(metrics)
            log=logs/'train';log.write_text(text)
            run=dict(run_id='r',train_log=str(log),initial_probe_version='r:0:v',input_sha256={},source_sha256={},validation_source_counts={'nq':10,**{k:9 for k in ('a','b','c','d','e','f')}},checkpoint_dir=str(base/'ckpts'),initial_model='/unused',started_unix=0)
            (logs/'run-manifest.json').write_text(json.dumps(run))
            (logs/'initial-probes-accepted.json').write_text(json.dumps(dict(run_id='r',accepted=True,version='r:0:v')))
            (logs/'five-updates-acceptance.json').write_text(json.dumps(dict(run_id='r',accepted=True,actual_updates=[dict(step=n,probe_version='r:0:v') for n in range(1,6)])))
            marker=json.dumps(dict(run_id='r',step=300,files_sha256={'weights':'original-hash'}));(ck/'complete.json').write_text(marker);(ck/'weights').write_bytes(b'modified')
            with patch.object(m,'validate_saved_checkpoint') as validate:
                with self.assertRaises(AssertionError):m.verify(base)
                validate.assert_not_called()
            self.assertEqual((ck/'complete.json').read_text(),marker)

if __name__=='__main__':unittest.main()
