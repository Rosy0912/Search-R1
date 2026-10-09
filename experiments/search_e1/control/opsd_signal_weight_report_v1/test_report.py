"""Model-free native-format contract tests, plus explicit synthetic visual QA."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
try:
    from .common import Reader, encode, MAX_AGGREGATE
    from .fixture import native_bundle, example_data
    from .validate import validate
    from .render import small_aggregate, render
except ImportError:
    from common import Reader, encode, MAX_AGGREGATE
    from fixture import native_bundle, example_data
    from validate import validate
    from render import small_aggregate, render


class ReportContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.ref,self.req,self.write=native_bundle(self.root/'input')

    def tearDown(self):self.tmp.cleanup()

    def mutated(self,object_ref,change,name):
        value=json.loads(Path(object_ref['path']).read_text());change(value)
        return self.write(name,value)

    def save_request(self):return self.write('modified-request.json',self.req)

    def update_coverage(self,change):
        p=self.req['coverage'];old=json.loads(Path(p['output']['path']).read_text());change(old)
        p['output']=self.write('coherent-modified-coverage.json',old)
        p['actual_stdout']=self.write('coherent-modified-coverage-stdout.json',dict(accepted=True,output=p['output']))
        p['execution']=self.mutated(p['execution'],lambda v:v.update(stdout=p['actual_stdout']),'coherent-modified-coverage-execution.json')

    def test_complete_native_format_dependency_graph(self):
        data=validate(self.ref)
        self.assertEqual(sum(len(x) for x in data['curves'].values()),100)
        self.assertEqual(len(data['models']),9)
        self.assertFalse(data['synthetic'])

    def test_reject_partial_coverage_even_if_accepted_true(self):
        p=self.req['coverage'];p['output']=self.mutated(p['output'],lambda v:v.update(final_accepted=False),'partial.json')
        p['actual_stdout']=self.write('partial-stdout.json',dict(accepted=True,output=p['output']))
        p['execution']=self.mutated(p['execution'],lambda v:v.update(stdout=p['actual_stdout']),'partial-execution.json')
        with self.assertRaisesRegex(ValueError,'final-accepted'):validate(self.save_request())

    def test_reject_wrong_full_counts(self):
        p=self.req['coverage'];p['output']=self.mutated(p['output'],lambda v:v.update(complete_outers=99),'wrongcounts.json')
        p['actual_stdout']=self.write('wrongcounts-stdout.json',dict(accepted=True,output=p['output']))
        p['execution']=self.mutated(p['execution'],lambda v:v.update(stdout=p['actual_stdout']),'wrongcounts-execution.json')
        with self.assertRaisesRegex(ValueError,'100/256000/4608'):validate(self.save_request())

    def test_reject_changed_bytes_even_same_size_metadata_restored(self):
        reader=Reader();reader.file(self.ref);path=Path(self.ref['path']);old=path.stat();raw=path.read_bytes()
        path.write_bytes(raw.replace(b'final-report-request',b'final-report-requesX',1));os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'SHA'):reader.stable_end()

    def test_reject_wrong_ledger_run(self):
        self.req['train_ledgers']['G_POS']=self.mutated(self.req['train_ledgers']['G_POS'],lambda v:v.update(run_id='wrong'),'wrong-run-ledger.json')
        cr=json.loads(Path(self.req['coverage_request']['path']).read_text())
        next(g for g in cr['groups'] if g['variant']=='G_POS')['train_ledger']=self.req['train_ledgers']['G_POS']
        self.req['coverage_request']=self.write('coherent-modified-coverage-request.json',cr)
        self.update_coverage(lambda v:v.update(request=self.req['coverage_request']))
        with self.assertRaisesRegex(ValueError,'original25'):validate(self.save_request())

    def test_reject_wrong_checkpoint_endpoint(self):
        entry=next(e for e in self.req['evaluations'] if 'G_POS-15-child' in e['child']['path'])
        entry['child']=self.mutated(entry['child'],lambda v:v.update(completed_outer=14),'wrongendpoint.json')
        with self.assertRaisesRegex(ValueError,'endpoint refs differ'):validate(self.save_request())

    def test_reject_incomplete_debug_retention(self):
        self.req['debug_retention']=self.mutated(self.req['debug_retention'],lambda v:v.update(sampling_workers=799),'short-debug.json')
        with self.assertRaisesRegex(ValueError,'debug retention'):validate(self.save_request())

    def test_reject_prefix_parameter_audit(self):
        self.req['parameter_movement']=self.mutated(self.req['parameter_movement'],lambda v:v.update(full_study_complete=False),'prefix-movement.json')
        with self.assertRaisesRegex(ValueError,'parameter movement'):validate(self.save_request())

    def test_reject_cli_report_relabel(self):
        p=self.req['analysis']['effects'];p['output']=self.req['analysis']['heldout_relationship']['output']
        with self.assertRaisesRegex(ValueError,'CLI'):validate(self.save_request())

    def test_aggregate_privacy_and_bound(self):
        data=example_data();data['debug']['private_raw_question']='DO NOT EXPORT';data['movement']['private_tensor']='DO NOT EXPORT'
        aggregate=small_aggregate(data);raw=encode(aggregate)
        self.assertLessEqual(len(raw),MAX_AGGREGATE)
        self.assertNotIn(b'DO NOT EXPORT',raw)
        def private_keys(value):
            if isinstance(value,dict):
                self.assertFalse(set(value)&{'question_id','question','response','trace','private_tensor','private_raw_question'})
                for child in value.values():private_keys(child)
            elif isinstance(value,list):
                for child in value:private_keys(child)
        private_keys(aggregate)
        self.assertFalse(aggregate['formal_report'])

    def test_synthetic_render_is_explicit_and_not_accepted(self):
        output=self.root/'visual';ref=render(example_data(),output);value=json.loads(Path(ref['path']).read_text())
        self.assertFalse(value['accepted']);self.assertFalse(value['formal_report']);self.assertTrue(value['synthetic_fixture'])
        self.assertEqual(len(list(output.glob('*.png'))),3);self.assertEqual(len(list(output.glob('*.svg'))),3)
        self.assertEqual(len(list(output.glob('*.csv'))),6)
        self.assertIn('合成排版样例', (output/'report-zh.md').read_text())
        with self.assertRaisesRegex(ValueError,'Exclusive'):render(example_data(),output)


if __name__=='__main__':unittest.main()
