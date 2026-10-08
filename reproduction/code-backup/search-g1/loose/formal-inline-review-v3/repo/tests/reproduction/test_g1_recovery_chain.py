"""Reject missing updates, wrong segment probes, changed labels and wrong64 IDs."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from reproduction.a800.recovery_binding import intervention_fingerprint,verify_interventions
from reproduction.a800.verify_continuation import segment_rows,bind_segment_versions,fixed64_identity
from reproduction.a800.validation_identity import record_validation

ROOT=Path(__file__).resolve().parents[2]
KEYS=['critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/ppo_kl','actor/grad_norm']


class RecoveryChain(unittest.TestCase):
    def test_actual_log_parser_and_exact_probe_boundaries(self):
        run=dict(run_id='new',initial_probe_version='new:200:init')
        text='SEARCH_G1_RUN new\n'
        for i in range(201,301):
            version='new:200:init' if i<=250 else 'new:250:refit'
            text+='SEARCH_G1_PROBE_VERSION='+version+'\nstep:'+str(i)+' - '+' - '.join(k+':0.1' for k in KEYS)+'\n'
        rows=segment_rows(text,run,201,300)
        bind_segment_versions(rows,run,200,['new:250:refit'],[250])
        for mutation in ('missing','header','probe'):
            if mutation=='missing':
                bad=text.replace('step:205 -','missing:205 -')
                with self.assertRaises(AssertionError):segment_rows(bad,run,201,300)
            elif mutation=='header':
                with self.assertRaises(AssertionError):segment_rows(text.replace('SEARCH_G1_RUN new','SEARCH_G1_RUN old'),run,201,300)
            else:
                bad=segment_rows(text.replace('new:200:init','old:150:stale'),run,201,300)
                with self.assertRaises(AssertionError):bind_segment_versions(bad,run,200,['new:250:refit'],[250])

    def test_preserved_dump_tamper_and_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a.npz').write_bytes(b'genuine');(p/'calibration-label-proof.json').write_text('{}')
            bound=intervention_fingerprint(p);self.assertEqual(verify_interventions(p,bound),bound)
            (p/'a.npz').write_bytes(b'changed')
            with self.assertRaises(AssertionError):verify_interventions(p,bound)
            (p/'a.npz').write_bytes(b'genuine');(p/'extra.npz').write_bytes(b'extra')
            with self.assertRaises(AssertionError):verify_interventions(p,bound)

    def test_fixed64_real_em_identity_order_and_final_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'logs').mkdir();run=dict(run_id='new',validation_question_ids=[str(i) for i in range(64)],validation_source_counts={'nq':64},input_sha256={'validation-64.parquet':'hash'})
            (p/'logs/run-manifest.json').write_text(json.dumps(run))
            ids=run['validation_question_ids'];sources=['nq']*64;scores=[1.]*17+[0.]*47
            evidence=record_validation(p,300,ids,sources,scores,final=True)
            validation=dict(counts={'nq':64},source_em={'nq':17/64})
            self.assertEqual(fixed64_identity(evidence,run,validation)['correct'],17)
            evidence['question_ids']=list(reversed(ids))
            with self.assertRaises(AssertionError):fixed64_identity(evidence,run,validation)
            with self.assertRaises(AssertionError):record_validation(p,205,list(reversed(ids)),sources,scores)
            with self.assertRaises(FileExistsError):record_validation(p,300,ids,sources,scores,final=True)

    def test_actual_final_counter_callsite_records300(self):
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='record_validation')
        from types import SimpleNamespace
        self_obj=SimpleNamespace(global_steps=301)
        expression=ast.Expression(call.args[1])
        self.assertEqual(eval(compile(expression,'actual-validation-step','eval'),{'self':self_obj,'final':True}),300)
        self.assertEqual(eval(compile(expression,'actual-validation-step','eval'),{'self':self_obj,'final':False}),301)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_validate' and any(k.arg=='final' and ast.literal_eval(k.value) is True for k in n.keywords)]
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
