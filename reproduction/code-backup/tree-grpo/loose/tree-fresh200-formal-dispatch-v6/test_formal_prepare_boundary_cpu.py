"""Pure metadata/map counterexamples only; no helper main/Hydra/assets."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
specification=importlib.util.spec_from_file_location('formal_cpu_common',HERE/'formal_cpu_common.py')
common=importlib.util.module_from_spec(specification);specification.loader.exec_module(common)


def nested(values):
    result={}
    for key,value in values.items():
        current=result
        parts=key.split('.')
        for part in parts[:-1]:
            current=current.setdefault(part,{})
        current[parts[-1]]=copy.deepcopy(value)
    return result


class BoundaryCPU(unittest.TestCase):
    def test_all_four_null_descriptors_refuse_before_any_file_read(self):
        with self.assertRaisesRegex(ValueError,'non-null'):
            common.require_observation_descriptors(dict.fromkeys(common.INPUT_NAMES))

    def test_wrong_stage_or_source_manifest_refuses_without_assets(self):
        for changed in ('stage','code_commit'):
            manifest=dict(stage=common.STAGE,training_protocol=common.PROTOCOL,code_commit=common.COMMIT)
            manifest[changed]='historical-zero-or-source'
            with self.assertRaisesRegex(ValueError,'Wrong formal'):
                common.validate_manifest(manifest)

    def test_own_checkpoint_and_dynamic_values_are_checked_in_plain_map(self):
        spec=json.loads((ROOT/'work/tree-fresh200-train-runtime-v6/CONFIGURATION_SPEC.json').read_text())
        # Synthetic sentinel map values: never production inputs or thresholds.
        probe={'negative_tail_limit':.123456789,'initial_actor_repeat_max_abs_delta':.987654321}
        expected=copy.deepcopy(spec['expected_resolved_values'])
        expected['trainer.default_local_dir']='/synthetic-metadata-only/checkpoints'
        expected['actor_rollout_ref.actor.policy_safety.negative_tail_limit']=probe['negative_tail_limit']
        expected['actor_rollout_ref.actor.policy_safety.sampling_probability_correction.initial_actor_repeat_max_abs_delta']=probe['initial_actor_repeat_max_abs_delta']
        resolved=nested(expected)
        self.assertEqual(len(common.check_resolved(resolved,spec,probe,Path(expected['trainer.default_local_dir']))),90)
        for key in ('trainer.default_local_dir','actor_rollout_ref.actor.policy_safety.negative_tail_limit',
                    'actor_rollout_ref.actor.policy_safety.sampling_probability_correction.initial_actor_repeat_max_abs_delta'):
            drift=copy.deepcopy(expected);drift[key]='wrong-unaccepted-map-value'
            with self.assertRaisesRegex(ValueError,'Resolved formal config'):
                common.check_resolved(nested(drift),spec,probe,Path(expected['trainer.default_local_dir']))
        drift=copy.deepcopy(expected);drift['trainer.nnodes']=True
        with self.assertRaisesRegex(ValueError,'Resolved formal config'):
            common.check_resolved(nested(drift),spec,probe,Path(expected['trainer.default_local_dir']))

    def test_exact_common_reuse_and_before_configuration_order(self):
        original=ast.parse((ROOT/'work/tree-fresh200-dispatch-v11/fresh200_dispatch_common.py').read_text())
        candidate=ast.parse((HERE/'formal_cpu_common.py').read_text())
        reuse=json.loads((HERE/'PURE_V11_REUSE_MANIFEST.json').read_text())
        for name,digest in reuse['reused_function_ast_sha256'].items():
            nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                   for tree in (original,candidate)]
            dumps=[ast.dump(n,include_attributes=False) for n in nodes]
            self.assertEqual(dumps[0],dumps[1]);self.assertEqual(hashlib.sha256(dumps[1].encode()).hexdigest(),digest)
        tree=ast.parse((HERE/'prepare_formal_tree_fresh200_package.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        call_nodes=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        one=lambda name:next(n.lineno for n in call_nodes if ast.unparse(n.func)==name)
        self.assertLess(one('c.require_observation_descriptors'),one('c.safe_extract'))
        self.assertLess(one('production_contract.observed_production_inputs'),one('c.verify_local_assets'))
        self.assertLess(one('production_contract.observed_production_inputs'),one('supervisor.configuration'))
        assignment=next(n for n in ast.walk(main) if isinstance(n,ast.Assign)
                        and any(ast.unparse(t)=='supervisor.PRODUCTION_THRESHOLDS' for t in n.targets))
        self.assertLess(assignment.lineno,one('supervisor.configuration'))
        calls={ast.unparse(n.func) for n in call_nodes}
        self.assertNotIn('supervisor.main',calls)
        self.assertNotIn('production_contract.production_inputs',calls)
        self.assertFalse(any(name.startswith(('subprocess.','remote.','torch.cuda.init','torch.cuda.set_device')) for name in calls))


if __name__=='__main__':
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryCPU))
    receipt=dict(status='PURE_METADATA_AST_COUNTEREXAMPLES_PASSED' if result.wasSuccessful() else 'PURE_COUNTEREXAMPLES_FAILED',
        actual_test_count=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_seconds=time.perf_counter()-start,
        actual_hydra_executed=False,actual_supervisor_imported=False,actual_helper_main_executed=False,
        actual_builder_executed=False,actual_official_assets_read=False,actual_remote_executed=False,
        actual_slurm_executed=False,actual_gpu_executed=False,actual_cpu_prepared=False,training_release=False,
        scope='4 pure AST/NULL metadata/plain map counterexample tests; synthetic sentinel map values are not production thresholds')
    (HERE/'PURE_BOUNDARY_CPU_RUN.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
