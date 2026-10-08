"""Prospective typed graph CPU fixtures; no launch or healthy admission."""
import copy,json,tempfile,unittest
from pathlib import Path
from reproduction.common8.formal_contract import FORMAL_MATHEMATICS,assert_formal_run,assert_scientific_run
from reproduction.common8.policy_calibration_plan import MATHEMATICS,document_sha256
from reproduction.common8.test_configuration import ConfigurationTests
from reproduction.common8.warmup_measurement_plan import (
    ROLE,PHASE_BUDGET,HEALTH,THRESHOLDS,TAIL_METHOD,METHOD_ROLE,PLAN_PATH,PARENT_PATH,METHOD_PATH,
    INITIALIZATION,PREREQUISITES,scientific_identity,verify_parent,verify_method,load_plan,
)
from reproduction.common8.verify_data import digest


def fixture(root):
    base,parent=ConfigurationTests().fixture(root)
    parent.update(run_id='a'*32,source_commit='b'*40,started_unix=100.,
        initial_model_sha256={'config.json':'c'*64},input_sha256={'manifest.json':parent['budget']['common_manifest_sha256']},
        calibration_question_ids=['CPU question hash '+str(i) for i in range(300)])
    run=copy.deepcopy(parent);run.update(run_id='d'*32,policy_calibration_role=ROLE,
        measurement_outer_count=2,measurement_parent_run_id=parent['run_id'],started_unix=101.)
    method=dict(schema_version=2,role=METHOD_ROLE,created_unix=99.,phase_budget=copy.deepcopy(PHASE_BUDGET),
        health_rule=copy.deepcopy(HEALTH),negative_tail_method=TAIL_METHOD,candidate_thresholds=copy.deepcopy(THRESHOLDS),
        mathematics_sha256=document_sha256(FORMAL_MATHEMATICS))
    return base,parent,run,method


class WarmupMeasurementPlanTests(unittest.TestCase):
    def test_science_is_shared_but_typed_phase_is_not_formal_or_old_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            _,parent,run,method=fixture(root)
            self.assertEqual(scientific_identity(run),scientific_identity(parent))
            assert_scientific_run(run,expected_role=ROLE);verify_parent(run,parent);verify_method(method,run)
            with self.assertRaises(AssertionError):assert_formal_run(run)
            self.assertEqual(run['target_updates'],200);self.assertEqual(run['acceptance_update'],5)
            self.assertEqual(MATHEMATICS['target_actual_outer_updates'],300)
            self.assertEqual(MATHEMATICS['reference_kl_coefficient'],.001)

    def test_prospective_method_rejects_late_old_schema_changed_threshold_and_fake_health(self):
        with tempfile.TemporaryDirectory() as root:
            _,_,run,method=fixture(root)
            variants=[dict(created_unix=102.),dict(schema_version=1),dict(negative_tail_method='ignore_tail'),
                dict(phase_budget=dict(PHASE_BUDGET,optimizer_writes_per_rank=20)),
                dict(health_rule=dict(method='unpaired',allowed_drop=dict(weighted_em=.1))),
                dict(candidate_thresholds=dict(THRESHOLDS,target_old_policy_kl=.02))]
            for variant in variants:
                bad=copy.deepcopy(method);bad.update(variant)
                with self.subTest(variant=variant),self.assertRaises(ValueError):verify_method(bad,run)

    def test_parent_must_be_actual_separate_fresh_and_exact_scientific_graph(self):
        with tempfile.TemporaryDirectory() as root:
            _,parent,run,_=fixture(root)
            for key,value in (('run_id',run['run_id']),('completed_updates',1),('source_commit','e'*40),
                              ('calibration_question_ids',['wrong']*300)):
                bad=copy.deepcopy(parent);bad[key]=value
                with self.subTest(key=key),self.assertRaises((AssertionError,ValueError)):verify_parent(run,bad)

    def test_plan_hash_bytes_parent_method_and_actual_prerequisite_are_all_bound(self):
        with tempfile.TemporaryDirectory() as root:
            base,parent,run,method=fixture(root);(base/'preparation').mkdir()
            for relative,document in ((PARENT_PATH,parent),(METHOD_PATH,method)):
                (base/relative).write_text(json.dumps(document))
            run['measurement_parent_manifest_sha256']=digest(base/PARENT_PATH)
            run['measurement_method_sha256']=digest(base/METHOD_PATH)
            prerequisites={}
            for name in PREREQUISITES:
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"CPU_fixture":true}')
                prerequisites[name]=digest(path)
            prerequisite=base/'logs/native-allocation-admission.json'
            run['initial_probe_version']=run['run_id']+':0:CPU-fixture-only'
            plan=dict(schema_version=2,role=ROLE,run_id=run['run_id'],parent_production_run_id=parent['run_id'],
                identity=scientific_identity(run),mathematics=copy.deepcopy(FORMAL_MATHEMATICS),phase_budget=PHASE_BUDGET,
                candidate_thresholds=THRESHOLDS,production_admission=False,healthy_checkpoint_accepted=False,slurm_job_id='123',
                parent_manifest_sha256=run['measurement_parent_manifest_sha256'],method_sha256=run['measurement_method_sha256'],
                method=method,prerequisite_sha256=prerequisites,initialization=INITIALIZATION,
                initial_probe_version=run['initial_probe_version'],created_unix=102.)
            (base/PLAN_PATH).write_text(json.dumps(plan));run['warmup_measurement_plan_sha256']=digest(base/PLAN_PATH)
            self.assertEqual(load_plan(base,run,expected_slurm_job_id='123')[0],plan)
            with self.assertRaises(ValueError):load_plan(base,run,expected_slurm_job_id='124')
            for path in (base/PLAN_PATH,base/PARENT_PATH,base/METHOD_PATH,prerequisite):
                original=path.read_bytes();path.write_bytes(original+b' ')
                with self.subTest(path=path),self.assertRaises(ValueError):load_plan(base,run)
                path.write_bytes(original)


if __name__=='__main__':unittest.main()
