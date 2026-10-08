"""Stdlib CPU protocol fixtures; no physical execution or healthy admission."""
import copy
import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirmation_v3', HERE / 'submit_policy_confirmation_v3_native.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    raw = (HERE / m.METHOD_NAME).read_bytes()
    method = m.strict_json(raw)
    launch_raw = (HERE / m.DISCOVERY_LAUNCH_NAME).read_bytes()
    launch = m.strict_json(launch_raw)
    freeze = lambda document: dict(document=document, sha256=m.sha(m.canonical(document)))
    selection = dict(schema_version=1, role='policy_safety_confirmation_orchestration_selection',
        selected_by='root_explicit_selection', selection_provenance={'scope': 'fabricated CPU fixture only'},
        parent_base=launch['parent_base'], discovery_base=launch['base'],
        parent_production_run_id=method['parent_production_run_id'], discovery_run_id=method['discovery_run_id'],
        discovery_slurm_job_id=launch['job_id'], source_commit=m.FROZEN_COMMIT,
        source_map_sha256=m.sha(m.canonical(launch['source_sha256'])), scientific_identity_sha256='b'*64,
        discovery_manifest_sha256='c'*64, discovery_plan_file_sha256='d'*64, discovery_plan_sha256='e'*64,
        discovery_evidence_sha256='f'*64, selection_method_sha256=m.FROZEN_METHOD_SHA256,
        negative_tail_limit=.25, health_rule=freeze(method['health_rule_document']))
    selection['threshold_selection'] = freeze(dict(role='policy_safety_threshold_selection',
        provenance={'scope': 'fabricated CPU candidate; not actual tail selection'}, criteria=dict(
        discovery_evidence_sha256=selection['discovery_evidence_sha256'],
        health_rule_sha256=selection['health_rule']['sha256'],
        candidate_thresholds=dict(method['fixed_candidates'], negative_tail_limit=.25))))
    return selection, launch, raw, launch_raw


def segment(selection):
    observations = []
    for ordinal in range(40):
        mini = ordinal//2
        observations.append(dict(context=dict(logical_update=1, minibatch_index=mini, prior_optimizer_writes=mini,
            behavior_probability_gate_sha256='a'*64), stage=('before_minibatch','after_minibatch')[ordinal%2],
            run_id=selection['discovery_run_id'], calibration_sha256=selection['discovery_plan_sha256'],
            optimizer_writes=mini+ordinal%2, execution_role=m.DISCOVERY_ROLE,
            enforcement={'negative_tail':'observe_only'}, allowed=True, reasons=[],
            negative_tail_weight_fraction=.25 if ordinal == 39 else .1))
    return dict(behavior_probability_gate={'proof_sha256':'a'*64}, structural=dict(
        execution_role=m.DISCOVERY_ROLE, run_id=selection['discovery_run_id'],
        calibration_sha256=selection['discovery_plan_sha256'], outer_count=1, optimizer_writes_per_rank=20,
        enforcement={'negative_tail':'observe_only'}, global_monitors=observations))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.selection, self.launch, self.method_raw, self.launch_raw = fixture()

    def test_actual_prospective_original_bytes(self):
        method=m.method_binding(self.method_raw,self.selection)
        observed=(HERE/m.ZERO_OBSERVATION_NAME).read_bytes()
        self.assertEqual(m.prospective_binding(method,self.launch_raw,observed),self.launch)
        m.preregistration_binding((HERE/m.PREREGISTRATION_NAME).read_bytes(),method,self.launch)
        self.assertEqual(m.sha(self.method_raw),m.FROZEN_METHOD_SHA256)

    def test_observation_raw_drift_rejected(self):
        with self.assertRaises(ValueError):
            m.prospective_binding(m.strict_json(self.method_raw),self.launch_raw,(HERE/m.ZERO_OBSERVATION_NAME).read_bytes()+b' ')

    def test_launch_raw_drift_rejected(self):
        with self.assertRaises(ValueError):
            m.prospective_binding(m.strict_json(self.method_raw),self.launch_raw+b' ',(HERE/m.ZERO_OBSERVATION_NAME).read_bytes())

    def test_old_v2_method_rejected(self):
        with self.assertRaises(ValueError):
            m.method_binding((HERE/'policy-calibration-v2-selection-method.json').read_bytes(),self.selection)

    def test_preregistration_drift_rejected(self):
        with self.assertRaises(ValueError):
            m.preregistration_binding((HERE/m.PREREGISTRATION_NAME).read_bytes()+b' ',m.strict_json(self.method_raw),self.launch)

    def test_frozen_health_not_relaxed(self):
        bad=copy.deepcopy(self.selection);bad['health_rule']['document']['criteria']['allowed_drop']['weighted_em']=.01
        bad['health_rule']['sha256']=m.sha(m.canonical(bad['health_rule']['document']))
        with self.assertRaises(ValueError):m.method_binding(self.method_raw,bad)

    def test_new_selection_schema_and_source(self):
        m.selection_schema(self.selection,self.launch)
        bad=copy.deepcopy(self.selection);bad['source_commit']='130d17a0d6f0a3ab4b4aee730928f461e25105f0'
        with self.assertRaises(ValueError):m.selection_schema(bad,self.launch)

    def test_missing_selection_no_intent_or_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            with patch.object(m,'METHOD_PATH',HERE/m.METHOD_NAME), patch.object(m,'CPU_PATH',HERE/m.CPU_NAME),patch.object(m,'SOURCE_CLEANUP_PATH',prep/'none'),patch.object(m,'remote') as remote:
                with self.assertRaises(FileNotFoundError):m.submit(prep)
                remote.assert_not_called();self.assertEqual(list(prep.iterdir()),[])

    def test_existing_intent_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory);(prep/'policy-confirmation-v3-native-submit-intent.json').write_text('{}')
            with patch.object(m,'remote') as remote:
                with self.assertRaises(ValueError):m.submit(prep)
                remote.assert_not_called()

    def test_actual_222_cpu_full_map(self):
        proof=m.strict_json((HERE/m.CPU_NAME).read_bytes());m.cpu_contract(proof,self.launch)
        self.assertEqual(len(proof['source_sha256']),353)

    def test_cpu_old_count_skips_or_missing_module_rejected(self):
        proof=m.strict_json((HERE/m.CPU_NAME).read_bytes())
        for mutate in (lambda p:p['modules'].pop(), lambda p:p['source_sha256'].pop(next(iter(p['source_sha256']))),
                       lambda p:p.update(stdout=json.dumps(dict(m.strict_json(p['stdout']),tests_run=147))),
                       lambda p:p.update(stdout=json.dumps(dict(m.strict_json(p['stdout']),skipped=['fixture'])))):
            bad=copy.deepcopy(proof);mutate(bad)
            with self.assertRaises(ValueError):m.cpu_contract(bad,self.launch)

    def test_exact_40_tail_max_new_gate_context(self):
        self.assertEqual(m.measured_selection(self.selection,segment(self.selection)),.25)

    def test_missing_drift_gate_partial_rejected(self):
        for modify in (lambda s:s['structural']['global_monitors'][0]['context'].pop('behavior_probability_gate_sha256'),
                       lambda s:s['structural']['global_monitors'][39]['context'].update(behavior_probability_gate_sha256='b'*64),
                       lambda s:s['structural']['global_monitors'].pop(),
                       lambda s:s['structural']['global_monitors'][0].update(allowed=False)):
            bad=segment(self.selection);modify(bad)
            with self.assertRaises(ValueError):m.measured_selection(self.selection,bad)

    def test_tail_not_generated_or_given_margin(self):
        bad=copy.deepcopy(self.selection);bad['negative_tail_limit']=.26
        with self.assertRaises(ValueError):m.measured_selection(bad,segment(self.selection))

    def test_actual_public_reader_api_wiring_cpu_boundary_only(self):
        calls=[];gate=dict(proof_sha256='a'*64,private_trace_sha256='b'*64,included_actor_token_count=1,max_abs_discrepancy=0.0)
        fake=types.ModuleType('reproduction.common8.behavior_runtime')
        def read(base,run,step,*,update_record):calls.append(('read',step,update_record));return gate
        def links(base,gates,first,last):calls.append(('links',gates,first,last))
        fake.read_gate_receipt=read;fake.verify_write_receipt_links=links
        with patch.dict(sys.modules,{'reproduction.common8.behavior_runtime':fake}):
            m.behavior_links(Path('/CPU-only'),{}, {'behavior_probability_gate':gate},[{'CPU_fixture':True}])
            self.assertEqual(calls[1],('links',{'1':gate},1,1))
            with self.assertRaises(ValueError):m.behavior_links(Path('/CPU-only'),{}, {'behavior_probability_gate':dict(gate,proof_sha256='c'*64)},[{}])

    def test_native_resource_precedes_heavy_claim_model(self):
        source=inspect.getsource(m.target_evidence)
        self.assertLess(source.index('native_preparation()'),source.index('verify_child(discovery'))
        bind=inspect.getsource(m.native_bind)
        self.assertLess(bind.index('target_evidence('),bind.index("'preparation/confirmation-bind-claim.json'"))
        light=inspect.getsource(m.source_precheck)
        self.assertNotIn('verify_child(',light);self.assertNotIn('_checkpoint(',light)
        self.assertIn('full_checkpoint_verification_pending=True',light)
        self.assertIn('behavior_links(discovery',light);self.assertIn('behavior_links(discovery',source)

    def test_embedded_codes_compile_and_independent_v3_paths(self):
        compile(m.verification_code(),'<light>','exec')
        for name in ('PREPARE_CODE','SUBMIT_CODE','START_SOURCE_CODE','OBSERVE_CODE','CLEANUP_SOURCE_CODE'):
            compile(getattr(m,name),'<'+name+'>','exec')
        text=Path(m.__file__).read_text()
        self.assertNotIn('policy-confirmation-v2-',text)
        self.assertNotIn('submit_policy_confirmation_v2_native.py',text)
        self.assertIn('--gres=gpu:h100:8',text);self.assertIn('--mem=512G',text)
        self.assertEqual(m.ray_socket_budget('/mnt/data/shaoshitong/g1c-12345678')['worst_plasma_socket_bytes'],105)


if __name__=='__main__':unittest.main()
