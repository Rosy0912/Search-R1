"""New namespace/lineage adaptation tests only; no native, model or GPU claim."""
import base64
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import deployment_contract_raw_v3 as c

LOCAL = Path(__file__).resolve().parents[3]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'


def candidate():
    return json.loads((OUT / 'CALIBRATION-RAW-CANDIDATE-2262942-v3.json').read_text())


def fixture():
    scope=c.NATIVE_SCOPE;job='9999';binding_sha='a'*64
    guard={k:dict(path=str(c.NATIVE_HELPERS/name),sha256=digest) for k,name,digest in [
        ('resource_guard','resource_guard_v2.py','9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
        ('resource_ownership','resource_ownership_v2.py','7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed')]}
    outer=dict(optimizer_writes=10,gradient_micro_forwards=160,before_monitor_forwards=160,after_monitor_forwards=160,
               original_local_trajectory_rows=320,mini_denominators=[32.]*10)
    ranks=[dict(status='PASS',rank=rank,world=8,code_commit=c.SOURCE,binding_sha256=binding_sha,native_scope=scope,
        native_initialization='OFFICIAL_MODEL_INITIALIZATION',native_actual_restore=False,formal_updates_contributed=0,
        official_3b=True,official_revision=c.REVISION,actual_fsdp=True,flash_attention2=True,remove_padding=True,manual_offload=True,
        prefix_4096_action500_forward_backward=True,old_ref_rank4_action16_no_grad_maximum=True,
        mapping_formula_value_and_gradient_equal=True,state_save_restore_next_outer_equal=True,native_hf_semantic_reload_equal=True,
        real_sampler_executed=False,real_fixed_reference_executed=False,hf_engine_probability_acceptance=False,
        production_thresholds_selected=False,scheduler_at_retained_boundary=2,adam_at_retained_boundary=20,
        outer_records=[copy.deepcopy(outer) for _ in range(3)],rank5_refusals={k:dict(all8_refused=True,extra_adam=0)
                                                                             for k in ('temperature','output_shape')}) for rank in range(8)]
    run=str(c.ROOT/('runs/gigpo-e1-action-prefix-native-job'+job));cg='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_'+job
    result=dict(status='PASS',validation_scope=c.ADMISSION_SCOPE,native_scope=scope,code_commit=c.SOURCE,
        job=job,node='h100-8',binding_sha256=binding_sha,native_initialization='OFFICIAL_MODEL_INITIALIZATION',
        native_actual_restore=False,formal_updates_contributed=0,fixtures_passed=True,cpu_exit=0,gpu_exit=0,
        rank_records=ranks,production_probability_acceptance=False,cleanup_residual_states={},final_gpu={'processes':[]},
        guard_identity=guard,cgroup=cg+'/step_0')
    binding=dict(code_commit=c.SOURCE,helpers=str(c.NATIVE_HELPERS),controls=str(c.NATIVE_CONTROLS),native_scope=scope,native_initialization='OFFICIAL_MODEL_INITIALIZATION',
        native_actual_restore=False,formal_updates_contributed=0,official_revision=c.REVISION,allowed_safety_modes=['native_fixture'],
        restore_step=25,restore_checkpoint=c.CHECKPOINT,restore_manifest_sha256=c.MANIFEST,**guard)
    terminal=dict(fields=dict(JobId=job,NodeList='h100-8',JobState='COMPLETED',ExitCode='0:0',Requeue='0',Restarts='0'))
    acceptance=dict(status='PASS',code_commit=c.SOURCE,native_scope=scope,job=job,binding_sha256=binding_sha,
        result_path=run+'/result.json',result_sha256='b'*64,review_path=str(c.ROOT/'logs/new-native-review-9999.json'),review_sha256='c'*64,
        native_binding_path=str(c.NATIVE_BINDING),native_binding_sha256=binding_sha,
        terminal_path=str(c.ROOT/'logs/new-native-terminal-9999.json'),terminal_sha256='d'*64,
        physical_payload_path=str(c.ROOT/'logs/new-native-physical-9999.json.gz'),physical_payload_sha256='e'*64,
        terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True,guard_identity=guard)
    originals=[]
    for key,data in [('result',result),('native_binding',binding),('terminal',terminal)]:
        payload=json.dumps(data).encode();digest=hashlib.sha256(payload).hexdigest();acceptance[key+'_sha256']=digest
        originals.append(dict(path=acceptance[key+'_path'],bytes=len(payload),sha256=digest,content_base64=base64.b64encode(payload).decode()))
    # Bind the native binding's actual bytes SHA everywhere, exactly like the runtime.
    binding_sha=acceptance['native_binding_sha256'];result['binding_sha256']=acceptance['binding_sha256']=binding_sha
    for rank in result['rank_records']:rank['binding_sha256']=binding_sha
    payload=json.dumps(result).encode();acceptance['result_sha256']=hashlib.sha256(payload).hexdigest()
    originals[0].update(bytes=len(payload),sha256=acceptance['result_sha256'],content_base64=base64.b64encode(payload).decode())
    review=dict(status='PASS',code_commit=c.SOURCE,native_scope=scope,job=job,binding_sha256=binding_sha,
        result_sha256=acceptance['result_sha256'],native_initialization='OFFICIAL_MODEL_INITIALIZATION',native_actual_restore=False,
        formal_updates_contributed=0,terminal_path=acceptance['terminal_path'],terminal_sha256=acceptance['terminal_sha256'],
        terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True,
        physical_payload_path=acceptance['physical_payload_path'],physical_payload_sha256=acceptance['physical_payload_sha256'])
    physical=dict(status='ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED',job=job,run=run,result=result,terminal=terminal,
        own_cgroups=[],own_job_queue='',current_compute={'returncode':0,'stdout':'123 other-job'},current_node_queue='123 other-job RUNNING',
        cgroup_release=dict(job_id=job,path=cg,result_cgroup=result['cgroup'],ancestor_path='/sys/fs/cgroup/freezer/slurm_h100-8',
            ancestor_readable=True,stat_outcome='ENOENT',scope_stat_outcome='ENOENT',removed=True),originals=originals,
        physical=[{k:originals[0][k] for k in ('path','bytes','sha256')}])
    return acceptance,result,review,binding,guard,terminal,physical


class RawV3AdaptationTests(unittest.TestCase):
    def test_actual_v2_native_parent_and_raw_v3_candidate_exact_closure(self):
        value = candidate()
        parent = json.loads((OUT / 'ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v2.json').read_text())
        physical = json.loads((OUT / 'ACTION-PREFIX-NATIVE-PHYSICAL-2262942-v2.json').read_text())
        self.assertEqual(c.sha(OUT / 'ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v2.json'), c.PARENT_CANDIDATE_SHA)
        self.assertEqual(c.sha(OUT / 'ACTION-PREFIX-NATIVE-PHYSICAL-2262942-v2.json'), c.PARENT_PHYSICAL_SHA)
        self.assertTrue(c.validate_candidate(value))
        self.assertTrue(c.validate_inheritance(value, parent, physical))
        self.assertEqual(value['resource_guard']['path'], str(c.NATIVE_HELPERS / 'resource_guard_v2.py'))
        self.assertEqual(value['helpers'], str(c.HELPERS))
        self.assertEqual(value['controls'], str(c.CONTROLS))
        self.assertEqual(value['source_native_acceptance'], None)
        with self.assertRaises(ValueError):
            c.validate_ready(value)

    def test_old_raw_v2_namespace_and_parent_refused(self):
        old = json.loads((OUT / 'CALIBRATION-RAW-CANDIDATE-2262942-v2.json').read_text())
        with self.assertRaises(ValueError):
            c.validate_candidate(old)
        value = candidate()
        for key, old_value in [('helpers', old['helpers']), ('controls', old['controls']),
                               ('native_parent_candidate', old['native_parent_candidate']),
                               ('inherited_physical', old['inherited_physical'])]:
            changed = copy.deepcopy(value); changed[key] = old_value
            with self.assertRaises(ValueError):
                c.validate_candidate(changed)

    def test_native_v2_acceptance_contract_positive_not_actual_gpu(self):
        self.assertTrue(c.validate_native_payload(*fixture()))

    def test_native_v1_binding_helper_or_controls_rejected(self):
        for key, value in [('helpers', str(c.ROOT / 'tools/action-prefix-native-2262942-v1')),
                           ('controls', str(c.ROOT / 'tools/action-prefix-native-2262942-control-v1'))]:
            values = fixture(); values[3][key] = value
            with self.assertRaises(ValueError):
                c.validate_native_payload(*values)
        values = fixture()
        values[0]['native_binding_path'] = str(c.ROOT / 'logs/ACTION-PREFIX-NATIVE-BINDING-2262942-v1.json')
        with self.assertRaises(ValueError):
            c.validate_native_payload(*values)

    def test_native_v1_guard_path_rejected_even_same_bytes(self):
        value = candidate()
        value['resource_guard']['path'] = str(c.ROOT / 'tools/action-prefix-native-2262942-v1/resource_guard_v2.py')
        with self.assertRaises(ValueError):
            c.validate_candidate(value)

    def test_failed3297_native_cannot_be_borrowed(self):
        values = fixture()
        for obj in values[:3]:
            obj['job'] = '3297'
        with self.assertRaises(ValueError):
            c.validate_native_payload(*values)

    def test_old_namespace_submit_rejected_before_sbatch(self):
        import submit_once_raw_v3 as submit
        with patch.object(submit.socket, 'gethostname', return_value='h100-8'), patch.object(submit.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Not exact raw-v3 binding path'):
                submit.main(str(c.ROOT / 'logs/policy-safety-calibration-raw-2262942-v2-reviewed-binding.json'), 'a'*64, 'unused', 'b'*64)
            run.assert_not_called()

    def test_unchanged_all18_runtime_local_bytes(self):
        runtime = Path(__file__).resolve().parent.with_name('safety-calibration-raw-v2')
        for name, identity in c.RUNTIME_IDENTITIES.items():
            path = runtime / name
            self.assertEqual(path.stat().st_size, identity['bytes'])
            self.assertEqual(c.sha(path), identity['sha256'])


if __name__ == '__main__':
    unittest.main()
