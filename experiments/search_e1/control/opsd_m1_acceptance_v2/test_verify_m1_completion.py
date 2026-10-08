import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import verify_m1_completion as v

REGION = Path(__file__).resolve().parent.parent / 'opsd_token_region_analysis/summarize_regions.py'
M4 = Path(__file__).resolve().parent.parent / 'opsd_native512_eval_v2'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(v.canonical(obj))


def terminal_fixture():
    children = [dict(pid=20+i, uid=1005, starttick=50+i) for i in range(3)]
    start = dict(verified_live=True, m1_run_id='test', captured_unix=15,
                 controller=dict(pid=10, uid=1005, starttick=40, ppid=9, pgid=9, state='S'),
                 pane_process=dict(pid=9, uid=1005, starttick=39, pgid=9, state='S'),
                 tmux_pane_id='%7', tmux_session='test-pane', tmux_start_command='python main.py')
    config = dict(run_id='test', controller_pid=10, controller_uid=1005, started_unix=12, output='/test/run')
    final = dict(status='WORKERS_COMPLETE', run_id='test', controller_pid=10, output='/test/run', children=children,
                 child_exit_codes={str(c['pid']): 0 for c in children}, no_process_signals=True, ended_unix=20)
    reading = dict(pane_id='%7', pane_pid=9, session_name='test-pane', pane_start_command='python main.py', pane_dead=True, pane_dead_status=0)
    return start, config, final, children, reading


def probe_fixture(directory, version, name, origin, hard=False, value=.3):
    response = '<answer>x</answer>'; spans = [[0, len(response)]]
    teacher = not hard; rows = []
    dist = dict(target_logprob=-2., target_probability=.135, entropy=1.5,
                topk=[dict(token_id=i, logprob=-2., probability=.135) for i in range(32)])
    for i, char in enumerate(response):
        rows.append(dict(token_index=i, token_id=ord(char), character_offset=[i, i+1], text=char,
                         policy_mask=True, information_mask=False, distributions={k: copy.deepcopy(dist) for k in
                         (['student', 'base_ordinary'] + (['teacher_privileged'] if teacher else []))},
                         teacher_student_clipped_kl=value if teacher else None,
                         teacher_student_unclipped_kl=value+.1 if teacher else None,
                         base_ordinary_to_student_kl=value+.2, clipped_vocabulary_contributions=1 if teacher else None,
                         clipped_teacher_probability_mass=.125 if teacher else None,
                         actual_masked_trajectory_loss_contribution=value/len(response) if teacher else None))
    path = directory / ('tokens-'+version+'-'+name) / 'tokens.jsonl'; path.parent.mkdir(parents=True)
    data = b''.join(v.canonical(x)+b'\n' for x in rows); path.write_bytes(data)
    summary = dict(kind='fixed_response_teacher_forcing', actual_autoregressive_generation=False, status='complete',
        case_id='fixture', version=version, response_origin=origin, original_training_objective=origin=='preserved_student',
        response_tokens=len(rows), valid_policy_tokens=len(rows), reference_status='successful_original_reference' if teacher else 'not_applicable_no_successful_reference',
        implicit_temperature=1., contribution_upper_clip=10., topk=32, max_context=8192, token_reduction='valid policy token mean per trajectory',
        all_token_metrics_finite=True, tokens_file=dict(path=str(path), bytes=len(data), rows=len(rows), sha256=v.digest(data)),
        full_vocabulary_dump=dict(status='not_requested'), trajectory_mean=value if teacher else None,
        frozen_pair_loss=value if teacher else None, per_token_contribution_sum_float64=value if teacher else None,
        clipped_policy_vocabulary_contributions=len(rows) if teacher else None,
        frozen_pair_loss_check=dict(passed=True, per_token_contribution_sum_matches=True) if teacher else dict(status='not_applicable_no_successful_reference'))
    if name.startswith('original-'):
        summary['preserved_trajectory_name'] = name
    write(directory / ('token-summary-'+version+'-'+name+'.json'), summary)
    return summary, response, spans, path


def case_fixture(directory, hard=False):
    response = '<answer>x</answer>'; spans = [[0, len(response)]]
    pair = None if hard else dict(row=dict(student=response, student_policy_spans=spans))
    case = dict(case_id='fixture', original_pair=pair,
                original_siblings=[dict(row=dict(sibling=i, response=response, policy_spans=spans)) for i in range(5)])
    def gen(version, teacher=False):
        return dict(kind='actual_autoregressive_generation', response=response, policy_spans=spans, protocol_em=1,
                    active_adapter='disabled' if teacher else version, fixed_weight_source='source100',
                    role='frozen_source100_privileged_teacher' if teacher else 'ordinary_student', privileged_reference=teacher,
                    temperature=0., max_new_tokens_per_turn=500, max_context=8192, max_search_calls=4,
                    topk_documents=3, teacher_forced=False, termination='answer', search_calls=0)
    result = dict(versions={})
    if hard:
        teacher = dict(kind='actual_autoregressive_generation', status='not_applicable_no_successful_reference', privileged_reference=False)
        result['actual_teacher_response_teacher_forced_probe'] = None
    else:
        teacher = gen('source100', True)
        summary, _, _, _ = probe_fixture(directory, 'source100', 'actual-teacher-generation', 'new_teacher_generation')
        result['actual_teacher_response_teacher_forced_probe'] = summary
    result['teacher_generation'] = teacher; write(directory / 'teacher-generation.json', teacher)
    for vi, version in enumerate(v.VERSIONS):
        generation = gen(version); write(directory / ('student-generation-'+version+'.json'), generation)
        originals = []
        for i in range(5 if hard else 1):
            name = 'original-failed-sibling-'+str(i) if hard else 'original-mined-student'
            summary, _, _, _ = probe_fixture(directory, version, name, 'preserved_failed_sibling' if hard else 'preserved_student', hard, .3+vi/10)
            originals.append(summary)
        own, _, _, _ = probe_fixture(directory, version, 'actual-student-generation', 'new_student_generation', hard)
        result['versions'][version] = dict(generation_em=1, generation_termination='answer', generation_search_calls=0,
            token_probes=originals, actual_student_response_teacher_forced_probe=own)
    return case, result


class AcceptanceTests(unittest.TestCase):
    def test_actual_m4_checked_json_and_exit_interface(self):
        # Actual frozen M4 functions, unmodified and unmocked. Only small CPU
        # evidence fixtures are consumed; this is no real terminal claim.
        sys.path.insert(0, str(M4))
        try:
            spec = importlib.util.spec_from_file_location('m1_fixture_actual_m4_utils', M4/'utils.py')
            utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
            spec = importlib.util.spec_from_file_location('m1_fixture_actual_m4_gates', M4/'gates.py')
            gates = importlib.util.module_from_spec(spec); spec.loader.exec_module(gates)
            preservation = Path(__file__).resolve().parent/'fixture-tmux-preservation-binding.json'
            pproof = v.file_proof(preservation, v.PRESERVATION_SHA)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); controller = dict(pid=10,uid=1005,starttick=20,exit_code=0)
                final = root/'controller-final.json'; write(final, dict(status='WORKERS_COMPLETE'))
                final_ref = v.file_proof(final)
                exit_path = root/'tmux-exit-proof.json'
                exit_value = v.exit_evidence('fixture',controller,{'pane_dead':True,'pane_dead_status':0},pproof,{'fixture_only':True})
                write(exit_path,exit_value); exit_ref=v.file_proof(exit_path)
                release=dict(controller=controller,m1_run_id='fixture',**v.release_references(final_ref,exit_path))
                self.assertEqual(utils.checked_json(release['controller_final'])[0]['status'],'WORKERS_COMPLETE')
                actual_exit,_=utils.checked_json(release['tmux_exit_proof'])
                self.assertEqual(utils.checked_json(actual_exit['preservation_receipt'])[1]['sha256'],v.PRESERVATION_SHA)
                gates.validate_exit_proof(release,actual_exit)
                with self.assertRaises((TypeError,KeyError)):
                    utils.checked_json(str(final))
        finally:
            sys.path.pop(0)

    def test_student_privileged_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);case,result=case_fixture(directory);module,_=v.load_region(REGION)
            path=directory/'student-generation-update1.json';gen=json.loads(path.read_text());gen['privileged_reference']=True;write(path,gen)
            with self.assertRaisesRegex(ValueError,'privileged reference'):
                v.case_probes(module,case,result,directory,{},Counter(),Counter())

    def test_real_exit_zero_required(self):
        args = terminal_fixture(); self.assertEqual(v.validate_terminal(*args)['exit_code'], 0)
        for value in (None, '', 1, False):
            changed = copy.deepcopy(args); changed[-1]['pane_dead_status'] = value
            with self.assertRaises(ValueError): v.validate_terminal(*changed)

    def test_foreign_pane_or_nonzero_child_rejected(self):
        args = terminal_fixture(); args[-1]['pane_pid'] = 123
        with self.assertRaises(ValueError): v.validate_terminal(*args)
        args = terminal_fixture(); args[2]['child_exit_codes']['20'] = None
        with self.assertRaises(ValueError): v.validate_terminal(*args)

    def test_absence_only_termination_and_reuse(self):
        identity = dict(pid=12, uid=1005, starttick=99)
        def missing(pid): raise FileNotFoundError()
        self.assertEqual(v.terminated([identity], missing)[0]['state'], 'ABSENT')
        with self.assertRaises(ValueError): v.terminated([identity], lambda pid: dict(identity, state='Z'))
        with self.assertRaises(ValueError): v.terminated([identity], lambda pid: dict(identity, uid=0, state='S'))
        self.assertEqual(v.terminated([identity], lambda pid: dict(identity, starttick=100))[0]['state'], 'PID_REUSED')

    def test_sources_fresh_hash_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.py'; path.write_text('x=1\n')
            proof = v.fresh_sources(tmp, {'a.py': v.digest(path.read_bytes())}, 1)
            path.write_text('x=2\n')
            with self.assertRaises(ValueError): v.source_stats_unchanged(proof)
            with self.assertRaises(ValueError): v.fresh_sources(tmp, {'../a.py': '0'*64}, 1)

    def test_skip_requires_actual_boundary(self):
        item = dict(status='skipped', reason=v.SKIP_REASON, response_tokens=0, valid_policy_tokens=0,
                    ordinary_prefix_tokens=10, privileged_prefix_tokens=None)
        self.assertEqual(v.check_skip(item)['status'], 'allowed_skip')
        item.update(response_tokens=10, valid_policy_tokens=10)
        with self.assertRaises(ValueError): v.check_skip(item)
        item['ordinary_prefix_tokens'] = 8190; self.assertEqual(v.check_skip(item)['status'], 'allowed_skip')

    def test_complete9_artifacts_and_teacher_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); case, result = case_fixture(directory); module, _ = v.load_region(REGION)
            counts = Counter(); item = v.case_probes(module, case, result, directory, {}, counts, Counter())
            self.assertEqual(counts['complete'], 9)
            self.assertEqual(set(item['teacher_invariance']), {'update1','update5','epoch1'})
            self.assertTrue(all(x['status']=='verified' for x in item['teacher_invariance'].values()))

    def test_hard24_artifacts_null_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); case, result = case_fixture(directory, True); module, _ = v.load_region(REGION)
            counts = Counter(); v.case_probes(module, case, result, directory, {}, counts, Counter())
            self.assertEqual(counts['complete'], 24)
            result['actual_teacher_response_teacher_forced_probe'] = {}
            with self.assertRaises(ValueError): v.case_probes(module, case, result, directory, {}, Counter(), Counter())

    def test_missing_artifact_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); case, result = case_fixture(directory); module, _ = v.load_region(REGION)
            (directory/'student-generation-update5.json').unlink()
            with self.assertRaises(FileNotFoundError): v.case_probes(module, case, result, directory, {}, Counter(), Counter())

    def test_native_mask_or_sha_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); case, result=case_fixture(directory); module,_=v.load_region(REGION)
            summary=result['versions']['source100']['token_probes'][0]
            summary['frozen_pair_loss']=.9
            write(directory/'token-summary-source100-original-mined-student.json',summary)
            with self.assertRaisesRegex(ValueError,'native trajectory'): v.case_probes(module,case,result,directory,{},Counter(),Counter())

    def test_fresh_weights12_plus8_one_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/'base'; base.mkdir(); files=[]
            for i in range(12):
                path=base/(str(i)+'.json');path.write_text('base'+str(i));files.append(dict(path=str(path),relative_path=path.name,sha256=v.digest(path.read_bytes()),**v.stat(path)))
            adapters={}
            for version in v.VERSIONS:
                d=root/version;d.mkdir();c=d/'adapter_config.json';m=d/'adapter_model.safetensors';c.write_text('config');m.write_text(version)
                adapters[version]=dict(path=str(d),adapter_config_sha256=v.digest(c.read_bytes()),adapter_model_sha256=v.digest(m.read_bytes()),config_stat=v.stat(c),model_stat=v.stat(m))
            proof=v.fresh_weights(dict(source100=dict(path=str(base),files=files),adapters=adapters))
            self.assertEqual(len(proof['source100']),12);self.assertEqual(len(proof['adapters']),4)
            (base/'0.json').write_text('changed')
            with self.assertRaises(ValueError):v.source_stats_unchanged(proof)

    def test_full_vocab_cap_skip_distinct_from_probe_skip(self):
        counts=Counter();summary=dict(full_vocabulary_dump=dict(status='cap_skip',bytes_requested_with_header_allowance=8,worker_bytes_reserved=5,worker_cap_bytes=10))
        self.assertEqual(v.full_vocab(summary,Path('.'),{},counts)['status'],'cap_skip');self.assertEqual(counts['cap_skip'],1)
        summary['full_vocabulary_dump']['worker_cap_bytes']=20
        with self.assertRaises(ValueError):v.full_vocab(summary,Path('.'),{},Counter())

    def test_nonfinite_rejected_before_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json';path.write_text('{"a":1e400}')
            with self.assertRaises(ValueError):v.json_file(path)


if __name__=='__main__': unittest.main()
