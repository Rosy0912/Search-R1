"""Synthetic fast raw-census/subset CPU tests; no real GPU journal replay."""
import copy
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('failed3294_diag',HERE/'diagnose_policy_discovery_v5_special_tokens_v4.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
REPO=HERE.parent/'repo'
launch=m.strict_json((HERE/'policy-discovery-v5-native-launch.json').read_bytes())
source=REPO/'reproduction/common8/behavior_trace.py'
assert m.sha(source.read_bytes())==launch['source_sha256']['reproduction/common8/behavior_trace.py']
helper_spec=importlib.util.spec_from_file_location('original_e670_pure_trace',source)
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)


class TokenizerFixture:
    eos_token_id=9;pad_token_id=0;all_special_ids=[0,8,9]
    fragments={0:'<|pad|>',1:'<answer>CPU answer',2:'</answer>',3:'<search>CPU query',4:'</search>',
        5:' tail',6:'ordinary',8:'<|other_special|>',9:'<|eos|>'}
    def decode(self,ids,*,skip_special_tokens=False,clean_up_tokenization_spaces=False):
        assert not clean_up_tokenization_spaces
        if skip_special_tokens: ids=[token for token in ids if token not in self.all_special_ids]
        return ''.join(self.fragments[token] for token in ids)


def sampling():
    return dict(n=1,best_of=1,do_sample=True,temperature=1.,top_p=1.,top_k=-1,min_p=0.,
        presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,logprobs=1,
        min_tokens=0,logits_processors=[],logprob_semantics=helper.LOGPROB_SEMANTICS,max_tokens=500)


def record(raw,*,row='CPU-record',ordinal=0,lp=None):
    return dict(rollout_call_id='CPU-call-'+str(ordinal),actual_request_token_ids=[6,6],
        raw_generated_token_ids=raw,raw_generated_token_count=len(raw),
        selected_logprobs=[-.1]*len(raw) if lp is None else lp,finish_reason='stop' if raw[-1:]==[9] else 'length',
        stop_reason=None,eos_token_id=9,pad_token_id=0,sampling=sampling(),policy_snapshot_id='CPU:1:policy-fixture',
        schema_version=1,request_id='CPU-request-'+row,row_id=row,rollout_ordinal=ordinal)


def journal(records):
    return dict(run_id='CPU',source_sha256={'CPU.py':'1'*64},slurm_job_id='1',logical_update=1,
        policy_snapshot_id='CPU:1:policy-fixture',role='common8_failed_partial_generation',
        actual_sampling_records=records,failure_reason='BehaviorTraceError',complete_outer=False)


def diagnose(records,tokenizer=None):
    return m.journal_diagnosis(journal(records),tokenizer or TokenizerFixture(),helper,
        run_id='CPU',source_sha256={'CPU.py':'1'*64},job_id='1')


def actual_paths():
    names=dict(launch='policy-discovery-v5-native-launch.json',birth='policy-discovery-v5-observation-1791254632444188.json',
        terminal='policy-discovery-v5-observation-1791256551399966.json')
    return {label:dict(path=str(HERE/name),sha256=m.sha((HERE/name).read_bytes())) for label,name in names.items()}


class PrivateDiagnosisTests(unittest.TestCase):
    def test_actual_e670_git_empty_source_set_is_exact_and_authenticated(self):
        observed=set()
        for name,digest in launch['source_sha256'].items():
            path=REPO/name
            if path.stat().st_size==0:
                observed.add(name);self.assertEqual(digest,m.EMPTY_SOURCE_SHA256)
                original=subprocess.check_output(['git','show',m.SOURCE_COMMIT+':'+name],cwd=REPO)
                self.assertEqual(original,b'');self.assertEqual(m.sha(original),digest)
        self.assertEqual(observed,m.EMPTY_SOURCE_FILES)

    def test_empty_source_permission_is_explicit_exact_hash_path_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve()
            for name in m.EMPTY_SOURCE_FILES:
                path=base/'repo'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'')
                raw,receipt=m.readonly_bytes(path,100,expected_sha256=m.EMPTY_SOURCE_SHA256,empty_source_relative=name)
                self.assertEqual(raw,b'');self.assertEqual(receipt['size_bytes'],0)
                self.assertEqual(receipt['sha256'],m.EMPTY_SOURCE_SHA256);self.assertEqual(path.read_bytes(),b'')
                with self.assertRaisesRegex(ValueError,'bounded'):m.readonly_bytes(path,100,expected_sha256=m.EMPTY_SOURCE_SHA256)
                with self.assertRaisesRegex(ValueError,'authenticated package source'):
                    m.readonly_bytes(path,100,expected_sha256='0'*64,empty_source_relative=name)
                with self.assertRaisesRegex(ValueError,'authenticated package source'):
                    m.readonly_bytes(path,100,private=True,expected_sha256=m.EMPTY_SOURCE_SHA256,empty_source_relative=name)

    def test_empty_metadata_and_journal_rejected_even_if_expected_sha_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();base.chmod(0o700)
            for name in ('manifest.json','outer-1.json','unknown.py'):
                path=base/name;path.write_bytes(b'');path.chmod(0o600)
                with self.assertRaisesRegex(ValueError,'bounded'):
                    m.readonly_bytes(path,100,expected_sha256=m.EMPTY_SOURCE_SHA256)
                with self.assertRaisesRegex(ValueError,'authenticated package source'):
                    m.readonly_bytes(path,100,expected_sha256=m.EMPTY_SOURCE_SHA256,
                        empty_source_relative='search_r1/__init__.py')
            unknown=base/'repo/unknown.py';unknown.parent.mkdir();unknown.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'authenticated package source'):
                m.readonly_bytes(unknown,100,expected_sha256=m.EMPTY_SOURCE_SHA256,empty_source_relative='unknown.py')

    def test_official_extra_eos_distinct_from_tokenizer_and_captured_explicit_stops(self):
        value=record([1,8]);value.update(finish_reason='stop',stop_reason=8)
        result=m.journal_diagnosis(journal([value]),TokenizerFixture(),helper,run_id='CPU',
            source_sha256={'CPU.py':'1'*64},job_id='1',generation_config={'eos_token_id':[9,8]})
        self.assertEqual(result['existing_helper_error_counts'],{'included_special_or_pad_token':1})
        self.assertEqual(result['official_generation_config_eos_not_tokenizer_eos_count'],1)
        self.assertEqual(result['counts']['native_generation_eos_not_in_captured_explicit_stops'],1)
        self.assertEqual(result['counts']['finish_stop_and_final_raw_matches_official_generation_eos'],1)
        detail=result['details'][0]
        self.assertEqual(detail['special_positions'][0]['category'],'other_special')
        self.assertTrue(detail['special_positions'][0]['in_official_generation_config_eos'])
        self.assertTrue(detail['special_positions'][0]['inside_original_retained_prefix'])
        self.assertFalse(result['engine_effective_stop_set_authenticated'])
        self.assertFalse(result['accepted'])

    def test_official_extra_eos_schema_does_not_infer_unknown_or_boolean_ids(self):
        self.assertEqual(m.generation_eos_ids({}),set())
        self.assertEqual(m.generation_eos_ids({'eos_token_id':9}),{9})
        for value in (True,[9,False],[],['9'],-1):
            with self.assertRaisesRegex(ValueError,'EOS schema'):
                m.generation_eos_ids({'eos_token_id':value})

    def test_actual_arithmetic_before512_summary_omits_rows_and_no_health_claim(self):
        # Actual hash-bound e670 aggregate, selected by AST; no model/Torch evaluation.
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'repo/reproduction/common8').mkdir(parents=True)
            src=REPO/'reproduction/common8/calibration_evaluation.py'
            (base/'repo/reproduction/common8/calibration_evaluation.py').write_bytes(src.read_bytes())
            aggregate=m.evaluation_aggregate(base,launch['source_sha256'])
            ids=[m.sha(str(i).encode()) for i in range(512)];sources=['nq']*256+['hotpotqa']*256
            rows=[dict(question_id=ids[i],data_source=sources[i],em=int(i<91),f1=float(i<91),
                actual_calls=0,valid_action=1,turns=1,answer_sha256='1'*64,answer_present=True,
                response_token_count=2,model_token_count=2) for i in range(512)]
            calculated=aggregate(rows,ids,sources)
            run=dict(run_id='CPU',parent_production_run_id='parent',policy_calibration_slurm_job_id='1',
                policy_calibration_plan_sha256='2'*64,policy_calibration_role='policy_safety_discovery')
            validation=dict(validation_sha256='3'*64,question_ids=ids,data_sources=sources)
            value=calculated|dict(run_id='CPU',parent_production_run_id='parent',slurm_job_id='1',
                plan_sha256='2'*64,step=0,validation_sha256='3'*64,phase_role='policy_safety_discovery')
            result=m.before512_projection(value,run,validation,aggregate)
            self.assertEqual(result['weighted_em'],91/512);self.assertEqual(result['source_counts'],{'nq':256,'hotpotqa':256})
            self.assertNotIn('rows',result);self.assertNotIn('question_ids',result)
            self.assertFalse(result['health_evaluated']);self.assertFalse(result['accepted'])
            for key,changed in (('weighted_em',1),('step',1),('plan_sha256','4'*64)):
                with self.assertRaises(ValueError):m.before512_projection(value|{key:changed},run,validation,aggregate)
            missing=m.before512_diagnosis(base,run,launch['source_sha256'])
            self.assertEqual(missing,dict(exists=False,health_evaluated=False,accepted=False))

    def test_real_failed_terminal_binding_local_only_no_rpc(self):
        reader,actual,terminal=m.input_contract(actual_paths())
        self.assertEqual(actual['source_commit'],m.SOURCE_COMMIT)
        self.assertEqual(terminal['slurm_terminal_state'],'FAILED');self.assertEqual(terminal['job_id'],'3294')
        self.assertEqual(terminal['exit_code'],'1:0');self.assertFalse(terminal['production_admission'])
        code=m.remote_code(reader);namespace={};exec(code.split('p=json.load(sys.stdin)',1)[0],namespace)
        self.assertEqual(namespace['PRIVATE_FIELDS'],m.PRIVATE_FIELDS)
        self.assertNotIn('torch',namespace);self.assertNotIn('AutoTokenizer',namespace)

    def test_guard_rejects_default_execution_before_input_or_rpc(self):
        with patch.object(m,'input_contract',side_effect=AssertionError('Must not inspect inputs')):
            with self.assertRaisesRegex(ValueError,'explicit execute'):
                m.execute_once({},HERE/'must-not-write.json',rpc=lambda *args:self.fail('Must not RPC'))
        result=subprocess.run([sys.executable,'-B',str(HERE/'diagnose_policy_discovery_v5_special_tokens_v4.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('--terminal',result.stderr)

    def test_mismatched_original_sha_and_missing_failure_inputs_reject(self):
        for label in ('launch','birth','terminal'):
            value=actual_paths();value[label]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'failed3294 SHA'):m.input_contract(value)
        with self.assertRaisesRegex(ValueError,'three explicit'):m.input_contract({})

    def test_internal_pad_is_real_stored_raw_position_and_original_helper_rejects(self):
        result=diagnose([record([1,0,2])]);detail=result['details'][0]
        self.assertEqual(result['existing_helper_error_counts'],{'included_special_or_pad_token':1})
        self.assertEqual(result['retained_special_token_class_counts'],{'pad':1})
        self.assertEqual(detail['special_positions'][0]['raw_position'],1)
        self.assertTrue(detail['special_positions'][0]['captured_pre_padding_position'])
        self.assertTrue(detail['special_positions'][0]['inside_original_retained_prefix'])
        self.assertEqual(detail['diagnostic_projection']['retained_count'],3)
        self.assertTrue(detail['diagnostic_projection']['decoded_prefix_stable'])
        self.assertFalse(detail['context_audit_accepted']);self.assertFalse(result['accepted'])

    def test_internal_eos_and_other_special_classification_no_removal(self):
        result=diagnose([record([1,9,2]),record([1,8,2],row='CPU-other')])
        self.assertEqual(result['retained_special_token_class_counts'],{'eos':1,'other_special':1})
        self.assertEqual(result['existing_helper_error_counts']['included_special_or_pad_token'],2)
        self.assertEqual(result['counts']['observed_diagnostic_retained_token_count'],6)

    def test_true_trailing_eos_excluded_only_in_original_helper_scope(self):
        result=diagnose([record([1,2,9])]);detail=result['details'][0]
        self.assertEqual(result['candidate_helper_replay_count'],0)
        self.assertIsNone(detail['diagnostic_projection'])
        self.assertFalse(detail['original_helper_replayed'])
        self.assertEqual(result['raw_special_token_class_counts'],{'eos':1})
        self.assertEqual(result['retained_special_token_class_counts'],{})
        self.assertTrue(detail['special_positions'][0]['trailing_eos_excluded'])
        self.assertFalse(detail['special_positions'][0]['inside_original_retained_prefix'])
        self.assertTrue(detail['trailing_eos_has_native_stop_evidence'])
        changed=record([1,2,9]);changed['finish_reason']='length'
        self.assertEqual(diagnose([changed])['counts']['trailing_eos_without_native_stop'],1)

    def test_pad_equals_eos_remains_explicit_ambiguous_class(self):
        tokenizer=TokenizerFixture();tokenizer.pad_token_id=9
        value=record([1,9,2]);value['pad_token_id']=9
        result=diagnose([value],tokenizer)
        self.assertTrue(result['tokenizer_eos_equals_pad'])
        self.assertEqual(result['retained_special_token_class_counts'],{'eos_and_pad':1})

    def test_special_after_closing_raw_prefix_is_not_falsely_included(self):
        result=diagnose([record([1,2,8,6])]);detail=result['details'][0]
        self.assertEqual(result['counts']['existing_helper_pass'],1)
        self.assertEqual(detail['diagnostic_projection']['retained_count'],2)
        self.assertFalse(detail['special_positions'][0]['inside_original_retained_prefix'])
        self.assertEqual(result['retained_special_token_class_counts'],{})

    def test_crossing_whole_token_projection_and_future_context_hashed_only(self):
        tokenizer=TokenizerFixture();tokenizer.fragments=tokenizer.fragments|{2:'</answer> trailing'}
        result=diagnose([record([1,2,8])],tokenizer);detail=result['details'][0]
        self.assertEqual(detail['diagnostic_projection']['suffix_class'],'nonwhitespace')
        self.assertEqual(detail['diagnostic_projection']['retained_count'],2)
        self.assertTrue(detail['diagnostic_projection']['decoded_prefix_stable'])
        self.assertNotIn('CPU answer',m.canonical(result).decode());self.assertNotIn('trailing',m.canonical(result).decode().replace('trailing_eos',''))

    def test_all_record_census_not_just_first_failure_and_bounded_details(self):
        records=[record([1,0,2],row='CPU-'+str(i)) for i in range(70)]
        result=diagnose(records)
        self.assertEqual(result['sampling_record_count'],70)
        self.assertEqual(result['existing_helper_error_counts']['included_special_or_pad_token'],64)
        self.assertEqual(result['candidate_record_count'],70)
        self.assertEqual(result['candidate_helper_replay_omitted_count'],6)
        self.assertEqual(len(result['details']),64);self.assertEqual(result['detail_records_omitted'],6)
        self.assertEqual(result['counts']['records_with_included_special'],64)
        self.assertEqual(result['raw_special_token_class_counts'],{'pad':70})
        self.assertEqual(result['retained_special_token_class_counts'],{'pad':64})

    def test_actual_lp_values_never_returned_but_finite_length_census_observed(self):
        value=record([1,0,2],lp=[-.123456789,{'__behavior_trace_nonfinite_float__':'nan'},.5])
        value['raw_generated_token_count']=2
        result=diagnose([value]);detail=result['details'][0]
        self.assertEqual(result['selected_lp_nonfinite_or_typed_marker_count'],1)
        self.assertEqual(result['selected_lp_positive_count'],1)
        self.assertEqual(result['raw_count_or_lp_length_mismatch_count'],1)
        self.assertFalse(detail['stored_raw_count_matches_lp']);self.assertFalse(detail['selected_lp_all_finite'])
        self.assertNotIn('-.123456789',m.canonical(result).decode())
        self.assertFalse(detail['special_positions'][0]['captured_pre_padding_position'])

    def test_environment_tokens_and_missing_masks_are_explicit_not_synthesized(self):
        event=dict(role='actual_environment_event',rollout_call_id='CPU-call-0',row_ids=['CPU1','CPU2'],
            observation_or_correction_token_ids=[[3,4],[6]],is_search=[1,0],final_turn=False)
        result=diagnose([record([1,0,2]),event])
        self.assertEqual(result['environment_row_count'],2);self.assertEqual(result['environment_token_count'],3)
        self.assertEqual(result['counts']['observation_tokens'],2);self.assertEqual(result['counts']['correction_tokens'],1)
        self.assertEqual(result['stored_final_actor_mask_count'],0);self.assertEqual(result['stored_sampling_tensor_mask_count'],0)

    def test_private_schema_crossrun_unknown_and_fake_masks_reject(self):
        original=journal([record([1,0,2])])
        for key,value in (('run_id','wrong'),('complete_outer',True),('unknown',0)):
            changed=original|{key:value}
            with self.assertRaisesRegex(ValueError,'partial outer1'):
                m.journal_diagnosis(changed,TokenizerFixture(),helper,run_id='CPU',source_sha256={'CPU.py':'1'*64},job_id='1')
        value=record([1,0,2]);value['sampling_mask']=[1,1,1]
        with self.assertRaisesRegex(ValueError,'record/mask schema'):diagnose([value])

    def test_prefix_ambiguity_is_counted_and_not_repaired_by_extending_tokens(self):
        class Ambiguous(TokenizerFixture):
            def decode(self,ids,**kwargs):
                if ids==[1]:return '<answer>CPU</answer>\ufffd'
                if ids==[1,2]:return '<answer>CPU</answer>different'
                return super().decode(ids,**kwargs)
        result=diagnose([record([1,2,8])],Ambiguous())
        self.assertEqual(result['existing_helper_error_counts'],{'ambiguous_decoding_boundary':1})
        self.assertEqual(result['diagnostic_prefix_unstable_count'],1)
        self.assertEqual(result['details'][0]['diagnostic_projection']['retained_count'],1)

    def test_all2560_ordinary_vectors_have_zero_decoder_or_helper_prefix_replays(self):
        class NoDecode(TokenizerFixture):
            def decode(self,*args,**kwargs):raise AssertionError('Ordinary records must not be decoded')
        records=[record([6]*500,row='CPU-'+str(i)) for i in range(2560)]
        result=diagnose(records,NoDecode())
        self.assertEqual(result['sampling_record_count'],2560)
        self.assertEqual(result['counts']['stored_raw_token_count'],1280000)
        self.assertEqual(result['candidate_helper_replay_count'],0)
        self.assertEqual(result['ordinary_or_single_canonical_trailing_eos_records_not_decoded'],2560)
        self.assertEqual(result['lengths']['retained'],dict(count=0,minimum=None,maximum=None,sum=None))
        self.assertFalse(result['accepted']);self.assertFalse(result['context_verified'])

    def test_all2560_candidate_vectors_are_censused_with_exact64_original_replays(self):
        records=[record([1,8,2],row='CPU-'+str(i)) for i in range(2560)]
        result=diagnose(records)
        self.assertEqual(result['sampling_record_count'],2560)
        self.assertEqual(result['raw_special_token_class_counts'],{'other_special':2560})
        self.assertEqual(result['candidate_helper_replay_count'],64)
        self.assertEqual(result['candidate_helper_replay_omitted_count'],2496)
        self.assertEqual(result['existing_helper_error_counts'],{'included_special_or_pad_token':64})
        self.assertEqual(result['lengths']['retained']['count'],64)
        self.assertEqual(len(result['details']),64)
        self.assertFalse(result['context_verified'])

    def test_canonical_trailing_eos_only_records_are_raw_census_not_acceptance(self):
        records=[record([6,9],row='CPU-'+str(i)) for i in range(2560)]
        result=diagnose(records)
        self.assertEqual(result['raw_special_token_class_counts'],{'eos':2560})
        self.assertEqual(result['candidate_helper_replay_count'],0)
        self.assertEqual(result['counts']['trailing_eos_without_native_stop'],0)
        self.assertEqual(result['initial_before512'] if 'initial_before512' in result else None,None)
        self.assertFalse(result['accepted']);self.assertFalse(result['healthy_checkpoint_accepted'])

    def test_private_original_bytes_hash_stat_modes_and_readonly_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory).resolve()/'private';parent.mkdir(mode=0o700)
            path=parent/'outer-1.json';raw=m.canonical(journal([record([1,0,2])]))
            path.write_bytes(raw);path.chmod(0o600)
            found,receipt=m.readonly_bytes(path,len(raw),private=True)
            self.assertEqual(found,raw);self.assertEqual(receipt['sha256'],m.sha(raw));self.assertEqual(path.read_bytes(),raw)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError,'private ownership'):m.readonly_bytes(path,len(raw),private=True)
            path.chmod(0o600);link=parent/'link';link.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'symlink'):m.readonly_bytes(link,len(raw),private=True)
            with self.assertRaisesRegex(ValueError,'bounded'):m.readonly_bytes(path,len(raw)-1,private=True)

    def test_no_remote_launcher_weights_or_raw_publication_paths(self):
        code=Path(m.__file__).read_text()
        self.assertNotIn("['sbatch'",code);self.assertNotIn('get_tensor(',code);self.assertNotIn('safe_open(',code)
        self.assertNotIn('torch.load(',code);self.assertNotIn('signal.',code);self.assertNotIn('save_pretrained',code)
        self.assertEqual(m.TOKENIZER_FILES,('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json'))
        result=diagnose([record([1,0,2])])
        def keys(value):
            if type(value) is dict:
                for key,item in value.items():yield key;yield from keys(item)
            if type(value) is list:
                for item in value:yield from keys(item)
        forbidden={'raw_generated_token_ids','actual_request_token_ids','selected_logprobs','execution_text','decoded_retained_prefix','prompt'}
        self.assertFalse(forbidden.intersection(keys(result)))



class DualProjectionTests(unittest.TestCase):
    def test_special_inside_query_is_not_cleaned_authored_text(self):
        result=diagnose([record([3,8,4,5,9])])['details'][0]['dual_projection_comparison']
        self.assertTrue(result['whole_prefix_counts_equal'])
        self.assertFalse(result['execution_hashes_equal'])
        self.assertEqual(result['visible']['first_complete_action_kind'],'search')
        self.assertEqual(result['upstream_skip_special']['first_complete_action_kind'],'search')
        self.assertEqual(result['visible']['retained_special_token_count'],1)
        self.assertFalse(result['reward_cleaned']);self.assertFalse(result['accepted'])
        self.assertNotEqual(result['visible']['first_complete_action_payload_sha256'],result['upstream_skip_special']['first_complete_action_payload_sha256'])

    def test_special_removal_can_join_closing_without_raw_token_removal(self):
        class Split(TokenizerFixture):
            fragments=dict(TokenizerFixture.fragments, **{})
            fragments.update({10:'<search>q</sea',11:'rch>'})
        result=m.dual_projection([10,8,11],Split(),9,set(Split.all_special_ids))
        self.assertTrue(result['special_removal_creates_projected_closing'])
        self.assertEqual(result['visible']['closing_tag_kind'],'none')
        self.assertEqual(result['upstream_skip_special']['closing_tag_kind'],'search')
        self.assertEqual(result['upstream_skip_special']['earliest_whole_prefix_count'],3)
        self.assertFalse(result['raw_ids_or_lp_changed'])

    def test_search_priority_and_first_answer_are_independently_projected(self):
        result=m.dual_projection([1,2,8,3,4,9],TokenizerFixture(),9,set(TokenizerFixture.all_special_ids))
        for mode in ('visible','upstream_skip_special'):
            self.assertEqual(result[mode]['closing_tag_kind'],'search')
            self.assertEqual(result[mode]['first_complete_action_kind'],'answer')
            self.assertEqual(result[mode]['earliest_whole_prefix_count'],5)
        self.assertTrue(result['first_complete_action_kinds_equal'])

    def test_special_after_closing_keeps_original_prefix_and_eos_rule(self):
        raw=[1,2,8,9]
        result=m.dual_projection(raw,TokenizerFixture(),9,set(TokenizerFixture.all_special_ids))
        self.assertEqual(raw,[1,2,8,9])
        for mode in ('visible','upstream_skip_special'):
            self.assertEqual(result[mode]['earliest_whole_prefix_count'],2)
            self.assertEqual(result[mode]['retained_special_token_count'],0)
        self.assertTrue(result['execution_hashes_equal'])
        diagnosis=diagnose([record(raw)])
        self.assertEqual(diagnosis['existing_helper_error_counts'],{})
        self.assertEqual(diagnosis['candidate_helper_replay_count'],1)

    def test_original_executor_regex_ast_agrees_with_diagnostic(self):
        import ast
        tree=ast.parse((REPO/'search_r1/llm_agent/generation.py').read_text())
        patterns=[node.value for node in ast.walk(tree) if isinstance(node,ast.Constant)
            and isinstance(node.value,str) and node.value==r'<(search|answer)>(.*?)</\1>']
        self.assertEqual(len(patterns),1)
        result=m.dual_projection([1,2,8,3,4],TokenizerFixture(),9,set(TokenizerFixture.all_special_ids))
        self.assertEqual(result['upstream_skip_special']['first_complete_action_kind'],'answer')

    def test_no_tag_and_nonstable_projection_remain_observations_not_acceptance(self):
        class Nonstable(TokenizerFixture):
            def decode(self,ids,**kwargs):
                if ids==[3,8,4]:return '<search>q</search>'
                return super().decode(ids,**kwargs)
        result=m.dual_projection([3,8,4,5],Nonstable(),9,set(Nonstable.all_special_ids))
        self.assertFalse(result['visible']['decoded_prefix_stable'])
        self.assertFalse(result['context_verified'])
        plain=m.dual_projection([6,8],TokenizerFixture(),9,set(TokenizerFixture.all_special_ids))
        self.assertEqual(plain['visible']['closing_tag_kind'],'none')
        self.assertEqual(plain['upstream_skip_special']['earliest_whole_prefix_count'],2)
        text=json.dumps(result)
        self.assertNotIn('CPU query',text);self.assertNotIn('<|other_special|>',text)

if __name__=='__main__':unittest.main()
