"""Owner synthetic CPU completion/refusal tests, never actual model evidence."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import probe_package as package
from test_dispatch_window_once import synthetic_envelopes


def synthetic_ready(rank, bundle, evidence):
    anchors = [dict(x,status='observed') for x in package.window_anchors() if x['rank'] == rank]
    return dict(status=package.WINDOW_READY_STATUS,rank=rank,world_size=8,
        source_and_input_bundle_identity=bundle['identity'],window_profile=package.WINDOW_PROFILE,
        bundle_sha256='d'*64,window_inputs_sha256=bundle['window_inputs_sha256'],
        original_generation_call=5,original_prompt_rows=320,generation_invocations=1,
        actor_scoring_native_forwards=80,warmup_native_forwards=1,actor_native_forwards=81,
        maximum_new_emitted_tokens=160000,actual_new_emitted_tokens=6400,
        optimizer_committed=0,optimizer_instantiated=False,scheduler_instantiated=False,
        scheduler_advanced=False,backward_called=False,probability_accepted=False,
        training_accepted=False,GPU_capacity_proven=False,original_master_state_unchanged=True,
        original_modes_restored=True,actual_loaded_binary_identity_proven=False,
        cpu_preparation=dict(status='ACTUAL_CALL5_CPU_INPUTS_VERIFIED_BEFORE_CUDA',actual_tensor_rows_verified=True,
            cuda_initialized=False,rank=rank,local_rows=320,original_generation_call=5,expected_actor_micro4_calls=80,
            source_snapshot=bundle['window_inputs']['ranks'][rank]['source_snapshot'],
            source_record=bundle['window_inputs']['ranks'][rank]['source_record'],
            source_full_input_identity=bundle['window_inputs']['ranks'][rank]['source_full_input_identity'],
            window_inputs_sha256=bundle['window_inputs_sha256'],model_forwards=0,optimizer_writes=0),
        services=dict(ray_initialized=False,vllm_engine_required=True,optimizer_instantiated=False),
        actual_runtime=dict(configs=dict(parallel_config=dict(tensor_parallel_size=1))),
        wrappers=dict(counts=dict(sync=1,output=1,execute=20),registered_requests=320),
        observer=dict(status='PASSIVE_WINDOW_OBSERVATION_COMPLETE',observer_evidence_complete=True,
            hooks_restored=True,errors=[],probability_accepted=False,training_accepted=False,
            observer_replaced_native_outputs=False,observer_changed_sampling_parameters=False,
            generated=[dict(input_row=i,true_response_length=20) for i in range(320)],anchors=anchors),
        live_window_weights=dict(status='LIVE_SYNC_AND_PRE_OFFLOAD_WEIGHTS_OBSERVED_NO_PROBABILITY_ACCEPTANCE',
            buffers_compared=True,probability_accepted=False,training_accepted=False,
            historical_3268_identity_proven=False,
            pre=dict(serialized_source_keys=435,serialized_target_keys=255,tensor_parallel_size=1,
                actual_device_is_cuda=True,all_parameter_values_match=False,max_abs_delta=1.5),
            post=dict(status='LIVE_PRE_OFFLOAD_BYTES_AND_ALIAS_UNCHANGED',same_window_tie=True,
                CPU_offload_snapshot_used=False)),
        raw_probability_observations=dict(max_abs_delta=1.5),tensor_evidence=evidence,
        anchor_observations=[dict(anchor_id=a['id'],status='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED',
            input_row=a['input_row'],response_offset=a['response_offset'],probability_accepted=False,training_accepted=False,
            vllm_context_token_ids=[1,2],actor_context_token_ids=[1,2],vllm_context_positions=[0,1],actor_context_positions=[0,1],
            vllm_target_token_id=2,actor_target_token_id=2,vocab_shape=[4],
            tensor_identities={k:dict(shape=[4]) for k in ('head','sampler_input','actor')},
            **{k:dict(count=4,max_abs=1.5) for k in ('full_vocab_actor_minus_vllm_head',
                 'full_vocab_actor_minus_vllm_head_after_constant','full_vocab_sampler_input_minus_head')}) for a in anchors])


class WindowReadyTests(unittest.TestCase):
    def setUp(self):
        self.bundle=synthetic_envelopes()[0]
        self.bundle['identity']={'synthetic_cpu_fixture':True}
        self.row=synthetic_ready(4,self.bundle,dict(path='/unused/synthetic',sha256='a'*64,bytes=4))

    def test_finite_weights_and_logprob_disagreement_is_valid_measurement(self):
        self.assertFalse(self.row['live_window_weights']['pre']['all_parameter_values_match'])
        self.assertIs(package.validate_window_ready(self.row,4,self.bundle,'d'*64),self.row)
        self.assertFalse(self.row['probability_accepted'])

    def test_exact_count_and_boolean_count_refusals(self):
        for field,value in [('generation_invocations',5),('generation_invocations',True),
            ('original_prompt_rows',319),('actor_scoring_native_forwards',79),
            ('warmup_native_forwards',0),('actor_native_forwards',80),('maximum_new_emitted_tokens',159999),
            ('optimizer_committed',False),('actual_new_emitted_tokens',160001)]:
            with self.subTest(field=field,value=value):
                row=copy.deepcopy(self.row);row[field]=value
                with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_old_or_wrong_bundle_ready_refused(self):
        for field,value in [('status','SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE'),
            ('bundle_sha256','e'*64),('window_inputs_sha256','e'*64),('window_profile','native_only'),
            ('rank',3),('original_master_state_unchanged',False),('original_modes_restored',False)]:
            row=copy.deepcopy(self.row);row[field]=value
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_no_default_probability_training_binary_or_preparation_acceptance(self):
        for field in ('optimizer_instantiated','scheduler_instantiated','scheduler_advanced','backward_called',
                      'probability_accepted','training_accepted','GPU_capacity_proven','actual_loaded_binary_identity_proven'):
            row=copy.deepcopy(self.row);row[field]=True
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)
        row=copy.deepcopy(self.row);row['cpu_preparation']={}
        with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_wrong_tp1_or_native_window_counts_refused(self):
        for change in (dict(tensor_parallel_size=2),dict(tensor_parallel_size=True)):
            row=copy.deepcopy(self.row);row['actual_runtime']['configs']['parallel_config']=change
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)
        for key,value in [('sync',2),('sync',True),('output',2),('execute',0)]:
            row=copy.deepcopy(self.row);row['wrappers']['counts'][key]=value
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_incomplete_generated_rows_lengths_or_observer_refused(self):
        for kind in ('missing','duplicate','length','bool_row','hooks','error','mutated'):
            row=copy.deepcopy(self.row);o=row['observer']
            if kind=='missing':o['generated'].pop()
            elif kind=='duplicate':o['generated'][1]['input_row']=0
            elif kind=='length':o['generated'][1]['true_response_length']=501
            elif kind=='bool_row':o['generated'][0]['input_row']=False
            elif kind=='hooks':o['hooks_restored']=False
            elif kind=='error':o['errors']=['actual metadata failure']
            else:o['observer_replaced_native_outputs']=True
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_anchor_early_stop_absence_is_explicitly_valid(self):
        row=copy.deepcopy(self.row);anchor=row['observer']['anchors'][1]
        anchor.update(status='absent_native_early_finish',true_response_length=10,finish_reason='stop')
        row['anchor_observations'][1]=dict(anchor_id=anchor['id'],status='absent_native_early_finish',
            input_row=anchor['input_row'],response_offset=anchor['response_offset'],probability_accepted=False,
            true_response_length=10,finish_reason='stop')
        row['observer']['generated'][160]['true_response_length']=10
        row['actual_new_emitted_tokens']-=10
        self.assertIs(package.validate_window_ready(row,4,self.bundle,'d'*64),row)

    def test_missing_duplicate_or_unexplained_anchor_refused(self):
        for kind in ('missing','duplicate','geometry','unexplained','not_early'):
            row=copy.deepcopy(self.row);a=row['observer']['anchors']
            if kind=='missing':a.pop()
            elif kind=='duplicate':a[1]=copy.deepcopy(a[0])
            elif kind=='geometry':a[1]['input_row']=159
            elif kind=='unexplained':a[1]['status']='missing'
            else:a[1].update(status='absent_native_early_finish',true_response_length=18,finish_reason='stop')
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_incomplete_or_mismatched_full_vocabulary_pair_refused(self):
        for kind in ('missing','causal','position','target','shape','count','accepted'):
            row=copy.deepcopy(self.row);p=row['anchor_observations'][0]
            if kind=='missing':row['anchor_observations'].pop()
            elif kind=='causal':p['actor_context_token_ids']=[2,1]
            elif kind=='position':p['actor_context_positions']=[0,2]
            elif kind=='target':p['actor_target_token_id']=1
            elif kind=='shape':p['tensor_identities']['actor']['shape']=[3]
            elif kind=='count':p['full_vocab_actor_minus_vllm_head']['count']=3
            else:p['probability_accepted']=True
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_incomplete_live_weight_provenance_refused(self):
        for branch,key,value in [('pre','serialized_source_keys',434),('pre','serialized_target_keys',254),
                                 ('pre','actual_device_is_cuda',False),('post','same_window_tie',False),
                                 ('post','CPU_offload_snapshot_used',True)]:
            row=copy.deepcopy(self.row);row['live_window_weights'][branch][key]=value
            with self.assertRaises(ValueError):package.validate_window_ready(row,4,self.bundle,'d'*64)

    def test_all_eight_same_fd_ready_and_evidence_then_missing_or_tampered_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()
            for rank in range(8):
                folder=output/('rank-%02d'%rank);folder.mkdir();evidence=folder/'window.pt'
                data=b'SYNTHETIC_BYTES_NOT_A_REAL_PT';evidence.write_bytes(data)
                pin=dict(path=str(evidence),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
                package.atomic_json(folder/'ready.json',synthetic_ready(rank,self.bundle,pin))
            self.assertEqual(len(package.rank_records(output,self.bundle,'d'*64)),8)
            (output/'rank-07/window.pt').write_bytes(b'TAMPER')
            with self.assertRaises(ValueError):package.rank_records(output,self.bundle,'d'*64)
            (output/'rank-07/ready.json').unlink()
            with self.assertRaises(FileNotFoundError):package.rank_records(output,self.bundle,'d'*64)


if __name__=='__main__':unittest.main(verbosity=2)
