"""Real V2 validator calls on tiny fixed vocabularies; no model/assets/Ray/GPU."""
import ast
import copy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import search_r1.baselines.tree_raw_tokens as raw
import test_tree_raw_tokens_cpu as raw_fixture
from test_tree_raw_tokens_cpu import Tokenizer


def rehash(action):
    action['action_binding_sha256']=raw.fingerprint({k:v for k,v in action.items()
        if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})


def worker_registration_statement():
    tree=ast.parse(Path('verl/workers/fsdp_workers.py').read_text())
    nodes=[node for node in ast.walk(tree) if isinstance(node,ast.If) and any(
        isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and
        child.func.id=='register_validation_tokenizer' for child in ast.walk(node))]
    # Keep only the small opt-in statement, not an enclosing model function.
    node=min(nodes,key=lambda x:len(list(ast.walk(x))))
    return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        'actual-fsdp-tokenizer-registration','exec')


def registry_child(path,pieces,state_payload,identity,queue):
    try:
        assert not raw._VALIDATION_TOKENIZERS  # spawn has a fresh Python process
        state=NS(**state_payload)
        try:raw.trajectory_projection(state,4096)
        except ValueError as error:missing=str(error)
        else:raise AssertionError('missing process registration accepted')
        tok=Tokenizer(path);tok.parts=pieces
        config=NS(actor=dict(baseline_method='tree_grpo',policy_safety=dict(
            sampling_probability_correction=dict(enabled=True))))
        worker=NS(config=config,tokenizer=tok)
        statement=worker_registration_statement();exec(statement,dict(self=worker))
        assert worker._tree_raw_validation_tokenizer_identity==identity
        projection=raw.trajectory_projection(state,4096)
        second=Tokenizer(path);second.parts=copy.deepcopy(pieces)
        worker.tokenizer=second;exec(statement,dict(self=worker))
        assert raw._validation_tokenizer(identity) is tok
        assert raw.trajectory_projection(state,4096)==projection
        queue.put(dict(ok=True,missing_registry_refused=missing,actual_worker_statement=True,
            duplicate_keeps_first=True,semantic_ids=projection['semantic_response_token_ids']))
    except Exception as error:queue.put(dict(ok=False,error=repr(error)))


class Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=raw_fixture.RawTokenTests();self.fixture.setUp()
        self.fixture.tok.parts.update({200:b'<answer>\xe4',201:b'\xb8\xad</answer>tail'})
        self.state=self.fixture.state([200,201,151645])
        self.action=self.state.raw_ledger[0]['action']
    def tearDown(self):self.fixture.tearDown()

    def test_resealed_single_double_substring_fakeanswer_semantics_refuse_every_boundary(self):
        actual=self.action['canonical_semantic_token_ids']
        for candidate in ('<answer>WRONG</answer>','中','<answer>中</answer>tail','<search>fake</search>'):
            for fields in ('ids','reward','both'):
                changed=copy.deepcopy(self.action)
                if fields in ('ids','both'):changed['canonical_semantic_token_ids']=self.fixture.tok.encode(candidate)
                if fields in ('reward','both'):changed['legacy_reward_action_text']=candidate
                rehash(changed)
                state=copy.deepcopy(self.state);state.raw_ledger[0]['action']=changed
                with self.subTest(candidate=candidate,fields=fields):
                    with self.assertRaises(ValueError):raw.validate_action_utf8_proof(changed,self.fixture.tok)
                    with self.assertRaises(ValueError):raw.trajectory_projection(state,4096)
                    # Build all projection hashes manually after the real helper
                    # refused; this also probes the actual reward reader.
                    item=self.fixture.item(self.state)
                    p=item.non_tensor_batch[raw.FIELD];p['raw_ledger'][0]['action']=changed
                    p['semantic_response_token_ids']=changed['canonical_semantic_token_ids'].copy()
                    p['semantic_model_mask']=[1]*len(p['semantic_response_token_ids'])
                    p['binding_sha256']=raw.fingerprint({k:v for k,v in p.items() if k!='binding_sha256'})
                    with self.assertRaises(ValueError):raw.semantic_spans(item,self.fixture.tok)
        self.assertEqual(self.action['canonical_semantic_token_ids'],actual)
        self.assertEqual(raw.semantic_spans(self.fixture.item(self.state),self.fixture.tok),['<answer>中</answer>'])

    def test_resealed_canonical_text_cannot_replace_actual_raw_text(self):
        changed=copy.deepcopy(self.action);fake='<answer>WRONG</answer>'
        changed.update(canonical_text=fake,canonical_text_sha256=raw.fingerprint(fake),
            canonical_semantic_token_ids=self.fixture.tok.encode(fake),legacy_reward_action_text=fake)
        rehash(changed)
        with self.assertRaises(ValueError):raw.validate_action_utf8_proof(changed,self.fixture.tok)

    def test_path_file_backend_and_default_flag_drift_refuse_registered_authority(self):
        tok=self.fixture.tok
        cases=(('path',None),('file',None),('vocab',None),('normalizer',None),('flags',None),('added_tokens',None))
        for label,_ in cases:
            with self.subTest(label=label):
                if label=='path':
                    other=Path(tok.name_or_path)/'other';other.mkdir(exist_ok=True)
                    (other/'tokenizer.json').write_bytes((Path(tok.name_or_path)/'tokenizer.json').read_bytes())
                    with patch.object(tok,'name_or_path',str(other)),self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
                elif label=='file':
                    path=Path(tok.name_or_path)/'tokenizer.json';original=path.read_bytes();path.write_bytes(b'{"changed":true}')
                    try:
                        with self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
                    finally:path.write_bytes(original)
                elif label=='vocab':
                    with patch.dict(tok.parts,{2:'WRONG'}),self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
                elif label=='normalizer':
                    with patch.object(tok,'backend_normalizer',dict(type='Lowercase')),self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
                elif label=='flags':
                    with patch.object(tok,'split_special_tokens',True,create=True),self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
                else:
                    with patch.object(tok,'all_special_ids',[151643,151645]),self.assertRaises(ValueError):raw.trajectory_projection(self.state,4096)
        self.assertEqual(raw.semantic_spans(self.fixture.item(self.state),tok),['<answer>中</answer>'])

    def test_normal_padding_truncation_changes_are_allowed_without_vocabulary_drift(self):
        tok=self.fixture.tok;before=raw.trajectory_projection(self.state,4096)
        tok.backend_padding=dict(strategy='BatchLongest',pad_id=151643)
        tok.backend_truncation=dict(max_length=500,strategy='LongestFirst')
        self.assertEqual(raw.trajectory_projection(self.state,4096),before)
        self.assertEqual(raw.semantic_spans(self.fixture.item(self.state),tok),['<answer>中</answer>'])

    def test_actual_worker_registers_already_loaded_tokenizer_in_independent_spawn_process(self):
        context=multiprocessing.get_context('spawn');queue=context.Queue()
        state=dict(prompt=self.state.prompt,response=self.state.response,model_mask=self.state.model_mask,
                   raw_ledger=self.state.raw_ledger)
        child=context.Process(target=registry_child,args=(self.fixture.tok.name_or_path,
            self.fixture.tok.parts,state,self.action['tokenizer_identity'],queue))
        child.start();child.join(30)
        if child.is_alive():child.terminate();child.join();self.fail('CPU registry child timed out')
        self.assertEqual(child.exitcode,0);result=queue.get(timeout=2)
        self.assertTrue(result['ok'],result);self.assertTrue(result['actual_worker_statement'])

    def test_inherited_two_turn_chain_is_valid_and_resealed_ancestor_change_refuses(self):
        first=copy.deepcopy(self.state);first.response+=[10];first.model_mask+=[0]
        first.raw_ledger[0]['observation_token_ids']=[10]
        first.raw_ledger[0]['observation_text']='X'
        second,ledger=self.fixture.action([1,2,3,151645])
        context=first.prompt+first.response
        ledger['native_emission']['prompt_token_ids']=context.copy();rehash(ledger)
        first.raw_ledger.append(dict(turn=1,action=ledger,observation_token_ids=[],observation_text='',
            generation_context_token_ids=context.copy(),generation_context_positions=list(range(len(context))),
            generation_context_attention=[1]*len(context)))
        first.response+=second;first.model_mask+=[1]*len(second)
        self.assertEqual(raw.semantic_spans(self.fixture.item(first),self.fixture.tok),
                         ['<answer>中</answer>','<answer>Paris</answer>'])
        child=first.branch();ancestor=child.raw_ledger[0]['action']
        ancestor['canonical_semantic_token_ids']=self.fixture.tok.encode('<answer>WRONG</answer>')
        ancestor['legacy_reward_action_text']='<answer>WRONG</answer>';rehash(ancestor)
        with self.assertRaises(ValueError):raw.trajectory_projection(child,4096)
        self.assertEqual(raw.semantic_spans(self.fixture.item(first),self.fixture.tok),
                         ['<answer>中</answer>','<answer>Paris</answer>'])

    def test_full_backend_serialization_once_per_identity_within_one_trajectory(self):
        state=copy.deepcopy(self.state);action=copy.deepcopy(self.action)
        context=state.prompt+state.response;action['native_emission']['prompt_token_ids']=context;rehash(action)
        state.raw_ledger.append(dict(turn=1,action=action,observation_token_ids=[],observation_text='',
            generation_context_token_ids=context,generation_context_positions=list(range(len(context))),
            generation_context_attention=[1]*len(context)))
        state.response+=action['action_token_ids'];state.model_mask+=[1]*action['cut']
        actual=self.fixture.tok.backend_tokenizer.to_str
        with patch.object(self.fixture.tok.backend_tokenizer,'to_str',wraps=actual) as snapshot:
            raw.trajectory_projection(state,4096);self.assertEqual(snapshot.call_count,1)
        self.assertGreaterEqual(self.action['prefix_scan_tokenizer_validation_cpu_seconds'],0.)


if __name__=='__main__':unittest.main()
