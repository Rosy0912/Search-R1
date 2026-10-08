"""Independent bounded synthetic CPU checks of exactly the sealed helper."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from types import SimpleNamespace

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
OUT = ROOT / 'work/tree-fresh200-utf8-cache-none-review'
SOURCE = ROOT / 'work/tree-fresh200-utf8-cache-source/search_r1/baselines/tree_raw_tokens.py'
REVIEW = OUT / 'INDEPENDENT_UTF8_CACHE_SOURCE_REVIEW.json'
SOURCE_SHA = '5ebffa13473b111f5b73992679370d1c90603f2f9da5bdfee52db8b5736750c6'
REVIEW_SHA = 'b31c86c1b81258200580b0a2bb0346c24aba9e4ba53332e4f732bbfe0b1efdbf'
COMMIT = 'a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(SOURCE.read_bytes()) == SOURCE_SHA
assert sha(REVIEW.read_bytes()) == REVIEW_SHA
assert subprocess.check_output(['git', '-C', str(SOURCE.parents[2]), 'rev-parse', 'HEAD']).decode().strip() == COMMIT
review = json.loads(REVIEW.read_text())
assert review['blocker_count'] == 0 and review['source_review_accepted'] is True

calls = Counter()
def profile(frame, event, arg):
    if event == 'call' and frame.f_code.co_filename == str(SOURCE):
        calls[frame.f_code.co_name] += 1
sys.setprofile(profile)
spec = importlib.util.spec_from_file_location('independent_sealed_utf8_cache_helper', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Native synthetic decode consumes original fixture bytes independently of the
# helper's alphabet inversion; none of these tokens are official/model assets.
visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
byte_spelling = {b: chr(b) for b in visible}
for i, b in enumerate(b for b in range(256) if b not in byte_spelling):
    byte_spelling[b] = chr(256 + i)
DECODER = dict(type='ByteLevel', add_prefix_space=False, trim_offsets=False, use_regex=False)

def old_normalized_sha(serialized):
    state = json.loads(serialized)
    state.pop('padding', None)
    state.pop('truncation', None)
    return sha(json.dumps(state, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())

class Backend:
    def __init__(self, tok):
        self.tok = tok
        self.calls = 0
        self.override = None
        self.pretty = False
        self.decoder = SimpleNamespace(__getstate__=lambda: json.dumps(DECODER).encode())
    def to_str(self):
        self.calls += 1
        if self.override is not None:
            return self.override
        return json.dumps(self.tok.state, ensure_ascii=False, sort_keys=self.pretty, indent=2 if self.pretty else None)

class Tokenizer:
    pad_token_id = 151643
    eos_token_id = 151645
    bos_token_id = None
    clean_up_tokenization_spaces = False
    split_special_tokens = False
    add_prefix_space = False
    do_lower_case = False
    bos_token = None
    eos_token = '<eos>'
    pad_token = '<pad>'
    unk_token = None
    def __init__(self, directory, pieces):
        self.name_or_path = str(directory)
        self.pieces = dict(pieces)
        self.all_special_ids = [151643, 151645, 152000]
        self.additional_special_tokens = ['<special>']
        self.all_special_tokens = ['<pad>', '<eos>', '<special>']
        self.native_calls = []
        self.mismatch = None
        self.state = dict(version='synthetic-only', model=dict(type='SyntheticByteModel', pieces={str(k): v.hex() for k, v in self.pieces.items()}), normalizer=None, pre_tokenizer=None, post_processor=None, decoder=copy.deepcopy(DECODER), added_tokens=[dict(id=i, special=True) for i in self.all_special_ids], padding=None, truncation=None)
        self.backend_tokenizer = Backend(self)
    def token_bytes(self, token):
        return bytes([token - 1000]) if 1000 <= token < 1256 else self.pieces[token]
    def convert_ids_to_tokens(self, token):
        return ''.join(byte_spelling[b] for b in self.token_bytes(token))
    def decode(self, ids, skip_special_tokens=False):
        ids = list(ids)
        self.native_calls.append((ids, skip_special_tokens))
        if skip_special_tokens and self.mismatch == ids:
            return 'deliberate synthetic mismatch'
        raw = bytearray()
        for token in ids:
            if token in self.all_special_ids:
                if not skip_special_tokens:
                    raw.extend(('<eos>' if token == self.eos_token_id else '<special>').encode())
            else:
                raw.extend(self.token_bytes(token))
        return bytes(raw).decode('utf-8', errors='replace')
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1000 + b for b in text.encode('utf-8')]

results = []
def case(name, category, fn):
    start = time.perf_counter()
    try:
        evidence = fn() or {}
        results.append(dict(id=name, category=category, passed=True, elapsed_seconds=time.perf_counter()-start, evidence=evidence))
    except Exception as exc:
        results.append(dict(id=name, category=category, passed=False, elapsed_seconds=time.perf_counter()-start, error=repr(exc)))

def refused(fn, expected):
    try:
        fn()
    except ValueError as exc:
        assert expected in str(exc), str(exc)
        return str(exc)
    raise AssertionError('Expected ValueError refusal did not occur')

with tempfile.TemporaryDirectory(prefix='synthetic-utf8-cache-', dir=OUT) as tmp:
    directory = Path(tmp)
    (directory / 'tokenizer.json').write_text('{"independent_synthetic_fixture":true}\n')
    file_sha = sha((directory / 'tokenizer.json').read_bytes())
    def make(pieces):
        tok = Tokenizer(directory, dict(enumerate(pieces, 200)))
        identity = m.tokenizer_identity(tok, file_sha)
        return tok, identity
    def action(tok, identity):
        ids = list(tok.pieces) + [tok.eos_token_id]
        values = [-.25 * (i + 1) for i in range(len(ids))]
        sampling = dict(call_id='independent-synthetic', rank=0, request_id='fixture', physical_row=0)
        emission = dict(protocol=m.PROTOCOL, schema_version=1, raw_emitted_token_ids=ids, true_emitted_length=len(ids), raw_selected_logprobs=values, sampling_identity=sampling, sample_id=m.fingerprint(sampling))
        native_text = b''.join(tok.pieces.values()).decode('utf-8', errors='replace')
        legacy_text = native_text.split('</search>')[0]+'</search>' if '</search>' in native_text else native_text.split('</answer>')[0]+'</answer>' if '</answer>' in native_text else native_text
        before_ids = copy.deepcopy(ids); before_values = copy.deepcopy(values)
        value = m.sampled_action(tok, emission, ids+[tok.pad_token_id]*(500-len(ids)), tok.encode(legacy_text), legacy_text, identity)
        assert ids == before_ids and values == before_values
        return value, before_ids, before_values

    partitions = [(b'\xef\xbf\xbd',), (b'\xef', b'\xbf\xbd'), (b'\xef\xbf', b'\xbd'), (b'\xef', b'\xbf', b'\xbd')]
    for i, pieces in enumerate(partitions):
        def literal(pieces=pieces):
            tok, identity = make(pieces)
            (retained, ledger), ids, scalars = action(tok, identity)
            assert ledger['canonical_text'] == '\ufffd' and retained == ids and ledger['selected_logprobs'] == scalars
            proof = ledger['utf8_boundary_proof']
            assert proof['original_emitted_byte_sha256'] == sha(b'\xef\xbf\xbd')
            assert proof['native_prefix_agreement_count'] == len(ids) and proof['final_utf8_complete'] is True
            assert [seq for seq, skip in tok.native_calls if skip] == [ids[:n] for n in range(1, len(ids)+1)]
            m.validate_action_utf8_proof(ledger, tok)
            return dict(partition_lengths=list(map(len, pieces)), native_prefixes_verified=len(ids), pending_prefix_count=len(proof['pending_prefix_events']), retained_original_tokens=True, retained_original_scalars=True)
        case('literal_efbfbd_partition_'+str(i+1), 'utf8', literal)

    def tag_boundary():
        pieces=(b'<answer>X</answer>\xef', b'\xbf', b'\xbd', b'tail')
        tok, identity=make(pieces)
        (retained, ledger), ids, scalars=action(tok, identity)
        assert retained == ids[:3] and ledger['selected_logprobs'] == scalars[:3]
        assert ledger['canonical_text'] == '<answer>X</answer>' and ledger['indivisible_extra_text'] == '\ufffd'
        assert ledger['cut'] == 3 and ledger['utf8_boundary_proof']['complete_original_token_boundaries'] == [3,4,5]
        assert ledger['utf8_boundary_proof']['original_emitted_byte_sha256'] == sha(b''.join(pieces))
        m.validate_action_utf8_proof(ledger, tok)
        return dict(cut=3, native_prefixes_verified=5, literal_extra_kept_in_original_token_prefix=True)
    case('literal_after_tag_waits_for_whole_complete_boundary', 'utf8', tag_boundary)

    illegal=[b'\x80', b'\xefX', b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xef\xbf\xbd\x80']
    for i, data in enumerate(illegal):
        def bad(data=data):
            tok, identity=make((data,))
            error=refused(lambda:action(tok, identity),'Illegal original UTF8 byte sequence')
            return dict(synthetic_original_bytes_hex=data.hex(), refusal=error)
        case('strict_invalid_original_bytes_'+str(i+1), 'utf8', bad)
    for i,data in enumerate((b'\xef\xbf', b'<answer>X</answer>\xef\xbf')):
        def incomplete(data=data):
            tok, identity=make((data,))
            return dict(refusal=refused(lambda:action(tok, identity),'Final original UTF8 sequence is incomplete'))
        case('strict_final_incomplete_'+str(i+1), 'utf8', incomplete)
    def native_mismatch():
        tok, identity=make((b'\xef\xbf\xbd',));tok.mismatch=[200]
        return dict(refusal=refused(lambda:action(tok, identity),'Native prefix decode differs'))
    case('legal_literal_does_not_bypass_native_prefix_match', 'utf8', native_mismatch)

    def exact_hit():
        tok,_=make((b'A',));backend=tok.backend_tokenizer
        m._ENCODER_SEMANTICS_CACHE=None
        parses=[];fingerprints=[]
        original_loads=m.json.loads; original_fingerprint=m.fingerprint
        def counted_loads(value,*args,**kwargs):
            parses.append(True);return original_loads(value,*args,**kwargs)
        def counted_fingerprint(value):
            fingerprints.append(True);return original_fingerprint(value)
        expected=old_normalized_sha(backend.to_str());start_calls=backend.calls
        m.json.loads=counted_loads;m.fingerprint=counted_fingerprint
        try:
            first=m._encoder_binding(tok);entry=m._ENCODER_SEMANTICS_CACHE
            second=m._encoder_binding(tok)
        finally:
            m.json.loads=original_loads;m.fingerprint=original_fingerprint
        assert first == second and first['backend_semantics_sha256'] == expected
        assert backend.calls-start_calls == 2 and len(parses) == len(fingerprints) == 1
        assert m._ENCODER_SEMANTICS_CACHE is entry and type(entry) is tuple and len(entry)==2 and all(type(x) is str for x in entry)
        return dict(binding_calls=2,live_full_to_str_calls=2,json_loads_calls=1,normalized_fingerprint_calls=1,immutable_tuple=True)
    case('exact_hit_calls_live_serializer_and_reuses_only_sha', 'cache', exact_hit)

    def ignored_options():
        tok,identity=make((b'A',));old_entry=m._ENCODER_SEMANTICS_CACHE
        tok.state['padding']=dict(strategy='BatchLongest');tok.state['truncation']=dict(max_length=7)
        value=m._encoder_binding(tok)
        assert value['backend_semantics_sha256'] == identity['encoder_binding']['backend_semantics_sha256'] == old_normalized_sha(tok.backend_tokenizer.to_str())
        assert m._ENCODER_SEMANTICS_CACHE is not old_entry
        m._TokenizerValidationContext(tok,identity)
        return dict(full_string_changed=True,normalization_matches_previous_rule=True,old_identity_preserved=True)
    case('padding_truncation_changed_string_recomputes_same_semantics', 'cache', ignored_options)
    def layout_change():
        tok,identity=make((b'A',));old_entry=m._ENCODER_SEMANTICS_CACHE
        tok.backend_tokenizer.pretty=True
        value=m._encoder_binding(tok)
        assert m._ENCODER_SEMANTICS_CACHE is not old_entry
        assert value['backend_semantics_sha256']==identity['encoder_binding']['backend_semantics_sha256']==old_normalized_sha(tok.backend_tokenizer.to_str())
        return dict(full_json_layout_changed=True,normalized_sha_unchanged=True)
    case('full_json_layout_miss_keeps_original_normalized_sha', 'cache', layout_change)
    def semantic_change():
        tok,identity=make((b'A',));old_sha=identity['encoder_binding']['backend_semantics_sha256']
        tok.state['normalizer']=dict(type='SyntheticChangedNormalizer')
        value=m._encoder_binding(tok)
        assert value['backend_semantics_sha256'] != old_sha
        assert value['backend_semantics_sha256']==old_normalized_sha(tok.backend_tokenizer.to_str())
        return dict(refusal=refused(lambda:m._TokenizerValidationContext(tok,identity),'identity drifted'), semantic_sha_changed=True)
    case('encoder_semantic_bytes_changed_miss_and_old_identity_refused', 'cache', semantic_change)

    bad_states=[('malformed_json','{broken'),('non_dict','[]'),('missing_key',json.dumps(dict(model={},normalizer=None,pre_tokenizer=None,post_processor=None,decoder=DECODER))),('nan',json.dumps(dict(model=dict(bad=float('nan')),normalizer=None,pre_tokenizer=None,post_processor=None,decoder=DECODER,added_tokens=[])))]
    for name, serialized in bad_states:
        def malformed(serialized=serialized):
            tok,_=make((b'A',));entry=m._ENCODER_SEMANTICS_CACHE
            tok.backend_tokenizer.override=serialized
            try:m._encoder_binding(tok)
            except ValueError:pass
            else:raise AssertionError('Invalid encoder state accepted')
            assert m._ENCODER_SEMANTICS_CACHE is entry
            return dict(refused=True,good_cache_tuple_not_replaced=True)
        case('failed_state_'+name+'_cannot_publish', 'cache', malformed)

    class StrSubclass(str):pass
    for name, wrap in [('bytes',lambda s:s.encode()),('str_subclass',StrSubclass)]:
        def nonstr(wrap=wrap):
            tok,_=make((b'A',));entry=m._ENCODER_SEMANTICS_CACHE
            serialized=entry[0];expected=old_normalized_sha(serialized)
            tok.backend_tokenizer.override=wrap(serialized)
            original_loads=m.json.loads;parses=[]
            def counted(value,*args,**kwargs):parses.append(True);return original_loads(value,*args,**kwargs)
            m.json.loads=counted
            try:value=m._encoder_binding(tok)
            finally:m.json.loads=original_loads
            assert len(parses)==1 and m._ENCODER_SEMANTICS_CACHE is entry and value['backend_semantics_sha256']==expected
            return dict(parsed_again=True,cache_not_published=True,old_normalized_sha_matches=True)
        case('non_builtin_str_'+name+'_never_hits_or_publishes', 'cache', nonstr)

    for name, mutate in [('scalar',lambda t:setattr(t,'split_special_tokens',True)),('list_replacement',lambda t:setattr(t,'additional_special_tokens',['<changed>']))]:
        def flag(mutate=mutate):
            tok,identity=make((b'A',));entry=m._ENCODER_SEMANTICS_CACHE
            mutate(tok)
            error=refused(lambda:m._TokenizerValidationContext(tok,identity),'identity drifted')
            assert m._ENCODER_SEMANTICS_CACHE is entry
            return dict(exact_serialized_string_hit=True,refusal=error)
        case('live_hf_flag_'+name+'_mutation_detected_on_hit', 'cache', flag)

    def concurrent_snapshot():
        tok_a,_=make((b'A',));entry_a=m._ENCODER_SEMANTICS_CACHE
        tok_b,_=make((b'B',));entry_b=m._ENCODER_SEMANTICS_CACHE
        assert entry_a[0]!=entry_b[0] and entry_a[1]!=entry_b[1]
        m._ENCODER_SEMANTICS_CACHE=entry_a
        ready=threading.Event();done=threading.Event();traced=[]
        def writer():
            if ready.wait(2):
                m._ENCODER_SEMANTICS_CACHE=entry_b
                done.set()
        thread=threading.Thread(target=writer,daemon=True);thread.start()
        def trace(frame,event,arg):
            if frame.f_code is m._encoder_binding.__code__ and event=='line' and frame.f_lineno==152:
                assert frame.f_locals['cached'] is entry_a
                ready.set();assert done.wait(2),'controlled cache writer timeout'
                traced.append(True)
            return trace
        sys.settrace(trace)
        try:value=m._encoder_binding(tok_a)
        finally:sys.settrace(None);thread.join(2)
        assert traced==[True] and not thread.is_alive()
        assert m._ENCODER_SEMANTICS_CACHE is entry_b and value['backend_semantics_sha256']==entry_a[1]
        return dict(replacement_after_local_snapshot=True,returned_sha_matches_local_serialized_state=True,global_now_contains_other_successfully_validated_tuple=True)
    case('controlled_concurrent_tuple_replacement_keeps_matching_pair', 'cache', concurrent_snapshot)

sys.setprofile(None)
assert sha(SOURCE.read_bytes())==SOURCE_SHA and sha(REVIEW.read_bytes())==REVIEW_SHA
failed=[c for c in results if not c['passed']]
report=dict(schema_version=1,status='BOUNDED_LOCAL_CPU_PASS' if not failed else 'BOUNDED_LOCAL_CPU_FAILED',source=dict(path=str(SOURCE),commit=COMMIT,sha256=SOURCE_SHA),independent_source_review=dict(path=str(REVIEW),sha256=REVIEW_SHA),fixture=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes()),synthetic_only=True,temporary_synthetic_tokenizer_file_cleaned=True),execution=dict(actual_frozen_module_imported=True,actual_frozen_functions_executed=True,cases_executed=len(results),cases_passed=len(results)-len(failed),cases_failed=len(failed),cases_skipped=0,cases_by_category=dict(Counter(c['category'] for c in results)),actual_source_function_calls=dict(calls),old_full_suite_executed=False,official_tokenizer_or_model_or_dataset_assets_loaded=False,raw_historical_assets_loaded=False,optimizer_or_adam_or_ray_or_gpu_or_remote_or_slurm_executed=False,source_or_core_or_repository_test_files_modified=False),cases=results,acceptance=dict(bounded_synthetic_cpu=not failed,official_tokenizer_runtime=False,actual_3317_500_token_replay=False,full_tree_mu=False,ready=False,native24=False,full2560_mu=False,full_zero=False,formal_first5=False,training_release=False),historical_stale_test_note='Existing test_tree_raw_tokens_cpu.py:262 legal EF BF BD negative expectation was not executed or edited.',next_scope='Standalone actual-h100 row96/500 replay producer must pass source-only review before any remote execution.')
result=OUT/'INDEPENDENT_UTF8_CACHE_BOUNDED_CPU_RESULT.json'
result.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],cases_executed=len(results),cases_passed=len(results)-len(failed),cases_failed=len(failed),cases_by_category=report['execution']['cases_by_category'],result_path=str(result),result_sha256=sha(result.read_bytes()),fixture_sha256=report['fixture']['sha256'],failed_cases=failed),ensure_ascii=False,indent=2))
if failed:sys.exit(1)
