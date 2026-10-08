"""Independent SOURCE/CPU probe. Uses only synthetic bytes and synthetic tokenizer."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'work/tree-behavior-mu-utf8-v2'
RAW = SOURCE / 'search_r1/baselines/tree_raw_tokens.py'
spec = importlib.util.spec_from_file_location('independent_raw', RAW)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)

# Independent construction of the known reversible ByteLevel alphabet.
visible_bytes = [b for b in range(256) if 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255]
other_bytes = [b for b in range(256) if b not in visible_bytes]
alphabet = {b: chr(b) for b in visible_bytes}
alphabet.update({b: chr(256 + i) for i, b in enumerate(other_bytes)})

class SyntheticTokenizer:
    pad_token_id = 151643
    eos_token_id = 151645
    bos_token_id = None
    clean_up_tokenization_spaces = False
    all_special_ids = [151643, 151645]
    def __init__(self, path):
        self.name_or_path = str(path)
        self.backend_tokenizer = NS(decoder=NS(__getstate__=lambda: b'{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}'))
        self.pieces = {1: b'<answer>\xe4', 2: b'\xb8\xad</answer>tail',
                       42: b'prompt', 300: '<answer>中</answer>'.encode(),
                       301: b'<answer>WRONG</answer>'}
        # V2 requires the actual full backend state, so the tiny fixed fixture
        # explicitly exposes its complete declared vocabulary/configuration.
        self.backend_tokenizer.to_str = lambda: json.dumps(dict(model=dict(type='SyntheticFixed', pieces={str(k):v.hex() for k,v in self.pieces.items()}), normalizer=None, pre_tokenizer=None, post_processor=None, decoder={'type':'ByteLevel','add_prefix_space':False,'trim_offsets':False,'use_regex':False}, added_tokens=[dict(id=x,special=True) for x in self.all_special_ids], padding=None, truncation=None))
    def convert_ids_to_tokens(self, token):
        return ''.join(alphabet[b] for b in self.pieces[token])
    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for token in ids:
            if token in self.all_special_ids: pieces.append(b'' if skip_special_tokens else b'<SPECIAL>')
            else: pieces.append(self.pieces[token])
        return b''.join(pieces).decode('utf8', errors='replace')
    def encode(self, text, add_special_tokens=False):
        if text == '<answer>中</answer>': return [300]
        if text == '<answer>WRONG</answer>': return [301]
        raise ValueError('Synthetic encode covers only two explicit semantic candidates')

# Avoid optional dataproto dependencies; semantic_spans's actual CPU API uses these fields.
def make_item(action):
    state = NS(prompt=[42], response=action['action_token_ids'].copy(), model_mask=[1] * action['cut'])
    state.raw_ledger = [dict(turn=0, action=action, observation_token_ids=[], observation_text='',
        generation_context_token_ids=[42], generation_context_attention=[1], generation_context_positions=[0])]
    projection = raw.trajectory_projection(state, 4096)
    return NS(batch=dict(prompts=torch.tensor([42]), responses=torch.tensor(state.response),
        attention_mask=torch.ones(1 + len(state.response), dtype=torch.long),
        info_mask=torch.ones(1 + len(state.response), dtype=torch.long)),
        non_tensor_batch={raw.FIELD: projection}, meta_info={'tree_raw_token_protocol': raw.PROTOCOL})

def rehash_action(action):
    action['action_binding_sha256'] = raw.fingerprint({k: v for k, v in action.items()
        if not k.startswith('prefix_scan_') and k != 'action_binding_sha256'})

def outcome(call):
    try: return {'accepted': True, 'result': call()}
    except Exception as exc: return {'accepted': False, 'error_type': type(exc).__name__, 'error': str(exc)}

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    (path / 'tokenizer.json').write_text('{"strictly_synthetic":true}')
    tokenizer = SyntheticTokenizer(path)
    sha = hashlib.sha256((path / 'tokenizer.json').read_bytes()).hexdigest()
    identity = raw.tokenizer_identity(tokenizer, sha)
    emitted = [1, 2, 151645]
    emission = dict(protocol=raw.PROTOCOL, schema_version=1, raw_emitted_token_ids=emitted,
        raw_selected_logprobs=[-.25, -.5, -.75], true_emitted_length=3,
        sampling_identity={'synthetic_only': True}, prompt_token_ids=[42])
    emission['sample_id'] = raw.fingerprint(emission['sampling_identity'])
    _, action = raw.sampled_action(tokenizer, emission, emitted + [151643] * 497,
        [300], '<answer>中</answer>', identity)
    results = dict(actual_gpu_executed=False, actual_remote_executed=False,
        actual_official_assets_loaded=False, synthetic_tokenizer_json_sha256=sha,
        source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
        baseline=outcome(lambda: raw.semantic_spans(make_item(action), tokenizer)))
    corrupt_proof = copy.deepcopy(action)
    corrupt_proof['utf8_boundary_proof']['pending_prefix_events'][0]['pending_utf8_hex'] = 'ff'
    rehash_action(corrupt_proof)
    results['rehashed_pending_byte_mutation'] = outcome(lambda: raw.semantic_spans(make_item(corrupt_proof), tokenizer))
    corrupt_semantics = copy.deepcopy(action)
    corrupt_semantics['canonical_semantic_token_ids'] = [301]
    corrupt_semantics['legacy_reward_action_text'] = '<answer>WRONG</answer>'
    rehash_action(corrupt_semantics)
    results['rehashed_semantic_token_substitution'] = outcome(lambda: raw.semantic_spans(make_item(corrupt_semantics), tokenizer))
    results['substituted_action_canonical_text'] = corrupt_semantics['canonical_text']
    results['substituted_policy_token_ids'] = corrupt_semantics['action_token_ids']
    results['substituted_utf8_proof_equal_to_original'] = corrupt_semantics['utf8_boundary_proof'] == action['utf8_boundary_proof']
    # Genuine malformed bytes and incomplete final bytes must refuse creation.
    for label, pieces in [('invalid_bytes', {1:b'\xff', 2:b'X'}), ('terminal_pending_bytes', {1:b'A', 2:b'\xe4'})]:
        saved = tokenizer.pieces.copy();tokenizer.pieces.update(pieces)
        results[label] = outcome(lambda: raw.sampled_action(tokenizer, emission,
            emitted + [151643] * 497, [300], '<answer>中</answer>', identity))
        tokenizer.pieces = saved
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
