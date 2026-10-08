"""Streaming full checkpoint integrity checks; canonical/tied-head rules unchanged."""
import json
import torch
from safetensors import safe_open


def validate_saved_checkpoint(directory, canonical, run_started):
    """Match canonical serialized keys/shapes; allow reshards and stream finite checks."""
    index_path=directory/'model.safetensors.index.json'
    saved=json.loads(index_path.read_text())['weight_map']
    expected=json.loads((canonical/'model.safetensors.index.json').read_text())['weight_map']
    # This Qwen profile permits one FSDP/HF serialization redundancy, never
    # a missing canonical parameter or an arbitrary extra model parameter.
    alias='lm_head.weight'
    embedding='model.embed_tokens.weight'
    extra=set(saved)-set(expected)
    tied_alias=False
    if extra=={alias} and embedding in expected and alias not in expected:
        base_config_path=canonical/'config.json'
        saved_config_path=directory/'config.json'
        if base_config_path.is_file() and saved_config_path.is_file():
            configs=[json.loads(p.read_text()) for p in [base_config_path,saved_config_path]]
            tied_alias=all(c.get('model_type')=='qwen2' and c.get('tie_word_embeddings') is True
                           and c.get('architectures')==['Qwen2ForCausalLM'] for c in configs)
    permitted={alias} if tied_alias else set()
    assert expected and set(saved)-permitted==set(expected), 'Checkpoint keys differ from canonical model'
    paths={directory/f for f in saved.values()} | {index_path}
    assert all(p.is_file() and p.stat().st_size>0 and p.stat().st_mtime>=run_started for p in paths), 'Missing or stale checkpoint file'
    shapes={}
    for filename in set(expected.values()):
        with safe_open(str(canonical/filename),framework='pt',device='cpu') as reader:
            for key in (k for k,v in expected.items() if v==filename):
                shapes[key]=tuple(reader.get_slice(key).get_shape())
    if tied_alias:
        shapes[alias]=shapes[embedding]
    for filename in set(saved.values()):
        with safe_open(str(directory/filename),framework='pt',device='cpu') as reader:
            keys={k for k,v in saved.items() if v==filename}
            assert set(reader.keys())==keys, 'Incomplete checkpoint shard'
            for key in keys:
                assert tuple(reader.get_slice(key).get_shape())==shapes[key], 'Checkpoint shape differs: '+key
                tensor=reader.get_tensor(key)
                assert torch.isfinite(tensor).all(), 'Nonfinite checkpoint tensor: '+key
                del tensor
    if tied_alias:
        with safe_open(str(directory/saved[embedding]),framework='pt',device='cpu') as reader:
            input_weight=reader.get_tensor(embedding)
        with safe_open(str(directory/saved[alias]),framework='pt',device='cpu') as reader:
            output_weight=reader.get_tensor(alias)
        assert input_weight.dtype==output_weight.dtype and torch.equal(input_weight,output_weight), 'Inconsistent tied Qwen embedding/head'
        del input_weight,output_weight
    return len(saved)
