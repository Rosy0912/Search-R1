"""Generic CPU mathematics/tokenizer readers only; no inherited job/run/source policy."""
import hashlib,json,math
from pathlib import Path

def require(value,message):
    if not value:raise ValueError(message)

def finite(value):return type(value)in(int,float)and math.isfinite(value)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def safetensor_map(root):
    from safetensors import safe_open
    found = {}
    for path in sorted(root.glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in found:
                    raise ValueError('Duplicate weight key: '+key)
                found[key] = path
    if not found:
        raise ValueError('No safetensors in '+str(root))
    return found

def resolve_qwen2_tied_alias(base, checkpoint, before, after):
    """Only admit the exact observed redundant Qwen2 tied-head serialization."""
    if set(before) == set(after):
        return {}
    if set(before) - set(after) or set(after) - set(before) != {'lm_head.weight'}:
        raise ValueError('Base/checkpoint tensor key sets differ')
    embed, head = 'model.embed_tokens.weight', 'lm_head.weight'
    if embed not in before or embed not in after or head in before:
        raise ValueError('Invalid Qwen2 tied alias pattern')
    configs = [json.loads((root / 'config.json').read_text()) for root in (base, checkpoint)]
    for config in configs:
        if (config.get('model_type') != 'qwen2' or config.get('tie_word_embeddings') is not True
                or config.get('architectures') != ['Qwen2ForCausalLM']):
            raise ValueError('Extra head requires verified tied Qwen2 configurations')
    shape = [configs[0]['vocab_size'], configs[0]['hidden_size']]
    if shape != [configs[1]['vocab_size'], configs[1]['hidden_size']]:
        raise ValueError('Tied embedding dimensions differ')
    import torch
    from safetensors import safe_open
    checked = 0
    with safe_open(str(after[embed]), framework='pt', device='cpu') as f, safe_open(str(after[head]), framework='pt', device='cpu') as g:
        if (f.get_slice(embed).get_shape() != shape or g.get_slice(head).get_shape() != shape
                or f.get_slice(embed).get_dtype() != g.get_slice(head).get_dtype()):
            raise ValueError('Serialized tied weights shape/dtype differ')
        for start in range(0, shape[0], 1024):
            a = f.get_slice(embed)[start:start+1024]
            b = g.get_slice(head)[start:start+1024]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a, b):
                raise ValueError('Serialized tied weights must be entirely finite and exactly equal')
            checked += a.numel()
    before[head] = before[embed]
    return dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED',
                base_logical_key=head, base_storage_key=embed,
                checkpoint_extra_key=head, checkpoint_equal_key=embed,
                shape=shape, elements_compared=checked, exact_equal=True)

def tokenizer_proof(left,right):
    lv,rv=left.get_vocab(),right.get_vocab()
    vh=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    require(lv==rv and left.bos_token_id==right.bos_token_id and left.eos_token_id==right.eos_token_id,'Official Instruct tokenizer differs')
    require(left.pad_token_id==right.pad_token_id and left.special_tokens_map==right.special_tokens_map,'Official pad/special tokens differ')
    require(bool(left.chat_template)and left.chat_template==right.chat_template,'Official Instruct chat_template differs')
    def added_metadata(tokenizer):
        return {str(key):dict(content=tok.content,single_word=tok.single_word,lstrip=tok.lstrip,rstrip=tok.rstrip,normalized=tok.normalized,special=tok.special)for key,tok in tokenizer.added_tokens_decoder.items()}
    left_added,right_added=added_metadata(left),added_metadata(right)
    require(left_added==right_added,'AddedToken normalization/special metadata differs')
    conversation=[{'role':'user','content':'What is the capital of France?'}]
    left_chat=left.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True)
    right_chat=right.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True)
    require(isinstance(left_chat,list)and left_chat and left_chat==right_chat,'Real Instruct chat-template tokenization differs')
    token=dict(full_vocabulary_equal=True,bos_equal=True,eos_equal=True,base_vocab_sha256=vh(lv),checkpoint_vocab_sha256=vh(rv),
        base_vocab_size=len(lv),checkpoint_vocab_size=len(rv),bos=left.bos_token_id,eos=left.eos_token_id,pad=left.pad_token_id,
        pad_equal=True,special_tokens_equal=True,special_tokens_map=left.special_tokens_map,added_token_metadata_equal=True,added_token_metadata_sha256=vh(left_added),
        chat_template_equal=True,chat_template_sha256=hashlib.sha256(json.dumps(left.chat_template,sort_keys=True).encode()).hexdigest(),
        real_chat_template_tokenization_equal=True,real_chat_token_ids=left_chat,
        scope='REAL_TOKENIZER_PAIR_EQUALITY_REQUIRES_OFFICIAL_BASE_BINDING')
    return token
