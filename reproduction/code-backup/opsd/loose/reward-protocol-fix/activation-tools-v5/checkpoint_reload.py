"""Read-only acceptance of complete real HF export, tokenizer identity and student change."""
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from safetensors import safe_open
out=Path(os.environ['OPSD_EVIDENCE']);checkpoint=Path(os.environ['OPSD_CHECKPOINT']);base=Path(os.environ['OPSD_TEACHER'])
start=time.monotonic()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
shapes_by_directory={}
def mapping(directory):
    index=directory/'model.safetensors.index.json'
    if index.exists(): declared=json.loads(index.read_text())['weight_map']
    else:
        files=list(directory.glob('*.safetensors'));assert len(files)==1
        with safe_open(str(files[0]),framework='pt',device='cpu') as f:declared={key:files[0].name for key in f.keys()}
    actual={}; shapes={}
    for filename in set(declared.values()):
        assert Path(filename).name==filename
        with safe_open(str(directory/filename),framework='pt',device='cpu') as f:
            for key in f.keys():
                assert key not in actual, 'Duplicate tensor in shards'
                actual[key]=filename
                shapes[key]=f.get_slice(key).get_shape()
    assert declared==actual, 'Shard actual tensors differ from index'
    shapes_by_directory[str(directory)]=shapes
    return declared
base_config=AutoConfig.from_pretrained(str(base),local_files_only=True)
saved_config=AutoConfig.from_pretrained(str(checkpoint),local_files_only=True)
# Serialization provenance and explicit attention precision are runtime choices.
# The existing trainer deliberately overrides model BOS/EOS/PAD from its tokenizer.
config_runtime_fields={'_name_or_path','_commit_hash','transformers_version','torch_dtype',
    '_attn_implementation_internal','_attn_implementation_autoset',
    'bos_token_id','eos_token_id','pad_token_id'}
base_semantics={k:v for k,v in base_config.to_dict().items() if k not in config_runtime_fields}
saved_semantics={k:v for k,v in saved_config.to_dict().items() if k not in config_runtime_fields}
assert base_semantics==saved_semantics, 'Model computation configuration changed'
tied=bool(base_config.tie_word_embeddings)
base_map=mapping(base);saved_map=mapping(checkpoint)
def canonical(keys): return set(keys)-({'lm_head.weight'} if tied else set())
assert canonical(base_map)==canonical(saved_map), 'Export lacks model tensors'
for key in canonical(base_map):
    assert shapes_by_directory[str(base)][key]==shapes_by_directory[str(checkpoint)][key], ('Tensor shape changed',key)
if tied:
    for directory,weights_map in ((base,base_map),(checkpoint,saved_map)):
        assert 'model.embed_tokens.weight' in weights_map
        if 'lm_head.weight' in weights_map:
            with safe_open(str(directory/weights_map['lm_head.weight']),framework='pt',device='cpu') as f:x=f.get_tensor('lm_head.weight')
            with safe_open(str(directory/weights_map['model.embed_tokens.weight']),framework='pt',device='cpu') as f:y=f.get_tensor('model.embed_tokens.weight')
            assert torch.equal(x,y), 'Declared tied aliases have different values'
a=AutoTokenizer.from_pretrained(str(base),local_files_only=True);b=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
assert a.get_vocab()==b.get_vocab() and (a.bos_token_id,a.eos_token_id,a.pad_token_id)==(b.bos_token_id,b.eos_token_id,b.pad_token_id)
for field in ('bos_token_id','eos_token_id','pad_token_id'):
    assert getattr(saved_config,field)==getattr(b,field), ('Saved model/tokenizer ID mismatch',field)
assert a.special_tokens_map==b.special_tokens_map and a.chat_template==b.chat_template
assert a.padding_side==b.padding_side and a.truncation_side==b.truncation_side
backend_a=json.loads(a.backend_tokenizer.to_str());backend_b=json.loads(b.backend_tokenizer.to_str())
# Last-call padding/truncation state is runtime state; structural token processing must match.
transient_backend={key:dict(base=backend_a.pop(key,None),checkpoint=backend_b.pop(key,None)) for key in ('padding','truncation')}
assert backend_a==backend_b, 'Tokenizer structural rules changed'
tokenizer_semantic_sha256=hashlib.sha256(json.dumps(backend_a,sort_keys=True).encode()).hexdigest()
changes={}
for key in ('model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.v_proj.weight','model.layers.0.mlp.down_proj.weight','model.embed_tokens.weight' if tied else 'lm_head.weight'):
    with safe_open(str(base/base_map[key]),framework='pt',device='cpu') as f:x=f.get_tensor(key).float()
    with safe_open(str(checkpoint/saved_map[key]),framework='pt',device='cpu') as f:y=f.get_tensor(key).float()
    assert x.shape==y.shape and torch.isfinite(y).all()
    delta=(x-y).abs();changes[key]=dict(changed_elements=int((delta!=0).sum()),max_abs_delta=float(delta.max()),mean_abs_delta=float(delta.mean()))
assert sum(v['changed_elements'] for v in changes.values())>0, 'No student parameter change after warmup'
weights={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in checkpoint.glob('*.safetensors')}
model,loading=AutoModelForCausalLM.from_pretrained(str(checkpoint),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',output_loading_info=True)
assert all(not loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), loading
assert canonical(model.state_dict())==canonical(saved_map), 'Loaded architecture does not contain all exported tensors'
if tied: assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
model=model.to('cuda')
assert model.config._attn_implementation=='flash_attention_2' and next(model.parameters()).dtype==torch.bfloat16
assert all(torch.isfinite(p).all().item() for p in model.parameters()), 'Reloaded tensor contains nonfinite values'
model.eval();inputs=b('The capital of France is',return_tensors='pt').to('cuda')
with torch.no_grad():
    logits=model(**inputs,use_cache=False).logits;assert torch.isfinite(logits).all()
    output=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=b.pad_token_id)
assert output.shape[1]>inputs.input_ids.shape[1]
result=dict(checkpoint=str(checkpoint),tensor_count=len(saved_map),config_computation_equal=True,config_runtime_fields=sorted(config_runtime_fields),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),tokenizer_equal=True,tokenizer_semantic_sha256=tokenizer_semantic_sha256,transient_backend=transient_backend,loading_info=loading,all_loaded_parameters_finite=True,tied_embeddings=tied,node=__import__('socket').gethostname(),gpu_name=torch.cuda.get_device_name(),student_change=changes,weights=weights,
    generation=b.decode(output[0]),seconds=time.monotonic()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),slurm_job_id=os.environ['SLURM_JOB_ID'])
(out/'checkpoint-reload.json').write_text(json.dumps(result,indent=2));print('REAL_3B_HF_CHECKPOINT_RELOAD_AND_STUDENT_CHANGE_OK',json.dumps(result),flush=True)
