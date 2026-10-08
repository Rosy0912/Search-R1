from pathlib import Path
import json,hashlib
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env')
paths=sorted(root.glob('lib/python*/site-packages/vllm/engine/llm_engine.py'))
assert len(paths)==1,paths
site=paths[0].parents[1];files=['engine/llm_engine.py','transformers_utils/tokenizer_group/tokenizer_group.py','sampling_params.py','sequence.py','core/scheduler.py']
out={}
for name in files:
 p=site/name
 if not p.is_file():continue
 content=p.read_text();lines=content.splitlines();indexes=[i for i,l in enumerate(lines)if any(x in l for x in ['def get_tokenizer(', 'def get_lora_tokenizer(', 'def add_request(', 'def update_from_generation_config(', 'generation_config_fields','all_stop_token_ids','self.tokenizer =','eos_token_id=', 'self.waiting', 'def add_seq_group(', 'def get_seqs('])]
 selected=set()
 for i in indexes:selected.update(range(max(0,i-3),min(len(lines),i+48)))
 out[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'excerpts':'\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected))}
print(json.dumps(out))
