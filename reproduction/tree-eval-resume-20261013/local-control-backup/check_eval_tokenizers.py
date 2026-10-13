import importlib.util,shlex
from pathlib import Path
s=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
code='''import json,hashlib
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-branpo-fresh200-3360';cfg=OmegaConf.load(run/'resolved.yaml');base=cfg.actor_rollout_ref.model.path;hf=run/'checkpoints/actor/global_step_200'
a=AutoTokenizer.from_pretrained(base,local_files_only=True);b=AutoTokenizer.from_pretrained(hf,local_files_only=True)
print(json.dumps(dict(base=base,base_sha=hashlib.sha256((Path(base)/'tokenizer.json').read_bytes()).hexdigest(),expected=cfg.baselines.tree.token_preservation.tokenizer_sha256,hf_sha=hashlib.sha256((hf/'tokenizer.json').read_bytes()).hexdigest(),backend_equal=json.loads(a.backend_tokenizer.to_str())==json.loads(b.backend_tokenizer.to_str()),vocab_equal=a.get_vocab()==b.get_vocab(),special_equal=a.special_tokens_map==b.special_tokens_map,chat_equal=a.chat_template==b.chat_template,flags_equal=(a.pad_token_id,a.eos_token_id,a.clean_up_tokenization_spaces)==(b.pad_token_id,b.eos_token_id,b.clean_up_tokenization_spaces))))
'''
r=m.run(shlex.quote('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python')+' -c '+shlex.quote(code),timeout=60);print(r.stdout,r.stderr,r.returncode);Path('outputs/tree-eval-resume-20261013/TOKENIZER_ALIAS_CHECK.txt').write_text(r.stdout+r.stderr+'\nexit '+str(r.returncode))
