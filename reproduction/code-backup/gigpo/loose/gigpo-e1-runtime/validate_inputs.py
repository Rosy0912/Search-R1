"""Actual target-node common-data consumer and official tokenizer boundary."""
import os,json,hashlib,subprocess,inspect
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.instruct300 import prepare_sampler
import search_r1.baselines.instruct300 as source
root=Path(os.environ['GIGPO_E1_ROOT'])
with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'),version_base=None):
 cfg=compose(config_name='ppo_gigpo_instruct300')
split=prepare_common_holdout(cfg)
sampler=prepare_sampler(cfg,split['train_rows'])
from transformers import AutoTokenizer
import pandas as pd
tokenizer=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,local_files_only=True,trust_remote_code=False)
prompt=pd.read_parquet(cfg.data.val_files).iloc[0]['prompt'].tolist()
text=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
value=dict(status='PASS',node=os.uname().nodename,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_file=inspect.getfile(source),split=split,resolved_config=OmegaConf.to_container(cfg,resolve=True),tokenizer_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),golden_prompt=text,golden_ids=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True),sampler_size=len(sampler))
(root/'logs/actual-config-data-tokenizer-validation.json').write_text(json.dumps(value,indent=2))
files=[]
tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
for path in [root/'repo'/p for p in tracked]+list((root/'tools').glob('*')):
 if path.is_file():files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
binding=dict(code_commit=value['code_commit'],files=files,status='NATIVE_VALIDATION_ONLY')
(root/'logs/native-binding-v2.json').write_text(json.dumps(binding,indent=2))
print(json.dumps(dict(status=value['status'],commit=value['code_commit'],input_mode=split['input_mode'],counts=split.get('counts'),files=len(files),source_file=value['source_file']),indent=2))
