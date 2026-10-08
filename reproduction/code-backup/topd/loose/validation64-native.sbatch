#!/bin/bash
#SBATCH --job-name=topd-validation64-loader
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/validation64-loader-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export SEARCH_R1_ASSETS="$AGENT_ROOT/assets-root"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/repo"
cd "$AGENT_ROOT"
"$AGENT_ROOT/env/bin/python" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
root=Path(os.environ['AGENT_ROOT'])
assert not torch.cuda.is_initialized()
commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
assert commit=='438af3529b40663aae7b8293080e09fda57f6a87'
with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'),version_base=None):
    cfg=compose(config_name='ppo_baselines',overrides=['baselines.method=topd','data.val_data_num=64'])
data=root/'assets-root/data/paper-grpo/validation-64.parquet'
assert Path(cfg.data.val_files).resolve()==data.resolve()
assert cfg.data.val_batch_size==cfg.data.val_data_num==64
tokenizer=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,local_files_only=True)
dataset=RLHFDataset(parquet_files=cfg.data.val_files,tokenizer=tokenizer,prompt_key=cfg.data.prompt_key,
 max_prompt_length=cfg.data.max_prompt_length,filter_prompts=True,
 return_raw_chat=cfg.data.get('return_raw_chat',False),truncation='error',cache_dir=str(root/'loader-check-cache'))
original_rows=len(dataset)
assert original_rows==64
dataset.dataframe=dataset.dataframe.sample(cfg.data.val_data_num,random_state=42)
loader=DataLoader(dataset=dataset,batch_size=cfg.data.val_batch_size,shuffle=False,drop_last=False,collate_fn=collate_fn)
batches=list(loader)
assert len(batches)==1 and batches[0]['input_ids'].shape==(64,cfg.data.max_prompt_length)
lengths=batches[0]['attention_mask'].sum(-1).tolist()
assert len(lengths)==64 and min(lengths)>0 and max(lengths)<=cfg.data.max_prompt_length
assert not torch.cuda.is_initialized()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(status='PASS_CPU_DATA_LOADING_ONLY_NOT_TRAINING',code_commit=commit,
 input_path=str(data),input_sha256=sha(data),input_rows=original_rows,actual_dataset_rows=len(dataset),
 actual_batches=len(batches),actual_batch_rows=64,prompt_width=cfg.data.max_prompt_length,
 prompt_token_lengths=lengths,val_data_num=cfg.data.val_data_num,val_batch_size=cfg.data.val_batch_size,
 dataset_source_sha256=sha(root/'repo/verl/utils/dataset/rl_dataset.py'),
 trainer_source_sha256=sha(root/'repo/verl/trainer/ppo/ray_trainer.py'),
 tokenizer_files={p.name:sha(p) for p in Path(cfg.actor_rollout_ref.model.path).iterdir() if p.is_file() and ('token' in p.name or p.name in ['vocab.json','merges.txt','special_tokens_map.json'])},
 torch_version=torch.__version__,cuda_initialized=False,actual_topd_outer_updates=0)
target=root/'logs/validation64-loader-result.json'
with target.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,sort_keys=True))
print('ACTUAL_VALIDATION64_DATA_LOADER_PASS_NO_GPU_NO_TRAINING')
PY
