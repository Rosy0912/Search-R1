"""Compose exact native argv and check our schema/tokenizer without GPU or training."""
import json, hashlib, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig
import pandas as pd

root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
from launch_observed import compose_launch_config
config,reload_only=compose_launch_config()
assert not reload_only
assert config.trainer.resume_mode=='disable'
assert config.trainer.total_training_steps is None
assert config.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=='observe_worker.defer_install'
assert config.trainer.test_freq==config.trainer.save_freq==5
argv=json.loads((root/'tools/author-cli.json').read_text())
resolved=OmegaConf.to_container(config,resolve=True)
(root/'tools/resolved-author.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
model=str(config.actor_rollout_ref.model.path)
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
model_config=AutoConfig.from_pretrained(model,local_files_only=True)
result={'argv_count':len(argv),'configuration_composed':True,
        'has_chat_template':bool(tokenizer.chat_template),'model_path':model,
        'max_position_embeddings':model_config.max_position_embeddings,
        'actor_scheduler_config':resolved['actor_rollout_ref']['actor']['optim'],
        'ranker_scheduler_config':resolved['reranker_actor_rollout_ref']['actor']['optim'],
        'data':{},'gpu_training_started':False}
for label,paths in [('train',config.data.train_files),('validation',config.data.val_files)]:
    rows=pd.read_parquet(paths[0]); selected=rows.iloc[:3]
    assert len(rows)==(166222 if label=='train' else 64)
    lengths=[]
    for _,row in selected.iterrows():
        question=row['extra_info']['question']
        prompt=list(row['prompt'])
        assert question and question in prompt[-1]['content']
        assert len(row['reward_model']['ground_truth']['target'])>0
        if tokenizer.chat_template:
            lengths.append(len(tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=True)))
    result['data'][label]={'rows':len(rows),'sample_prompt_lengths':lengths,
      'file_sha256':hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()}
result['passed']=result['has_chat_template']
(root/'tools/cpu-launch-gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert result['passed'],'Our base tokenizer needs an explicitly recorded compatible chat template'
