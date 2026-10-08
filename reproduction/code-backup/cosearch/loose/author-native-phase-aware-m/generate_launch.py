"""Extract author's pinned CLI verbatim; override infrastructure/inputs only."""
import json,re,shlex
from pathlib import Path
repo=Path(__file__).resolve().parents[1]/'cosearch-author-review'
script=(repo/'scripts/train_co_search_grpo.sh').read_text()
root='/dataset/shaoshitong/search-r1-rl/cosearch/author-native'
asset='/dataset/shaoshitong/search-r1-rl/cosearch'
values=dict(PROJECT_ROOT=root+'/repo',CHECKPOINT_PATH=asset+'/assets-root/assets/qwen2.5-3b',RERANKER_CHECKPOINT_PATH=asset+'/assets-root/assets/qwen2.5-3b',TRAIN_DATA='["'+root+'/data/train.parquet"]',VAL_DATA='["'+root+'/data/validation-64.parquet"]',TOOL_CONFIG=root+'/tools/tool.yaml',AGENT_LOOP_CONFIG=root+'/repo/config/co_search_agent_loop_config.yaml',NNODES='1',N_GPUS_PER_NODE='8',EXP_NAME='author-native-raw-five',CHECKPOINT_DIR=root+'/runs',PROJECT_NAME='co_search')
for key,value in re.findall(r'^([A-Z][A-Z0-9_]*)=(.*)$',script,re.M):
 if key in values:continue
 parsed=shlex.split(value,comments=True)
 if len(parsed)!=1:continue
 value=parsed[0]
 for k,v in values.items():value=value.replace('${'+k+'}',v)
 if '$' not in value:values[key]=value
lines=script[script.index('python main_co_search_ppo.py \\\n'):].splitlines()
command_lines=[]
for line in lines:
 command_lines.append(line)
 if not line.rstrip().endswith('\\'):break
text='\n'.join(command_lines).replace('\\\n',' ')
for k in sorted(values,key=len,reverse=True):
 text=text.replace('${'+k+'}',values[k]);text=re.sub(r'\$'+re.escape(k)+r'\b',lambda _:values[k],text)
assert '$' not in text,text
argv=shlex.split(text)[2:]
assert all('=' in arg and not arg.startswith('#') for arg in argv),argv
# Keep natural full-epoch scheduler horizon. Stop via external wrapper after step5.
replacements={'trainer.logger':'[console]','trainer.test_freq':'5','trainer.save_freq':'5','trainer.resume_mode':'disable','trainer.default_local_dir':root+'/runs/author-native-raw-five/checkpoints','trainer.rollout_data_dir':root+'/runs/author-native-raw-five/rollouts','trainer.validation_data_dir':root+'/runs/author-native-raw-five/validation'}
argv=[k.split('=',1)[0]+'='+replacements[k.split('=',1)[0]] if k.split('=',1)[0] in replacements else k for k in argv]
argv.append('data.seed=42')
Path(__file__).with_name('author-cli.json').write_text(json.dumps(argv,indent=2)+'\n')
Path(__file__).with_name('launch-differences.json').write_text(json.dumps({'source_commit':'763bf8c4d7caa363ad845d39ddfd53b81ae377bd','same_base_actor_and_ranker':True,'author_optimization_hyperparameters_retained':True,'natural_epoch_scheduler_horizon_retained':True,'infrastructure_and_acceptance_overrides':replacements,'raw_author_prompt_cap':16384,'native_source_patched':False},indent=2)+'\n')
print('Prepared',len(argv),'author CLI overrides')
