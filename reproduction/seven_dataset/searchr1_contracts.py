"""Official Search-R1 GRPO v0.3 identity; fixed three-repeat common evaluation."""
import json
from pathlib import Path
import re
from .contracts import DATA_SHA, IDS_SHA, digest, require, sha, validate_model_files

REVISION = '4b8abb9beb4efc748019e956266ada547e276381'
MODEL = 'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3'
FILES = {'added_tokens.json': '58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b', 'config.json': 'eaf8dde6f40e2f1ac0b314640c93a01c8d3077d4b989df17e3ea49e6dcb30bd2', 'generation_config.json': 'c71477a85f061ff54313cdaad39ead45f00942f4cbbcd9ab07bfc137cb7d38c2', 'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5', 'model-00001-of-00003.safetensors': 'bdaac3b63ac6f88d2db877661e42019adc0bb7b0c1ed1483e26ed3ae416f48e0', 'model-00002-of-00003.safetensors': '1147c245cf129ad6470294b6ec86d80cb007ac7614fbe19d3aad3b2018b6d86b', 'model-00003-of-00003.safetensors': '268f7d83a0ef536bcfd51dd7de8d71425e103194166bfe1559e1e199597893cd', 'model.safetensors.index.json': '3a899e7f6ef595e15c34b5ffc7e4a1df6131d7c371418b1a6e8f823ab8a7302d', 'special_tokens_map.json': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd', 'tokenizer.json': '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa', 'tokenizer_config.json': '57fea41fa99fce34f91450643e48b3bfbb078c32c6831a3c06834f122c200510', 'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
INHERITED = ('base','backend','data_root','data_manifest_sha256','final_content_identity',
             'ordered_ids_sha256','retriever_receipt','retriever_receipt_sha256')
OWN_FIELDS = {'protocol','group','outer','baseline_model','model_revision','model','model_files',
              'checkpoint_identity','protocol_identity','protocol_template_config_sha256',
              'template_plan','template_plan_sha256','evaluation_code_commit','evaluation_run_id',
              'job','output','repeat_index','repeat_count','canonical_tokenizer','canonical_tokenizer_files','inference_model'}


def model_identity():
    return digest(dict(model=MODEL,revision=REVISION,model_files=FILES,data=DATA_SHA))


def protocol_identity(template):
    return digest(dict(version='native_greedy_search_full51713_v1',
        template_config_sha256=template['training_config_sha256'],data=DATA_SHA,
        final_content_identity=template['final_content_identity'],val_batch_size=1024,
        gpu_memory_utilization=0.65,workers=8,do_sample=False))


def output_root(plan):
    return (Path(plan['base'])/'baseline-seven-evaluations'/plan['checkpoint_identity']
            /plan['protocol_identity']/('r'+str(plan['repeat_index'])))


def validate_searchr1_plan(plan,template):
    require(set(plan)==set(INHERITED)|OWN_FIELDS,'Unknown or missing baseline plan fields')
    require(plan['protocol']=='searchr1_seven_repeats_v1' and plan['group']=='Search-R1'
            and type(plan['outer']) is int and plan['outer']==0,'Not the pinned official Search-R1 baseline')
    require(plan['baseline_model']==MODEL and plan['model_revision']==REVISION
            and plan['model_files']==FILES,'Baseline weights/revision differ')
    require(template['protocol']=='tree_seven_milestone_v1' and template['group']=='original'
            and template['outer']==200 and template['backend']=='slurm_lumia'
            and not any(k in template for k in ('repeat','replaces_job_id','pj_retired_submission')),
            'Protocol template must be the first original200 plan')
    require(all(plan[k]==template[k] for k in INHERITED),'Shared data/retrieval/backend changed')
    require(Path(plan['model'])==Path(plan['base'])/'official-baselines'/'searchr1-grpo-v03'/REVISION,'Unbound official model path')
    require(plan['canonical_tokenizer']==template['canonical_tokenizer'] and plan['canonical_tokenizer_files']==template['canonical_tokenizer_files'],'Common tokenizer identity changed')
    require(Path(plan['inference_model'])==Path(plan['base'])/'official-baseline-views'/model_identity(),'Unbound inference view')
    require(plan['ordered_ids_sha256']==IDS_SHA and
            plan['protocol_template_config_sha256']==template['training_config_sha256'],
            'Protocol configuration identity changed')
    require(plan['checkpoint_identity']==model_identity()
            and plan['protocol_identity']==protocol_identity(template),'Baseline/protocol identity changed')
    require(type(plan['repeat_index']) is int and plan['repeat_index'] in (1,2,3)
            and type(plan['repeat_count']) is int and plan['repeat_count']==3,
            'Exactly three independent baseline repeats requested')
    require(re.fullmatch('[0-9a-f]{40}',plan['evaluation_code_commit']) is not None
            and re.fullmatch('[0-9a-f]{32}',plan['evaluation_run_id']) is not None,
            'Missing source/execution identity')
    require(plan['job']=='zhangjunming-ttt-e-sr1-r'+str(plan['repeat_index'])+'-'+model_identity()[:4]
            and Path(plan['output'])==output_root(plan)/'output','Baseline namespace/name changed')
    return output_root(plan)


def inspect_searchr1(plan):
    require(sha(plan['template_plan'])==plan['template_plan_sha256'],'Protocol template plan changed')
    template=json.loads(Path(plan['template_plan']).read_text())
    root=validate_searchr1_plan(plan,template)
    validate_model_files(plan)  # Real official file contents, never a fake training manifest.
    from .entry import inspect_inputs
    cfg,expected=inspect_inputs(template)  # Independently validate the reference science/data contract.
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg,False)
    from .model_view import validate_view
    validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
    validate_tokenizer_equivalence(plan)
    cfg.actor_rollout_ref.model.path=plan['inference_model']
    cfg.actor_rollout_ref.ref.model_path=plan['inference_model']
    cfg.trainer.default_local_dir=str(Path(plan['output'])/'unused-checkpoints')
    require(cfg.baselines.tree.token_preservation.tokenizer_sha256==plan['canonical_tokenizer_files']['tokenizer.json'],
            'Base tokenizer differs from the common inference tokenizer')
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    return cfg,expected,root


def validate_tokenizer_equivalence(plan):
    from verl.utils import hf_tokenizer
    from search_r1.baselines.tree_raw_tokens import _encoder_binding
    original=hf_tokenizer(plan['model']);common=hf_tokenizer(plan['canonical_tokenizer'])
    require(original.get_vocab()==common.get_vocab(), 'Official token IDs differ from common tokenizer')
    for key in ('pad_token_id','eos_token_id','bos_token_id','all_special_ids','chat_template'):
        require(getattr(original,key)==getattr(common,key),'Official tokenizer semantic flag differs: '+key)
    require(_encoder_binding(original)==_encoder_binding(common),'Official encoder semantics differ')
    probes=['<think>Paris</think><search>Paris\nFrance</search><information>é中文</information><answer>Paris</answer>', 'A\u0085B\nC', 'foo   bar\t!']
    for text in probes:
        require(original.encode(text)==common.encode(text),'Official encoding differs')
        ids=original.encode(text)
        require(original.decode(ids)==common.decode(ids),'Official decoding differs')
    return True
