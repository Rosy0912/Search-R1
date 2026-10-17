"""Independent, untrained Qwen baseline identity and fixed three-repeat contract."""
import json
from pathlib import Path
import re
from .contracts import DATA_SHA, IDS_SHA, digest, require, sha, validate_model_files

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MODEL = 'Qwen/Qwen2.5-3B-Instruct'
# Actual official files independently hashed before authorizing baseline repeats.
FILES = {
    'config.json': 'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b',
    'generation_config.json': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'model-00001-of-00002.safetensors': '67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2',
    'model-00002-of-00002.safetensors': 'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d',
    'model.safetensors.index.json': 'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}
INHERITED = ('base','backend','data_root','data_manifest_sha256','final_content_identity',
             'ordered_ids_sha256','retriever_receipt','retriever_receipt_sha256')
OWN_FIELDS = {'protocol','group','outer','baseline_model','model_revision','model','model_files',
              'checkpoint_identity','protocol_identity','protocol_template_config_sha256',
              'template_plan','template_plan_sha256','evaluation_code_commit','evaluation_run_id',
              'job','output','repeat_index','repeat_count'}


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


def validate_base_plan(plan,template):
    require(set(plan)==set(INHERITED)|OWN_FIELDS,'Unknown or missing baseline plan fields')
    require(plan['protocol']=='base_seven_repeats_v1' and plan['group']=='base'
            and type(plan['outer']) is int and plan['outer']==0,'Not the untrained baseline')
    require(plan['baseline_model']==MODEL and plan['model_revision']==REVISION
            and plan['model_files']==FILES,'Baseline weights/revision differ')
    require(template['protocol']=='tree_seven_milestone_v1' and template['group']=='original'
            and template['outer']==200 and template['backend']=='slurm_lumia'
            and not any(k in template for k in ('repeat','replaces_job_id','pj_retired_submission')),
            'Protocol template must be the first original200 plan')
    require(all(plan[k]==template[k] for k in INHERITED),'Shared data/retrieval/backend changed')
    require(plan['model']==template['canonical_tokenizer'],'Base must be the archived initialization model')
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
    require(plan['job']=='zhangjunming-ttt-e-base-r'+str(plan['repeat_index'])+'-'+model_identity()[:4]
            and Path(plan['output'])==output_root(plan)/'output','Baseline namespace/name changed')
    return output_root(plan)


def inspect_base(plan):
    require(sha(plan['template_plan'])==plan['template_plan_sha256'],'Protocol template plan changed')
    template=json.loads(Path(plan['template_plan']).read_text())
    root=validate_base_plan(plan,template)
    validate_model_files(plan)  # Real official file contents, never a fake training manifest.
    from .entry import inspect_inputs
    cfg,expected=inspect_inputs(template)  # Independently validate the reference science/data contract.
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg,False)
    cfg.actor_rollout_ref.model.path=plan['model']
    cfg.actor_rollout_ref.ref.model_path=plan['model']
    cfg.trainer.default_local_dir=str(Path(plan['output'])/'unused-checkpoints')
    require(cfg.baselines.tree.token_preservation.tokenizer_sha256==FILES['tokenizer.json'],
            'Base tokenizer differs from the common inference tokenizer')
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    return cfg,expected,root
