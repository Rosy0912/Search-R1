import json
from pathlib import Path

ROOT = Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1')
OLD = Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
PROFILE = 'qwen25_7b_four_way_v1'
REPOSITORY = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
TOKENIZER = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'


def model_contract(baselines):
    return (baselines.get('scale_profile') == PROFILE
            and baselines.get('model_repository') == REPOSITORY
            and baselines.get('model_revision') == REVISION
            and baselines.tree.token_preservation.tokenizer_sha256 == TOKENIZER
            and 'recursive_ablation' not in baselines
            and 'question_curriculum' not in baselines)


def models():
    return json.loads(Path(__file__).with_name('models.json').read_text())['models']


def model_path(label):
    return ROOT/'models'/label/models()[label]['revision']


def verify_model(label):
    from reproduction.seven_dataset.contracts import validate_model_files
    m = models()[label]
    files = {f['file']: f['sha256'] for f in m['files']}
    path = model_path(label)
    validate_model_files(dict(model=str(path), model_files=files))
    c = json.loads((path/'config.json').read_text())
    if (c['model_type'], c['hidden_size'], c['num_hidden_layers'],
            c['num_attention_heads'], c['num_key_value_heads']) != ('qwen2',3584,28,28,4):
        raise ValueError('Not the pinned Qwen2.5 7B architecture')
    return files


def make_config(root, model, host, *, evaluation=False):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(__file__).with_name('original-template.yaml'))
    OmegaConf.set_struct(cfg, False)
    cfg.actor_rollout_ref.model.path = str(model)
    cfg.actor_rollout_ref.ref.model_path = str(model)
    cfg.critic.model.tokenizer_path = str(model)
    cfg.reward_model.model.input_tokenizer = str(model)
    cfg.baselines.model_repository = REPOSITORY
    cfg.baselines.model_revision = REVISION
    cfg.baselines.scale_profile = PROFILE
    cfg.baselines.local_runtime_host = host
    cfg.trainer.experiment_name = 'sevenb-original'
    cfg.trainer.default_local_dir = str(Path(root)/'checkpoints')
    if evaluation:
        cfg.data.val_batch_size = 1024
        cfg.actor_rollout_ref.rollout.gpu_memory_utilization = .65
        cfg.retriever.url = 'http://127.0.0.1:29147/retrieve'
    return cfg


def verify_data():
    from reproduction.seven_dataset.contracts import DATA_SHA, sha
    data = OLD/'common-data'
    if sha(data/'manifest.json') != DATA_SHA:
        raise ValueError('Common data manifest changed')
    m = json.loads((data/'manifest.json').read_text())
    for split, count in [('train',169028),('validation',512),('stage',64),('final',51713)]:
        row = m['splits'][split]
        if row['rows'] != count or sha(data/row['path']) != row['sha256']:
            raise ValueError('Common split changed: '+split)
    return data, m


def validate_plan(plan):
    from reproduction.seven_dataset.contracts import digest
    if plan['profile'] != PROFILE or plan['base'] != str(ROOT):
        raise ValueError('Different 7B study')
    kind = plan['kind']
    if kind not in ('original','treegrpo','base','searchr1'):
        raise ValueError('Unrequested group; no ablations')
    if kind in ('original','treegrpo'):
        expected = ROOT/'runs'/kind
        if plan['repeat_index'] != 0 or plan['model_label'] != 'base':
            raise ValueError('Only one training run per method')
    else:
        if type(plan['repeat_index']) is not int or plan['repeat_index'] not in (1,2,3):
            raise ValueError('Exactly three fixed repeats')
        expected = ROOT/'evaluations'/kind/('r'+str(plan['repeat_index']))
        if plan['model_label'] != kind:
            raise ValueError('Wrong baseline model')
    if plan['root'] != str(expected) or plan['output'] != str(expected/'output'):
        raise ValueError('Output namespace differs')
    if plan['job'] != 'zhangjunming-7b-'+kind+'-'+str(plan['repeat_index']):
        raise ValueError('Job name differs')
    m = models()[plan['model_label']]
    if plan['model'] != str(model_path(plan['model_label'])) or plan['model_revision'] != m['revision']:
        raise ValueError('Model revision/path differs')
    if plan['model_files'] != {f['file']:f['sha256'] for f in m['files']}:
        raise ValueError('Model file hashes differ')
    expected_identity = digest(dict(profile=PROFILE,model=m['repo'],revision=m['revision'],files=plan['model_files']))
    if plan['checkpoint_identity'] != expected_identity:
        raise ValueError('Model identity differs')
    return expected
