"""CPU-only verification of the isolated immutable Search-E1 pure OPSD profile."""
import hashlib
import os
from pathlib import Path
from .sharedholdout import unique_json, _sha256_field, _logical_path

COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
OFFICIAL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'

PROFILE_CONFIGS = {'e1_300': 'ppo_opsd_e1_300', 'author200': 'ppo_opsd_e1_author200',
                   'multitraj200': 'ppo_opsd_e1_multitraj200'}


def profile_name(config):
    """An absent selector preserves the frozen original E1-300 profile."""
    baselines = config.get('baselines') if hasattr(config, 'get') else config.baselines
    settings = baselines.get('opsd_e1') if hasattr(baselines, 'get') else baselines.opsd_e1
    name = settings.get('profile', 'e1_300') if hasattr(settings, 'get') else getattr(settings, 'profile', 'e1_300')
    profile_contract(name)  # Reject unknown selectors instead of guessing a budget.
    return name


def profile_contract(name='e1_300'):
    if type(name) is not str or name not in PROFILE_CONFIGS:
        raise ValueError('Unknown isolated OPSD profile')
    if name == 'multitraj200':
        return dict(horizon=200, warmup_steps=57, lr=1e-6, grad_clip=1,
                    mini_batch=256, actor_micro_batch=32, log_prob_micro_batch=64,
                    responses=5, sampling_temperature=1, top_p=1, top_k=-1,
                    component_clip=None, optimizer_steps=10,
                    samples_per_rank=320, save_freq=50)
    if name == 'author200':
        return dict(horizon=200, warmup_steps=0, lr=5e-6, grad_clip=.1,
                    mini_batch=32, responses=1, sampling_temperature=1.1,
                    top_p=.95, top_k=20, component_clip=.05,
                    optimizer_steps=16, samples_per_rank=64)
    return dict(horizon=300, warmup_steps=57, lr=1e-6, grad_clip=1,
                mini_batch=128, responses=5, sampling_temperature=1,
                top_p=1, top_k=-1, component_clip=None,
                optimizer_steps=20, samples_per_rank=320)

def pinned_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha256_field(expected, 'external pin'):
        raise ValueError('Pinned JSON SHA differs')
    return unique_json(raw)

def validate_profile(cfg):
    from omegaconf import OmegaConf
    d = OmegaConf.to_container(cfg, resolve=True)
    def get(key):
        x = d
        for part in key.split('.'): x = x[part]
        return x
    contract = profile_contract(profile_name(cfg))
    expected = {
        'baselines.method':'opsd','baselines.seed':42,
        'baselines.local_runtime_only':True,'baselines.local_runtime_host':'h100-10',
        'baselines.holdout.input_mode':'prebuilt_manifest','baselines.holdout.manifest_sha256':COMMON_SHA,
        'data.train_batch_size':512,'data.val_batch_size':512,
        'data.train_data_num':None,'data.val_data_num':None,
        'actor_rollout_ref.actor.ppo_mini_batch_size':contract['mini_batch'],
        'actor_rollout_ref.actor.ppo_micro_batch_size':contract.get('actor_micro_batch',16),
        'actor_rollout_ref.actor.grad_clip':contract['grad_clip'],'actor_rollout_ref.actor.clip_ratio':0.2,
        'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.actor.entropy_coeff':0,
        'actor_rollout_ref.actor.use_kl_loss':False,'actor_rollout_ref.actor.kl_loss_coef':0,
        'actor_rollout_ref.actor.optim.lr':contract['lr'],'actor_rollout_ref.actor.optim.lr_warmup_steps':contract['warmup_steps'],
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio':0,'actor_rollout_ref.actor.optim.warmup_style':'constant',
        'actor_rollout_ref.rollout.n_agent':contract['responses'],'actor_rollout_ref.rollout.n':1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size':contract.get('log_prob_micro_batch',32),
        'actor_rollout_ref.ref.log_prob_micro_batch_size':contract.get('log_prob_micro_batch',32),
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.temperature':contract['sampling_temperature'],'actor_rollout_ref.rollout.top_p':contract['top_p'],'actor_rollout_ref.rollout.top_k':contract['top_k'],
        'algorithm.kl_ctrl.kl_coef':0,'trainer.total_training_steps':contract['horizon'],
        'trainer.n_gpus_per_node':8,'trainer.nnodes':1,
        'trainer.save_freq':contract.get('save_freq',25),'trainer.test_freq':50,
        'trainer.acceptance_step':5,'trainer.val_before_train':True,
        'trainer.smoke_only':False,'data.max_prompt_length':4096,'data.max_start_length':2048,
        'data.max_response_length':500,'data.max_obs_length':500,'max_turns':4,'force_final_answer':False,
        'retriever.topk':3,'baselines.distillation.teacher_device':'cuda',
        'baselines.distillation.privileged_answers':True,'baselines.distillation.teacher_prompt_mode':'user_context',
        'baselines.distillation.beta':0,'baselines.distillation.temperature':1,
        'baselines.distillation.component_clip':contract['component_clip'],'baselines.distillation.chunk_size':16,
        'baselines.distillation.loss_checkpointing':True,'baselines.distillation.max_student_logits_gib':8.0,
        'baselines.opsd_e1.enabled':True,'baselines.opsd_e1.gate_required':True,
        'baselines.opsd_e1.acceptance_step':5,'baselines.opsd_e1.stage_num':64,'baselines.opsd_e1.stage_freq':50,
    }
    if profile_name(cfg) == 'multitraj200':
        expected['baselines.distillation.response_only_logits'] = False
        expected['baselines.distillation.teacher_forward_micro_batch'] = 2
    for key, value in expected.items():
        actual = get(key)
        # bools must not pass as integer identity.
        if actual != value or (isinstance(value,(bool,int)) and type(actual) is not type(value)):
            raise ValueError('E1 fixed profile differs: ' + key)
    if get('actor_rollout_ref.model.path') != get('baselines.distillation.teacher_path'):
        raise ValueError('Frozen teacher must be the same official initial model')
    from urllib.parse import urlparse
    endpoint=urlparse(get('retriever.url'))
    if endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','::1','localhost') or endpoint.path!='/retrieve' or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError('E1 retrieval must be same-node loopback /retrieve')
    root = Path(get('baselines.opsd_e1.run_root'))
    if not root.is_absolute(): raise ValueError('Explicit absolute own run root required')
    for key, name in [('baselines.opsd_e1.train_log_path','train.log'),('baselines.opsd_e1.receipt_path','acceptance-5-receipt.json')]:
        if Path(get(key)) != root/name: raise ValueError('Own run path differs: ' + key)
    return d

def verify_official_assets(model_root, manifest_path):
    manifest = pinned_json(manifest_path, OFFICIAL_SHA)
    if manifest.get('repo') != 'Qwen/Qwen2.5-3B-Instruct' or manifest.get('revision') != REVISION:
        raise ValueError('Official model identity differs')
    records = manifest.get('files')
    if not isinstance(records,list) or len(records)!=11: raise ValueError('Official manifest inventory differs')
    root = Path(model_root).resolve()
    seen = set()
    from .holdout import file_hash
    for rec in records:
        name = rec['file']
        if name in seen: raise ValueError('Duplicate official model file')
        seen.add(name)
        path = _logical_path(root,name)
        if rec.get('revision') != REVISION or type(rec.get('bytes')) is not int or path.stat().st_size!=rec['bytes']:
            raise ValueError('Official model file bytes/revision differ: '+name)
        if file_hash(path)!=_sha256_field(rec['sha256'],'model file'):
            raise ValueError('Official model file SHA differs: '+name)
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    metadata = {'VERIFIED'}
    verified_manifest_path = Path(manifest_path).resolve()
    if verified_manifest_path.parent == root:
        metadata.add(verified_manifest_path.name)  # Already pinned_json checked the exact official manifest bytes.
    if actual_files - seen - metadata:
        raise ValueError('Unmanifested official model files are not allowed')
    # A marker is inventory metadata only, never evidence of acceptance.
    return manifest

def compose_profile(profile='e1_300'):
    from hydra import compose, initialize_config_dir
    profile_contract(profile)
    config_dir=Path(__file__).resolve().parents[2]/'verl/trainer/config'
    with initialize_config_dir(config_dir=str(config_dir),version_base=None):
        return compose(config_name=PROFILE_CONFIGS[profile])

def main():
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--validate-assets',action='store_true')
    parser.add_argument('--profile',choices=tuple(PROFILE_CONFIGS),default='e1_300');args=parser.parse_args()
    cfg=compose_profile(args.profile);resolved=validate_profile(cfg)
    if args.validate_assets:
        from .sharedholdout import prepare_prebuilt_holdout
        verify_official_assets(cfg.actor_rollout_ref.model.path,os.environ['OPSD_E1_OFFICIAL_MANIFEST'])
        pinned_json(cfg.baselines.opsd_e1.binding_manifest_path,cfg.baselines.opsd_e1.binding_manifest_sha256)
        pinned_json(cfg.baselines.opsd_e1.public_key_path,cfg.baselines.opsd_e1.public_key_sha256)
        import contextlib,sys
        with contextlib.redirect_stdout(sys.stderr):
            record=prepare_prebuilt_holdout(cfg)
        if record['train_rows']!=169028: raise ValueError('Actual common train rows differ')
        resolved=validate_profile(cfg)  # Record the actual physical post-manifest paths.
    print(json.dumps(resolved,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
