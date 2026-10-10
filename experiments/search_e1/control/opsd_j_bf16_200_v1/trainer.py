"""One online outer round, eight persistent replicated LoRA/Adam ranks.

The controller collects a full current-student 512x5 cohort before this entry.
Import is stdlib only. main() is the only model/CUDA/distributed entry point.
Checkpoint manifests become visible only after all eight complete states and
the adapter have durable hashes; failed attempts never replace accepted state.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

from objective import canonical_sha, encoded_pair, pair_objective, validate_component_clip
from precision import PrecisionContract, SyncedGradScaler
from targets import SharedTargets
from miner_v1 import verify_pairs
from checkpoint import science_contract


RNG_NAMES = ['python', 'numpy', 'torch_cpu', 'cuda_all']


def validate_science(config):
    s = config['scientific']
    fixed = dict(loss_reduction='trajectory_mean', learning_rate=5e-6 if config['variant']=='C0' else 1e-6,
                 micro_batch=1, global_batch=32, lora_rank=16, lora_alpha=32,
                 grad_norm_clip=1, chunk=32, max_context=8192, seed=42,
                 warmup_updates=0)
    for name, expected in fixed.items():
        if s.get(name) != expected:
            raise ValueError('Unapproved scientific setting: ' + name)
    if 'component_clip' not in s:
        raise ValueError('component_clip requires explicit human choice')
    validate_component_clip(s['component_clip'])
    if s['component_clip'] != 10 or s['outer_iterations'] != 200:
        raise ValueError('All seven groups require upper component clip10 and ten online outers')
    if type(s.get('outer_iterations')) is not int or s['outer_iterations'] <= 0:
        raise ValueError('outer_iterations requires explicit positive human choice')
    opt = s['optimizer']
    if opt != dict(kind='AdamW', betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01):
        raise ValueError('Original AdamW settings must be explicitly bound')
    log = config['logging']
    if log['topk'] != 32 or type(log['topk_max_tokens_per_outer']) is not int or log['topk_max_tokens_per_outer'] < 0:
        raise ValueError('Explicit top32 logging cap required (0 disables top32 only)')
    return s



def verify_sampling_identity(outer, config, prior_checkpoint=None):
    """Bind sampled weight to the *start* of the outer, never to resumed mid-outer."""
    identity = outer['sampling_model_identity']
    previous = outer.get('previous_complete_checkpoint')
    if identity['source_model_identity'] != config['source_model_manifest']['sha256']:
        raise ValueError('Sampling source100 identity changed')
    if identity['checkpoint'] != previous or identity['optimizer_updates'] != outer['cumulative_optimizer_updates_before']:
        raise ValueError('Sampling did not use original latest outer-start checkpoint')
    if previous is None:
        if (outer['outer_index'] != 0 or outer['explicit_fresh'] is not True or identity['optimizer_updates'] != 0 or
                identity['adapter_tensor_sha256'] != config['common_initial_adapter_tensor_sha256']):
            raise ValueError('Initial sampling must be source100 zero-effect, no prior Adam')
    elif (prior_checkpoint is None or outer['explicit_fresh'] is not False or
          prior_checkpoint['optimizer_updates'] != identity['optimizer_updates'] or
          prior_checkpoint['all_rank_adapter_sha256'][0] != identity['adapter_tensor_sha256'] or
          prior_checkpoint['cursor']['outer_index'] != outer['outer_index']-1 or
          prior_checkpoint['cursor']['next_pair_cursor'] != prior_checkpoint['cursor']['usable_pair_count']):
        raise ValueError('Sampling identity not latest fully completed preceding outer')
    return identity


def verify_outer_old(outer, config, resumed=None):
    expected = outer.get('previous_complete_checkpoint') or config['common_initial_adapter_manifest']
    if outer.get('outer_old_adapter') != expected:
        raise ValueError('Outer-old must be the exact original outer-start checkpoint/template')
    if resumed and resumed['cursor']['outer_index'] == outer['outer_index']:
        if resumed.get('outer_old_adapter') != expected:
            raise ValueError('Mid-outer resume changed original frozen old adapter')
    return expected


def actual_adam_state(state, updates, expected_parameters=None):
    """Inspect all actual moments and every step; no declaration-only acceptance."""
    import torch
    entries = state['state']
    parameters = [i for g in state['param_groups'] for i in g['params']]
    if len(parameters) != len(set(parameters)) or (expected_parameters is not None and len(parameters) != expected_parameters):
        raise ValueError('Adam parameter inventory mismatch')
    if updates == 0:
        if entries: raise ValueError('Fresh Adam unexpectedly has moments')
        return []
    if set(entries) != set(parameters): raise ValueError('Missing Adam moments for trainable parameters')
    for value in entries.values():
        if set(value) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Incomplete/unapproved Adam state')
        if float(value['step']) != updates or value['exp_avg'].shape != value['exp_avg_sq'].shape:
            raise ValueError('Actual Adam step/shape mismatch')
        if any(not torch.isfinite(t).all().item() for t in value.values()):
            raise FloatingPointError('Nonfinite saved Adam step/moments')
    return [float(updates)]


def verify_rank_state(state, manifest, rank, config, optimizer_state=None):
    """Verify saved state semantic metadata before loading optimizer or RNG."""
    expected = dict(schema='opsd-online-rank-state-v1',rank=rank,
        run_id=manifest['run_id'],config_sha256=manifest['config_sha256'] if manifest.get('config_sha256')==config.get('continuation',{}).get('prefix_config_sha256') and manifest['optimizer_updates']==50 and manifest['cursor']['outer_index']==4 else config['_config_sha256'],
        optimizer_updates=manifest['optimizer_updates'],cumulative_pair_exposure=manifest['cumulative_pair_exposure'],
        cursor=manifest['cursor'],source_model_identity=manifest['source_model_identity'],
        runtime_identity=manifest['runtime_identity'],training_input=manifest['training_input'],
        adapter_tensor_sha256=manifest['all_rank_adapter_sha256'][rank],
        optimizer_tensor_sha256=manifest['all_rank_optimizer_tensor_sha256'][rank],
        contract=science_contract(config), outer_old_adapter=manifest['outer_old_adapter'],
        scaler=manifest['all_rank_scaler_state'][rank])
    if any(state.get(k) != v for k,v in expected.items()):
        raise ValueError('Rank state semantic metadata differs from complete manifest')
    validate_rng(state['rng'])
    actual_adam_state(state['optimizer'], manifest['optimizer_updates'])
    if optimizer_tensor_digest(state['optimizer']) != expected['optimizer_tensor_sha256']:
        raise ValueError('Saved Adam moments/groups do not match manifest digest')
    if optimizer_state is not None and optimizer_tensor_digest(optimizer_state) != expected['optimizer_tensor_sha256']:
        raise ValueError('Restored Adam differs from saved actual state')
    return True



def validate_rng(rng):
    import numpy as np
    import torch
    if set(rng) != set(RNG_NAMES): raise ValueError('Incomplete saved all-RNG state')
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    values=[rng['torch_cpu']]+list(rng['cuda_all'])
    if not rng['cuda_all'] or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.uint8 or t.device.type!='cpu' or t.ndim!=1 or t.numel()==0 for t in values):
        raise ValueError('Malformed/incomplete actual Torch CPU/CUDA-all RNG')


def verify_committed_receipts(refs, manifest, outer_before, checked_json, checked_file, order=None):
    """Only manifest-referenced minis are committed; preserve failed attempt logs."""
    cursor=manifest['cursor']['next_pair_cursor']
    expected_count=math.ceil(cursor/32)
    if len(refs)!=expected_count or manifest['optimizer_updates']!=outer_before+expected_count:
        raise ValueError('Committed cursor/update/receipt inventory differs')
    exposure=0; tokens=0
    for number,ref in enumerate(refs,1):
        value,_=checked_json(ref)
        actual=min(32,cursor-exposure)
        if (value['schema']!='opsd-online-real-update-v1' or value['accepted'] is not True or
            value['run_id']!=manifest['run_id'] or value['outer_index']!=manifest['cursor']['outer_index'] or
            value['optimizer_updates']!=outer_before+number or value['global_actual_pairs']!=actual):
            raise ValueError('Committed update receipt has wrong run/order/exposure')
        ranks=value['all_rank_actual_token_metrics']
        if len(ranks)!=8 or {r['rank'] for r in ranks}!=set(range(8)):
            raise ValueError('Committed update missing/duplicate rank token metrics')
        pairs=[pair for rank in ranks for pair in rank['pairs']]
        if len(pairs)!=actual or len({p['pair_index'] for p in pairs})!=actual:
            raise ValueError('Committed pair exposure missing/duplicated')
        if order is not None and {p['pair_index'] for p in pairs}!=set(order[exposure:exposure+actual]):
            raise ValueError('Committed receipts differ from exact shuffled pair order')
        for pair in pairs: checked_file(pair['token_log'])
        count=sum(p['valid_tokens'] for p in pairs)
        if count!=value['global_valid_tokens'] or any(p['valid_tokens']<=0 for p in pairs):
            raise ValueError('Committed effective-token count mismatch')
        exposure+=actual; tokens+=count
    if exposure!=cursor: raise ValueError('Committed receipt exposure differs from exact cursor')
    return tokens



def verify_source100(config, checked_json, checked_file):
    """Fresh CPU full SHA of twelve actual source files before every trainer load."""
    manifest,_=checked_json(config['source_model_manifest'])
    files=manifest.get('files',manifest.get('source_files'))
    root=Path(config['paths']['model']).resolve()
    if not isinstance(files,list) or len(files)!=12: raise ValueError('Source100 must bind twelve actual model files')
    paths=[Path(f['path']) for f in files]
    if len({p.resolve() for p in paths})!=12 or any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError('Source100 file inventory aliased/escaped/symlinked')
    actual={p.resolve() for p in root.rglob('*') if p.is_file()}
    if actual!={p.resolve() for p in paths}: raise ValueError('Source100 actual model directory differs from bound twelve-file inventory')
    return [checked_file(f) for f in files]


def verify_source_metadata(proofs):
    for proof in proofs:
        stat=Path(proof['path']).stat()
        expected=dict(device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
        if any(proof.get(k)!=v for k,v in expected.items()): raise ValueError('Actual source100 file changed since fresh SHA load binding')
    return True


def runtime_rank_identity(torch, device, rank, config):
    """Bind actual visible CUDA rank to nvidia-smi UUID (Torch2.4 has no props.uuid)."""
    import subprocess
    physical_gpus=config['owner']['physical_gpus'];expected_uuids=config['owner']['gpu_uuids']
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if (len(physical_gpus)!=8 or len(expected_uuids)!=8 or
        len(set(physical_gpus))!=8 or len(set(expected_uuids))!=8 or
        len(visible)!=8 or any(not item.strip().isdigit() for item in visible)):
        raise ValueError('Actual eight-GPU visible-device mapping required')
    mapping=[int(item.strip()) for item in visible]
    if mapping!=physical_gpus or device.index!=rank or not 0<=rank<8:
        raise ValueError('CUDA visible rank differs from authorized physical GPU mapping')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    identities={}
    for line in rows.splitlines():
        if not line.strip():continue
        fields=[field.strip() for field in line.split(',')]
        if len(fields)!=2 or not fields[0].isdigit() or not fields[1].startswith('GPU-'):
            raise ValueError('Malformed actual nvidia-smi GPU UUID inventory')
        index=int(fields[0])
        if index in identities:raise ValueError('Duplicate actual nvidia-smi physical GPU index')
        identities[index]=fields[1]
    if any(identities.get(index)!=expected for index,expected in zip(physical_gpus,expected_uuids)):
        raise ValueError('Actual physical GPU UUID differs from authorized eight-GPU inventory')
    physical_gpu=mapping[device.index]
    stat=Path('/proc/self/stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    free,total=torch.cuda.mem_get_info(device)
    return dict(rank=rank,pid=os.getpid(),uid=os.getuid(),starttick=int(tail[19]),
        local_rank=device.index,physical_gpu=physical_gpu,gpu_uuid=identities[physical_gpu],actual_free_bytes=free,total_bytes=total,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(device),reserved_peak_bytes=torch.cuda.max_memory_reserved(device))


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def rank_batch(batch, rank, world):
    if world <= 0 or not 0 <= rank < world:
        raise ValueError('Invalid rank partition')
    return batch[rank::world]


def batches(order, cursor=0, global_batch=32):
    if not 0 <= cursor <= len(order) or (cursor < len(order) and cursor % global_batch):
        raise ValueError('Resume cursor must be a committed mini-batch boundary')
    return [order[start:start+global_batch] for start in range(cursor, len(order), global_batch)]


def verify_resume_cursor(state, outer, order_sha, encoded_sha, usable_count):
    cursor = state['cursor']
    if cursor['outer_index'] != outer or cursor['order_sha256'] != order_sha or cursor['encoded_sha256'] != encoded_sha:
        raise ValueError('Resume data/order/outer identity mismatch')
    if cursor['usable_pair_count'] != usable_count:
        raise ValueError('Resume usable pair count changed')
    batches(list(range(usable_count)), cursor['next_pair_cursor'])
    return cursor['next_pair_cursor']


def capture_rng(torch, np):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), cuda_all=torch.cuda.get_rng_state_all())


def restore_rng(torch, np, rng):
    if set(rng) != set(RNG_NAMES):
        raise ValueError('Incomplete RNG state')
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state_all(rng['cuda_all'])


def tensor_digest(named_tensors):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizer_tensor_digest(state):
    import torch
    h = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            h.update(tensor_digest({'tensor': value}).encode())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode()); add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: add(item)
        else:
            h.update(repr(value).encode())
    add(state)
    return h.hexdigest()


def state_to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: state_to_cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [state_to_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(state_to_cpu(v) for v in value)
    return value


def save_torch(path, state, torch):
    with Path(path).open('xb') as handle:
        torch.save(state, handle); handle.flush(); os.fsync(handle.fileno())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def use_frozen_runtime(config, checked_file):
    manifest = config['runtime_manifest']
    checked_file(manifest)
    runtime = Path(config['paths']['runtime']).resolve()
    package = runtime / 'search_e1'
    manifest_value=json.loads(Path(manifest['path']).read_text())
    entries = manifest_value.get('files',manifest_value)
    for name in ('search_e1/core.py', 'search_e1/opsd.py'):
        entry = entries.get(name)
        if not entry:
            raise ValueError('Frozen runtime lacks scientific module ' + name)
        wanted = entry['sha256'] if isinstance(entry, dict) else entry
        checked_file(dict(path=str(runtime / name), sha256=wanted))
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'upstream/Search-R1'))
    from scoring import verify_scoring
    config['_scoring_proof']=verify_scoring(config)
    core = importlib.import_module('search_e1.core')
    if Path(core.__file__).resolve() != package / 'core.py':
        raise ValueError('Wrong scientific core imported')
    return core



def atomic_checkpoint_directory(pending, destination, manifest, atomic_json):
    """Publish no final directory until its complete manifest is durable inside it."""
    pending,destination=Path(pending),Path(destination)
    if destination.exists(): raise ValueError('Committed destination exists')
    atomic_json(pending/'manifest.json',manifest)
    fsync_directory(pending/'adapter'); fsync_directory(pending)
    os.rename(pending,destination); fsync_directory(destination.parent)


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'outer-manifest', 'outer-manifest-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume-checkpoint'); p.add_argument('--resume-checkpoint-sha256')
    p.add_argument('--retry-capsule-manifest'); p.add_argument('--retry-capsule-sha256')
    a = p.parse_args()
    from checkpoint import checked_json, checked_file, fileproof, atomic_json, check_checkpoint, prune_checkpoints
    config, config_proof = checked_json(dict(path=a.config, sha256=a.config_sha256))
    config['_config_sha256'] = config_proof['sha256']
    if config['schema'] != 'opsd-ablation-config-v1': raise ValueError('Wrong config schema')
    science = validate_science(config)
    outer, outer_proof = checked_json(dict(path=a.outer_manifest, sha256=a.outer_manifest_sha256))
    if outer['schema'] != 'opsd-online-outer-manifest-v1' or outer['config_sha256'] != a.config_sha256:
        raise ValueError('Wrong outer/config identity')
    run_id = config['run_id']; outer_index = outer['outer_index']
    if outer['run_id'] != run_id or os.environ.get('E1_RUN_ID') != run_id or os.getuid() != 1005:
        raise ValueError('Private new-run identity / UID1005 required')
    if not 0 <= outer_index < science['outer_iterations']:
        raise ValueError('Outer index exceeds authorized budget')
    world = int(os.environ.get('WORLD_SIZE', '0')); rank = int(os.environ.get('RANK', '-1'))
    if world != 8 or not 0 <= rank < world: raise ValueError('Eight physical ranks required')
    for name in ('questions', 'rollouts', 'pairs'): checked_file(outer[name])
    questions = read_jsonl(outer['questions']['path']); rollouts = read_jsonl(outer['rollouts']['path'])
    if len(questions) != 512 or len({x['question_id'] for x in questions}) != 512 or len(rollouts) != 2560:
        raise ValueError('Full unique512 x5 cohort required')
    grouped = {}
    for row in rollouts: grouped.setdefault(row['question_id'], []).append(row)
    if set(grouped) != {x['question_id'] for x in questions} or any(len(v)!=5 or {r['sibling'] for r in v}!=set(range(5)) for v in grouped.values()):
        raise ValueError('Missing, duplicate or wrong-question siblings')
    core = use_frozen_runtime(config, checked_file)
    pairs = read_jsonl(outer['pairs']['path'])
    verify_pairs(rollouts,pairs,config['variant'],outer_index,core)
    if len(pairs)>512 or len({x['question_id'] for x in pairs})!=len(pairs):
        raise ValueError('Pairs missing or duplicated')
    checked_file(config['source_model_manifest'])
    prior_checkpoint = check_checkpoint(outer['previous_complete_checkpoint'],config,run_id)[0] if outer.get('previous_complete_checkpoint') else None
    verify_sampling_identity(outer,config,prior_checkpoint)
    import numpy as np
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    import transformers, peft
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    if (torch.__version__.split('+')[0],transformers.__version__,peft.__version__) != ('2.4.0','4.46.3','0.13.2'):
        raise ValueError('Frozen native Torch2.4.0/Transformers4.46.3/PEFT0.13.2 required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(hours=24))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    def reject(failed, message):
        flag = torch.tensor(int(failed), device=device, dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item(): raise FloatingPointError(message)
    def gather(value):
        result = [None]*world; dist.all_gather_object(result, value); return result
    def collectively(operation, label, rank_zero=False):
        # CPU/artifact failures rendezvous before any rank enters the next collective.
        result=None; error=None
        try:
            if not rank_zero or rank==0: result=operation()
        except Exception as failure: error=repr(failure)
        errors=gather(error)
        if any(errors): raise RuntimeError(label+': '+str(errors))
        return result

    source_started=time.time()
    source_proofs=collectively(lambda:verify_source100(config,checked_json,checked_file),'Fresh actual source100 twelve-file SHA',rank_zero=True)
    source_proofs=gather(source_proofs)[0]
    collectively(lambda:verify_source_metadata(source_proofs),'Source100 immediately before model load')
    random.seed(science['seed']); np.random.seed(science['seed']); torch.manual_seed(science['seed']); torch.cuda.manual_seed_all(science['seed'])
    tokenizer = AutoTokenizer.from_pretrained(config['paths']['model'], use_fast=True)
    precision = PrecisionContract(config['precision']['mode'])
    base = AutoModelForCausalLM.from_pretrained(config['paths']['model'],
        torch_dtype=torch.float32 if precision.mode=='fp32_master_fp16' else torch.bfloat16,
        attn_implementation='sdpa').to(device)
    collectively(lambda:verify_source_metadata(source_proofs),'Source100 immediately after model load')
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM'))
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
        if hasattr(module, 'attention_dropout') and module.attention_dropout != 0:
            raise ValueError('Nonzero functional attention dropout is unapproved')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
    trainable = [x for x in model.parameters() if x.requires_grad]
    opt = science['optimizer']
    optimizer = torch.optim.AdamW(trainable, lr=science['learning_rate'], betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'])
    previous_ref = outer.get('previous_complete_checkpoint')
    if bool(a.resume_checkpoint) != bool(a.resume_checkpoint_sha256): raise ValueError('Resume needs checkpoint plus SHA')
    resume_ref = dict(path=a.resume_checkpoint, sha256=a.resume_checkpoint_sha256) if a.resume_checkpoint else previous_ref
    resumed = None; updates = 0; exposure = 0
    if resume_ref:
        resumed = collectively(lambda:check_checkpoint(resume_ref,config,run_id)[0],'Complete resume checkpoint SHA and identity')
        def restore_accepted_rank():
            from safetensors.torch import load_file
            adapter = load_file(str(Path(resumed['adapter']['path'])/'adapter_model.safetensors'), device='cpu')
            set_peft_model_state_dict(model, adapter)
            state = torch.load({x['rank']:x['file']['path'] for x in resumed['ranks']}[rank], map_location='cpu', weights_only=False)
            from state_storage import hydrate_rank_optimizer
            state=hydrate_rank_optimizer(state,resumed,
                lambda path:torch.load(path,map_location='cpu',weights_only=False),optimizer_tensor_digest)
            verify_rank_state(state,resumed,rank,config)
            optimizer.load_state_dict(state['optimizer'])
            verify_rank_state(state,resumed,rank,config,optimizer.state_dict())
            updates, exposure = resumed['optimizer_updates'], resumed['cumulative_pair_exposure']
            for group in optimizer.param_groups:
                if (group['lr']!=science['learning_rate'] or tuple(group['betas'])!=tuple(opt['betas']) or group['eps']!=opt['eps'] or group['weight_decay']!=opt['weight_decay']):
                    raise ValueError('Restored actual Adam groups differ from authorized science')
            if tensor_digest(get_peft_model_state_dict(model)) != resumed['all_rank_adapter_sha256'][rank]:
                raise ValueError('Adapter state did not restore exactly')
            if optimizer_tensor_digest(optimizer.state_dict()) != state['optimizer_tensor_sha256']:
                raise ValueError('Adam moments/groups did not restore exactly')
            restore_rng(torch, np, state['rng'])
            del state,adapter
            return updates,exposure
        updates,exposure=collectively(restore_accepted_rank,'Actual accepted rank Adam/adapter/groups/RNG restoration')
    else:
        from safetensors.torch import load_file
        initial, _ = checked_json(config['common_initial_adapter_manifest'])
        collectively(lambda:[checked_file(f) for f in initial['payload_files']], 'Common initial payload SHA', rank_zero=True)
        initial_weights=collectively(lambda:load_file(initial['adapter_model']['path'],device='cpu'),'Load common initial tensor file')
        collectively(lambda:set_peft_model_state_dict(model,initial_weights),'Load actual shared initial504 LoRA')
        reject(tensor_digest(get_peft_model_state_dict(model)) != config['common_initial_adapter_tensor_sha256'],
               'Fresh initial504 LoRA tensors differ from common immutable template')
        del initial_weights
        adapter=get_peft_model_state_dict(model)
        if not adapter or not all(torch.isfinite(t).all().item() for t in adapter.values()) or not all(torch.count_nonzero(t).item()==0 for name,t in adapter.items() if 'lora_B' in name):
            raise ValueError('Fresh LoRA is not finite zero-effect source100')
        if not any('lora_B' in name for name in adapter): raise ValueError('Missing fresh zero-effect B tensors')
        del adapter
    # The old adapter is the original outer-start state, never the resumed mid-outer state.
    old_ref = verify_outer_old(outer,config,resumed)
    if not old_ref: raise ValueError('Explicit outer-old adapter manifest required')
    old_value, _ = checked_json(old_ref)
    if 'adapter' in old_value:
        old_file = next(f for f in old_value['adapter']['files'] if Path(f['path']).name=='adapter_model.safetensors')
        old_digest = old_value['all_rank_adapter_sha256'][0]
    else:
        old_file = old_value['adapter_model']; old_digest = old_value['adapter_tensor_sha256']
    collectively(lambda:checked_file(old_file),'Outer-old payload SHA',rank_zero=True)
    from safetensors.torch import load_file
    old_weights=collectively(lambda:load_file(old_file['path'],device='cpu'),'Load exact outer-old tensor file')
    def load_outer_old():
        model.add_adapter('outer_old',LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,
            target_modules=science['target_modules'],task_type='CAUSAL_LM'))
        # Preserve the existing C1 FP32 named-LoRA storage when adding the frozen old adapter.
        for name,parameter in model.named_parameters():
            if '.outer_old.' in name:parameter.data=parameter.data.to(dtype=torch.float32)
        set_peft_model_state_dict(model,old_weights,adapter_name='outer_old')
        precision.install(model)
        return SharedTargets(model,precision,old_adapter='outer_old',
            old_identity=dict(adapter_tensor_sha256=old_digest,manifest=old_ref))
    targets=collectively(load_outer_old,'Load/check exact frozen outer-old and install precision')
    del old_weights
    reject([id(x) for x in model.parameters() if x.requires_grad] != [id(x) for x in trainable],
           'Old adapter leaked into optimizer trainable inventory')
    scaler=SyncedGradScaler(enabled=precision.mode=='fp32_master_fp16')
    if resumed:
        def restore_after_targets():
            scaler.load_state_dict(resumed['all_rank_scaler_state'][rank])
            restored=torch.load({x['rank']:x['file']['path'] for x in resumed['ranks']}[rank],map_location='cpu',weights_only=False)
            restore_rng(torch,np,restored['rng'])
        collectively(restore_after_targets,'Restore accepted scaler/RNG after named-adapter setup')
    else:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    if not resume_ref and (not outer.get('explicit_fresh') or outer_index!=0 or outer['cumulative_optimizer_updates_before']!=0):
        raise ValueError('Fresh LoRA/Adam allowed only explicit first outer')
    if previous_ref and not a.resume_checkpoint and updates != outer['cumulative_optimizer_updates_before']:
        raise ValueError('Adam reset or stale prior outer state')
    if resumed and resumed['cursor']['outer_index'] not in (outer_index-1,outer_index):
        raise ValueError('Resume outer checkpoint is stale or from a future outer')
    output = Path(config['paths']['output_root']).resolve()
    if Path(outer['outer_dir']).resolve()!=output/'outers'/('outer-%05d'%outer_index):
        raise ValueError('Outer output root differs from immutable manifest')
    outer_dir = output/'outers'/('outer-%05d'%outer_index)
    training = outer_dir/'training'; training.mkdir(parents=True, exist_ok=True)
    usable, skipped = [], {}
    for pair in pairs:
        item, reason = encoded_pair(tokenizer, pair, core, 8192)
        if item is None: skipped[reason] = skipped.get(reason, 0)+1
        else:
            item['pair_weight']=pair['pair_weight']; item['mining']=pair['mining']; usable.append(item)
    K = len(usable); encoding = [x['sha256'] for x in usable]
    encoding_value = dict(sha256=canonical_sha(encoding), rows=encoding, P=len(pairs), K=K, skipped=skipped)
    encoding_path = training/'encoding.json'
    same_outer = resumed is not None and resumed['cursor']['outer_index']==outer_index
    if same_outer:
        order = json.loads(Path(resumed['training_input']['order']['path']).read_text())['pair_indices']
    else:
        if resumed and resumed['cursor']['next_pair_cursor']!=resumed['cursor']['usable_pair_count']:
            raise ValueError('Cannot advance outer before previous outer completed')
        order = list(range(K)); random.shuffle(order)
    if sorted(order)!=list(range(K)): raise ValueError('Shuffled pair order missing/duplicate')
    order_value = dict(pair_indices=order, outer_index=outer_index, encoded_sha256=encoding_value['sha256'])
    order_path=training/'order.json'
    def persist_encoding_order():
        for path,value in [(encoding_path,encoding_value),(order_path,order_value)]:
            if path.exists():
                if json.loads(path.read_text())!=value: raise ValueError('Existing outer data/order changed')
            else: atomic_json(path,value)
    collectively(persist_encoding_order,'Data/order persistence',rank_zero=True)
    order_proof=fileproof(order_path); encoding_proof=fileproof(encoding_path)
    cursor = verify_resume_cursor(resumed,outer_index,order_proof['sha256'],encoding_value['sha256'],K) if same_outer else 0
    if bool(a.retry_capsule_manifest)!=bool(a.retry_capsule_sha256):
        raise ValueError('Explicit retry manifest needs path and SHA')
    if a.retry_capsule_manifest:
        if not same_outer: raise ValueError('Overflow state can only restore the same accepted outer/cursor')
        from retry_state import load_rank_capsule
        capsule=collectively(lambda:load_rank_capsule(
            dict(path=a.retry_capsule_manifest,sha256=a.retry_capsule_sha256),resumed,resume_ref,config,rank,
            tensor_digest(get_peft_model_state_dict(model)),optimizer_tensor_digest(optimizer.state_dict())),
            'Actual eight-rank overflow capsule identity/Adam/adapter readback')
        collectively(lambda:validate_rng(capsule['rng']),'Complete actual retry RNG')
        def restore_retry():
            scaler.load_state_dict(capsule['step']['scaler_after']); restore_rng(torch,np,capsule['rng'])
        collectively(restore_retry,'Restore synchronized retry scale and actual CUDA RNG'); del capsule
    if same_outer and resumed['training_input']['outer_manifest']['sha256']!=outer_proof['sha256']:
        raise ValueError('Same-outer immutable manifest changed during resume')
    attempt=gather(uuid.uuid4().hex)[0]; attempt_dir=training/('attempt-'+attempt); (attempt_dir/('rank-%d'%rank)).mkdir(parents=True,exist_ok=True)
    outer_receipts=list(resumed.get('outer_update_receipts',[])) if same_outer else []
    source_load_path=attempt_dir/'source100-load-proof.json'
    source_load_evidence=dict(schema='opsd-online-source100-fresh-load-v1',run_id=run_id,outer_index=outer_index,
        source_model_identity=config['source_model_manifest']['sha256'],started_unix=source_started,ended_unix=time.time(),
        actual_files=source_proofs,strict_scorer=config['_scoring_proof'],
        scope='Fresh CPU full-file SHA once by rank0, every rank metadata before and after actual configured base/master load; exact305 QA import independently bound')
    collectively(lambda:atomic_json(source_load_path,source_load_evidence),'Persist fresh teacher load proof',rank_zero=True)
    source_load_proof=fileproof(source_load_path)
    outer_before = outer['cumulative_optimizer_updates_before']
    cumulative_valid_tokens=verify_committed_receipts(outer_receipts,resumed,outer_before,checked_json,checked_file,order) if same_outer else 0
    prior_exposure=prior_checkpoint['cumulative_pair_exposure'] if prior_checkpoint else 0
    if exposure!=prior_exposure+cursor:
        raise ValueError('Resume cumulative exposure differs from prior outer plus committed cursor')
    top_remaining=config['logging']['topk_max_tokens_per_outer']; top_by_pair={}
    for index in order:
        positions=[t for t,m in enumerate(usable[index]['mask']) if m]
        chosen=positions[:top_remaining]; top_remaining-=len(chosen); top_by_pair[index]=chosen
    # Before inspecting losses: one seed/hash selected eligible trajectory per group,
    # first two valid positions, only outer0. This cannot become an outcome-selected dump.
    vocab_pair=min(range(K),key=lambda i:hashlib.sha256(('vocab-v1:42:'+usable[i]['identity']['question_id']).encode()).hexdigest()) if False else None
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(cursor_now, outer_end, update_receipts):
        destination=output/'checkpoints'/('update-%08d-outer-%05d'%(updates,outer_index))
        pending=destination.with_name(destination.name+'.pending-'+attempt)
        def create_pending():
            if shutil.disk_usage(output).free < config['resources']['minimum_disk_bytes']:
                raise ValueError('Actual free disk below authorized checkpoint reserve; preserve latest complete state')
            if destination.exists(): raise ValueError('Refuse overwrite committed checkpoint; resume it explicitly')
            pending.mkdir(parents=True,exist_ok=False)
        collectively(create_pending,'Create pending checkpoint',rank_zero=True)
        def capture_actual_state():
            adapter_state=get_peft_model_state_dict(model);digest=tensor_digest(adapter_state)
            finite=all(torch.isfinite(t).all().item() for t in adapter_state.values())
            optimizer_state=state_to_cpu(optimizer.state_dict());odigest=optimizer_tensor_digest(optimizer_state)
            steps=actual_adam_state(optimizer_state,updates,len(trainable))
            return adapter_state,digest,finite,optimizer_state,odigest,steps,capture_rng(torch,np)
        adapter_state,digest,finite,optimizer_state,odigest,steps,rank_rng=collectively(capture_actual_state,'Capture actual adapter/Adam/RNG before checkpoint')
        reject(not finite, 'Nonfinite adapter after update; no committed checkpoint')
        current_cursor=dict(outer_index=outer_index,next_pair_cursor=cursor_now,usable_pair_count=K,order_sha256=order_proof['sha256'],encoded_sha256=encoding_value['sha256'])
        all_proofs=gather(dict(rank=rank,adapter_digest=digest,optimizer_digest=odigest,steps=steps,scaler=scaler.state_dict()))
        reject(len({x['adapter_digest'] for x in all_proofs})!=1 or len({x['optimizer_digest'] for x in all_proofs})!=1, 'Eight ranks differ in adapter or Adam tensors/groups')
        shared=[None]
        def persist_shared():
            from state_storage import optimizer_payload
            path=pending/'optimizer-shared.pt'
            save_torch(path,optimizer_payload(optimizer_state,all_proofs,run_id,a.config_sha256,updates,odigest),torch)
            shared[0]=fileproof(path);shared[0]['path']=str(destination/'optimizer-shared.pt')
        collectively(persist_shared,'Persist full Adam once after exact eight-rank equality',rank_zero=True)
        dist.broadcast_object_list(shared,src=0)
        state=dict(schema='opsd-online-rank-state-v1',rank=rank,run_id=run_id,config_sha256=a.config_sha256,
                   optimizer_storage='shared_exact_eight_rank_optimizer_v1',optimizer_ref=shared[0],optimizer_tensor_sha256=odigest,rng=rank_rng,
                   scaler=scaler.state_dict(),contract=science_contract(config),outer_old_adapter=old_ref,
                   optimizer_updates=updates,cumulative_pair_exposure=exposure,cursor=current_cursor,
                   source_model_identity=config['source_model_manifest']['sha256'],runtime_identity=config['runtime_manifest']['sha256'],
                   adapter_tensor_sha256=digest,training_input=dict(outer_manifest=outer_proof,pairs=outer['pairs'],order=order_proof,encoding=encoding_proof))
        collectively(lambda:save_torch(pending/('rank-%d-state.pt'%rank),state,torch),'Persist eight distinct full RNG/cursor states with exact complete Adam reference')
        def commit_checkpoint():
            model.save_pretrained(pending/'adapter',safe_serialization=True,selected_adapters=['default'])
            # Durable adapter/state files precede atomic directory commit.
            for path in (pending/'adapter').iterdir():
                if path.is_file():
                    with path.open('rb') as h: os.fsync(h.fileno())
            def final_proof(path):
                ref=fileproof(path); ref['path']=str(destination/path.relative_to(pending)); return ref
            reasons=[]
            if updates==0: reasons.append('initial')
            if updates in (1,5): reasons.append('update%d'%updates)
            if outer_end: reasons.append('outer_end')
            manifest=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=run_id,config_sha256=a.config_sha256,
                runtime_identity=config['runtime_manifest']['sha256'],source_model_identity=config['source_model_manifest']['sha256'],
                contract=science_contract(config),common_initial_adapter=config['common_initial_adapter_manifest'],
                outer_old_adapter=old_ref,all_rank_scaler_state=[x['scaler'] for x in all_proofs],
                optimizer_storage='shared_exact_eight_rank_optimizer_v1',shared_optimizer=shared[0],
                optimizer_updates=updates,cumulative_pair_exposure=exposure,complete_rank_count=8,equal_optimizer_steps=True,
                adapter=dict(path=str(destination/'adapter'),files=[final_proof(x) for x in sorted((pending/'adapter').iterdir()) if x.is_file()]),
                ranks=[dict(rank=r,file=final_proof(pending/('rank-%d-state.pt'%r))) for r in range(8)],
                all_rank_adapter_sha256=[x['adapter_digest'] for x in all_proofs],all_rank_optimizer_tensor_sha256=[x['optimizer_digest'] for x in all_proofs],
                all_rng=RNG_NAMES,scheduler=dict(kind='none',warmup_updates=0),cursor=current_cursor,
                training_input=state['training_input'],sampling_model_identity=outer['sampling_model_identity'],
                protection_reason=reasons,outer_update_receipts=update_receipts,
                actual_teacher_fixed=True,no_grpo_optimizer_restored=True,source100_fresh_load=source_load_proof)
            atomic_checkpoint_directory(pending,destination,manifest,atomic_json)
        collectively(commit_checkpoint,'Atomically commit complete checkpoint',rank_zero=True)
        ref=fileproof(destination/'manifest.json')
        collectively(lambda:check_checkpoint(ref,config,run_id,updates,outer_index),'Readback complete checkpoint',rank_zero=True)
        # Pruning is after durable commit and limited by the separately bound new-run policy.
        collectively(lambda:prune_checkpoints(output,run_id,a.config_sha256,ref),'Authorized rolling retention',rank_zero=True)
        return ref

    checkpoint_ref=resume_ref if same_outer else None
    if not same_outer:
        checkpoint_ref=save_checkpoint(0,K==0,outer_receipts)
    for batch in batches(order,cursor):
        retry=0
        while True:
            actual=len(batch); next_update=updates+1
            optimizer.zero_grad(set_to_none=True)
            local_failed=False; local_error=None; local_metrics=[]
            before_digest=tensor_digest(get_peft_model_state_dict(model))
            for index in rank_batch(batch,rank,world):
                item=usable[index]; log_path=attempt_dir/('rank-%d'%rank)/('update-%08d-retry-%04d-pair-%03d.jsonl'%(next_update,retry,index))
                try:
                    with log_path.open('x') as handle:
                        handle.write(json.dumps(dict(schema='opsd-online-training-token-header-v1',run_id=run_id,outer_index=outer_index,
                            optimizer_update=next_update,rank=rank,pair_index=index,encoded_sha256=item['sha256'],identity=item['identity'],
                            sampling_model_identity=outer['sampling_model_identity'],student_pre_update_adapter_sha256=before_digest,
                            frozen_teacher_identity=config['source_model_manifest']['sha256'],component_clip=science['component_clip'],
                            contract=science_contract(config),outer_old_adapter=old_ref,pair_weight=item['pair_weight'],mining=item['mining'],retry=retry,
                            token_ids=item['response'],policy_mask=item['mask'],token_offsets=item['offsets'],prefix_token_lengths=[len(x) for x in item['prefixes']],
                            loss_scope='actual_training_forward_before_optimizer',topk_cap=config['logging']['topk_max_tokens_per_outer'],
                            full_vocab_scope='preselected original-forward snapshots; see complete/cap_skip rows, other positions not requested',
                            full_vocab_study_cap_bytes=24*(1<<30)),ensure_ascii=False)+'\n')
                        record=lambda value:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
                        from vocabulary_dump import VocabularyDump
                        vocabulary_positions=top_by_pair[index][:2] if index==vocab_pair else []
                        if index==vocab_pair:
                            vocabulary_positions=[i for i,m in enumerate(item['mask']) if m][:2]
                        dump=VocabularyDump(config['logging']['full_vocab_study_root'],
                            attempt_dir/('rank-%d'%rank)/('vocabulary-update-%08d-retry-%04d-pair-%03d'%(next_update,retry,index)),
                            dict(run_id=run_id,variant=config['variant'],outer_index=outer_index,question_id=item['identity']['question_id'],
                                selection='seed42/qid SHA minimum eligible pair, first two valid positions, outer0',
                                current_adapter=before_digest,outer_old=old_ref),record) if vocabulary_positions else None
                        with targets.current_graph():
                            loss,metrics=pair_objective(model,item,science['component_clip'],32,actual,record,top_by_pair[index],tokenizer,
                                precision=precision,targets=targets,pair_weight=item['pair_weight'],use_mixture=config['variant'] in ('M','J'),
                                vocabulary_record=dump,vocabulary_positions=vocabulary_positions)
                            if not torch.isfinite(loss).all().item(): raise FloatingPointError('Nonfinite trajectory mean')
                            targets.mark_backward(); scaler.scale_loss(loss/actual).backward()
                        handle.flush(); os.fsync(handle.fileno())
                    local_metrics.append(dict(pair_index=index,token_log=fileproof(log_path),**metrics))
                except Exception as error:
                    local_failed=True; local_error=repr(error); break
            def reduce_max(value):
                t=torch.tensor(value,device=device,dtype=torch.int64)
                dist.all_reduce(t,op=dist.ReduceOp.MAX); return int(t.item())
            def agree_scale(value):
                if any(x!=value for x in gather(value)):raise ValueError('All-rank scaler state differs')
            step=scaler.finish_step(trainable,optimizer,
                lambda t:dist.all_reduce(t,op=dist.ReduceOp.SUM),reduce_max,agree_scale,local_failed)
            if not step['accepted']:
                capsule_dir=attempt_dir/('retry-update-%08d-attempt-%04d'%(next_update,retry))
                collectively(lambda:capsule_dir.mkdir(exist_ok=False),'Create durable retry capsule',rank_zero=True)
                def persist_rank_retry():
                    capsule=dict(run_id=run_id,contract=science_contract(config),outer_index=outer_index,
                        rank=rank,optimizer_updates=updates,cursor=cursor,order_sha256=order_proof['sha256'],
                        adapter_tensor_sha256=before_digest,outer_old_adapter=old_ref,step=step,
                        optimizer_tensor_sha256=optimizer_tensor_digest(optimizer.state_dict()),
                        rng=capture_rng(torch,np),attempt_metrics=local_metrics,error=local_error,
                        actual_adam_unchanged=True,committed_cursor_unchanged=True)
                    save_torch(capsule_dir/('rank-%d.pt'%rank),capsule,torch)
                collectively(persist_rank_retry,'Capture/persist eight actual retry RNG and optimizer identities')
                all_scalers=gather(step['scaler_after'])
                reject(any(x!=all_scalers[0] for x in all_scalers),'Retry scaler differs across ranks')
                def persist_retry_manifest():
                    from checkpoint import exclusive_json
                    exclusive_json(capsule_dir/'manifest.json',dict(schema='opsd-ablation-overflow-retry-v1',
                        accepted=True,run_id=run_id,config_sha256=a.config_sha256,contract=science_contract(config),
                        checkpoint=checkpoint_ref,cursor=json.loads(Path(checkpoint_ref['path']).read_text())['cursor'],
                        outer_old_adapter=old_ref,scaler_after=all_scalers[0],failure_kind=step['failure_kind'],
                        created_unix=time.time(),ranks=[dict(rank=r,file=fileproof(capsule_dir/('rank-%d.pt'%r))) for r in range(8)]))
                collectively(persist_retry_manifest,'Commit separate full-RNG overflow recovery manifest',rank_zero=True)
                if step['failure_kind']!='gradient_overflow' or not scaler.enabled:
                    raise FloatingPointError('Actual forward/logging failure preserved without Adam: '+str(gather(local_error)))
                retry+=1
                if retry>=16:raise FloatingPointError('Sixteen synchronized gradient overflows preserved; no fallback or Adam reset')
                continue
            norm=step['grad_norm']
            updates+=1; exposure+=actual; cursor+=actual
            torch.cuda.synchronize()
            actual_rank=collectively(lambda:runtime_rank_identity(torch,device,rank,config),'Actual rank physical GPU UUID identity')
            all_metrics=gather(dict(rank=rank,pairs=local_metrics,grad_norm=float(norm),actual=actual_rank,scaler=scaler.state_dict()))
            if sum(len(x['pairs']) for x in all_metrics)!=actual: raise ValueError('Missing real trajectory exposure')
            valid=sum(y['valid_tokens'] for x in all_metrics for y in x['pairs']); cumulative_valid_tokens+=valid
            reject(not all(torch.isfinite(x).all().item() for x in get_peft_model_state_dict(model).values()),'Nonfinite adapter; refuse checkpoint')
            reject(tensor_digest(get_peft_model_state_dict(model))==before_digest,
                   'Actual Adam did not change any current adapter tensor')
            update_path=attempt_dir/('update-%08d.json'%updates)
            def persist_update_receipt():
                receipt=dict(schema='opsd-online-real-update-v1',accepted=True,run_id=run_id,outer_index=outer_index,
                    optimizer_updates=updates,global_actual_pairs=actual,global_valid_tokens=valid,cumulative_pair_exposure=exposure,
                    pre_update_adapter_sha256=before_digest,post_update_adapter_sha256=tensor_digest(get_peft_model_state_dict(model)),
                    contract=science_contract(config),outer_old_adapter=old_ref,scaler=step,retries=retry,
                    learning_rate=optimizer.param_groups[0]['lr'],loss_reduction='trajectory_mean',micro_batch=1,
                    all_rank_actual_token_metrics=all_metrics,loss_scope='actual_training_forward_before_optimizer',
                    reference_student_prefix_shared=True,frozen_teacher=True,
                    durability='Committed only when referenced by an atomically complete checkpoint manifest')
                if update_path.exists(): raise ValueError('Refuse overwrite prior committed update receipt')
                atomic_json(update_path,receipt)
            collectively(persist_update_receipt,'Persist attempt-bound actual update receipt',rank_zero=True)
            outer_receipts.append(fileproof(update_path))
            checkpoint_ref=save_checkpoint(cursor,cursor==K,outer_receipts)
            def persist_live():
                atomic_json(training/'live-update.json',dict(run_id=run_id,outer_index=outer_index,optimizer_updates=updates,next_pair_cursor=cursor,K=K,checkpoint=checkpoint_ref))
                print('REAL_ONLINE_OPSD_UPDATE',json.dumps(dict(outer_index=outer_index,optimizer_updates=updates,actual_pairs=actual,valid_tokens=valid,checkpoint=checkpoint_ref)),flush=True)
            collectively(persist_live,'Persist latest live checkpoint pointer',rank_zero=True)
            break
    collectively(lambda:verify_source_metadata(source_proofs),'Source100 unchanged at outer training end')
    final_scalers=gather(scaler.state_dict())
    def persist_precision_proof():
        from checkpoint import exclusive_json
        exclusive_json(attempt_dir/('precision-rank-%d.json'%rank),dict(mode=precision.mode,
            events=precision.events,events_dropped=precision.events_dropped,
            dtype_counts=[dict(module=k[0],role=k[1],stage=k[2],input_dtypes=list(k[3]),output_dtypes=list(k[4]),count=v)
                for k,v in precision.dtype_counts.items()],
            scaler=scaler.state_dict(),optimizer_master_dtypes=sorted({str(x.dtype) for x in trainable}),
            actual_model_parameter_dtypes=sorted({str(x.dtype) for x in model.parameters()}),
            count_scope='All installed module observations counted; detailed events cap4096'))
    collectively(persist_precision_proof,'Persist actual precision/recomputation evidence')
    def persist_completion():
        completion=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=run_id,outer_index=outer_index,P=len(pairs),K=K,
            skipped=skipped,optimizer_updates_delta=math.ceil(K/32),optimizer_updates=updates,cumulative_pair_exposure=exposure,
            scheduler_advanced_updates=0,checkpoint=checkpoint_ref,empty_outer_no_adam=K==0,
            training_input=dict(outer_manifest=outer_proof,pairs=outer['pairs'],order=order_proof,encoding=encoding_proof),
            contract=science_contract(config),outer_old_adapter=old_ref,all_rank_scaler_state=final_scalers,
            global_valid_tokens=sum(sum(y['valid_tokens'] for x in json.loads(Path(ref['path']).read_text())['all_rank_actual_token_metrics'] for y in x['pairs']) for ref in outer_receipts),
            actual_optimizer_updates_since_this_invocation=updates-(resumed['optimizer_updates'] if resumed else 0),
            no_grpo_optimizer_restored=True,retention='GC controlled by separately verified new-run checkpoint helper')
        if updates!=outer_before+math.ceil(K/32): raise ValueError('Outer true update count/cursor mismatch')
        atomic_json(training/'outer-complete.json',completion)
    collectively(persist_completion,'Persist outer completion',rank_zero=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
