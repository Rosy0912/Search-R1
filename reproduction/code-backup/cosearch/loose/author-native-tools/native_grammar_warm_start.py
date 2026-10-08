"""Separate approved native grammar supervised initialization, same3B;0 RL."""
import os,json
assert os.environ.get('SLURM_JOB_NODELIST')=='h100-16' and os.environ.get('SLURM_NNODES')=='1'
assert os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==8 and len({v['uuid'] for v in leases})==8
os.environ['CUDA_VISIBLE_DEVICES']=','.join(v['uuid'] for v in leases)
os.environ['COSEARCH_RANKER_LEASE_UUIDS']=json.dumps([v['uuid'] for v in leases])
os.environ['COSEARCH_RANKER_GPU_UUID']=leases[7]['uuid']
for key in ('ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):os.environ.pop(key,None)
import hashlib,time,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from search_r1.baselines.cosearch import GenerativeRanker,eval_decoder_checkpoint
from search_r1.baselines.warm_start import SupervisedSequenceLoss
from native_supervised_parity import run_native_supervised_parity
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');native=root/'author-native'
corpus_root=native/'native-grammar-warm-start-fulltext-v2'
source_manifest=json.loads((corpus_root/'manifest.json').read_text())
corpus_file=corpus_root/'corpus.json'
def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert file_hash(corpus_file)==source_manifest['corpus_sha256']
corpus=json.loads(corpus_file.read_text());actor_examples=corpus['actor'];ranker_examples=corpus['ranker']
assert len(actor_examples)==256 and len(ranker_examples)==128
run=native/'runs/author-native-grammar-warm-start-v2';run.mkdir(exist_ok=False)
destination=run/'warm-start';destination.mkdir();started=time.perf_counter()
(run/'slurm-allocation.txt').write_text(os.popen('scontrol show job '+os.environ['SLURM_JOB_ID']).read())
(run/'leases.json').write_text(json.dumps(leases,indent=2))
torch.set_num_threads(32);torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
parity=run_native_supervised_parity([7,0,1,2,3,4,5,6]);(destination/'native-supervised-parity.json').write_text(json.dumps(parity,indent=2));torch.manual_seed(42)
base=root/'assets-root/assets/qwen2.5-3b'
cfg=OmegaConf.load(root/'repo/verl/trainer/config/ppo_baselines.yaml').baselines.ranker
cfg.model_path=str(base);cfg.device='cpu';cfg.cpu_threads=32;cfg.attn_implementation='sdpa'
cfg.phase_gpu=True;cfg.phase_gpu_devices=8;cfg.phase_seed=42;cfg.gradient_checkpointing=True;cfg.full_context_extrapolation=True
cfg.lr=1e-5;cfg.weight_decay=0.;cfg.keep=5
ranker=GenerativeRanker(cfg);tokenizer=ranker.tokenizer
budget=dict(source_manifest,roles={},acceptance_updates=0,objective='supervised response-only sequence-mean NLL',
 lr=1e-5,weight_decay=0.,parameters_dtype='float32',source_weights='originalsame3Bbase',
 same_static_rope=True,reuse_N_warm_weights=False,job=os.environ['SLURM_JOB_ID'],
 supervised_script_sha256=file_hash(Path(__file__)),fresh_RL_optimizers_required=True)
for role, examples in (('actor', actor_examples), ('ranker', ranker_examples)):
    if role == 'actor': model = ranker
    else:
        del model, ranker; gc.collect()
        model = GenerativeRanker(cfg)
    initial_config = model.model.config.to_dict()
    initial_freq = model.model.model.rotary_emb.inv_freq.clone()
    initial_probes = {name: param.detach().reshape(-1)[:8].clone()
                      for name, param in model.model.named_parameters()}
    optimizer_steps = 0; tokens = 0; traces = []
    role_started = time.perf_counter()
    checkpoint_counts={'calls':0}
    with model.execution_phase(training=True):
        assert model.model.model.training and model.model.model.gradient_checkpointing
        original_checkpoint=model.model.model._gradient_checkpointing_func
        def observed_checkpoint(*args,**kwargs):
            checkpoint_counts['calls']+=1
            return original_checkpoint(*args,**kwargs)
        model.model.model._gradient_checkpointing_func=observed_checkpoint
        for offset in range(0, len(examples), 8):
            batch = examples[offset:offset+8]
            tensors = [torch.tensor(row['input_ids']) for row in batch]
            width = max(map(len, tensors)); lengths = torch.tensor([row['response_length'] for row in batch], device=model.device)
            input_ids = torch.stack([F.pad(t, (width-len(t), 0), value=tokenizer.pad_token_id) for t in tensors]).to(model.device)
            masks = torch.stack([F.pad(torch.ones_like(t), (width-len(t), 0)) for t in tensors]).to(model.device)
            model.optimizer.zero_grad(set_to_none=True)
            objective = torch.nn.DataParallel(SupervisedSequenceLoss(model.model),
                device_ids=model.phase_devices[:len(batch)], output_device=model.device.index)
            loss = objective(input_ids, masks, lengths, max_response_length=int(lengths.max())).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite supervised NLL')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.)
            if not torch.isfinite(norm): raise ValueError('Nonfinite supervised gradient')
            model.optimizer.step(); optimizer_steps += 1; tokens += sum(map(len, tensors))
            trace = dict(role=role, supervised_optimizer_steps=optimizer_steps, checkpoint_calls=checkpoint_counts['calls'], master_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device), loss=float(loss.detach()),
                preclip_gradient_norm=float(norm), wall_seconds=time.perf_counter()-role_started, acceptance_updates=0)
            traces.append(trace); print('COSEARCH_WARM_START_PROGRESS_JSON='+json.dumps(trace), flush=True)
            del objective, loss, input_ids, masks, lengths
    assert checkpoint_counts['calls']>0,'Real full model checkpointing did not execute'
    assert model.model.config.to_dict() == initial_config
    torch.testing.assert_close(model.model.model.rotary_emb.inv_freq.cpu(), initial_freq, atol=0, rtol=0)
    saved = destination / role; model.save(saved)
    probe_delta = sum(float((param.detach().reshape(-1)[:8]-initial_probes[name]).abs().sum())
                      for name,param in model.model.named_parameters())
    assert probe_delta > 0, 'Supervised parameters did not change'
    hashes = {p.name: file_hash(p) for p in saved.iterdir() if p.is_file()}
    budget['roles'][role] = dict(supervised_optimizer_steps=optimizer_steps, tokens=tokens, traces=traces,
        wall_seconds=time.perf_counter()-role_started, hf_path=str(saved), files_sha256=hashes,
        parameter_probe_delta_from_original_base_l1=probe_delta, phase_gpu_cost=model.cost)
    (destination/'manifest.json').write_text(json.dumps(budget, indent=2))
assert budget['roles']['actor']['supervised_optimizer_steps'] == 32
assert budget['roles']['ranker']['supervised_optimizer_steps'] == 16
budget.update(completed=True, wall_seconds=time.perf_counter()-started)
(destination/'manifest.json').write_text(json.dumps(budget, indent=2))
print('COSEARCH_WARM_START_COMPLETE_JSON='+json.dumps(budget), flush=True)
