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
format_cohort=native/'native-format-t1-next-cohort-v4.json'
assert file_hash(format_cohort)=='b672188acbd6abb4b772388cedb00e2dad8aa14de0ec5bce9d0d0358676afed6'
cohort=json.loads(format_cohort.read_text());assert len(cohort['questions'])==16 and cohort['created_before_v4_training'] and cohort['warm128_and_K16_excluded']
run=native/'runs/author-native-grammar-actor-continue-v4';run.mkdir(exist_ok=False)
destination=run/'warm-start';destination.mkdir();started=time.perf_counter()
(run/'slurm-allocation.txt').write_text(os.popen('scontrol show job '+os.environ['SLURM_JOB_ID']).read())
(run/'leases.json').write_text(json.dumps(leases,indent=2))
torch.set_num_threads(32);torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
parity=run_native_supervised_parity([7,0,1,2,3,4,5,6]);(destination/'native-supervised-parity.json').write_text(json.dumps(parity,indent=2));torch.manual_seed(42)
base=native/'runs/author-native-grammar-actor-continue-v3/warm-start/actor'
prior=json.loads((base.parent/'manifest.json').read_text())
assert prior['completed'] and prior['roles']['actor']['supervised_optimizer_steps']==32 and prior['cumulative_extra_sft_updates']==80
assert json.loads((base.parent.parent/'driver-exit.json').read_text())['returncode']==0
for name,expected in prior['roles']['actor']['files_sha256'].items():
 if name!='optimizer.pt':assert file_hash(base/name)==expected,(name,'source warm actor changed')
assert prior['ranker_initialization_identity']['supervised_optimizer_steps']==16
for name,expected in prior['ranker_initialization_identity']['files_sha256'].items():
 if name!='optimizer.pt':assert file_hash(native/'runs/author-native-grammar-warm-start-v2/warm-start/ranker'/name)==expected,(name,'unchanged warm ranker identity')
cfg=OmegaConf.load(root/'repo/verl/trainer/config/ppo_baselines.yaml').baselines.ranker
cfg.model_path=str(base);cfg.device='cpu';cfg.cpu_threads=32;cfg.attn_implementation='sdpa'
cfg.phase_gpu=True;cfg.phase_gpu_devices=8;cfg.phase_seed=42;cfg.gradient_checkpointing=True;cfg.full_context_extrapolation=True
cfg.lr=1e-5;cfg.weight_decay=0.;cfg.keep=5
ranker=GenerativeRanker(cfg);tokenizer=ranker.tokenizer
budget=dict(source_preparation=source_manifest,training_not_started=False,roles={},acceptance_updates=0,objective='supervised response-only sequence-mean NLL',
 lr=1e-5,weight_decay=0.,parameters_dtype='float32',source_weights='native-grammar-actor-continue-v3/actor; originalsame3Bbase ancestry'
 ,extra_actor_supervised_updates=32,prior_supervised_updates=dict(actor=64,ranker=16),
 ranker_unchanged_path=str(native/'runs/author-native-grammar-warm-start-v2/warm-start/ranker'),
 supervised_Adam='fresh extra SFT Adam; saved previous SFT Adam not loaded',
 prior_warm_manifest_sha256=file_hash(base.parent/'manifest.json'),
 ranker_initialization_identity=prior['ranker_initialization_identity'],cumulative_extra_sft_updates=112,
 same_static_rope=True,reuse_N_warm_weights=False,job=os.environ['SLURM_JOB_ID'],
 supervised_script_sha256=file_hash(Path(__file__)),fresh_RL_optimizers_required=True,next_format_cohort_path=str(format_cohort),next_format_cohort_sha256=file_hash(format_cohort),development_format_probe_K_already_used=True,post_epoch_required_stop='No automatic further training; check original T1/parser on new unused TRAIN development format questions, exclude K16 and warm128; no validation or final sets')
for role, examples in (('actor', actor_examples),):
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
        parameter_probe_delta_from_incoming_actor_l1=probe_delta, incoming_actor_path=str(base), phase_gpu_cost=model.cost)
    (destination/'manifest.json').write_text(json.dumps(budget, indent=2))
assert budget['roles']['actor']['supervised_optimizer_steps'] == 32
assert list(budget['roles'])==['actor'];budget['ranker_additional_supervised_updates']=0
budget['not_automatically_authorized_further_epochs']=True
budget.update(completed=True, wall_seconds=time.perf_counter()-started)
(destination/'manifest.json').write_text(json.dumps(budget, indent=2))
print('COSEARCH_WARM_START_COMPLETE_JSON='+json.dumps(budget), flush=True)
