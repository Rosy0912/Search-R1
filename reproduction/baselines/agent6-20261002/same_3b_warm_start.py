"""Authorized training-only supervised initialization; actual RL count remains zero."""
import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(json.loads(os.environ['COSEARCH_RANKER_LEASE_UUIDS']))
import hashlib, time, gc
from pathlib import Path
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from search_r1.baselines.cosearch import GenerativeRanker, eval_decoder_checkpoint, check_candidate_context, answer_available
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.warm_start import SupervisedSequenceLoss, training_ranking
from search_r1.llm_agent.generation import LLMGenerationManager
from native_supervised_parity import run_native_supervised_parity

root = Path('/dataset/shaoshitong/search-r1-rl/cosearch')
run = Path(os.environ['COSEARCH_PROBE_RUN'])
destination = run / 'warm-start'
destination.mkdir(exist_ok=False)
started = time.perf_counter()
torch.set_num_threads(32)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
base = root / 'assets-root/assets/qwen2.5-3b'
native = run_native_supervised_parity([7,0,1,2,3,4,5,6])
(destination/'native-supervised-parity.json').write_text(json.dumps(native, indent=2))
torch.manual_seed(42)
cfg = OmegaConf.load(root / 'repo/verl/trainer/config/ppo_baselines.yaml').baselines.ranker
cfg.model_path = str(base)
cfg.device = 'cpu'; cfg.cpu_threads = 32; cfg.attn_implementation = 'sdpa'
cfg.phase_gpu = True; cfg.phase_gpu_devices = 8; cfg.phase_seed = 42
cfg.gradient_checkpointing = True; cfg.full_context_extrapolation = True
cfg.lr = 1e-5; cfg.weight_decay = 0.; cfg.keep = 5
ranker = GenerativeRanker(cfg)
tokenizer = ranker.tokenizer
def ids(text):
    return tokenizer.encode(text, add_special_tokens=False)
def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
data_root = root / 'assets-root/data/paper-grpo'
resolved = OmegaConf.load(run / 'pre-warm-resolved.yaml')
split = prepare_common_holdout(resolved)
retained_path = Path(resolved.data.train_files)
stage_path = Path(resolved.baselines.bapo.stage_validation_files)
train = pd.read_parquet(retained_path)
def content(row): return row['prompt'][0]['content']
excluded = set()
for path in (stage_path, data_root/'validation-64.parquet', data_root/'test.parquet'):
    frame = pd.read_parquet(path)
    excluded.update(content(row).strip() for _, row in frame.iterrows())
    del frame
# Only the retained common training split may supply supervised labels.
selection = []
seen = set()
ordered = sorted(range(len(train)), key=lambda i: hashlib.sha256(content(train.iloc[i]).encode()).hexdigest())
for i in ordered:
    row = train.iloc[i]; text = content(row).strip()
    aliases = list(row['reward_model']['ground_truth']['target'])
    if text in seen or text in excluded or not aliases or not aliases[0].strip(): continue
    selection.append((i, row, aliases)); seen.add(text)
    if len(selection) == 128: break
assert len(selection) == 128
assert not seen.intersection(excluded)
queries = [content(row).rsplit('Question:', 1)[1].strip() for _, row, _ in selection]
response = requests.post('http://127.0.0.1:18086/retrieve',
    json=dict(queries=queries, topk=50, return_scores=True), timeout=60)
response.raise_for_status(); candidates = response.json()['result']
assert len(candidates) == 128 and all(len(row) == 50 for row in candidates)
actor_examples = []; ranker_examples = []; records = []
for (index, row, aliases), query, docs in zip(selection, queries, candidates):
    chat = row['prompt'].tolist() if hasattr(row['prompt'], 'tolist') else row['prompt']
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False) if tokenizer.chat_template else chat[0]['content']
    prompt = ids(text)
    assert len(prompt) <= 2048, 'Do not silently crop warm-start initial prompts'
    action = f'<think>I need evidence to answer this question.</think><search>{query}</search>'
    action_ids = ids(action)
    ranking = training_ranking(docs, aliases, keep=5)
    passages = LLMGenerationManager._passages2string(None, [docs[i-1] for i in ranking])
    observation = f'\n\n<information>{passages.strip()}</information>\n\n'
    # Match the original actor's 500-token observation budget. Ranker full50 stays intact.
    observation_ids = ids(observation)[:500]
    next_prompt = prompt + action_ids + observation_ids
    assert len(next_prompt) <= 4096
    final = f'<think>I will provide a concise answer.</think><answer>{aliases[0]}</answer>'
    for prefix, target in ((prompt, action), (next_prompt, final)):
        target_ids = ids(target) + [tokenizer.eos_token_id]
        assert len(target_ids) <= 500
        actor_examples.append(dict(input_ids=prefix + target_ids, response_length=len(target_ids)))
    question = tokenizer.decode(prompt, skip_special_tokens=True)
    full_passages = '\n'.join(f'Doc {i+1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
    rank_prompt = (f'Question context:\n{question}\nSearch query: {query}\nCandidates:\n{full_passages}\n'
        'Select exactly 5 distinct document IDs ordered by relevance. '
        'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
    prefix = tokenizer(rank_prompt, return_token_type_ids=False)['input_ids']
    check_candidate_context([prefix], 256, 32768, [question], [query], [docs], True)
    target = f'<ranking>{json.dumps(ranking)}</ranking>'
    target_ids = ids(target) + [tokenizer.eos_token_id]
    assert len(target_ids) <= 256
    ranker_examples.append(dict(input_ids=prefix + target_ids, response_length=len(target_ids)))
    records.append(dict(train_row=index, source_index=int(row['extra_info']['index']), question=query, aliases=aliases, candidates=docs,
        full50_answer_available=answer_available([d['document']['contents'] for d in docs],aliases),
        selected5_answer_available=answer_available([docs[i-1]['document']['contents'] for i in ranking],aliases),
        visible500_answer_available=answer_available([tokenizer.decode(observation_ids,skip_special_tokens=True)],aliases),
        supervised_ranking=ranking, actor_observation_tokens=len(observation_ids), ranker_prompt_tokens=len(prefix)))
corpus = destination / 'training-only-corpus.json'
corpus.write_text(json.dumps(records, ensure_ascii=False))
budget = dict(acceptance_updates=0, objective='supervised response-only sequence-mean NLL',
    questions=128, actor_examples=256, ranker_examples=128, epochs=1, batch_size=8,
    lr=1e-5, weight_decay=0., parameters_dtype='float32', model_revision_unchanged=True,
    original_static_rope_unchanged=True, ranker_full50=True, keep=5,
    actor_observation_cap=500, gold_used_only_for_supervised_training_targets=True,
    train_sha256=file_hash(retained_path), corpus_sha256=file_hash(corpus),
    retained_train_path=str(retained_path), stage_holdout_path=str(stage_path),
    split_content_identity=split['content_identity']['sha256'], common_split_manifest=split,
    stage_periodic_and_final_questions_excluded=True, exact_joint_resume=False, roles={})
(destination/'manifest.json').write_text(json.dumps(budget, indent=2))
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
    with model.execution_phase(training=True):
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
            trace = dict(role=role, supervised_optimizer_steps=optimizer_steps, loss=float(loss.detach()),
                preclip_gradient_norm=float(norm), wall_seconds=time.perf_counter()-role_started, acceptance_updates=0)
            traces.append(trace); print('COSEARCH_WARM_START_PROGRESS_JSON='+json.dumps(trace), flush=True)
            del objective, loss, input_ids, masks, lengths
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
