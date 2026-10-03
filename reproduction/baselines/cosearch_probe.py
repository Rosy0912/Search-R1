"""Standalone real-model CoSearch capability probe; not an actor training smoke."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import requests
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer
from search_r1.baselines.cosearch import GenerativeRanker

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--train', required=True)
parser.add_argument('--url', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--questions', type=int, default=3)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
args = parser.parse_args()
if min(args.threads, args.questions, args.repeats) < 1:
    parser.error('threads, questions and repeats must be positive')
torch.set_num_threads(args.threads)
torch.manual_seed(42)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
report = dict(scope='standalone rank generation only; no actor training, no fabricated reward',
              model=args.model, train=args.train, seed=42, threads=args.threads, attention=args.attention,
              candidates=50, keep=3, max_new_tokens=256, queries=[], generations=[])

def save():
    divisor = 1024 ** 2 if sys.platform == 'darwin' else 1024
    report['process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

original_excepthook = sys.excepthook
def record_failure(kind, error, traceback):
    report.update(status='failed', error_type=kind.__name__, error_message=str(error))
    save()
    original_excepthook(kind, error, traceback)
sys.excepthook = record_failure

tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
limit = AutoConfig.from_pretrained(args.model, local_files_only=True).max_position_embeddings
report['model_context_limit'] = limit
data = pd.read_parquet(args.train).head(args.questions)
if data.empty:
    raise ValueError('Cannot probe an empty training dataset')
inputs = []
for _, row in data.iterrows():
    chat = row['prompt'].tolist() if hasattr(row['prompt'], 'tolist') else row['prompt']
    initial = (tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
               if tokenizer.chat_template else chat[0]['content'])
    initial_ids = tokenizer(initial, add_special_tokens=False)['input_ids']
    if len(initial_ids) > 4096:
        raise ValueError('Training prompt exceeds the fixed 4096-token dataset budget')
    # The production ranker decodes SearchRollout's unpadded max_start_length prompt.
    initial = tokenizer.decode(initial_ids[-2048:], skip_special_tokens=True)
    extra = row.get('extra_info', {})
    question = extra.get('question') if isinstance(extra, dict) else None
    if not question:
        content = chat[-1]['content']
        if 'Question: ' not in content:
            raise ValueError('Cannot identify question in the fixed Search-R1 prompt')
        question = content.rsplit('Question: ', 1)[1].strip()
    query = question
    started = time.perf_counter()
    result = requests.post(args.url, json=dict(queries=[query], topk=50, return_scores=True),
                           timeout=(5, 60))
    result.raise_for_status()
    batch = result.json()['result']
    if len(batch) != 1 or len(batch[0]) != 50:
        raise ValueError('Real retriever must return exactly 50 candidates per query')
    docs = batch[0]
    passages = '\n'.join(f'Doc {i + 1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
    prompt = (f'Question context:\n{initial}\nSearch query: {query}\nCandidates:\n{passages}\n'
              'Select exactly 3 distinct document IDs ordered by relevance. '
              'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
    n = len(tokenizer(prompt)['input_ids'])
    report['queries'].append(dict(query=query, candidate_count=50, prompt_tokens=n,
                                  retrieval_wall_seconds=time.perf_counter() - started,
                                  context_fits=n + 256 <= limit))
    inputs.append((initial, query, docs))
    save()
if not all(q['context_fits'] for q in report['queries']):
    report['status'] = 'context_overflow_user_choice_required_no_truncation_applied'
    save()
    raise SystemExit(2)

config = OmegaConf.create(dict(model_path=args.model, device='cpu', attn_implementation=args.attention, candidates=50, keep=3,
                              max_new_tokens=256, temperature=1., lr=1e-6,
                              weight_decay=.01, clip_ratio=.2, grad_clip=1., epochs=1,
                              batch_size=8, similarity=.8, min_group_size=3,
                              threshold=.5, split_availability=False))
started, cpu_started = time.perf_counter(), time.process_time()
ranker = GenerativeRanker(config)
report['load_wall_seconds'] = time.perf_counter() - started
report['load_cpu_seconds'] = time.process_time() - cpu_started
ranker.question_uids = [str(i) for i in range(len(inputs))]
save()
for i, (initial, query, docs) in enumerate(inputs):
    for repeat in range(args.repeats):
        started, cpu_started = time.perf_counter(), time.process_time()
        ranker.rank(initial, query, docs, SimpleNamespace(source=i, trajectory=f'{i}-{repeat}'))
        event = ranker.events[-1]
        report['generations'].append(dict(question=i, repeat=repeat, ranking=event['ranking'],
             legal=event['ranking'] is not None, response_tokens=event['response_length'],
             response=ranker.tokenizer.decode(event['input_ids'][-event['response_length']:],
                                              skip_special_tokens=True),
             wall_seconds=time.perf_counter() - started,
             cpu_seconds=time.process_time() - cpu_started,
             old_logprob_finite=bool(torch.isfinite(event['old_logp']).all())))
        report['cost'] = ranker.cost
        report['legal_count'] = sum(g['legal'] for g in report['generations'])
        report['status'] = 'rank_generation_in_progress'
        save()
report['status'] = 'standalone_generation_complete_joint_smoke_still_required'
save()
