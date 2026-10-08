"""Fresh post-GRPO rollouts and EM evaluation, using a fixed Search-R1 retriever."""
import argparse
import json
import re
from pathlib import Path
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from .core import prompt, reward
from .io import read_jsonl, write_json

class StopTags(StoppingCriteria):
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
        return '</search>' in text or '</answer>' in text

class SearchAgent:
    def __init__(self, model, tokenizer, url, max_new_tokens=500, max_context=8192):
        self.model, self.tokenizer, self.url = model, tokenizer, url
        self.max_new_tokens, self.max_context = max_new_tokens, max_context
    @torch.no_grad()
    def run(self, question, temperature=1.):
        prefix = self.tokenizer.apply_chat_template(prompt(question), tokenize=False, add_generation_prompt=True)
        self.model.eval()
        response, calls = '', 0
        policy_spans = []
        for turn in range(5):
            forced = '\n<information>Search budget exhausted. Give the final answer now inside <answer> and </answer>.</information>\n' if turn == 4 else ''
            text = prefix + response + forced
            ids = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(next(self.model.parameters()).device)
            if ids.shape[1] + self.max_new_tokens > self.max_context:
                return {'response': response, 'search_calls': calls, 'policy_spans': policy_spans, 'termination': 'context_limit'}
            options = dict(max_new_tokens=self.max_new_tokens, do_sample=temperature > 0,
                           pad_token_id=self.tokenizer.eos_token_id,
                           stopping_criteria=StoppingCriteriaList([StopTags(self.tokenizer, ids.shape[1])]))
            if temperature > 0:
                options.update(temperature=temperature, top_p=1.)
            output = self.model.generate(ids, **options)
            segment = self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            ends = [segment.find(tag) + len(tag) for tag in ('</search>', '</answer>') if tag in segment]
            if ends:
                segment = segment[:min(ends)]
            response += forced
            start = len(response)
            response += segment
            policy_spans.append((start, len(response)))
            if '</answer>' in segment:
                return {'response': response, 'search_calls': calls, 'policy_spans': policy_spans, 'termination': 'answer'}
            match = re.search(r'<search>(.*?)</search>', segment, re.S)
            if not match or calls == 4:
                return {'response': response, 'search_calls': calls, 'policy_spans': policy_spans, 'termination': 'invalid_or_budget'}
            res = requests.post(self.url, json={'queries': [match.group(1).strip()], 'topk': 3, 'return_scores': True}, timeout=120)
            res.raise_for_status()
            passages = res.json()['result'][0]
            formatted = []
            for i, passage in enumerate(passages):
                doc = passage.get('document', passage)
                contents = doc.get('contents', doc.get('text', ''))
                formatted.append(f'Doc {i+1}: {contents}')
            info = '\n'.join(formatted)
            # Match Search-R1's max_obs_length=500; always close the structural information tag.
            info_ids = self.tokenizer.encode(info, add_special_tokens=False)[:480]
            response += '\n<information>\n' + self.tokenizer.decode(info_ids) + '\n</information>\n'
            calls += 1
        return {'response': response, 'search_calls': calls, 'policy_spans': policy_spans, 'termination': 'budget'}

def evaluate(agent, rows):
    values = []
    for row in rows:
        result = agent.run(row['question'], temperature=0)
        values.append(reward(result['response'], row['answers'], result['policy_spans']))
    return sum(values) / len(values) if values else 0.

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--retriever-url', default='http://127.0.0.1:8000/retrieve')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--backend', choices=['hf','vllm'], default='hf')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rows = read_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    if a.workers < 1 or a.k < 1:
        raise ValueError('Workers and K must be positive')
    if a.workers > 1:
        from .parallel_rollout import run_parallel
        run_parallel(a, rows)
        return
    if a.backend == 'vllm':
        from .vllm_rollout import batched_rollouts
        results = batched_rollouts(a.model, rows, a.k, a.retriever_url, a.temperature, a.seed, a.batch_size)
    else:
        tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        agent = SearchAgent(model, tokenizer, a.retriever_url)
        def generate():
            for row in rows:
                for sibling in range(a.k):
                    result = dict(agent.run(row['question'], a.temperature), **row, sibling=sibling)
                    result['token_count'] = len(tokenizer.encode(result['response'], add_special_tokens=False))
                    yield result
        results = generate()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, 'x') as f:
        for result in results:
            result['reward'] = reward(result['response'], result['answers'], result['policy_spans'])
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({'question_id': result['question_id'], 'sibling':result['sibling']}), flush=True)
    records = read_jsonl(a.output)
    by_dataset = {}
    for r in records:
        by_dataset.setdefault(r.get('data_source', 'unknown'), []).append(r['reward'])
    metrics = {k: sum(v) / len(v) for k, v in by_dataset.items()}
    write_json(a.output + '.metrics.json', {'em': metrics, 'macro_em': sum(metrics.values())/len(metrics), 'count': len(records)})
if __name__ == '__main__':
    main()
