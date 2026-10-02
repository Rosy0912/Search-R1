"""CoSearch auxiliary generative ranker, with its own optimizer and checkpoint.

Portable first backend: an unsharded HF ranker on a dedicated device (CPU default).
This is a joint-system adaptation, not policy-only GRPO. No gold enters ranking.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import re
import torch
import torch.nn.functional as F
from .advantages import grouped_center
from .bapo import normalize, token_f1


def parse_ranking(text, candidate_count, keep):
    matches = list(re.finditer(r'<ranking>\s*(\[[^\]]*\])\s*</ranking>', text, re.DOTALL))
    if len(matches) != 1:
        return None
    try:
        ids = json.loads(matches[0].group(1))
    except (TypeError, ValueError):
        return None
    if not isinstance(ids, list) or len(ids) != keep or any(type(i) is not int for i in ids):
        return None
    if len(set(ids)) != len(ids) or any(i < 1 or i > candidate_count for i in ids):
        return None
    return ids


def contains_answer(document, answers):
    if isinstance(answers, str):
        answers = [answers]
    text = ' ' + normalize(document) + ' '
    return any(normalize(a) and (' ' + normalize(a) + ' ') in text for a in answers)


def ranker_reward(candidates, ranking, answers, main_reward, threshold=.5, format_penalty=-.2):
    if ranking is None:
        return format_penalty
    available = any(contains_answer(d, answers) for d in candidates)
    selected = [candidates[i - 1] for i in ranking]
    relevance = sum(any(contains_answer(d, answers) for d in selected[:k]) for k in (1, 3, 5)) / 3
    return relevance if available and relevance <= threshold else relevance + main_reward if available else main_reward


def semantic_groups(events, similarity=.8, split_availability=False):
    """Greedy token-F1 groups within the same initial question, never across tasks."""
    buckets = defaultdict(list)
    for i, e in enumerate(events):
        key = (e['question_uid'], e.get('answer_available') if split_availability else None)
        buckets[key].append(i)
    keys = [None] * len(events)
    group = 0
    for indices in buckets.values():
        representatives = []
        for i in indices:
            for j, key in representatives:
                if token_f1(events[i]['query'], events[j]['query']) >= similarity:
                    keys[i] = key
                    break
            else:
                keys[i] = group
                representatives.append((i, group))
                group += 1
    return keys


class GenerativeRanker:
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = torch.device(config.device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.float32,
                                                        attn_implementation='eager').to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.events = []
        self.cost = dict(generated_tokens=0, prefill_tokens=0, optimizer_steps=0, score_forward_tokens=0,
                         training_forward_tokens=0, validation_generated_tokens=0, validation_prefill_tokens=0)
        self.question_uids = []
        self.validation = False

    def rank(self, initial_question, query, docs, state):
        keep = min(self.config.keep, len(docs))
        if not keep:
            return []
        passages = '\n'.join(f'Doc {i + 1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
        prompt = (f'Question context:\n{initial_question}\nSearch query: {query}\nCandidates:\n{passages}\n'
                  f'Select exactly {keep} distinct document IDs ordered by relevance. '
                  'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        if inputs.input_ids.shape[1] + self.config.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Ranker candidate context exceeds model limit')
        self.model.eval()
        # Checkpoints may ship repetition_penalty=1.05 or forced tokens. A fresh
        # config keeps rollout exactly the temperature policy scored below.
        from transformers import GenerationConfig
        sampling = GenerationConfig(max_new_tokens=self.config.max_new_tokens,
                                    do_sample=not self.validation, temperature=self.config.temperature,
                                    top_p=1., top_k=0, num_beams=1,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.model.generation_config.eos_token_id,
                                    bos_token_id=self.model.generation_config.bos_token_id)
        with torch.no_grad():
            generated = self.model.generate(**inputs, generation_config=sampling)
            response = generated[:, inputs.input_ids.shape[1]:]
            # Same eval-mode distribution and temperature as optimization.
            logits = self.model(generated, use_cache=False).logits[:, inputs.input_ids.shape[1] - 1:-1]
            logp = F.log_softmax(logits.float() / self.config.temperature, -1).gather(-1, response[..., None]).squeeze(-1)
        ranking = parse_ranking(self.tokenizer.decode(response[0], skip_special_tokens=True), len(docs), keep)
        prefix = 'validation_' if self.validation else ''
        self.cost[prefix + 'prefill_tokens'] += inputs.input_ids.numel()
        self.cost[prefix + 'generated_tokens'] += response.numel()
        self.cost['score_forward_tokens'] += generated.numel()
        if not self.validation:
            self.events.append(dict(question_uid=self.question_uids[state.source], trajectory_uid=state.trajectory,
                                    query=query, candidates=[d['document']['contents'] for d in docs], ranking=ranking,
                                    input_ids=generated[0].cpu(), response_length=response.shape[1],
                                    old_logp=logp[0].cpu()))
        # Deterministic fallback keeps search operational; invalid action remains -1 ranker reward.
        selection = ranking if ranking is not None else list(range(1, keep + 1))
        return [docs[i - 1] for i in selection]

    def execute(self, manager, predictions, states, do_search):
        actions, contents = manager.postprocess_predictions(predictions)
        indices = [i for i, action in enumerate(actions) if action == 'search']
        docs = []
        if indices and do_search:
            import requests
            payload = dict(queries=[contents[i] for i in indices], topk=self.config.candidates, return_scores=True)
            response = requests.post(manager.config.search_url, json=payload, timeout=60)
            response.raise_for_status()
            docs = response.json()['result']
            if len(docs) != len(indices):
                raise ValueError('Retriever returned wrong candidate batch size')
        observation = [''] * len(states)
        done, valid, searches = [], [], []
        doc_index = 0
        for i, action in enumerate(actions):
            if action == 'search':
                if do_search:
                    initial = manager.tokenizer.decode(states[i].prompt, skip_special_tokens=True)
                    selected = self.rank(initial, contents[i], docs[doc_index], states[i])
                    doc_index += 1
                    text = manager._passages2string(selected)
                else:
                    text = ''
                observation[i] = f'\n\n<information>{text.strip()}</information>\n\n'
                done.append(0); valid.append(1); searches.append(1)
            elif action == 'answer':
                done.append(1); valid.append(1); searches.append(0)
            else:
                observation[i] = ('\nMy previous action is invalid. If I want to search, I should put the query '
                                  'between <search> and </search>. If I want to give the final answer, I should '
                                  'put the answer between <answer> and </answer>. Let me try again.\n')
                done.append(0); valid.append(0); searches.append(0)
        return observation, done, valid, searches

    def update(self, rewards_by_trajectory):
        events = []
        for e in self.events:
            main, targets, legal = rewards_by_trajectory[e['trajectory_uid']]
            if not legal:
                continue
            e['reward'] = ranker_reward(e['candidates'], e['ranking'], targets, main, self.config.threshold)
            e['answer_available'] = any(contains_answer(d, targets) for d in e['candidates'])
            events.append(e)
        self.events.clear()
        keys = semantic_groups(events, self.config.similarity, self.config.split_availability)
        counts = defaultdict(int)
        for key in keys:
            counts[key] += 1
        chosen = [i for i, key in enumerate(keys) if counts[key] >= self.config.min_group_size]
        values = torch.tensor([e['reward'] for e in events])
        advantage = grouped_center(values, keys) if events else values
        metrics = {'ranker/events': len(events), 'ranker/training_events': len(chosen),
                   'ranker/parameters': sum(p.numel() for p in self.model.parameters())}
        # Keep dropout disabled while taking gradients, matching rollout old logprobs.
        self.model.eval()
        for _ in range(self.config.epochs):
            for start in range(0, len(chosen), self.config.batch_size):
                rows = chosen[start:start + self.config.batch_size]
                total_tokens = sum(events[i]['response_length'] for i in rows)
                self.optimizer.zero_grad()
                for i in rows:
                    e = events[i]
                    ids = e['input_ids'].to(self.device)[None, :]
                    self.cost['training_forward_tokens'] += ids.numel()
                    length = e['response_length']
                    logits = self.model(ids, use_cache=False).logits[:, -length - 1:-1].float() / self.config.temperature
                    logp = F.log_softmax(logits, -1).gather(-1, ids[:, -length:, None]).squeeze(-1)
                    ratio = (logp - e['old_logp'].to(self.device)).exp()
                    clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    loss = -torch.minimum(ratio * advantage[i], clipped * advantage[i]).sum() / max(1, total_tokens)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.cost['optimizer_steps'] += 1
        metrics.update({f'ranker/{k}_cumulative': v for k, v in self.cost.items()})
        return metrics

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        torch.save(self.optimizer.state_dict(), os.path.join(directory, 'optimizer.pt'))
