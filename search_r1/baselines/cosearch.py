"""CoSearch auxiliary generative ranker, with its own optimizer and checkpoint.

Portable first backend: an unsharded HF ranker on a dedicated device (CPU default).
This is a joint-system adaptation, not policy-only GRPO. No gold enters ranking.
"""
from collections import Counter, defaultdict
import json
import os
import re
import unicodedata
import resource
import sys
import time
import regex
import torch
import torch.nn.functional as F
from .advantages import grouped_center


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


# Match the pinned author's SimpleTokenizer: punctuation and articles remain tokens.
_DOCUMENT_TOKENS = regex.compile(r"([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])",
                                 flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE)
_TRIVIAL_ANSWERS = frozenset(('yes', 'no', 'true', 'false'))


def contains_answer(document, answers):
    if isinstance(answers, str):
        answers = [answers]
    tokens = [m.group().lower() for m in _DOCUMENT_TOKENS.finditer(unicodedata.normalize('NFD', document))]
    for answer in answers:
        target = [m.group().lower() for m in _DOCUMENT_TOKENS.finditer(unicodedata.normalize('NFD', answer))]
        if target and any(tokens[i:i + len(target)] == target for i in range(len(tokens) - len(target) + 1)):
            return True
    return False


def answers_are_trivial(answers):
    if isinstance(answers, str):
        answers = [answers]
    return bool(answers) and all(a.lower().strip() in _TRIVIAL_ANSWERS for a in answers)


def answer_available(candidates, answers):
    return not answers_are_trivial(answers) and any(contains_answer(d, answers) for d in candidates)


def ranker_reward(candidates, ranking, answers, main_reward, threshold=.5, format_penalty=-.2):
    if ranking is None:
        return format_penalty
    available = answer_available(candidates, answers)
    selected = [candidates[i - 1] for i in ranking]
    relevance = (sum(any(contains_answer(d, answers) for d in selected[:k]) for k in (1, 3, 5)) / 3
                 if available else 0.)
    return relevance if available and relevance <= threshold else relevance + main_reward if available else main_reward


def normalize_query(text):
    text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def query_f1(left, right):
    left, right = normalize_query(left).split(), normalize_query(right).split()
    if not left or not right:
        return 0.
    return 2 * sum((Counter(left) & Counter(right)).values()) / (len(left) + len(right))


def sequence_ppo_loss(logp, old_logp, advantage, clip_ratio):
    """GRPO objective (Eq.3): each sampled ranking sequence has equal weight irrespective of length."""
    if not torch.isfinite(logp).all() or not torch.isfinite(old_logp).all():
        raise ValueError('Non-finite ranker log probabilities')
    ratio = (logp - old_logp).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Non-finite ranker importance ratios')
    return -torch.minimum(ratio * advantage,
                          ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).mean()


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
                if query_f1(events[i]['query'], events[j]['query']) >= similarity:
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
        started, cpu_started = time.perf_counter(), time.process_time()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = torch.device(config.device)
        attention = config.get('attn_implementation', 'eager')
        if attention not in ('eager', 'sdpa'):
            raise ValueError('Ranker attention must be eager or sdpa')
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.float32,
                                                        attn_implementation=attention).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr,
                                           weight_decay=config.get('weight_decay', .01))
        self.events = []
        self.cost = dict(generated_tokens=0, prefill_tokens=0, optimizer_steps=0, score_forward_tokens=0,
                         training_forward_tokens=0, validation_generated_tokens=0, validation_prefill_tokens=0,
                         model_load_wall_seconds=time.perf_counter() - started,
                         model_load_cpu_seconds=time.process_time() - cpu_started)
        self.question_uids = []
        self.validation = False

    def cost_metrics(self):
        # Driver RSS includes dependencies and trainer state, not only this ranker.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 ** 2 if sys.platform == 'darwin' else 1024
        metrics = {f'ranker/{k}_cumulative': v for k, v in self.cost.items()}
        metrics['ranker/driver_peak_rss_mib'] = rss / divisor
        return metrics

    def rank(self, initial_question, query, docs, state):
        started, cpu_started = time.perf_counter(), time.process_time()
        keep = min(self.config.keep, len(docs))
        if not keep:
            return []
        passages = '\n'.join(f'Doc {i + 1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
        prompt = (f'Question context:\n{initial_question}\nSearch query: {query}\nCandidates:\n{passages}\n'
                  f'Select exactly {keep} distinct document IDs ordered by relevance. '
                  'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
        inputs = self.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(self.device)
        prefix = 'validation_' if self.validation else ''
        self.cost[prefix + 'prompt_tokens_max'] = max(self.cost.get(prefix + 'prompt_tokens_max', 0),
                                                     inputs.input_ids.numel())
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
        for name, value in [('rank_calls', 1), ('legal_rankings', int(ranking is not None)),
                            ('fallback_rankings', int(ranking is None)),
                            ('rank_wall_seconds', time.perf_counter() - started),
                            ('rank_cpu_seconds', time.process_time() - cpu_started)]:
            self.cost[prefix + name] = self.cost.get(prefix + name, 0) + value
        self.cost[prefix + 'prefill_tokens'] += inputs.input_ids.numel()
        self.cost[prefix + 'generated_tokens'] += response.numel()
        self.cost.setdefault(prefix + 'score_forward_tokens', 0)
        self.cost[prefix + 'score_forward_tokens'] += generated.numel()
        if not self.validation:
            self.events.append(dict(question_uid=self.question_uids[state.source], trajectory_uid=state.trajectory,
                                    query=query, candidates=[d['document']['contents'] for d in docs], ranking=ranking,
                                    input_ids=generated[0].cpu(), response_length=response.shape[1],
                                    old_logp=logp[0].cpu()))
        # Deterministic fallback keeps search operational; invalid action earns -.2 ranker reward.
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
            for row in docs:
                if not isinstance(row, list) or len(row) != self.config.candidates:
                    raise ValueError(f'CoSearch requires exactly {self.config.candidates} candidates per query')
                if any(not isinstance(d, dict) or not isinstance(d.get('document'), dict)
                       or not isinstance(d['document'].get('contents'), str) for d in row):
                    raise ValueError('CoSearch candidate lacks document.contents text')
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
        started, cpu_started = time.perf_counter(), time.process_time()
        events = []
        for e in self.events:
            main, targets, legal = rewards_by_trajectory[e['trajectory_uid']]
            if not legal:
                continue
            e['reward'] = ranker_reward(e['candidates'], e['ranking'], targets, main, self.config.threshold)
            e['answer_available'] = answer_available(e['candidates'], targets)
            e['answers_are_trivial'] = answers_are_trivial(targets)
            events.append(e)
        self.events.clear()
        keys = semantic_groups(events, self.config.similarity, self.config.split_availability)
        counts = defaultdict(int)
        for key in keys:
            counts[key] += 1
        group_rewards = defaultdict(set)
        for key, event in zip(keys, events):
            group_rewards[key].add(event['reward'])
        informative = {key for key in counts if counts[key] >= self.config.min_group_size
                       and len(group_rewards[key]) > 1}
        chosen = [i for i, key in enumerate(keys) if key in informative and events[i]['response_length'] > 0]
        values = torch.tensor([e['reward'] for e in events])
        advantage = grouped_center(values, keys) if events else values
        metrics = {'ranker/events': len(events), 'ranker/training_events': len(chosen),
                   'ranker/parameters': sum(p.numel() for p in self.model.parameters()),
                   'ranker/same_reward_groups_skipped': sum(counts[k] >= self.config.min_group_size and len(v) == 1
                                                          for k, v in group_rewards.items()),
                   'ranker/trivial_answer_events': sum(e['answers_are_trivial'] for e in events),
                   'ranker/invalid_ranking_events': sum(e['ranking'] is None for e in events),
                   'ranker/effective_nonzero_advantage_events': sum(float(advantage[i]) != 0. for i in chosen)}
        metrics['ranker/informative_groups'] = len(informative)
        metrics['ranker/informative_groups_with_distinct_rankings'] = sum(
            len({tuple(events[i]['ranking']) if events[i]['ranking'] is not None else None
                 for i, key in enumerate(keys) if key == group}) > 1 for group in informative)
        metrics['ranker/gradient_norm_max'] = 0.
        metrics['ranker/parameter_probe_delta_l1'] = 0.
        # Keep dropout disabled while taking gradients, matching rollout old logprobs.
        self.model.eval()
        for _ in range(self.config.epochs):
            for start in range(0, len(chosen), self.config.batch_size):
                rows = chosen[start:start + self.config.batch_size]
                # Preserve zero-advantage members of informative groups; they count in the sequence expectation.
                self.optimizer.zero_grad()
                for i in rows:
                    e = events[i]
                    ids = e['input_ids'].to(self.device)[None, :]
                    self.cost['training_forward_tokens'] += ids.numel()
                    length = e['response_length']
                    logits = self.model(ids, use_cache=False).logits[:, -length - 1:-1].float() / self.config.temperature
                    logp = F.log_softmax(logits, -1).gather(-1, ids[:, -length:, None]).squeeze(-1)
                    loss = sequence_ppo_loss(logp, e['old_logp'].to(self.device), advantage[i].to(self.device),
                                             self.config.clip_ratio) / len(rows)
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite ranker gradients; optimizer step refused')
                metrics['ranker/gradient_norm_max'] = max(metrics['ranker/gradient_norm_max'], float(norm))
                # Bounded samples prove an actual change without duplicating 3B weights.
                # Delta alone can include AdamW/momentum/roundoff: do not call it RL success.
                probes = []
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        flat = parameter.detach().reshape(-1)
                        count = min(8, flat.numel())
                        index = (torch.arange(count, device=flat.device, dtype=torch.long)
                                 * (flat.numel() - 1) // max(count - 1, 1))
                        probes.append((parameter, index, flat[index].clone()))
                self.optimizer.step()
                metrics['ranker/parameter_probe_delta_l1'] += sum(
                    float((parameter.detach().reshape(-1)[index] - before).abs().sum())
                    for parameter, index, before in probes)
                self.cost['optimizer_steps'] += 1
        self.cost['update_wall_seconds'] = self.cost.get('update_wall_seconds', 0) + time.perf_counter() - started
        self.cost['update_cpu_seconds'] = self.cost.get('update_cpu_seconds', 0) + time.process_time() - cpu_started
        metrics.update(self.cost_metrics())
        return metrics

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        torch.save(self.optimizer.state_dict(), os.path.join(directory, 'optimizer.pt'))
        # A snapshot is useful for accounting; it does not claim joint trainer resume support.
        with open(os.path.join(directory, 'ranker_state.json'), 'w') as stream:
            json.dump(dict(cost=self.cost, device=str(self.device), resume_supported=False), stream, indent=2)
