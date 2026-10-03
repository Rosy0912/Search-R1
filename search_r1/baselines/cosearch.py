"""CoSearch auxiliary generative ranker, with its own optimizer and checkpoint.

Portable first backend: an unsharded HF ranker on a dedicated device (CPU default).
This is a joint-system adaptation, not policy-only GRPO. No gold enters ranking.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
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


def unpack_generation(output, input_ids, attention_mask, eos_token_id, temperature):
    """Unpad each sampled sequence and score raw (unwarped) cached logits.

    Include its first EOS; forced padding of finished batch members is not an
    action, even when PAD and EOS are identical. Temperature is applied once.
    """
    response = output.sequences[:, input_ids.shape[1]:]
    if len(output.logits) != response.shape[1]:
        raise ValueError('Generation raw logits do not cover the response')
    eos_ids = ([] if eos_token_id is None else list(eos_token_id)
               if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    scores = torch.stack([F.log_softmax(logits.float() / temperature, -1).gather(
        -1, response[:, t, None]).squeeze(-1) for t, logits in enumerate(output.logits)], dim=1)
    events = []
    for row in range(input_ids.shape[0]):
        length = response.shape[1]
        ends = [i for i, token in enumerate(response[row].tolist()) if token in eos_ids]
        if ends:
            length = ends[0] + 1
        tokens = response[row, :length]
        logp = scores[row, :length]
        if not torch.isfinite(logp).all():
            raise ValueError('Nonfinite generation old log probabilities')
        prompt = input_ids[row][attention_mask[row].bool()]
        events.append(dict(input_ids=torch.cat((prompt, tokens)).cpu(),
                           response_length=length, old_logp=logp.cpu()))
    return events


@contextmanager
def eval_decoder_checkpoint(model, enabled):
    """Checkpoint the decoder while attention/dropout modules remain in eval.

    HF Qwen2 gates checkpointing on the decoder's own training flag; setting
    that flag directly does not enable training-mode dropout in its children.
    """
    if not enabled:
        yield
        return
    decoder = model.model
    if model.config.model_type != 'qwen2' or any(m.training for m in model.modules()):
        raise ValueError('Eval checkpointing requires an entirely eval Qwen2 model')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    decoder.training = True
    try:
        yield
    finally:
        decoder.training = False


class RankerSequenceLoss(torch.nn.Module):
    """One scalar per sampled sequence; never gather vocabulary logits."""
    def __init__(self, model, temperature, clip_ratio):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.clip_ratio = clip_ratio

    def forward(self, input_ids, attention_mask, response_lengths, old_logp, advantages, max_response_length):
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        logits = self.model(input_ids, attention_mask=attention_mask, position_ids=positions,
                            use_cache=False, num_logits_to_keep=max_response_length + 1).logits[:, :-1]
        logp = F.log_softmax(logits.float() / self.temperature, -1).gather(
            -1, input_ids[:, -max_response_length:, None]).squeeze(-1)
        if not torch.isfinite(logp).all() or not torch.isfinite(old_logp).all():
            raise ValueError('Nonfinite parallel ranker scores')
        ratio = (logp - old_logp).exp()
        if not torch.isfinite(ratio).all():
            raise ValueError('Nonfinite parallel ranker ratios')
        mask = torch.arange(max_response_length, device=logp.device)[None] >= (
            max_response_length - response_lengths[:, None])
        objective = torch.minimum(ratio * advantages[:, None],
            ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * advantages[:, None])
        return -(objective * mask).sum(-1) / response_lengths


class GenerativeRanker:
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        started, cpu_started = time.perf_counter(), time.process_time()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = torch.device(config.device)
        if self.device.type == 'cpu' and config.get('cpu_threads') is not None:
            threads = config.cpu_threads
            if not isinstance(threads, int) or threads < 1:
                raise ValueError('Ranker CPU threads must be a positive integer')
            torch.set_num_threads(threads)
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

    @contextmanager
    def execution_phase(self, training=False):
        if not getattr(self.config, 'phase_gpu', False):
            yield
            return
        import fcntl
        import subprocess
        uid = os.environ.get('COSEARCH_RANKER_GPU_UUID', '')
        leased = json.loads(os.environ.get('COSEARCH_RANKER_LEASE_UUIDS', '[]'))
        if (os.environ.get('SLURM_JOB_NODELIST') != 'h100-16' or not os.environ.get('SLURM_JOB_ID')
                or uid not in leased or len(set(leased)) != 8 or not uid.startswith('GPU-')):
            raise RuntimeError('Ranker phase requires the owned h100-16 eight-GPU allocation')
        count = int(getattr(self.config, 'phase_gpu_devices', 1))
        if count not in (1,8):
            raise ValueError('Ranker phase count must be one or all eight owned GPUs')
        for lease_uid in leased if count==8 else [uid]:
            path = f'/tmp/search-r1-gpu-leases-{os.getuid()}/{lease_uid}.lock'
            fd = os.open(path, os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise RuntimeError('Ranker GPU parent lease is not held')
            finally:
                os.close(fd)
        expected_mask=','.join(leased) if count==8 else uid
        # torch.is_initialized(False) does not rule out an earlier cuInit.
        # The dedicated process must be bound before importing HF/Torch.
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_mask or torch.cuda.device_count()!=count:
            raise RuntimeError('Ranker CUDA context is not the startup physical GPU mask')
        torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        parameter_ids = [id(p) for p in self.model.parameters()]
        # Gradients were consumed; next update clears them before accumulation.
        self.optimizer.zero_grad(set_to_none=True)
        master = leased.index(uid) if count == 8 else 0
        self.phase_devices = [master] + [i for i in range(count) if i != master]
        self.device = torch.device('cuda', master)
        torch.cuda.set_device(master)
        if count==8 and not getattr(self,'parallel_rng_seeded',False):
            # Independent device streams; never reseed inside concurrent threads.
            for i in range(count):
                with torch.cuda.device(i):
                    torch.cuda.manual_seed(int(self.config.phase_seed)+i+1)
            self.parallel_rng_seeded=True
        try:
            self.model.to(self.device)
            if training:
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value) and key != 'step':
                            state[key] = value.to(self.device)
            if parameter_ids != [id(p) for p in self.model.parameters()]:
                raise RuntimeError('Model migration changed optimizer parameter identities')
            applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                   '--format=csv,noheader,nounits'], text=True)
            actual = {line.split(',')[0].strip() for line in applications.splitlines()
                      if ',' in line and line.split(',')[1].strip() == str(os.getpid())}
            if uid not in actual or not actual.issubset(set(leased)):
                raise RuntimeError(f'Ranker physical GPU mismatch: expected {uid}, actual {sorted(actual)}')
            torch.cuda.synchronize()
            self.cost['gpu_to_wall_seconds'] = self.cost.get('gpu_to_wall_seconds', 0.) + time.perf_counter() - started
            for i in self.phase_devices:
                torch.cuda.reset_peak_memory_stats(i)
            with eval_decoder_checkpoint(self.model, training and getattr(self.config, 'gradient_checkpointing', False)):
                yield
        finally:
            try:
                for i in self.phase_devices:
                    torch.cuda.synchronize(i)
                for i in self.phase_devices:
                    for kind in ('allocated','reserved'):
                        key=f'gpu_peak_{kind}_bytes_device_{i}'
                        value=getattr(torch.cuda,f'max_memory_{kind}')(i)
                        self.cost[key]=max(self.cost.get(key,0),value)
                self.cost['gpu_peak_allocated_bytes'] = max(self.cost.get('gpu_peak_allocated_bytes', 0),
                                                            torch.cuda.max_memory_allocated())
            finally:
                # CUDA failures may be unrecoverable. Attempt cleanup, then let
                # the exception fail the owned Slurm job rather than run actor.
                started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.to('cpu')
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.cpu()
                self.device = torch.device('cpu')
                for i in self.phase_devices:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                self.cost['gpu_from_wall_seconds'] = self.cost.get('gpu_from_wall_seconds', 0.) + time.perf_counter() - started
                print('COSEARCH_GPU_PHASE_JSON=' + json.dumps(dict(training=training, uuid=uid,
                      cost=self.cost, job=os.environ['SLURM_JOB_ID'])), flush=True)

    def response_logp(self, ids, length):
        # Project only the response positions when supported by the frozen HF
        # model. The full decoder still sees every candidate and prompt token.
        if self.config.get('response_only_logits', False):
            logits = self.model(ids, use_cache=False, num_logits_to_keep=length + 1).logits[:, :-1]
        else:
            logits = self.model(ids, use_cache=False).logits[:, -length - 1:-1]
        return F.log_softmax(logits.float() / self.config.temperature, -1).gather(
            -1, ids[:, -length:, None]).squeeze(-1)

    def cost_metrics(self):
        # Driver RSS includes dependencies and trainer state, not only this ranker.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 ** 2 if sys.platform == 'darwin' else 1024
        metrics = {f'ranker/{k}_cumulative': v for k, v in self.cost.items()}
        metrics['ranker/driver_peak_rss_mib'] = rss / divisor
        return metrics

    def _rank_legacy(self, initial_question, query, docs, state):
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
            logp = self.response_logp(generated, response.shape[1])
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
        calls = self.cost[prefix + 'rank_calls']
        if calls == 1 or calls % 10 == 0:
            print('COSEARCH_RANK_PROGRESS_JSON=' + json.dumps(dict(
                validation=self.validation, calls=calls,
                legal=self.cost[prefix + 'legal_rankings'],
                fallback=self.cost[prefix + 'fallback_rankings'],
                cumulative_wall_seconds=self.cost[prefix + 'rank_wall_seconds'],
                last_prompt_tokens=inputs.input_ids.numel())), flush=True)
        if not self.validation:
            self.events.append(dict(question_uid=self.question_uids[state.source], trajectory_uid=state.trajectory,
                                    query=query, candidates=[d['document']['contents'] for d in docs], ranking=ranking,
                                    input_ids=generated[0].cpu(), response_length=response.shape[1],
                                    old_logp=logp[0].cpu()))
        # Deterministic fallback keeps search operational; invalid action earns -.2 ranker reward.
        selection = ranking if ranking is not None else list(range(1, keep + 1))
        return [docs[i - 1] for i in selection]

    def rank(self, initial_question, query, docs, state):
        return self.rank_batch([initial_question], [query], [docs], [state])[0]

    def rank_batch(self, initial_questions, queries, documents, states):
        if not getattr(self.config, 'reuse_generation_logits', False):
            return [self._rank_legacy(q, query, docs, state)
                    for q, query, docs, state in zip(initial_questions, queries, documents, states)]
        with self.execution_phase():
            return self._rank_batch(initial_questions, queries, documents, states)

    def _rank_batch(self, initial_questions, queries, documents, states):
        from transformers import GenerationConfig
        started = time.perf_counter()
        prompts = []
        for question, query, docs in zip(initial_questions, queries, documents):
            keep = min(self.config.keep, len(docs))
            if not keep:
                raise ValueError('Batch ranker needs nonempty full candidates')
            passages = '\n'.join(f'Doc {i + 1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
            prompts.append(f'Question context:\n{question}\nSearch query: {query}\nCandidates:\n{passages}\n'
                           f'Select exactly {keep} distinct document IDs ordered by relevance. '
                           'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\n')
        encoded = self.tokenizer(prompts, return_token_type_ids=False)['input_ids']
        if any(len(ids) + self.config.max_new_tokens > self.model.config.max_position_embeddings for ids in encoded):
            raise ValueError('Ranker candidate context exceeds model limit')
        order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
        rows = [None] * len(order)
        batch_size = int(self.config.inference_batch_size)
        budget = int(self.config.inference_token_budget)
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        groups=[]
        offset=0
        while offset < len(order):
            group=[]
            for i in order[offset:offset+batch_size]:
                if group and (len(group)+1)*(len(encoded[i])+self.config.max_new_tokens)>budget:
                    break
                group.append(i)
            groups.append(group);offset+=len(group)
        devices = getattr(self, 'phase_devices', [self.device.index or 0]) if getattr(self.config, 'inference_parallel', False) else [self.device]
        replicas = None
        gpu_models = [self.model]
        generation_started=time.perf_counter()
        try:
            if len(devices)>1:
                if self.model.config.model_type != 'qwen2' or self.model.config.rope_scaling:
                    raise ValueError('Parallel ranker inference requires pinned static-RoPE Qwen2')
                replica_at=time.perf_counter()
                replicas=torch.nn.parallel.replicate(self.model, devices, detach=True)
                gpu_models=replicas
                for i,model in zip(devices,gpu_models):
                    if any(p.device.index != i for module in model.modules()
                           for p in module._former_parameters.values()):
                        raise RuntimeError('Ranker replica is on the wrong GPU')
                import subprocess
                applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                                      '--format=csv,noheader,nounits'],text=True)
                actual={line.split(',')[0].strip() for line in applications.splitlines()
                        if ',' in line and line.split(',')[1].strip()==str(os.getpid())}
                expected=set(json.loads(os.environ['COSEARCH_RANKER_LEASE_UUIDS']))
                if actual!=expected:
                    raise RuntimeError(f'Parallel replica physical GPUs differ: {sorted(actual)}')
                print('COSEARCH_GPU_IDENTITY_JSON='+json.dumps(dict(job=os.environ['SLURM_JOB_ID'],
                      pid=os.getpid(),replica_uuids=sorted(actual))),flush=True)
                self.cost['replication_wall_seconds']=self.cost.get('replication_wall_seconds',0.)+time.perf_counter()-replica_at
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                completed=0
                for wave in range(0,len(groups),len(devices)):
                    futures=[]
                    for j,group in enumerate(groups[wave:wave+len(devices)]):
                        inputs=self.tokenizer.pad({'input_ids':[encoded[i] for i in group]},padding=True,return_tensors='pt')
                        sampling=GenerationConfig(max_new_tokens=self.config.max_new_tokens,do_sample=not self.validation,
                            temperature=self.config.temperature,top_p=1.,top_k=0,num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.model.generation_config.eos_token_id,
                            bos_token_id=self.model.generation_config.bos_token_id,return_dict_in_generate=True,
                            output_logits=not self.validation)
                        futures.append((group,executor.submit(self._generate_batch, gpu_models[j], devices[j], inputs, sampling)))
                    for group,future in futures:
                        events,wall=future.result()
                        self.cost['generation_gpu_worker_seconds']=self.cost.get('generation_gpu_worker_seconds',0.)+wall
                        for i,event in zip(group,events):rows[i]=event
                        completed+=len(group)
                    if wave==0 or completed%100<sum(len(g) for g,_ in futures) or completed==len(order):
                        print('COSEARCH_BATCH_PROGRESS_JSON='+json.dumps(dict(validation=self.validation,
                              completed=completed,total=len(order),wall_seconds=time.perf_counter()-started)),flush=True)
                    futures.clear()
        finally:
            gpu_models.clear()
            replicas=None
            self.cost['generation_wall_seconds']=self.cost.get('generation_wall_seconds',0.)+time.perf_counter()-generation_started
        result = []
        prefix = 'validation_' if self.validation else ''
        # Restore retrieval/trajectory order before recording events or grouping.
        for i, event in enumerate(rows):
            docs = documents[i]
            response = event['input_ids'][-event['response_length']:]
            ranking = parse_ranking(self.tokenizer.decode(response, skip_special_tokens=True), len(docs), self.config.keep)
            for name, value in [('rank_calls', 1), ('legal_rankings', int(ranking is not None)),
                                ('fallback_rankings', int(ranking is None)), ('prefill_tokens', len(encoded[i])),
                                ('generated_tokens', event['response_length'])]:
                self.cost[prefix + name] = self.cost.get(prefix + name, 0) + value
            self.cost[prefix + 'prompt_tokens_max'] = max(self.cost.get(prefix + 'prompt_tokens_max', 0), len(encoded[i]))
            if not self.validation:
                event.update(question_uid=self.question_uids[states[i].source], trajectory_uid=states[i].trajectory,
                             query=queries[i], candidates=[d['document']['contents'] for d in docs], ranking=ranking)
                self.events.append(event)
            selection = ranking if ranking is not None else list(range(1, self.config.keep + 1))
            result.append([docs[j - 1] for j in selection])
        self.cost[prefix + 'rank_wall_seconds'] = self.cost.get(prefix + 'rank_wall_seconds', 0.) + time.perf_counter() - started
        print('COSEARCH_RANK_PROGRESS_JSON=' + json.dumps(dict(validation=self.validation,
            calls=self.cost[prefix + 'rank_calls'], legal=self.cost[prefix + 'legal_rankings'],
            fallback=self.cost[prefix + 'fallback_rankings'], cumulative_wall_seconds=self.cost[prefix + 'rank_wall_seconds'])), flush=True)
        return result

    def _generate_batch(self, model, device, inputs, sampling):
        started=time.perf_counter()
        device=torch.device('cuda',device) if isinstance(device,int) else device
        from contextlib import nullcontext
        # grad mode and active CUDA device are thread-local.
        with torch.cuda.device(device) if device.type=='cuda' else nullcontext():
            inputs=inputs.to(device)
            with torch.no_grad():
                output=model.generate(**inputs,generation_config=sampling)
                if not self.validation:
                    events=unpack_generation(output,inputs.input_ids,inputs.attention_mask,
                                             sampling.eos_token_id,self.config.temperature)
                else:
                    response=output.sequences[:,inputs.input_ids.shape[1]:]
                    eos=sampling.eos_token_id
                    eos_ids=eos if isinstance(eos,(list,tuple)) else [eos]
                    events=[]
                    for row in range(response.shape[0]):
                        ends=[t for t,token in enumerate(response[row].tolist()) if token in eos_ids]
                        length=ends[0]+1 if ends else response.shape[1]
                        prompt=inputs.input_ids[row][inputs.attention_mask[row].bool()]
                        events.append(dict(input_ids=torch.cat((prompt,response[row,:length])).cpu(),response_length=length))
            del output,inputs
        return events,time.perf_counter()-started

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
        selected_rows = []
        if indices and do_search:
            selected_rows = self.rank_batch(
                [manager.tokenizer.decode(states[i].prompt, skip_special_tokens=True) for i in indices],
                [contents[i] for i in indices], docs, [states[i] for i in indices])
        observation = [''] * len(states)
        done, valid, searches = [], [], []
        doc_index = 0
        for i, action in enumerate(actions):
            if action == 'search':
                if do_search:
                    selected = selected_rows[doc_index]
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
        self.model.eval()
        with self.execution_phase(training=True):
            return self._update(rewards_by_trajectory)

    def _update(self, rewards_by_trajectory):
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
        if not getattr(self.config, "gradient_checkpointing", False):
            self.model.eval()
        for _ in range(self.config.epochs):
            for start in range(0, len(chosen), self.config.batch_size):
                rows = chosen[start:start + self.config.batch_size]
                # Preserve zero-advantage members of informative groups; they count in the sequence expectation.
                self.optimizer.zero_grad()
                if getattr(self.config, 'training_parallel', False):
                    width=max(events[i]['input_ids'].numel() for i in rows)
                    response_width=max(events[i]['response_length'] for i in rows)
                    ids=torch.stack([F.pad(events[i]['input_ids'],(width-events[i]['input_ids'].numel(),0),
                                           value=self.tokenizer.pad_token_id) for i in rows]).to(self.device)
                    mask=torch.stack([F.pad(torch.ones_like(events[i]['input_ids']),
                                            (width-events[i]['input_ids'].numel(),0)) for i in rows]).to(self.device)
                    old=torch.stack([F.pad(events[i]['old_logp'],(response_width-events[i]['response_length'],0)) for i in rows]).to(self.device)
                    lengths=torch.tensor([events[i]['response_length'] for i in rows],device=self.device)
                    row_advantages=advantage[rows].to(self.device)
                    sequence_loss=RankerSequenceLoss(self.model,self.config.temperature,self.config.clip_ratio)
                    parallel=torch.nn.DataParallel(sequence_loss,device_ids=self.phase_devices[:len(rows)],output_device=self.device.index)
                    losses=parallel(ids,mask,lengths,old,row_advantages,max_response_length=response_width)
                    (losses.sum()/len(rows)).backward()
                    del losses,parallel,sequence_loss,ids,mask,old,lengths,row_advantages
                    self.cost['training_forward_tokens']+=sum(events[i]['input_ids'].numel() for i in rows)
                    self.cost['training_padding_tokens']=self.cost.get('training_padding_tokens',0)+width*len(rows)-sum(events[i]['input_ids'].numel() for i in rows)
                else:
                    for i in rows:
                        e = events[i]
                        ids = e['input_ids'].to(self.device)[None, :]
                        self.cost['training_forward_tokens'] += ids.numel()
                        length = e['response_length']
                        logp = self.response_logp(ids, length)
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
