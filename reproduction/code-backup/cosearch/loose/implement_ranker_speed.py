from pathlib import Path
p=Path('work/cosearch/search_r1/baselines/cosearch.py')
s=p.read_text().replace('from collections import Counter, defaultdict','from collections import Counter, defaultdict\nfrom contextlib import contextmanager')
pos=s.index('\nclass GenerativeRanker:')
s=s[:pos]+'''
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
    events = []
    for row in range(input_ids.shape[0]):
        length = response.shape[1]
        ends = [i for i, token in enumerate(response[row].tolist()) if token in eos_ids]
        if ends:
            length = ends[0] + 1
        tokens = response[row, :length]
        logp = torch.stack([F.log_softmax(output.logits[t][row].float() / temperature, -1)[tokens[t]]
                            for t in range(length)])
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

''' + s[pos:]
# phase wraps complete rank_batch and update, always release before next actor RPC.
pos=s.index('    def response_logp(')
s=s[:pos]+'''    @contextmanager
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
        path = f'/tmp/search-r1-gpu-leases-{os.getuid()}/{uid}.lock'
        fd = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError('Ranker GPU parent lease is not held')
        finally:
            os.close(fd)
        if not torch.cuda.is_initialized():
            # Only this CPU Ray driver is remapped. Eight actor workers keep
            # their Ray-provided masks and GPU reservations.
            os.environ['CUDA_VISIBLE_DEVICES'] = uid
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uid or torch.cuda.device_count() != 1:
            raise RuntimeError('Ranker CUDA context is not the selected physical GPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        parameter_ids = [id(p) for p in self.model.parameters()]
        self.device = torch.device('cuda:0')
        self.model.to(self.device)
        for parameter, state in self.optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(self.device)
        if parameter_ids != [id(p) for p in self.model.parameters()]:
            raise RuntimeError('Model migration changed optimizer parameter identities')
        torch.cuda.synchronize()
        self.cost['gpu_to_wall_seconds'] = self.cost.get('gpu_to_wall_seconds', 0.) + time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        try:
            with eval_decoder_checkpoint(self.model, training and getattr(self.config, 'gradient_checkpointing', False)):
                yield
        finally:
            torch.cuda.synchronize()
            self.cost['gpu_peak_allocated_bytes'] = max(self.cost.get('gpu_peak_allocated_bytes', 0),
                                                        torch.cuda.max_memory_allocated())
            started = time.perf_counter()
            self.model.to('cpu')
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cpu()
            self.device = torch.device('cpu')
            torch.cuda.empty_cache()
            self.cost['gpu_from_wall_seconds'] = self.cost.get('gpu_from_wall_seconds', 0.) + time.perf_counter() - started
            print('COSEARCH_GPU_PHASE_JSON=' + json.dumps(dict(training=training, uuid=uid,
                  cost=self.cost, job=os.environ['SLURM_JOB_ID'])), flush=True)

''' + s[pos:]
start=s.index('    def rank(self,')
end=s.index('    def execute(',start)
legacy=s[start:end].replace('    def rank(self,','    def _rank_legacy(self,',1)
new='''    def rank(self, initial_question, query, docs, state):
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
            passages = '\\n'.join(f'Doc {i + 1}: {d["document"]["contents"]}' for i, d in enumerate(docs))
            prompts.append(f'Question context:\\n{question}\\nSearch query: {query}\\nCandidates:\\n{passages}\\n'
                           f'Select exactly {keep} distinct document IDs ordered by relevance. '
                           'Think briefly, then output <ranking>[1, 2, ...]</ranking>.\\n')
        encoded = self.tokenizer(prompts, return_token_type_ids=False)['input_ids']
        if any(len(ids) + self.config.max_new_tokens > self.model.config.max_position_embeddings for ids in encoded):
            raise ValueError('Ranker candidate context exceeds model limit')
        order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
        rows = [None] * len(order)
        batch_size = int(self.config.inference_batch_size)
        budget = int(self.config.inference_token_budget)
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        offset = 0
        while offset < len(order):
            group = []
            for i in order[offset:offset + batch_size]:
                if group and (len(group) + 1) * (len(encoded[i]) + self.config.max_new_tokens) > budget:
                    break
                group.append(i)
            offset += len(group)
            inputs = self.tokenizer.pad({'input_ids': [encoded[i] for i in group]},
                                        padding=True, return_tensors='pt').to(self.device)
            sampling = GenerationConfig(max_new_tokens=self.config.max_new_tokens, do_sample=not self.validation,
                temperature=self.config.temperature, top_p=1., top_k=0, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.model.generation_config.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id, return_dict_in_generate=True,
                output_logits=not self.validation)
            generated_at = time.perf_counter()
            with torch.no_grad():
                output = self.model.generate(**inputs, generation_config=sampling)
                if self.validation:
                    # Validation consumes only the ranking, never a PPO action.
                    response = output.sequences[:, inputs.input_ids.shape[1]:]
                    eos = sampling.eos_token_id
                    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
                    events = []
                    for row, i in enumerate(group):
                        ends = [t for t, token in enumerate(response[row].tolist()) if token in eos_ids]
                        length = ends[0] + 1 if ends else response.shape[1]
                        events.append(dict(input_ids=torch.cat((torch.tensor(encoded[i]), response[row, :length].cpu())),
                                           response_length=length))
                else:
                    events = unpack_generation(output, inputs.input_ids, inputs.attention_mask,
                                               sampling.eos_token_id, self.config.temperature)
            self.cost['generation_wall_seconds'] = self.cost.get('generation_wall_seconds', 0.) + time.perf_counter() - generated_at
            for i, event in zip(group, events):
                rows[i] = event
            del output, inputs, events
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

'''
s=s[:start]+legacy+new+s[end:]
s=s.replace("        observation = [''] * len(states)", "        selected_rows = []\n        if indices and do_search:\n            selected_rows = self.rank_batch(\n                [manager.tokenizer.decode(states[i].prompt, skip_special_tokens=True) for i in indices],\n                [contents[i] for i in indices], docs, [states[i] for i in indices])\n        observation = [''] * len(states)")
s=s.replace("                    initial = manager.tokenizer.decode(states[i].prompt, skip_special_tokens=True)\n                    selected = self.rank(initial, contents[i], docs[doc_index], states[i])", "                    selected = selected_rows[doc_index]")
s=s.replace('    def update(self, rewards_by_trajectory):','    def update(self, rewards_by_trajectory):\n        self.model.eval()\n        with self.execution_phase(training=True):\n            return self._update(rewards_by_trajectory)\n\n    def _update(self, rewards_by_trajectory):',1)
# _update must not undo decoder-only checkpoint flag.
s=s.replace('        self.model.eval()\n        for _ in range(self.config.epochs):', '        if not getattr(self.config, "gradient_checkpointing", False):\n            self.model.eval()\n        for _ in range(self.config.epochs):')
p.write_text(s)
p=Path('work/cosearch/verl/trainer/config/ppo_baselines.yaml');s=p.read_text();s=s.replace('    response_only_logits: false', '    response_only_logits: false\n    phase_gpu: false\n    reuse_generation_logits: false\n    inference_batch_size: 1\n    inference_token_budget: 32768\n    gradient_checkpointing: false');p.write_text(s)
