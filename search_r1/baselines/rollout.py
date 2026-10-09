"""Resumable search trajectories, retaining the reviewed parser/retriever.

Only new algorithms use this engine. The original GRPO/validation path is kept.
All metadata is row-local, so DataProto permutations also permute provenance.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import random
import uuid
import numpy as np
import torch


def object_array(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def select_rows(data, indices):
    """Old verl __getitem__ always returns an Item; explicitly preserve batch APIs."""
    from verl import DataProto
    if isinstance(indices, list):
        indices = np.asarray(indices, dtype=np.int64)
    return DataProto(batch=data.batch[indices],
                     non_tensor_batch={k: v[indices] for k, v in data.non_tensor_batch.items()},
                     meta_info=copy.deepcopy(data.meta_info))


@dataclass
class State:
    source: int
    prompt: list
    rolling: list
    response: list = field(default_factory=list)
    model_mask: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    records: list = field(default_factory=list)
    observation: list = field(default_factory=list)
    turns: int = 0
    done: bool = False
    final_used: bool = False
    valid_actions: int = 0
    valid_searches: int = 0
    tree: str = field(default_factory=lambda: uuid.uuid4().hex)
    trajectory: str = field(default_factory=lambda: uuid.uuid4().hex)
    node: str = field(default_factory=lambda: uuid.uuid4().hex)

    sampling_row_id: str = ""
    sampling_turns: list = field(default_factory=list)
    sampling_logprobs: list = field(default_factory=list)
    rolling_position_offset: int = 0
    raw_sampling_diagnostics: list = field(default_factory=list)

    def branch(self):
        child = copy.deepcopy(self)
        child.trajectory = uuid.uuid4().hex
        return child


def padded(rows, pad, left=False):
    width = max(1, max(map(len, rows), default=0))
    out = torch.full((len(rows), width), pad, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            start = width - len(row) if left else 0
            out[i, start:start + len(row)] = torch.tensor(row)
    return out


class SearchRollout:
    def __init__(self, manager, anchor_mode='observation', ranker=None, seed=42, context_alignment='report'):
        self.preserve_sampling_tokens = getattr(manager.config, "preserve_sampling_tokens", False)
        self.manager = manager
        self.tokenizer = manager.tokenizer
        self.config = manager.config
        if anchor_mode not in ('observation', 'context'):
            raise ValueError('anchor_mode must be observation or context')
        self.anchor_mode = anchor_mode
        if context_alignment not in ('report', 'error'):
            raise ValueError('context_alignment must be report or error')
        self.context_alignment = context_alignment
        self.ranker = ranker
        self.rng = random.Random(seed)
        self.cost = dict(generated_tokens=0, prefill_tokens=0, retrieval_calls=0,
                         trajectories=0, padded_rows=0, context_mismatches=0)

    def initial_states(self, gen_batch):
        # The frozen manager retains the original padded prompt width, even
        # though generation removes its left padding before the first call.
        self.initial_prompt_width = min(gen_batch.batch['input_ids'].shape[1],
                                        self.config.max_start_length)
        states = []
        for i in range(len(gen_batch)):
            ids = gen_batch.batch['input_ids'][i][gen_batch.batch['attention_mask'][i].bool()].tolist()
            if self.preserve_sampling_tokens:
                from search_r1.llm_agent.sampling_trace import require
                require(0 < len(ids) <= self.config.max_start_length, 'initial BAPO context would be cropped')
                row_ids = gen_batch.non_tensor_batch.get('fresh200_row_id')
                require(row_ids is not None and len(set(map(str,row_ids))) == len(gen_batch), 'BAPO row identity missing')
                states.append(State(i, ids, ids, observation=ids, sampling_row_id=str(row_ids[i])))
            else:
                states.append(State(i, ids[-self.config.max_start_length:], ids, observation=ids))
                row_ids = gen_batch.non_tensor_batch.get('fresh200_row_id')
                states[-1].sampling_row_id = str(row_ids[i]) if row_ids is not None else uuid.uuid4().hex
        return states

    def pending(self, state):
        return not state.done and (state.turns < self.config.max_turns or
                                  (self.config.force_final_answer and not state.final_used))

    def advance(self, states, meta):
        if self.preserve_sampling_tokens:
            return self._advance_sampling(states, meta)
        from verl import DataProto
        matches = [s.prompt + s.response == s.rolling for s in states]
        mismatches = sum(not match for match in matches)
        self.cost['context_mismatches'] += mismatches
        if mismatches and self.context_alignment == 'error':
            raise ValueError('Actual rollout prefix differs from full-trajectory scoring prefix')
        pad = self.tokenizer.pad_token_id
        ids = padded([s.rolling for s in states], pad, left=True)
        attention = (ids != pad).long()
        positions = self.manager.tensor_fn.create_position_ids(attention)
        if getattr(self.config, 'baseline_aligned', False):
            offsets = torch.tensor([s.rolling_position_offset for s in states], dtype=positions.dtype)
            positions = positions + offsets[:, None] * attention
        batch = DataProto.from_dict(dict(input_ids=ids, attention_mask=attention,
                                        position_ids=positions), meta_info=meta.copy())
        self.cost['prefill_tokens'] += int(attention.sum())
        extra = (-len(states)) % self.config.num_gpus if self.config.num_gpus > 1 else 0
        self.cost['padded_rows'] += extra
        if extra:
            self.cost['prefill_tokens'] += extra * len(states[0].rolling)
            tensors = {k: torch.cat((v, v[:1].repeat(extra, 1)), 0) for k, v in batch.batch.items()}
            batch = DataProto.from_dict(tensors, meta_info=meta.copy())
        if getattr(self.config, 'capture_raw_sampling', False):
            batch.non_tensor_batch['fresh200_row_id'] = object_array(
                [s.sampling_row_id for s in states] + ['padding-' + uuid.uuid4().hex for _ in range(extra)])
            batch.meta_info.update(capture_raw_sampling=True, fresh200_call_id=uuid.uuid4().hex,
                                   fresh200_turn_index=max(len(s.raw_sampling_diagnostics) for s in states),
                                   fresh200_evaluation=not meta.get('do_sample', True))
        output = self.manager.actor_rollout_wg.generate_sequences(batch)
        self.output_meta = output.meta_info.copy()
        if getattr(self.config, 'baseline_aligned', False):
            self.output_meta = meta.copy()
            self.output_meta.update(output.meta_info)
        generated = output.batch['responses']
        # Include actual discarded padding generations, rather than estimate duplicates.
        self.cost['generated_tokens'] += int((generated != pad).sum())
        raw = generated[:len(states)]
        if getattr(self.config, 'baseline_aligned', False):
            # Native traces describe the raw sampler output. Retokenized policy
            # actions cannot inherit those selected-token probabilities.
            native = output.non_tensor_batch.get('fresh200_sampling')
            selected = output.batch.get('raw_sampling_log_probs', None)
            if selected is None:
                selected = output.batch.get('sampling_log_probs', None)
            for i, state in enumerate(states):
                diagnostic = dict(raw_generated_token_ids=raw[i].tolist(),
                                  probability_alignment='raw_sampler_only',
                                  capture_provenance=output.meta_info.get('raw_sampling_capture_provenance', 'tensor_fixture_or_uncaptured'),
                                  policy_snapshot_id=batch.meta_info.get('policy_snapshot_id'),
                                  row_id=state.sampling_row_id,
                                  actual_request_token_ids=state.rolling.copy())
                if native is not None:
                    diagnostic['native_trace'] = copy.deepcopy(native[i])
                if selected is not None:
                    diagnostic['raw_selected_logprobs'] = selected[i].tolist()
                state.raw_sampling_diagnostics.append(diagnostic)
        tokens, text = self.manager._postprocess_responses(raw)
        # A batch can contain normal and final-only steps. Never search for the latter.
        normal = [s.turns < self.config.max_turns for s in states]
        observations, dones, valid, searches = [''] * len(states), [0] * len(states), [0] * len(states), [0] * len(states)
        for do_search in (True, False):
            indices = [i for i, flag in enumerate(normal) if flag == do_search]
            if not indices:
                continue
            predictions = [text[i] for i in indices]
            if self.ranker is None:
                result = self.manager.execute_predictions(predictions, self.tokenizer.pad_token,
                                                           torch.ones(len(indices), dtype=torch.bool), do_search=do_search)
            else:
                result = self.ranker.execute(self.manager, predictions, [states[i] for i in indices], do_search)
            for j, i in enumerate(indices):
                observations[i], dones[i], valid[i], searches[i] = (part[j] for part in result)
        if getattr(self.config, 'baseline_aligned', False):
            # Archive final generation executes only for action statistics. Its
            # no-search observation (including invalid-action feedback) is discarded.
            observations = [obs if normal[i] else '' for i, obs in enumerate(observations)]
        obs_tokens = self.manager._process_next_obs(observations)
        for i, s in enumerate(states):
            action = tokens[i][tokens[i] != pad].tolist()
            obs = obs_tokens[i][obs_tokens[i] != pad].tolist()
            key_tokens = s.rolling if self.anchor_mode == 'context' else s.observation
            anchor = hashlib.sha256(np.asarray(key_tokens, dtype=np.int64).tobytes()).hexdigest()
            step = len(s.anchors)
            s.anchors.append(anchor)
            s.records.append(dict(text=text[i], action_tokens=len(action), observation_tokens=len(obs),
                                  scoring_context_matches_generation=matches[i]))
            s.response.extend(action + obs)
            s.model_mask.extend([1] * len(action) + [0] * len(obs))
            s.steps.extend([step] * len(action) + [-1] * len(obs))
            # Match base rolling and right-side caps exactly, including discarded tails.
            cap = self.config.max_prompt_length
            if getattr(self.config, 'baseline_aligned', False):
                # Archive assigns positions over the concatenation before its
                # tail crop; each new concatenation starts from position zero.
                s.rolling_position_offset = max(0, len(s.rolling) + len(action) + len(obs) - cap)
            s.rolling = (s.rolling + action + obs)[-cap:]
            s.response, s.model_mask, s.steps = s.response[:cap], s.model_mask[:cap], s.steps[:cap]
            s.observation = obs
            s.turns += int(normal[i])
            s.final_used |= not normal[i]
            s.done = bool(dones[i])
            s.node = uuid.uuid4().hex
            s.valid_actions += valid[i]
            s.valid_searches += searches[i]
            self.cost['retrieval_calls'] += int(searches[i] and normal[i])

    def _advance_sampling(self, states, meta):
        from verl import DataProto
        from search_r1.llm_agent.sampling_tokens import left_pad
        from search_r1.llm_agent.sampling_trace import require, prefix, sha
        require(self.ranker is None, 'fresh200 custom ranker unsupported')
        requests=[s.rolling for s in states]
        require(all(s.prompt+s.response==s.rolling for s in states), 'BAPO sampled/actor context mismatch')
        require(all(0<len(r)<=self.config.max_prompt_length for r in requests), 'BAPO rolling context would be cropped')
        ordinals={len(s.sampling_turns) for s in states}
        require(len(ordinals)==1, 'BAPO mixed native turn ordinal')
        extra=(-len(states))%self.config.num_gpus
        identities=[s.sampling_row_id for s in states]+['padding-'+uuid.uuid4().hex for _ in range(extra)]
        callmeta=meta.copy()
        callmeta.update(fresh200_call_id=uuid.uuid4().hex,fresh200_turn_index=next(iter(ordinals)),recompute_log_prob=False)
        batch=DataProto.from_dict(left_pad(requests+[requests[0]]*extra,self.tokenizer.pad_token_id),
            non_tensors={'fresh200_row_id':identities},meta_info=callmeta)
        self.cost['prefill_tokens']+=sum(map(len,requests))+extra*len(requests[0])
        self.cost['padded_rows']+=extra
        output=self.manager.actor_rollout_wg.generate_sequences(batch)
        require(len(output)==len(identities) and 'fresh200_sampling' in output.non_tensor_batch, 'BAPO missing actual native traces')
        records=output.non_tensor_batch['fresh200_sampling'].tolist()
        require([r['row_id'] for r in records]==identities, 'BAPO native row identity')
        self.output_meta=meta.copy()
        self.output_meta.update(output.meta_info)
        self.cost['generated_tokens']+=sum(len(r['raw_generated_token_ids']) for r in records)
        cuts=[];texts=[]
        for i,state in enumerate(states):
            record=records[i];raw=record['raw_generated_token_ids']
            require(record['actual_request_token_ids']==state.rolling and record['policy_snapshot_id']==meta['policy_snapshot_id'], 'BAPO actual native context')
            require(output.batch['responses'][i,:len(raw)].tolist()==raw,'BAPO raw token transport')
            require(torch.equal(output.batch['sampling_log_probs'][i,:len(raw)].float().cpu(),torch.tensor(record['selected_logprobs'],dtype=torch.float32)), 'BAPO raw selected LP transport')
            n,text=prefix(raw,lambda ts:self.tokenizer.decode(ts,skip_special_tokens=False,clean_up_tokenization_spaces=False))
            cuts.append(n);texts.append(text)
        normal=[s.turns<self.config.max_turns for s in states]
        observations,dones,valid,searches=['']*len(states),[0]*len(states),[0]*len(states),[0]*len(states)
        for do_search in (True,False):
            indices=[i for i,flag in enumerate(normal) if flag==do_search]
            if not indices:continue
            result=self.manager.execute_predictions([texts[i] for i in indices],self.tokenizer.pad_token,
                torch.ones(len(indices),dtype=torch.bool),do_search=do_search)
            for j,i in enumerate(indices):
                observations[i],dones[i],valid[i],searches[i]=(part[j] for part in result)
        for i,state in enumerate(states):
            n=cuts[i];record=records[i];action=record['raw_generated_token_ids'][:n]
            obs=self.tokenizer.encode(observations[i],add_special_tokens=False)[:self.config.max_obs_length]
            require(len(state.response)+len(action)+len(obs)<=self.config.max_prompt_length,'BAPO response would be truncated')
            key=state.rolling if self.anchor_mode=='context' else state.observation
            anchor=hashlib.sha256(np.asarray(key,dtype=np.int64).tobytes()).hexdigest();step=len(state.anchors)
            state.anchors.append(anchor)
            state.records.append(dict(text=texts[i],action_tokens=len(action),observation_tokens=len(obs),scoring_context_matches_generation=True))
            record.update(kept_token_count=n,environment_token_ids=obs);record['turn_sha256']=sha(record)
            state.sampling_turns.append(record);state.sampling_logprobs.extend(record['selected_logprobs'][:n]+[0.]*len(obs))
            state.response.extend(action+obs);state.model_mask.extend([1]*len(action)+[0]*len(obs));state.steps.extend([step]*len(action)+[-1]*len(obs))
            state.rolling=state.prompt+state.response;state.observation=obs
            state.turns+=int(normal[i]);state.final_used|=not normal[i];state.done=bool(dones[i]);state.node=uuid.uuid4().hex
            state.valid_actions+=valid[i];state.valid_searches+=searches[i]
            self.cost['retrieval_calls']+=int(searches[i] and normal[i])

    def complete(self, states, meta):
        snapshots = [[] for _ in states]
        while True:
            active_indices = [i for i, s in enumerate(states) if self.pending(s)]
            if not active_indices:
                break
            for i in active_indices:
                snapshots[i].append(copy.deepcopy(states[i]))
            self.advance([states[i] for i in active_indices], meta)
        self.cost['trajectories'] += len(states)
        return snapshots

    def compose(self, states):
        from verl import DataProto
        pad = self.tokenizer.pad_token_id
        prompts = padded([s.prompt for s in states], pad, left=True)
        if getattr(self.config, 'baseline_aligned', False) and prompts.shape[1] < self.initial_prompt_width:
            prompts = torch.cat((torch.full((len(states), self.initial_prompt_width - prompts.shape[1]),
                                           pad, dtype=prompts.dtype), prompts), dim=1)
        response = padded([s.response for s in states], pad)
        mask = padded([s.model_mask for s in states], 0)
        step_ids = padded([s.steps for s in states], -1)
        input_ids = torch.cat((prompts, response), 1)
        attention = (input_ids != pad).long()
        prompt_mask = (prompts != pad).long()
        if self.preserve_sampling_tokens:
            prompt_mask = torch.zeros_like(prompts)
            response_mask = torch.zeros_like(response)
            for i,state in enumerate(states):
                prompt_mask[i,-len(state.prompt):] = 1
                response_mask[i,:len(state.response)] = 1
            attention = torch.cat((prompt_mask,response_mask),1)
        info = torch.cat((prompt_mask, mask), 1)
        tensors = dict(prompts=prompts, responses=response,
                       responses_with_info_mask=torch.where(mask.bool(), response, pad),
                       input_ids=input_ids, attention_mask=attention, info_mask=info,
                       position_ids=self.manager.tensor_fn.create_position_ids(attention),
                       step_ids=step_ids)
        non_tensors = dict(anchors=object_array([s.anchors for s in states]),
                           action_records=object_array([s.records for s in states]),
                           trajectory_uid=object_array([s.trajectory for s in states]),
                           tree_uid=object_array([s.tree for s in states]))
        if getattr(self.config, 'baseline_aligned', False):
            non_tensors['raw_sampling_diagnostics'] = object_array([s.raw_sampling_diagnostics for s in states])
        if self.preserve_sampling_tokens:
            # padded() is integer-only upstream; transport selected probabilities explicitly.
            tensors['sampling_log_probs'] = torch.zeros(response.shape,dtype=torch.float32)
            for i,state in enumerate(states):
                tensors['sampling_log_probs'][i,:len(state.sampling_logprobs)] = torch.tensor(state.sampling_logprobs)
            tensors['sampling_mask'] = mask.bool()
            non_tensors.update(fresh200_row_id=object_array([s.sampling_row_id for s in states]),
                               fresh200_turns=object_array([s.sampling_turns for s in states]))
        result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        result.meta_info.update(self.output_meta)
        if getattr(self.config, 'baseline_aligned', False):
            result.meta_info['baseline_aligned'] = True
            result.meta_info['raw_sampling_probability_alignment'] = 'raw_sampler_only'
        result.meta_info.update(dict(turns_stats=[s.turns + int(s.final_used) for s in states],
                                     active_mask=[not s.done for s in states],
                                     valid_action_stats=[s.valid_actions for s in states],
                                     valid_search_stats=[s.valid_searches for s in states]))
        if self.preserve_sampling_tokens:
            from search_r1.llm_agent.sampling_tokens import audit_batch
            result.meta_info['sampling_context_audit'] = audit_batch(result)
        return result, np.asarray([s.source for s in states], dtype=np.int64)

    def run(self, gen_batch, tree=None):
        self.cost = dict.fromkeys(self.cost, 0)
        states = self.initial_states(gen_batch)
        self.tree_candidate_counts = []
        snapshots = self.complete(states, gen_batch.meta_info)
        if tree is not None:
            leaves_by_root = [[root] for root in states]
            counts_by_root = [[] for _ in states]
            # Select from each root's existing candidates before this round's
            # continuations run, then generate all roots together. A child keeps
            # its source/tree/prefix/remaining turn budget; only batching changes.
            for _ in range(tree.rounds):
                children, owners = [], []
                for root_index, nodes in enumerate(snapshots):
                    if nodes:
                        children.extend(self.rng.choice(nodes).branch() for _ in range(tree.expansions))
                        owners.extend([root_index] * tree.expansions)
                if not children:
                    break
                new_nodes = self.complete(children, gen_batch.meta_info)
                known_by_root = [{s.node for s in nodes} for nodes in snapshots]
                for child, root_index, path in zip(children, owners, new_nodes):
                    leaves_by_root[root_index].append(child)
                    nodes, known = snapshots[root_index], known_by_root[root_index]
                    for s in path:
                        if s.node not in known:
                            nodes.append(s)
                            known.add(s.node)
                for root_index in set(owners):
                    counts_by_root[root_index].append(len(snapshots[root_index]))
            # Keep diagnostic order root-major as in the original engine.
            self.tree_candidate_counts = [count for counts in counts_by_root for count in counts]
            selected = []
            for leaves in leaves_by_root:
                if len(leaves) < tree.leaves:
                    raise ValueError('Tree-GRPO has fewer sampled leaves than K; increase expansion budget')
                selected.extend(self.rng.sample(leaves, tree.leaves))
            states = selected
        self.last_states = states
        return self.compose(states)
