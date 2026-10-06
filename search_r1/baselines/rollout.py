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
    raw_actor_response: list = field(default_factory=list)
    raw_mask: list = field(default_factory=list)
    raw_steps: list = field(default_factory=list)
    action_prefix_records: list = field(default_factory=list)
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
    def __init__(self, manager, anchor_mode='observation', ranker=None, seed=42, context_alignment='report', method=None):
        self.manager = manager
        self.tokenizer = manager.tokenizer
        self.config = manager.config
        if anchor_mode not in ('observation', 'context'):
            raise ValueError('anchor_mode must be observation or context')
        self.anchor_mode = anchor_mode
        if context_alignment not in ('report', 'error'):
            raise ValueError('context_alignment must be report or error')
        self.context_alignment = context_alignment
        from .action_identity import IDENTITY_MODE, LEGACY_MODE
        self.actor_token_identity = getattr(self.config, 'actor_token_identity', LEGACY_MODE)
        if self.actor_token_identity not in (LEGACY_MODE, IDENTITY_MODE):
            raise ValueError('Unknown actor token identity mode')
        self.sampled_action_identity = self.actor_token_identity == IDENTITY_MODE
        if self.sampled_action_identity:
            if (method != 'gigpo' or anchor_mode != 'observation' or ranker is not None
                    or self.config.max_turns != 4 or self.config.force_final_answer
                    or self.config.max_response_length != 500 or self.config.max_obs_length != 500
                    or self.config.max_prompt_length != 4096 or self.config.max_start_length != 2048
                    or self.config.num_gpus != 8 or self.config.no_think_rl):
                raise ValueError('Sampled action identity requires the reviewed GiGPO observation/4-turn budget')
        self.ranker = ranker
        self.rng = random.Random(seed)
        self.cost = dict(generated_tokens=0, prefill_tokens=0, retrieval_calls=0,
                         trajectories=0, padded_rows=0, context_mismatches=0)

    def initial_states(self, gen_batch):
        states = []
        for i in range(len(gen_batch)):
            ids = gen_batch.batch['input_ids'][i][gen_batch.batch['attention_mask'][i].bool()].tolist()
            states.append(State(i, ids[-self.config.max_start_length:], ids, observation=ids))
        return states

    def pending(self, state):
        return not state.done and (state.turns < self.config.max_turns or
                                  (self.config.force_final_answer and not state.final_used))

    def advance(self, states, meta):
        if getattr(self.manager, "rollout_trace", None) is not None and self.ranker is not None:
            raise ValueError("Compact trajectory logging requires the GiGPO unranked rollout")
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
        batch = DataProto.from_dict(dict(input_ids=ids, attention_mask=attention,
                                        position_ids=positions), meta_info=meta.copy())
        self.cost['prefill_tokens'] += int(attention.sum())
        extra = (-len(states)) % self.config.num_gpus if self.config.num_gpus > 1 else 0
        self.cost['padded_rows'] += extra
        if extra:
            self.cost['prefill_tokens'] += extra * len(states[0].rolling)
            tensors = {k: torch.cat((v, v[:1].repeat(extra, 1)), 0) for k, v in batch.batch.items()}
            batch = DataProto.from_dict(tensors, meta_info=meta.copy())
        output = self.manager.actor_rollout_wg.generate_sequences(batch)
        self.output_meta = output.meta_info.copy()
        generated = output.batch['responses']
        # Include actual discarded padding generations, rather than estimate duplicates.
        if not self.sampled_action_identity:
            self.cost['generated_tokens'] += int((generated != pad).sum())
        raw = generated[:len(states)]
        tokens, text = self.manager._postprocess_responses(raw)
        action_identities = None
        if self.sampled_action_identity:
            from .action_identity import retain_sampled_action, sampled_ids_with_evidence
            if ('response_sampled_lengths' not in output.batch
                    or 'response_sample_mask' not in output.batch):
                raise ValueError('Raw action mode requires original engine sampled lengths and masks')
            lengths, sample_masks = output.batch['response_sampled_lengths'], output.batch['response_sample_mask']
            if (lengths.shape != (len(output),) or lengths.dtype != torch.int64
                    or sample_masks.shape != generated.shape or sample_masks.dtype not in (torch.bool, torch.int64)):
                raise ValueError('Invalid engine sampled length/mask tensor identity')
            for index in range(len(output)):
                sampled_ids_with_evidence(generated[index].tolist(), int(lengths[index]), sample_masks[index].tolist())
            self.cost['generated_tokens'] += int(lengths.sum())
            action_identities = [retain_sampled_action(self.tokenizer, raw[i].tolist(), int(lengths[i]),
                                                       sample_masks[i].tolist(), text[i])
                                 for i in range(len(states))]
        # A batch can contain normal and final-only steps. Never search for the latter.
        normal = [s.turns < self.config.max_turns for s in states]
        observations, dones, valid, searches = [''] * len(states), [0] * len(states), [0] * len(states), [0] * len(states)
        for do_search in (True, False):
            indices = [i for i, flag in enumerate(normal) if flag == do_search]
            if not indices:
                continue
            predictions = [text[i] for i in indices]
            if self.ranker is None:
                trace_options = {'record_trace': False} if getattr(self.manager, 'rollout_trace', None) is not None else {}
                result = self.manager.execute_predictions(predictions, self.tokenizer.pad_token,
                                                           torch.ones(len(indices), dtype=torch.bool), do_search=do_search,
                                                           **trace_options)
            else:
                result = self.ranker.execute(self.manager, predictions, [states[i] for i in indices], do_search)
            if len(result) != 4 or any(len(part) != len(indices) for part in result):
                raise ValueError('Environment result coverage changed')
            for j, i in enumerate(indices):
                observations[i], dones[i], valid[i], searches[i] = (part[j] for part in result)
        trace = getattr(self.manager, 'rollout_trace', None)
        if trace is not None:
            trace.record_active_turn([state.source for state in states], text, observations, dones, valid, searches)
        obs_tokens = self.manager._process_next_obs(observations)
        for i, s in enumerate(states):
            action = tokens[i][tokens[i] != pad].tolist()
            obs = obs_tokens[i][obs_tokens[i] != pad].tolist()
            key_tokens = s.rolling if self.anchor_mode == 'context' else s.observation
            anchor = hashlib.sha256(np.asarray(key_tokens, dtype=np.int64).tobytes()).hexdigest()
            step = len(s.anchors)
            if self.sampled_action_identity:
                from .action_identity import MAX_ACTIONS, MAX_PREFIX
                identity = action_identities[i]
                if step >= MAX_ACTIONS or len(s.rolling) > MAX_PREFIX or not s.rolling:
                    raise ValueError('Actual action/prefix exceeds the fixed sampled-action layout')
                raw_action = list(identity.actor_ids)
                offset = len(s.raw_actor_response)
                if offset + len(raw_action) + len(obs) > self.config.max_prompt_length:
                    raise ValueError('Raw actor timeline would truncate an action or observation')
                record = identity.metadata()
                record.update(step_id=step, generation_prefix=list(s.rolling),
                              prefix_position_ids=list(range(len(s.rolling))),
                              action_position_ids=list(range(len(s.rolling), len(s.rolling) + len(raw_action))),
                              prefix_attention_mask=[1] * len(s.rolling), action_attention_mask=[1] * len(raw_action),
                              actor_action_tokens=raw_action, actor_output_offset=offset,
                              canonical_text=text[i], canonical_action_tokens=list(action),
                              canonical_output_offset=len(s.response),
                              position_evidence_scope='DERIVED_NATURAL_POSITION_EXPECTATION')
                s.action_prefix_records.append(record)
                s.raw_actor_response.extend(raw_action + obs)
                s.raw_mask.extend([1] * len(raw_action) + [0] * len(obs))
                s.raw_steps.extend([step] * len(raw_action) + [-1] * len(obs))
            s.anchors.append(anchor)
            s.records.append(dict(text=text[i], action_tokens=len(action), observation_tokens=len(obs),
                                  scoring_context_matches_generation=matches[i]))
            s.response.extend(action + obs)
            s.model_mask.extend([1] * len(action) + [0] * len(obs))
            s.steps.extend([step] * len(action) + [-1] * len(obs))
            # Match base rolling and right-side caps exactly, including discarded tails.
            cap = self.config.max_prompt_length
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
        response = padded([s.response for s in states], pad)
        mask = padded([s.model_mask for s in states], 0)
        step_ids = padded([s.steps for s in states], -1)
        input_ids = torch.cat((prompts, response), 1)
        attention = (input_ids != pad).long()
        info = torch.cat(((prompts != pad).long(), mask), 1)
        tensors = dict(prompts=prompts, responses=response,
                       responses_with_info_mask=torch.where(mask.bool(), response, pad),
                       input_ids=input_ids, attention_mask=attention, info_mask=info,
                       position_ids=self.manager.tensor_fn.create_position_ids(attention),
                       step_ids=step_ids)
        non_tensors = dict(anchors=object_array([s.anchors for s in states]),
                           action_records=object_array([s.records for s in states]),
                           trajectory_uid=object_array([s.trajectory for s in states]),
                           tree_uid=object_array([s.tree for s in states]))
        result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        result.meta_info.update(self.output_meta)
        result.meta_info.update(dict(turns_stats=[s.turns + int(s.final_used) for s in states],
                                     active_mask=[not s.done for s in states],
                                     valid_action_stats=[s.valid_actions for s in states],
                                     valid_search_stats=[s.valid_searches for s in states]))
        if self.sampled_action_identity:
            return self._compose_sampled_actions(states, result), np.asarray([s.source for s in states], dtype=np.int64)
        return result, np.asarray([s.source for s in states], dtype=np.int64)

    def _compose_sampled_actions(self, states, canonical):
        """Raw loss targets and independent exact canonical prefixes per action.

        input_ids/position_ids are the row layout only: workers must dispatch
        actor_action_* to score each action at its own sampled canonical prefix.
        They cannot use the raw row concatenation as a replacement prefix.
        """
        from verl import DataProto
        from .action_identity import IDENTITY_MODE, MAX_ACTIONS, MAX_PREFIX, MAX_ACTION
        pad = self.tokenizer.pad_token_id
        count = len(states)
        prompts = canonical.batch['prompts']
        response = padded([s.raw_actor_response for s in states], pad)
        mask = padded([s.raw_mask for s in states], 0)
        steps = padded([s.raw_steps for s in states], -1)
        lengths = torch.tensor([len(s.raw_actor_response) for s in states], dtype=torch.long)
        response_attention = (torch.arange(response.shape[1])[None, :] < lengths[:, None]).long()
        prompt_attention = torch.zeros_like(prompts)
        for index, state in enumerate(states):
            prompt_attention[index, prompts.shape[1] - len(state.prompt):] = 1
        input_ids = torch.cat((prompts, response), 1)
        attention = torch.cat((prompt_attention, response_attention), 1)
        prefix_ids = torch.full((count, MAX_ACTIONS, MAX_PREFIX), pad, dtype=torch.long)
        action_ids = torch.full((count, MAX_ACTIONS, MAX_ACTION), pad, dtype=torch.long)
        action_lengths = torch.zeros((count, MAX_ACTIONS), dtype=torch.long)
        prefix_lengths = torch.zeros_like(action_lengths)
        offsets = torch.full_like(action_lengths, -1)
        for index, state in enumerate(states):
            if len(state.action_prefix_records) != len(state.anchors):
                raise ValueError('Missing complete sampled action provenance')
            for step, record in enumerate(state.action_prefix_records):
                prefix, action = record['generation_prefix'], record['actor_action_tokens']
                offset = record['actor_output_offset']
                if (record['step_id'] != step or not 0 < len(prefix) <= MAX_PREFIX or len(action) > MAX_ACTION
                        or not 0 <= offset <= len(state.raw_actor_response) - len(action)
                        or state.raw_actor_response[offset:offset + len(action)] != action
                        or state.raw_steps[offset:offset + len(action)] != [step] * len(action)
                        or state.raw_mask[offset:offset + len(action)] != [1] * len(action)):
                    raise ValueError('Sampled action output mapping changed')
                prefix_ids[index, step, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
                action_ids[index, step, :len(action)] = torch.tensor(action, dtype=torch.long)
                action_lengths[index, step], prefix_lengths[index, step], offsets[index, step] = len(action), len(prefix), offset
        tensors = dict(prompts=prompts, responses=response,
                       responses_with_info_mask=torch.where(mask.bool(), response, pad),
                       input_ids=input_ids, attention_mask=attention, info_mask=torch.cat((prompt_attention, mask), 1),
                       position_ids=self.manager.tensor_fn.create_position_ids(attention), step_ids=steps,
                       actor_action_prefix_ids=prefix_ids, actor_action_tokens=action_ids,
                       actor_action_lengths=action_lengths, actor_action_prefix_lengths=prefix_lengths,
                       actor_action_output_offsets=offsets,
                       actor_action_count=torch.tensor([len(s.action_prefix_records) for s in states], dtype=torch.long),
                       canonical_reward_responses=canonical.batch['responses'],
                       canonical_reward_attention_mask=canonical.batch['attention_mask'],
                       canonical_reward_info_mask=canonical.batch['info_mask'],
                       canonical_reward_step_ids=canonical.batch['step_ids'])
        non_tensors = dict(canonical.non_tensor_batch)
        non_tensors['action_prefix_records'] = object_array([s.action_prefix_records for s in states])
        result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors,
                                     meta_info=copy.deepcopy(canonical.meta_info))
        result.meta_info['actor_token_identity'] = IDENTITY_MODE
        return result

    def run(self, gen_batch, tree=None):
        if self.sampled_action_identity and tree is not None:
            raise ValueError('Sampled action identity is restricted to GiGPO without tree expansion')
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
