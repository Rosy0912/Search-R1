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
        self.audit = None
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
        self.cost['generated_tokens'] += int((generated != pad).sum())
        raw = generated[:len(states)]
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
        return result, np.asarray([s.source for s in states], dtype=np.int64)

    def run(self, gen_batch, tree=None):
        self.cost = dict.fromkeys(self.cost, 0)
        states = self.initial_states(gen_batch)
        self.tree_candidate_counts = []
        roots = states[:]
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
                        for _ in range(tree.expansions):
                            parent = self.rng.choice(nodes)
                            child = parent.branch()
                            if self.audit is not None:
                                self.audit.branch(parent, child)
                            children.append(child)
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
        output, sources = self.compose(states)
        if self.audit is not None:
            self.audit.finish(states, output, sources, roots, tree, self.tokenizer.pad_token_id)
        return output, sources
