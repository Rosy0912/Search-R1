"""Resumable search trajectories, retaining the reviewed parser/retriever.

Only new algorithms use this engine. The original GRPO/validation path is kept.
All metadata is row-local, so DataProto permutations also permute provenance.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import os
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
    raw_ledger: list = field(default_factory=list)
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
    def __init__(self, manager, anchor_mode='observation', ranker=None, seed=42, context_alignment='report', token_preservation=None, behavior_mu=None):
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
        self.token_preservation = token_preservation
        from .tree_behavior_mu import validate_options, runtime_provenance, require
        self.behavior_mu = validate_options(behavior_mu)
        require(self.behavior_mu is None or token_preservation is not None,
                'Behavior probabilities require raw-token dual representation')
        self.behavior_mu_provenance = runtime_provenance(os.environ) if self.behavior_mu else None
        self.tokenizer_identity = None
        self.seen_sampling_ids = set()
        self.failure_context = None
        if token_preservation is not None:
            from .tree_raw_tokens import tokenizer_identity, require
            require(self.tokenizer.pad_token_id == 151643 and self.tokenizer.eos_token_id == 151645,
                    'Frozen actual PAD/EOS identity changed')
            self.tokenizer_identity = tokenizer_identity(self.tokenizer, token_preservation['tokenizer_sha256'])
        self.cost = dict(generated_tokens=0, prefill_tokens=0, retrieval_calls=0,
                         trajectories=0, padded_rows=0, context_mismatches=0)
        if token_preservation is not None:
            self.cost.update(raw_slots=0, raw_emitted_tokens=0, raw_retained_tokens=0,
                raw_discarded_tail_tokens=0, raw_inherited_leaf_tokens=0, semantic_prefix_scan_cpu_seconds=0.,
                semantic_prefix_scan_wall_seconds=0.)

    def initial_states(self, gen_batch):
        states = []
        for i in range(len(gen_batch)):
            ids = gen_batch.batch['input_ids'][i][gen_batch.batch['attention_mask'][i].bool()].tolist()
            if self.token_preservation is not None:
                from .tree_raw_tokens import require
                require(self.tokenizer.pad_token_id not in ids and len(ids) <= self.config.max_start_length,
                        'Initial prompt PAD/left truncation is unsupported')
            states.append(State(i, ids[-self.config.max_start_length:], ids, observation=ids))
        return states

    def pending(self, state):
        return not state.done and (state.turns < self.config.max_turns or
                                  (self.config.force_final_answer and not state.final_used))

    def advance(self, states, meta):
        started=None
        if self.token_preservation is not None:
            import time
            started=time.perf_counter(),time.process_time()
            self.failure_context=dict(stage='before_generation',contexts=[s.rolling.copy() for s in states],
                scoring_contexts=[s.prompt+s.response for s in states],raw_ledgers=[s.raw_ledger for s in states])
        try:
            return self._advance(states, meta)
        except Exception as exc:
            if self.token_preservation is not None and os.environ.get('SEARCH_R1_BASELINE_OUTPUT'):
                from .training_state import atomic_save
                directory=os.path.join(os.environ['SEARCH_R1_BASELINE_OUTPUT'],'token-preservation-rejections')
                os.makedirs(directory,exist_ok=True)
                atomic_save(dict(status='CURRENT_ROLLOUT_REJECTED_BEFORE_RETURN',error=repr(exc),
                    cost=self.cost.copy(),failure_context=self.failure_context,
                    tokenizer_identity=self.tokenizer_identity,probability_accepted=False,
                    rejected_advance_wall_seconds=time.perf_counter()-started[0],
                    rejected_advance_cpu_seconds=time.process_time()-started[1]),
                    os.path.join(directory,uuid.uuid4().hex+'.json'),json_format=True)
            raise

    def _advance(self, states, meta):
        from verl import DataProto
        matches = [s.prompt + s.response == s.rolling for s in states]
        mismatches = sum(not match for match in matches)
        self.cost['context_mismatches'] += mismatches
        if mismatches and (self.context_alignment == 'error' or self.token_preservation is not None):
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
        prepared_actions = None
        if self.token_preservation is not None:
            from .tree_raw_tokens import EMISSION, sampled_action, require
            emissions=output.non_tensor_batch.get(EMISSION)
            require(emissions is not None and len(emissions) == len(generated),'Actual emitted length/finish metadata missing')
            self.cost['raw_slots'] += int(generated.numel())
            self.cost['raw_emitted_tokens'] += sum(record['true_emitted_length'] for record in emissions)
            self.failure_context=dict(native_emissions=emissions.tolist(),legacy_text=text,
                contexts=[s.rolling.copy() for s in states])
            prepared_actions=[]
            for i,s in enumerate(states):
                record=emissions[i]
                require(record['prompt_token_ids'] == s.rolling,'Actual native prompt differs from policy/scoring history')
                require(record['sample_id'] not in self.seen_sampling_ids,'Duplicate physical sampling identity')
                self.seen_sampling_ids.add(record['sample_id'])
                action,ledger=sampled_action(self.tokenizer,record,raw[i].tolist(),
                    tokens[i][tokens[i] != pad].tolist(),text[i],self.tokenizer_identity)
                prepared_actions.append((action,ledger))
                self.cost['raw_retained_tokens'] += len(action)
                self.cost['raw_discarded_tail_tokens'] += record['true_emitted_length']-len(action)
                self.cost['semantic_prefix_scan_cpu_seconds'] += ledger['prefix_scan_cpu_seconds']
                self.cost['semantic_prefix_scan_wall_seconds'] += ledger['prefix_scan_wall_seconds']
            self.cost['raw_discarded_tail_tokens'] += sum(record['true_emitted_length'] for record in emissions[len(states):])
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
            action = tokens[i][tokens[i] != pad].tolist() if prepared_actions is None else prepared_actions[i][0]
            obs = obs_tokens[i][obs_tokens[i] != pad].tolist()
            if prepared_actions is not None:
                from .tree_raw_tokens import require
                ledger=prepared_actions[i][1]
                require(len(s.response)+len(action)+len(obs) <= self.config.max_prompt_length and
                        len(s.rolling)+len(action)+len(obs) <= self.config.max_prompt_length,
                        'Score-cap or rolling left-crop requires a separate unambiguous projection protocol')
                semantic_count=sum(len(x['action']['canonical_semantic_token_ids'])+len(x['observation_token_ids'])
                                   for x in s.raw_ledger)+len(ledger['canonical_semantic_token_ids'])+len(obs)
                require(semantic_count <= self.config.max_prompt_length,'Legacy semantic timeline would cross score cap')
                s.raw_ledger.append(dict(turn=len(s.anchors),action=ledger,observation_token_ids=obs.copy(),
                    observation_text=observations[i],generation_context_token_ids=s.rolling.copy(),
                    generation_context_attention=[1]*len(s.rolling),
                    generation_context_positions=positions[i][attention[i].bool()].tolist()))
            key_tokens = s.rolling if self.anchor_mode == 'context' else s.observation
            anchor = hashlib.sha256(np.asarray(key_tokens, dtype=np.int64).tobytes()).hexdigest()
            step = len(s.anchors)
            s.anchors.append(anchor)
            s.records.append(dict(text=text[i], action_tokens=len(action), observation_tokens=len(obs),
                                  scoring_context_matches_generation=matches[i]))
            if os.environ.get('SEARCH_R1_TRAJECTORY_DIR'):
                s.records[-1].update(action_token_ids=action, observation_token_ids=obs,
                    raw_generation_token_ids=raw[i].tolist(),
                    observation_text=observations[i], generation_context_token_ids=s.rolling.copy(),
                    turn=step, valid_action=bool(valid[i]), valid_search=bool(searches[i]),
                    done=bool(dones[i]), search_enabled=bool(normal[i]))
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
        if self.token_preservation is not None:
            from .tree_raw_tokens import FIELD, trajectory_projection, require
            non_tensors[FIELD]=object_array([trajectory_projection(s,self.config.max_prompt_length) for s in states])
            occurrences={}
            total=0
            for state in states:
                for record in state.raw_ledger:
                    action=record['action'];sample=action['native_emission']['sample_id']
                    for offset,(token,logp) in enumerate(zip(action['action_token_ids'],action['selected_logprobs'])):
                        key=(sample,offset);value=(token,logp)
                        require(key not in occurrences or occurrences[key]==value,'Inherited sampling occurrence changed')
                        occurrences[key]=value;total+=1
            self.cost['raw_inherited_leaf_tokens']=total-len(occurrences)
        if self.behavior_mu is not None:
            from .tree_behavior_mu import compose_behavior_mu_rows, TENSOR, FIELD as MU_FIELD
            sampled, bindings = compose_behavior_mu_rows(states, non_tensors[FIELD],
                response_width=response.shape[1], expected_provenance=self.behavior_mu_provenance)
            tensors[TENSOR] = sampled
            non_tensors[MU_FIELD] = object_array(bindings)
        result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        result.meta_info.update(self.output_meta)
        if self.token_preservation is not None:
            from .tree_raw_tokens import PROTOCOL
            result.meta_info['tree_raw_token_protocol']=PROTOCOL
        if self.behavior_mu is not None:
            from .tree_behavior_mu import (PROTOCOL as MU_PROTOCOL, META, PROVENANCE_META,
                                           validate_behavior_mu_batch)
            result.meta_info[META] = MU_PROTOCOL
            result.meta_info[PROVENANCE_META] = copy.deepcopy(self.behavior_mu_provenance)
            result.meta_info['tree_behavior_mu_collection'] = validate_behavior_mu_batch(result,
                expected_provenance=self.behavior_mu_provenance)
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
