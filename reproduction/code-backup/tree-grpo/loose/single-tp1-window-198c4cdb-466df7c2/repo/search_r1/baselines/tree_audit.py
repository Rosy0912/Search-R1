"""Opt-in hash/count evidence only; no sampling, model, or metric mutations."""
import hashlib
import json
import os
import time
from pathlib import Path
from collections import Counter
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def state_signature(state):
    return {key: fingerprint(getattr(state, key)) for key in
            ('prompt', 'rolling', 'response', 'model_mask', 'steps', 'anchors', 'records', 'observation',
             'source', 'tree', 'node', 'turns', 'done', 'final_used', 'valid_actions', 'valid_searches')}


class TreeAudit:
    def __init__(self, directory, output_root, step, sequence, question_uids):
        self.directory = Path(directory).resolve()
        if not output_root or not self.directory.is_relative_to(Path(output_root).resolve()):
            raise ValueError('Tree audit must be inside own SEARCH_R1_BASELINE_OUTPUT')
        self.question_hashes = [fingerprint(str(uid)) for uid in question_uids]
        self.path = self.directory / ('step_%d_rollout_%d.json' % (step, sequence))
        self.report = dict(schema_version=1, global_step=int(step), rollout_sequence=int(sequence),
                           status='IN_PROGRESS', branches=[], leaves=[], question_groups=[],
                           diagnostic_check_cost={'scope':'hash and validation functions only; excludes generation waiting and publication IO'})
        self.closed = False

    def require(self, condition, code):
        if not condition:
            self.report.update(status='FAILED', failure_code=code)
            self.publish()
            raise ValueError('Tree audit invariant failed: '+code)

    def add_cost(self, phase, start):
        cost = self.report['diagnostic_check_cost']
        for suffix, elapsed in [('wall_seconds', time.perf_counter()-start[0]), ('process_cpu_seconds', time.process_time()-start[1])]:
            key = phase+'_'+suffix
            cost[key] = cost.get(key, 0.)+elapsed

    def branch(self, parent, child):
        start = time.perf_counter(), time.process_time()
        signatures = state_signature(parent), state_signature(child)
        row = dict(parent_node_hash=fingerprint(parent.node), parent_trajectory_hash=fingerprint(parent.trajectory),
                   child_trajectory_hash=fingerprint(child.trajectory), tree_hash=fingerprint(parent.tree),
                   source=int(parent.source), parent_fields=signatures[0], child_fields=signatures[1],
                   prefix_length=len(parent.rolling), inherited=signatures[0] == signatures[1])
        self.report['branches'].append(row)
        self.add_cost('branch', start)
        self.require(row['inherited'] and parent.trajectory != child.trajectory, 'branch_inheritance')

    def finish(self, states, output, sources, roots, tree, pad):
        start = time.perf_counter(), time.process_time()
        self.require(tree.leaves == 3, 'three_leaves_config')
        self.require(len(roots) == len(self.question_hashes), 'question_mapping_length')
        root_identity = {i: s.tree for i, s in enumerate(roots)}
        counts = Counter(s.source for s in states)
        self.require(counts == Counter({i: 3 for i in root_identity}), 'three_leaves_per_source')
        self.require(len({s.trajectory for s in states}) == len(states), 'unique_leaf_trajectories')
        self.require(len(set(root_identity.values())) == len(roots), 'unique_root_trees')
        groups = {}
        for i, question in enumerate(self.question_hashes):
            groups.setdefault(question, []).append(i)
        self.require(all(len(v) == 2 for v in groups.values()), 'two_roots_per_question')
        self.require(sources.tolist() == [s.source for s in states], 'compose_sources')
        width = output.batch['responses'].shape[1]
        self.require(output.batch['input_ids'].shape == output.batch['attention_mask'].shape == output.batch['info_mask'].shape, 'compose_input_shapes')
        self.require(output.batch['responses'].shape == output.batch['step_ids'].shape == output.batch['responses_with_info_mask'].shape, 'compose_response_shapes')
        for i, state in enumerate(states):
            self.require(state.tree == root_identity[state.source], 'leaf_tree_inheritance')
            n = len(state.response)
            self.require(n == len(state.model_mask) == len(state.steps), 'state_lengths')
            expected_actor, expected_step = [], []
            for step, record in enumerate(state.records):
                expected_actor.extend([1]*record['action_tokens']+[0]*record['observation_tokens'])
                expected_step.extend([step]*record['action_tokens']+[-1]*record['observation_tokens'])
            self.require(state.model_mask == expected_actor[:n] and state.steps == expected_step[:n], 'retained_record_mask')
            self.require(all((m == 0 and step == -1) or
                             (m == 1 and 0 <= step < len(state.anchors))
                             for m, step in zip(state.model_mask, state.steps)), 'observation_and_action_steps')
            mask = output.batch['info_mask'][i, -width:]
            steps = output.batch['step_ids'][i]
            response = output.batch['responses'][i]
            expected_mask = torch.tensor(state.model_mask + [0]*(width-n), dtype=mask.dtype, device=mask.device)
            expected_steps = torch.tensor(state.steps + [-1]*(width-n), dtype=steps.dtype, device=steps.device)
            expected_response = torch.tensor(state.response + [pad]*(width-n), dtype=response.dtype, device=response.device)
            self.require(torch.equal(mask, expected_mask) and torch.equal(steps, expected_steps)
                         and torch.equal(response, expected_response), 'compose_mask_steps_padding')
            self.require(torch.equal(output.batch['responses_with_info_mask'][i],
                                     torch.where(mask.bool(), response, pad)), 'compose_observation_zero')
            self.require(output.non_tensor_batch['tree_uid'][i] == state.tree and
                         output.non_tensor_batch['trajectory_uid'][i] == state.trajectory, 'compose_uid')
            self.report['leaves'].append(dict(source=state.source, question_hash=self.question_hashes[state.source],
                tree_hash=fingerprint(state.tree), trajectory_hash=fingerprint(state.trajectory),
                response_mask_hash=fingerprint(mask.tolist()), response_hash=fingerprint(response.tolist()),
                step_ids_hash=fingerprint(steps.tolist()), attention_response_hash=fingerprint(output.batch['attention_mask'][i, -width:].tolist()),
                response_length=n, action_count=sum(state.model_mask), observation_count=n-sum(state.model_mask),
                padding_count=width-n, anchor_count=len(state.anchors),
                retained_pad_token_count=state.response.count(pad),
                generation_prefix_hash=fingerprint(state.rolling), scoring_prefix_hash=fingerprint(state.prompt+state.response),
                context_matches=state.rolling == state.prompt+state.response))
        self.report['question_groups'] = [dict(question_hash=q, roots=len(indices), leaves=sum(counts[i] for i in indices)) for q,indices in groups.items()]
        self.add_cost('finish', start)
        self.report.update(status='ROLLOUT_DIAGNOSTIC_COMPLETE', branch_count=len(self.report['branches']), leaf_count=len(states))
        self.publish()


    def policy_batch(self, batch, global_step):
        start = time.perf_counter(), time.process_time()
        evidence = TreeAudit.__new__(TreeAudit)
        evidence.directory = self.directory
        evidence.path = self.path.with_name(self.path.stem+'_policy_batch.json')
        evidence.closed = False
        evidence.report = dict(schema_version=1, global_step=int(global_step),
                               rollout_sequence=self.report['rollout_sequence'], status='IN_PROGRESS',
                               rollout_file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), rows=[],
                               diagnostic_check_cost={'scope':'policy batch read-only checks and prior-rollout hash IO; excludes actor update and publication IO'})
        check = evidence.require
        check(self.report['status'] == 'ROLLOUT_DIAGNOSTIC_COMPLETE' and
              global_step == self.report['global_step'], 'policy_rollout_step')
        known = {row['trajectory_hash']:row for row in self.report['leaves']}
        observed = [fingerprint(str(uid)) for uid in batch.non_tensor_batch['trajectory_uid']]
        check(len(observed) == len(known) and len(set(observed)) == len(known) and set(observed) == set(known), 'policy_trajectory_set')
        width = batch.batch['responses'].shape[1]
        info = batch.batch['info_mask'][:, -width:]
        attention = batch.batch['attention_mask'][:, -width:]
        loss_mask = batch.batch['loss_mask']
        weights = batch.batch['policy_weights']
        check(info.shape == attention.shape == loss_mask.shape == weights.shape == batch.batch['old_log_probs'].shape == batch.batch['advantages'].shape, 'policy_tensor_shapes')
        check(torch.equal(loss_mask, info), 'policy_raw_loss_mask')
        check(bool(((info == 0) | (info == 1)).all()) and bool(((attention == 0) | (attention == 1)).all()), 'policy_binary_masks')
        effective = (attention*loss_mask).float()
        counts = effective.sum(-1, keepdim=True)
        expected = effective/counts.clamp_min(1)
        check(bool(torch.isfinite(weights).all()) and torch.allclose(weights, expected, atol=0., rtol=1e-6), 'policy_uniform_leaf_weights')
        check(torch.allclose(weights.sum(-1), (counts[:,0] > 0).float(), atol=1e-6, rtol=1e-6), 'policy_row_mass')
        selected = effective.bool()
        check(bool(torch.isfinite(batch.batch['old_log_probs'][selected]).all()) and
              bool(torch.isfinite(batch.batch['advantages'][selected]).all()), 'policy_finite_effective_values')
        for i, trajectory in enumerate(observed):
            previous = known[trajectory]
            check(fingerprint(str(batch.non_tensor_batch['tree_uid'][i])) == previous['tree_hash'] and
                  fingerprint(str(batch.non_tensor_batch['uid'][i])) == previous['question_hash'], 'policy_tree_question_binding')
            check(fingerprint(info[i].tolist()) == previous['response_mask_hash'] and
                  fingerprint(batch.batch['responses'][i].tolist()) == previous['response_hash'] and
                  fingerprint(batch.batch['step_ids'][i].tolist()) == previous['step_ids_hash'] and
                  fingerprint(attention[i].tolist()) == previous['attention_response_hash'], 'policy_row_tensor_binding')
            evidence.report['rows'].append(dict(trajectory_hash=trajectory, tree_hash=previous['tree_hash'],
                question_hash=previous['question_hash'], effective_tokens=int(counts[i,0]),
                raw_model_tokens=int(info[i].sum()), policy_weight_sum=float(weights[i].sum()),
                empty_leaf=bool(counts[i,0] == 0)))
        evidence.add_cost('policy', start)
        evidence.report.update(status='POLICY_BATCH_DIAGNOSTIC_COMPLETE', leaf_count=len(observed),
                               empty_leaves=int((counts == 0).sum()))
        evidence.publish()

    def publish(self):
        if self.closed:
            raise ValueError('Tree audit already published')
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.pending.%d' % os.getpid())
        with temporary.open('x') as stream:
            json.dump(self.report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, self.path)  # atomic publication, no replacing existing evidence
        finally:
            temporary.unlink()
        self.closed = True
