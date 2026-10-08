"""External metadata-only hooks around frozen trainer.fit's existing RPCs.

No batch reorder, token repair, forward pass, optimizer/scheduler write, RNG
restore or source-file edit occurs here. Failed identity stops before the
original update_actor RPC. This candidate still requires actual GPU review.
"""
import copy
import json
import os
from pathlib import Path

import torch
from calibration_state_v1 import state_digest
from formal_probability_v1 import validate_outer, validate_live_arms, processed_tokens_without_right_padding, require
from formal_probability_proofs_v1 import file_record, verify_record, manifest_records, validate_manifest_closure, read_json_record
from formal_probability_worker_v1 import scored_batch_state
from formal_audit_normalization_v1 import normalize_exact_audit_rows
from search_r1.baselines.recovery import atomic_json_save, atomic_torch_save, file_sha256


def raw_bound(record):
    path = Path(record['path'])
    require(path.is_file() and path.stat().st_size == record['bytes']
            and file_sha256(path) == record['sha256'], 'Actual raw sampling receipt changed')
    return torch.load(path, map_location='cpu', weights_only=False)


def actual_driver_boundary(trainer, batch):
    """Observe live post-sampling RNG/cursor/GiGPO and fixed scored batch only."""
    from search_r1.baselines.recovery import rng_state
    return dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                rng=rng_state(), sampler=trainer.recovery_sampler.state_dict(),
                global_steps=trainer.global_steps, actual_outer_updates=trainer.actual_outer_updates,
                kl_controller=vars(trainer.kl_ctrl), stage=trainer.baseline_controller.stage.state_dict(),
                gigpo_parameters=dict(trainer.config.baselines.gigpo),
                quality_health=copy.deepcopy(trainer.quality_health), fixed_batch=scored_batch_state(batch))


def reconstruct_actions(batch, turns, directory, binding, source, outer):
    require(len(batch) == 2560 and len(turns) > 0 and len(turns) <= 5,
            'Wrong actual full outer response count')
    row_ids = [int(value) for value in batch.non_tensor_batch['audit_trajectory_row']]
    require(set(row_ids) == set(range(2560)) and len(row_ids) == 2560,
            'Length balancing lost/duplicated actual trajectory mapping')
    by_row = {original: position for position, original in enumerate(row_ids)}
    records = [dict(source=index, question_id=str(batch.non_tensor_batch['index'][by_row[index]]),
                    response_group=index // 5, response_index=index % 5,
                    trajectory_uid=str(batch.non_tensor_batch['trajectory_uid'][by_row[index]]),
                    actions=[]) for index in range(2560)]
    require(all(len({records[index]['question_id'] for index in range(group*5, group*5+5)}) == 1
                for group in range(512)) and len({records[group*5]['question_id'] for group in range(512)}) == 512,
            'Actual512 question / five-response grouping differs')
    pad = binding['probability_identity']['pad_token_id']
    raw_files, raw_receipts = [], []
    for serial, observed in enumerate(turns):
        raws = []
        for rank in range(8):
            path = directory / 'raw' / f'call-{serial}-rank-{rank}.json'
            receipt = json.loads(path.read_text())
            require(receipt['rank'] == rank and receipt['serial'] == serial
                    and receipt['outer_step'] == outer and receipt['source_commit'] == source,
                    'Actual probability turn belongs to another outer/source/rank')
            raw = raw_bound(receipt['raw'])
            require(raw['rank'] == rank and raw['serial'] == serial and raw['outer_step'] == outer
                    and raw['source_commit'] == source and raw['pid'] == receipt['pid'],
                    'Actual raw sampler identity differs from receipt')
            require(raw['actual_pad_token_id'] == pad
                    and raw['actual_eos_token_id'] == binding['probability_identity']['eos_token_id'],
                    'Actual official tokenizer pad/EOS differs from accepted probability identity')
            raws.append(raw)
            raw_files.append(dict(rank=rank, serial=serial, **receipt['raw']))
            raw_receipts.append(dict(rank=rank, serial=serial, **file_record(path)))
        sizes = [len(raw['tensors']['responses']) for raw in raws]
        require(len(set(sizes)) == 1 and sum(sizes) >= len(observed)
                and sum(sizes) - len(observed) < 8, 'Unexpected raw DP padding/partition')
        for active_row, action in enumerate(observed):
            rank, local = divmod(active_row, sizes[0])
            raw = raws[rank]
            tensors = raw['tensors']
            prompt_width = tensors['prompts'].shape[1]
            sample_width = raw['sampled_ids'].shape[1]
            prompt_mask = tensors['attention_mask'][local, :prompt_width].bool()
            worker_prompt = tensors['prompts'][local][prompt_mask].tolist()
            response_attention = tensors['attention_mask'][local, prompt_width:prompt_width + sample_width].tolist()
            source_row = action['source']
            target = by_row[source_row]
            selected = batch.batch['step_ids'][target].eq(action['step_id']) & batch.batch['loss_mask'][target].bool()
            selected_positions = selected.nonzero().flatten().tolist()
            length = len(selected_positions)
            # For a zero-token action the unmodified advance observer is its
            # only scoring boundary; it has no actor probability mass. Nonempty
            # actions must occupy one contiguous model span in the scored data.
            if length:
                require(selected_positions == list(range(selected_positions[0], selected_positions[0] + length)),
                        'Effective action crosses a masked observation/span')
                boundary = batch.batch['prompts'].shape[1] + selected_positions[0]
                prefix_mask = batch.batch['attention_mask'][target, :boundary].bool()
                final_prefix = batch.batch['input_ids'][target, :boundary][prefix_mask].tolist()
            else:
                final_prefix = action['scoring_prefix']
            value = dict(action,
                sampler_runner_positions_observed=False,
                engine_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION',
                final_scoring_prefix=final_prefix,
                worker_prompt_token_ids=worker_prompt,
                engine_prompt_token_ids=raw['actual_engine_prompt_token_ids'][local],
                raw_sampled_tokens=raw['sampled_ids'][local].tolist(),
                raw_sampled_response_mask=response_attention,
                worker_response_tokens=tensors['responses'][local].tolist(),
                effective_scored_tokens=batch.batch['responses'][target][selected].tolist(),
                worker_prompt_position_ids=tensors['position_ids'][local, :prompt_width][prompt_mask].tolist(),
                worker_action_position_ids=tensors['position_ids'][local, prompt_width:prompt_width + length].tolist(),
                scoring_action_position_ids=batch.batch['position_ids'][target, -batch.batch['responses'].shape[1]:][selected].tolist(),
                worker_action_mask=tensors['attention_mask'][local, prompt_width:prompt_width + length].tolist(),
                scoring_action_mask=batch.batch['loss_mask'][target][selected].tolist(),
                sampling_parameters=raw['actual_sampling_parameters'],
                per_turn_scoring_temperature=raw['per_turn_scoring_temperature'],
                full_scoring_temperature=batch.meta_info['temperature'],
                full_old_log_probs=batch.batch['old_log_probs'][target][selected].double().tolist(),
                per_turn_old_log_probs=tensors['old_log_probs'][local, :length].double().tolist(),
                raw_sampled_log_probs=raw['sampled_log_probs'][local, :length].double().tolist())
            records[source_row]['actions'].append(value)
    return records, raw_files, raw_receipts


def reconstruct_reference_records(batch, directory, source, outer):
    from formal_protection_contract_v1 import MODEL
    reference_records = []
    reference_keys = ('input_ids', 'attention_mask', 'position_ids', 'prompts', 'responses')
    chunks = batch.chunk(chunks=8)
    for rank, chunk in enumerate(chunks):
        ref_path = directory / f'full-prefix-reference-rank-{rank}.json'
        reference = json.loads(ref_path.read_text())
        raw = raw_bound(reference['raw'])
        reference_digest = state_digest({key: chunk.batch[key] for key in reference_keys})
        require(reference['rank'] == rank and reference['outer_step'] == outer
                and reference['source_commit'] == source
                and reference['input_digest'] == reference_digest == state_digest(raw['tensors'])
                and reference['fixed_reference_identity'] == reference['reference_identity_after']
                and reference['fixed_reference_identity']['source'] == MODEL
                and reference['fixed_reference_identity']['finite'] is True
                and reference['temperature'] == 1
                and reference['probability_scope'] == 'EXISTING_FULL_SCORING_BATCH_NOT_NEW_RAW_PREFIX_REFERENCE_PROBE',
                'Fixed official reference did not score the exact same full batch/prefix')
        require(torch.equal(raw['ref_log_prob'], chunk.batch['ref_log_prob'])
                and torch.isfinite(raw['ref_log_prob'][chunk.batch['loss_mask'].bool()]).all(),
                'Actual effective fixed reference log-probs differ/nonfinite')
        reference_records.append(dict(reference, receipt=file_record(ref_path)))
    return reference_records


def collect_outer(trainer, batch, turns, directory, binding):
    source = os.environ['SEARCH_R1_BASELINE_COMMIT']
    outer = trainer.global_steps
    self_model_path(trainer)
    # Use the tokenizer that actually built this trainer, and bind it to the
    # independently accepted sampling identity rather than relying on a local
    # variable in a different reconstruction function.
    pad, eos = trainer.tokenizer.pad_token_id, trainer.tokenizer.eos_token_id
    accepted_tokenizer = binding['probability_identity']
    require(type(pad) is int and type(eos) is int
            and pad == accepted_tokenizer['pad_token_id']
            and eos == accepted_tokenizer['eos_token_id'],
            'Actual driver tokenizer PAD/EOS differs from bound sampling acceptance')
    records, raw_files, raw_receipts = reconstruct_actions(batch, turns, directory, binding, source, outer)
    # Preserve exact observed raw evidence and the actual scored data before
    # validation. A rejection cannot erase the structural/probability cause.
    atomic_json_save(dict(outer_step=outer, source_commit=source, records=records), directory / 'actions.json')
    fixed = atomic_torch_save(scored_batch_state(batch), directory / 'scored-global-batch.pt')
    checks = validate_outer(records, binding['probability_identity']['tolerances'], pad)
    reference_records = reconstruct_reference_records(batch, directory, source, outer)
    proof = binding['probability_identity']['acceptance']
    manifest = dict(schema=1, status='VERIFIED_BY_RUNTIME_NOT_INDEPENDENT_ACCEPTANCE',
        outer_step=outer, source_commit=source, job_id=os.environ['SLURM_JOB_ID'],
        run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        production_artifact_sha256=binding['production_artifact']['sha256'],
        probability_acceptance_sha256=proof['sha256'],
        actual_driver_tokenizer_identity=dict(pad_token_id=pad, eos_token_id=eos),
        global_batch_digest=state_digest(scored_batch_state(batch)),
        driver_post_sampling_state_digest=state_digest(actual_driver_boundary(trainer, batch)),
        observed_turns=file_record(directory / 'observed-turns.json'), actions=file_record(directory / 'actions.json'),
        scored_global_batch=fixed, raw_files=raw_files, raw_receipts=raw_receipts,
        reference_records=reference_records, checks=checks,
        chunks=[dict(rank=rank, rows=len(chunk), batch_digest=state_digest(scored_batch_state(chunk)))
                for rank, chunk in enumerate(batch.chunk(chunks=8))])
    manifest['evidence_files'] = [dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'])
                                  for row in manifest_records(manifest)]
    validate_manifest_closure(manifest, directory)
    path = directory / 'before-adam-manifest.json'
    atomic_json_save(manifest, path)
    return path, manifest


def self_model_path(trainer):
    from formal_protection_contract_v1 import MODEL
    require(trainer.config.actor_rollout_ref.ref.model_path == MODEL, 'Reference is not the official frozen model')
    return MODEL


class OuterProbabilityHooks:
    """Install once on an original trainer; reuse its complete fit/restore/gates."""
    def __init__(self, trainer, binding):
        self.trainer, self.binding = trainer, binding
        self.original_generate = trainer.baseline_controller.generate
        self.original_update = trainer.actor_rollout_wg.update_actor
        self.original_reference = trainer.ref_policy_wg.compute_ref_log_prob
        self.turns = None
        self.outer = None
        self.last_completed = 25

    def generate(self, manager, gen_batch, qa_batch, metrics):
        from search_r1.baselines.rollout import SearchRollout
        # Frozen source52 appends int64 audit IDs after DataProto construction.
        # Its select_rows interface requires object metadata. Preserve exact
        # IDs/order and reject malformed provenance before the original rollout.
        normalize_exact_audit_rows(qa_batch)
        outer = self.trainer.global_steps
        require(outer == self.last_completed + 1 and 26 <= outer <= 300,
                'Unexpected formal outer fetch/generation')
        require(len(qa_batch) == 2560 and list(qa_batch.non_tensor_batch['audit_trajectory_row']) == list(range(2560)),
                'Missing actual ordered512x5 source rows before generation')
        self.outer, self.turns = outer, []
        self.directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'probability-outer' / f'iter-{outer:06d}'
        require(not self.directory.exists(), 'Duplicate outer metadata directory; do not overwrite evidence')
        self.directory.mkdir(parents=True)
        begun = self.trainer.actor_rollout_wg.production_begin_outer(outer)
        require(len(begun) == 8 and {row['rank'] for row in begun} == set(range(8)),
                'Missing actual eight-rank generation hook')
        original = SearchRollout.advance

        def observed(engine, states, meta):
            require(engine.manager is manager, 'Observer attached to an unrelated generation manager')
            before = [dict(source=state.source, step_id=len(state.anchors),
                           generation_prefix=list(state.rolling),
                           scoring_prefix=list(state.prompt + state.response)) for state in states]
            original_postprocess = manager._postprocess_responses
            processed = []

            def postprocess(*args, **kwargs):
                result = original_postprocess(*args, **kwargs)
                processed.append([processed_tokens_without_right_padding(tokens.tolist(), manager.tokenizer.pad_token_id)
                                  for tokens in result[0]])
                return result

            manager._postprocess_responses = postprocess
            try:
                result = original(engine, states, meta)
            finally:
                manager._postprocess_responses = original_postprocess
            require(len(processed) == 1 and len(processed[0]) == len(states),
                    'Missing complete actual processed action observer')
            for index, (row, state) in enumerate(zip(before, states)):
                row['complete_processed_action_tokens'] = processed[0][index]
                row['retained_action_tokens'] = [token for token, step in zip(state.response, state.steps)
                                                 if step == row['step_id']]
            self.turns.append(before)
            atomic_json_save(self.turns, self.directory / 'observed-turns.json')
            return result

        SearchRollout.advance = observed
        try:
            output = self.original_generate(manager, gen_batch, qa_batch, metrics)
        finally:
            SearchRollout.advance = original
            self.trainer.actor_rollout_wg.production_end_sampling(outer)
        require(len(output[0]) == 2560, 'Incomplete actual generated response group')
        return output

    def update(self, batch):
        outer = self.trainer.global_steps
        require(self.outer == outer and self.last_completed + 1 == outer and self.turns is not None,
                'Original update RPC has no observed actual outer')
        try:
            path, manifest = collect_outer(self.trainer, batch, self.turns, self.directory, self.binding)
            receipt_sha = file_sha256(path)
            driver_before = state_digest(actual_driver_boundary(self.trainer, batch))
            require(driver_before == manifest['driver_post_sampling_state_digest'],
                    'Driver RNG/cursor/GiGPO/batch changed during probability check; zero Adam')
            armed = self.trainer.actor_rollout_wg.production_arm_outer(str(path), receipt_sha)
            validate_live_arms(armed, outer, receipt_sha)
            for row in armed:
                physical = read_json_record(row['arm_receipt'], self.directory)
                require(physical['manifest_sha256'] == receipt_sha
                        and physical['rank'] == row['rank'] and physical['pid'] == row['pid']
                        and physical['start_ticks'] == row['start_ticks']
                        and physical['complete_pause_state_sha256'] == row['complete_pause_state_sha256']
                        and physical['pause_state'] == row['pause_state'],
                        'Physical worker arm state differs from actual RPC identity')
                verify_record(physical['pause_state'], self.directory)
            require(state_digest(actual_driver_boundary(self.trainer, batch)) == driver_before,
                    'Driver RNG/cursor/GiGPO/batch changed while arming workers; zero Adam')
            arms_path = self.directory / 'armed-workers.json'
            atomic_json_save(dict(outer_step=outer, manifest_sha256=receipt_sha,
                driver_post_sampling_state_digest=driver_before, workers=armed), arms_path)
            arms_sha = file_sha256(arms_path)
            batch.meta_info['formal_probability_gate'] = dict(manifest_sha256=receipt_sha,
                armed_workers_path=str(arms_path), armed_workers_sha256=arms_sha)
        except Exception as error:
            failure = self.directory / 'before-adam-rejection.json'
            atomic_json_save(dict(status='ACTUAL_OUTER_REJECTED_BEFORE_FIRST_ADAM', outer_step=outer,
                source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                error_type=type(error).__name__, error=str(error), actual_adam_writes=0,
                evidence_directory=str(self.directory)), failure)
            rejected = self.trainer.actor_rollout_wg.production_reject_outer(
                outer, str(failure), file_sha256(failure))
            require(len(rejected) == 8 and all(row['status'] == 'ALL_RANKS_REJECTED_BEFORE_ORIGINAL_UPDATE_RPC'
                                             for row in rejected), 'Incomplete matched all-rank refusal')
            atomic_json_save(dict(status='MATCHED_ALL_RANK_REFUSAL_RETURNED',
                outer_step=outer, actual_adam_writes=0, rejected_workers=rejected),
                self.directory / 'before-adam-rejection-workers.json')
            raise
        # Only metadata is added; all original math tensors, rows, masks and
        # sampling/state RNG values remain untouched. Worker digests exclude
        # this declaration and inspect the real dispatched chunk before Adam.
        output = self.original_update(batch)
        self.last_completed = outer
        self.turns = None
        return output

    def reference(self, batch):
        tagged = self.trainer.ref_policy_wg.production_reference_outer(self.trainer.global_steps)
        require(len(tagged) == 8 and {row['rank'] for row in tagged} == set(range(8)),
                'Missing actual fixed-reference phase tags')
        return self.original_reference(batch)

    def install(self):
        require(self.trainer.baseline_controller.method == 'gigpo'
                and not self.trainer.use_critic and self.trainer.use_reference_policy,
                'Hooks are only the accepted strict GiGPO path')
        self.trainer.baseline_controller.generate = self.generate
        self.trainer.actor_rollout_wg.update_actor = self.update
        self.trainer.ref_policy_wg.compute_ref_log_prob = self.reference
        return self
