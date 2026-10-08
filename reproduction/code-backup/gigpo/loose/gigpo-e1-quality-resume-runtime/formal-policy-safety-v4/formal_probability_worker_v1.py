"""External production metadata hook; original FSDP update remains unchanged.

No native acceptance is implied by this local candidate. Every invocation first
reaches an all-rank identity decision, then calls the frozen worker method once.
The raw engine observer returns the original object and consumes no RNG itself.
"""
import copy
import json
import os
from pathlib import Path

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from search_r1.baselines.recovery import atomic_torch_save, atomic_json_save, file_sha256
from calibration_worker_v1 import CalibrationWorker
from calibration_state_v1 import state_digest
from formal_probability_v1 import collective_gate, collective_rejection_gate, validate_live_arms, require
from formal_probability_proofs_v1 import file_record, verify_record, validate_manifest_closure, read_json_record


def scored_batch_state(data):
    """Keep all tensor and row provenance values; no actor select/drop boundary yet."""
    return dict(tensors=dict(data.batch.items()), non_tensors=data.non_tensor_batch,
                metadata={k: v for k, v in data.meta_info.items() if k != 'formal_probability_gate'})


class ProductionProbabilityWorker(CalibrationWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_backend_identity(self):
        require(self._is_actor,'Backend metadata belongs to the actual actor rollout worker')
        from formal_backend_v1 import observe_backend_identity,require_flash_environment
        require_flash_environment()
        return observe_backend_identity(self.rollout.inference_engine,self.rank,
            os.environ['SEARCH_R1_BASELINE_COMMIT'],os.environ['GIGPO_PROTECTION_FORMAL_BINDING_SHA256'])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_begin_outer(self, outer_step):
        require(self._is_actor and self.world_size == 8 and 26 <= outer_step <= 300,
                'Wrong production worker/rank/outer scope')
        require(self.actor_lr_scheduler.last_epoch == outer_step - 1,
                'Scheduler is not the preceding complete outer')
        require(getattr(self, '_production_last_admitted', outer_step - 1) == outer_step - 1,
                'Duplicate/skipped production outer')
        self._production_outer_step = outer_step
        self._production_capture = True
        self._production_call = 0
        self._production_arm = None
        return dict(rank=self.rank, pid=os.getpid(), outer_step=outer_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_end_sampling(self, outer_step):
        require(self._production_outer_step == outer_step and self._production_capture,
                'No actual production sampling phase')
        self._production_capture = False
        return dict(rank=self.rank, pid=os.getpid(), outer_step=outer_step, calls=self._production_call)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        if not getattr(self, '_production_capture', False):
            return ActorRolloutRefWorker.generate_sequences(self, prompts)
        engine = self.rollout.inference_engine
        original = engine.generate
        captured = []

        def observe(*args, **kwargs):
            output = original(*args, **kwargs)
            params = kwargs['sampling_params']
            captured.append(dict(sampled_ids=output[0].detach().cpu().clone(),
                sampled_log_probs=output[1].detach().cpu().clone(),
                actual_engine_prompt_token_ids=copy.deepcopy(kwargs['prompt_token_ids']),
                actual_sampling_parameters=dict({key: getattr(params, key) for key in
                    ('temperature', 'top_p', 'top_k', 'n', 'logprobs', 'max_tokens',
                     'min_p', 'best_of', 'repetition_penalty', 'presence_penalty', 'frequency_penalty')},
                    logits_processors_count=0 if params.logits_processors is None else len(params.logits_processors))))
            return output

        engine.generate = observe
        try:
            # No extra forward/recompute/probe or RNG restoration is introduced.
            output = ActorRolloutRefWorker.generate_sequences(self, prompts)
        finally:
            engine.generate = original
        require(len(captured) == 1 and 'old_log_probs' in output.batch,
                'Missing actual raw sampler/per-turn actor probabilities')
        raw = captured[0]
        raw.update(tensors={key: output.batch[key].detach().cpu().clone() for key in
                   ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs')},
                   per_turn_scoring_temperature=output.meta_info['temperature'],
                   rank=self.rank, pid=os.getpid(), outer_step=self._production_outer_step,
                   serial=self._production_call,
                   source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                   actual_pad_token_id=self.tokenizer.pad_token_id,
                   actual_eos_token_id=self.tokenizer.eos_token_id)
        directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'probability-outer' / (
                    f'iter-{self._production_outer_step:06d}') / 'raw'
        directory.mkdir(parents=True, exist_ok=True)
        receipt = atomic_torch_save(raw, directory / f'call-{self._production_call}-rank-{self.rank}.pt')
        atomic_json_save(dict(rank=self.rank, pid=os.getpid(), serial=self._production_call,
                         outer_step=self._production_outer_step, source_commit=raw['source_commit'],
                         rows=len(output), raw=receipt),
                         directory / f'call-{self._production_call}-rank-{self.rank}.json')
        self._production_call += 1
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_reference_outer(self, outer_step):
        require(self._is_ref and self.world_size == 8 and 26 <= outer_step <= 300,
                'Wrong actual fixed reference scope')
        self._production_reference_outer = outer_step
        return dict(rank=self.rank, pid=os.getpid(), outer_step=outer_step)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        # Existing validation does not call reference scoring. If another source
        # path does, observe only production's explicitly tagged full batch.
        outer = getattr(self, '_production_reference_outer', None)
        if outer is None:
            return ActorRolloutRefWorker.compute_ref_log_prob(self, data)
        before = self.calibration_reference_identity()
        keys = ('input_ids', 'attention_mask', 'position_ids', 'prompts', 'responses')
        tensors = {key: data.batch[key].detach().cpu().clone() for key in keys}
        input_digest = state_digest(tensors)
        output = ActorRolloutRefWorker.compute_ref_log_prob(self, data)
        after = self.calibration_reference_identity()
        require(before == after, 'Fixed reference shards/buffers changed during existing full-prefix forward')
        expected = getattr(self, '_production_initial_reference', before)
        require(expected == before, 'Fixed official reference changed between outers')
        self._production_initial_reference = before
        directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'probability-outer' / f'iter-{outer:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        receipt = atomic_torch_save(dict(tensors=tensors, ref_log_prob=output.batch['ref_log_prob'].detach().cpu().clone()),
                                    directory / f'full-prefix-reference-rank-{self.rank}.pt')
        atomic_json_save(dict(rank=self.rank, pid=os.getpid(), outer_step=outer,
            source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], input_digest=input_digest,
            fixed_reference_identity=before, reference_identity_after=after,
            temperature=self.config.rollout.temperature,
            probability_scope='EXISTING_FULL_SCORING_BATCH_NOT_NEW_RAW_PREFIX_REFERENCE_PROBE', raw=receipt),
            directory / f'full-prefix-reference-rank-{self.rank}.json')
        self._production_reference_outer = None
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_arm_outer(self, manifest_path, manifest_sha256):
        require(not self._production_capture and self._production_arm is None,
                'Production probability gate cannot overwrite a live/reused arm')
        path = Path(manifest_path)
        require(path.is_file() and file_sha256(path) == manifest_sha256,
                'Actual outer probability manifest changed')
        manifest = json.loads(path.read_text())
        require(manifest['outer_step'] == self._production_outer_step
                and manifest['source_commit'] == os.environ['SEARCH_R1_BASELINE_COMMIT']
                and manifest['job_id'] == os.environ['SLURM_JOB_ID']
                and manifest['run_id'] == os.environ['SEARCH_R1_BASELINE_RUN_ID']
                and manifest['checks']['trajectories'] == 2560
                and manifest['checks']['prefix_mismatches'] == manifest['checks']['action_token_mismatches'] == 0
                and manifest['checks']['hf_position_mask_mismatches'] == 0
                and manifest['checks']['sampler_runner_positions_observed'] is False
                and manifest['checks']['engine_position_evidence_scope'] == 'DERIVED_BACKEND_CONTRACT_EXPECTATION',
                'Wrong/unverified actual full outer identity')
        require(len(manifest['chunks']) == 8
                and {row['rank'] for row in manifest['chunks']} == set(range(8)),
                'Missing actual eight DP chunks')
        validate_manifest_closure(manifest, path.parent, verify_bytes=False)
        # Driver has SHA-verified all raw bytes. Each real worker also verifies
        # its own unchanged raw records before advertising its local arm.
        own = [row for row in manifest['raw_files'] if row['rank'] == self.rank]
        require(len(own) == self._production_call and {r['serial'] for r in own} == set(range(self._production_call)),
                'Incomplete actual worker sampling ledger')
        for row in own:
            raw = Path(row['path'])
            require(raw.stat().st_size == row['bytes'] and file_sha256(raw) == row['sha256'],
                    'Actual raw sampled bytes changed before first Adam')
        pause = self._complete_pause_state()
        identity = pause['identity']
        pause_receipt = atomic_torch_save(pause, path.parent / f'arm-rank-{self.rank}.pt')
        arm_record = dict(rank=self.rank, pid=os.getpid(), start_ticks=identity['start_ticks'],
                    outer_step=self._production_outer_step, manifest_sha256=manifest_sha256,
                    complete_pause_state_sha256=state_digest(pause), pause_state=pause_receipt)
        arm_path = path.parent / f'arm-rank-{self.rank}.json'
        atomic_json_save(arm_record, arm_path)
        arm_receipt = file_record(arm_path)
        self._production_arm = dict(manifest=manifest, manifest_sha256=manifest_sha256,
                                    complete_pause_state=pause, arm_receipt=arm_receipt)
        return dict(arm_record, arm_receipt=arm_receipt)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def production_reject_outer(self, outer_step, failure_path, failure_sha256):
        """Driver-wide structural refusal is explicitly acknowledged by all ranks."""
        local = dict(rank=self.rank, pid=os.getpid(), start_ticks=None, outer_step=outer_step,
                     source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                     job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                     failure_path=str(failure_path), failure_sha256=failure_sha256, local_ok=False, error=None)
        try:
            local['start_ticks'] = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
            require(getattr(self, '_production_outer_step', None) == outer_step
                    and not getattr(self, '_production_capture', True), 'Wrong actual rejection boundary')
            path = Path(failure_path)
            require(path.is_absolute() and path.is_file() and not path.is_symlink()
                    and file_sha256(path) == failure_sha256, 'Actual rejection original changed/unreadable')
            local['local_ok'] = True
        except Exception as error:
            local['error'] = f'{type(error).__name__}: {error}'
        rows = [None] * self.world_size
        torch.distributed.all_gather_object(rows, local)
        # An evidence-write failure is part of the second matched decision.
        # There is no uncoordinated write after that decision before returning.
        try:
            directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']) / 'probability-admission'
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json_save(dict(status='PRE_REJECTION_RAW_RANK_PROPOSALS', local=local, all_ranks=rows),
                             directory / f'rejected-{outer_step}-rank-{self.rank}.json')
        except Exception as journal_error:
            local['local_ok'] = False
            local['error'] = f'Rejection raw journal failed: {type(journal_error).__name__}: {journal_error}'
        final_rows = [None] * self.world_size
        torch.distributed.all_gather_object(final_rows, local)
        receipt = collective_rejection_gate(final_rows)
        self._production_arm = None
        self._production_rejected = outer_step
        return receipt

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        # Catch every local structural/state error before the collective. A
        # rejecting rank must not throw alone while peers enter the optimizer.
        arm = getattr(self, '_production_arm', None)
        manifest = (arm or {}).get('manifest', {})
        local = dict(rank=self.rank, pid=os.getpid(), start_ticks=None, local_ok=False, error=None,
                     outer_step=getattr(self, '_production_outer_step', None),
                     source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                     job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                     global_manifest_sha256=(arm or {}).get('manifest_sha256'),
                     global_batch_digest=manifest.get('global_batch_digest'),
                     production_artifact_sha256=manifest.get('production_artifact_sha256'),
                     probability_acceptance_sha256=manifest.get('probability_acceptance_sha256'),
                     global_arm_sha256=None,
                     local_batch_digest=None)
        try:
            require(arm is not None and not self._production_capture,
                    'Unarmed update_actor is forbidden; zero Adam')
            require(getattr(self, '_production_rejected', None) != self._production_outer_step,
                    'Rejected outer cannot later be admitted')
            require(getattr(self, '_production_last_admitted', 25) == self._production_outer_step - 1,
                    'Reused/duplicate update admission')
            current = self._complete_pause_state()
            local['start_ticks'] = current['identity']['start_ticks']
            require(state_digest(current) == state_digest(arm['complete_pause_state']),
                    'Worker PID/start/model/internal/Adam/scheduler/RNG changed after raw admission')
            chunks = {row['rank']: row for row in manifest['chunks']}
            local['local_batch_digest'] = state_digest(scored_batch_state(data))
            require(len(data) == 320 and local['local_batch_digest'] == chunks[self.rank]['batch_digest'],
                    'Actual dispatched scored batch/trajectory provenance differs')
            require(data.meta_info['formal_probability_gate']['manifest_sha256'] == arm['manifest_sha256'],
                    'Actual RPC omitted/different full outer probability gate')
            gate = data.meta_info['formal_probability_gate']
            arm_path = Path(gate['armed_workers_path'])
            require(arm_path.is_file() and file_sha256(arm_path) == gate['armed_workers_sha256'],
                    'Actual all-worker live arm receipt changed')
            all_arms = json.loads(arm_path.read_text())
            require(all_arms['outer_step'] == self._production_outer_step
                    and all_arms['manifest_sha256'] == arm['manifest_sha256']
                    and all_arms['driver_post_sampling_state_digest'] == manifest['driver_post_sampling_state_digest'],
                    'Actual driver/all-worker live arm boundary differs')
            arms = validate_live_arms(all_arms['workers'], self._production_outer_step, arm['manifest_sha256'])
            own_arm = next(row for row in arms if row['rank'] == self.rank)
            require(own_arm['pid'] == os.getpid() and own_arm['start_ticks'] == local['start_ticks']
                    and own_arm['complete_pause_state_sha256'] == state_digest(arm['complete_pause_state'])
                    and own_arm['arm_receipt'] == arm['arm_receipt'],
                    'Actual worker arm PID/start/model/Adam/RNG identity differs')
            physical_arm = read_json_record(arm['arm_receipt'], arm_path.parent)
            require(physical_arm['pause_state'] == own_arm['pause_state'], 'Actual pause receipt differs')
            verify_record(physical_arm['pause_state'], arm_path.parent)
            physical_pause = torch.load(physical_arm['pause_state']['path'], map_location='cpu', weights_only=False)
            require(state_digest(physical_pause) == state_digest(arm['complete_pause_state']),
                    'Actual saved live model/Adam/RNG worker arm changed')
            local['global_arm_sha256'] = gate['armed_workers_sha256']
            require(data.meta_info['baseline_outer_step'] == self._production_outer_step,
                    'Actor outer counter differs')
            local['local_ok'] = True
        except Exception as error:
            local['error'] = f'{type(error).__name__}: {error}'
            # Keep PID/start meaningful even on an early unarmed refusal.
            if local['start_ticks'] is None:
                try:
                    local['start_ticks'] = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
                except Exception as start_error:
                    local['error'] += f'; unreadable actual PID/start: {type(start_error).__name__}'
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        # A local evidence write can also fail. Include it in a second matched
        # decision instead of letting one rank throw while others enter Adam.
        try:
            directory = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR']) / 'probability-admission'
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json_save(dict(status='PRE_DECISION_RAW_RANK_PROPOSALS', local=local, all_ranks=gathered),
                             directory / f'iter-{local["outer_step"]}-rank-{self.rank}.json')
        except Exception as journal_error:
            local['local_ok'] = False
            local['error'] = f'Admission raw evidence write failed: {type(journal_error).__name__}: {journal_error}'
        final_rows = [None] * self.world_size
        torch.distributed.all_gather_object(final_rows, local)
        decision = collective_gate(final_rows)
        self._production_last_admitted = self._production_outer_step
        self._production_arm = None  # One actual original update RPC only.
        output = ActorRolloutRefWorker.update_actor(self, data)
        # Persist the matched final proposals only after the original optimizer
        # RPC returned. No rank performs an uncoordinated journal write between
        # the last collective and Adam. A failure here is reported as an actual
        # post-update failure, never as zero Adam or an accepted complete outer.
        atomic_json_save(dict(status='MATCHED_ADMISSION_AND_ORIGINAL_UPDATE_RETURNED',
            outer_step=local['outer_step'], rank=self.rank, pid=os.getpid(),
            decision=decision, all_ranks=final_rows,
            actual_optimizer_steps=output.meta_info['metrics']['actor/optimizer_steps']),
            directory / f'complete-{local["outer_step"]}-rank-{self.rank}.json')
        # original worker owns offload, Adam, scheduler and returned metrics.
        return output
