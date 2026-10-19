"""Preserve the official loop, observe the first resumed batch and native cursor."""
import json
import os
from pathlib import Path
from rllm.trainer.verl.agent_ppo_branch_trainer import AgentPPOTrainer as OriginalTrainer
from .contract import sha


class AgentPPOTrainer(OriginalTrainer):
    def _load_checkpoint(self):
        assert self.config.trainer.total_steps == 250
        assert self.config.trainer.start_step == 100
        assert self.config.trainer.resume_mode == 'resume_path'
        assert self.config.trainer.val_before_train is False
        assert self.config.trainer.del_local_ckpt_after_load is False
        proof_path = Path(os.environ['BRANPO150_NATIVE_PROOF'])
        proof = json.loads(proof_path.read_text())
        result = super()._load_checkpoint()
        assert self.global_steps == 240
        state = self.train_dataloader.state_dict()
        expected = proof['dataloader']['state']
        assert state == expected, 'Actual StatefulDataLoader cursor restore differs'
        self._branpo150_first_batch = proof['dataloader']['next_ids']
        for rank in range(8):
            p = Path(os.environ['BRANPO_AUDIT_DIR'])/f'cold-restore-rank-{rank:02d}.json'
            row = json.loads(p.read_text())
            assert row['rank'] == rank and row['proof_sha256'] == sha(proof_path)
            assert all(row[k] for k in ('model_exact','adam_exact','scheduler_exact','saved_rng_exact'))
        p = Path(os.environ['BRANPO_AUDIT_DIR'])/'cold-driver-restore.json'
        with p.open('x') as stream:
            json.dump(dict(job_id=os.environ['SLURM_JOB_ID'],global_step=240,next_label=241,
                           dataloader_exact=True,proof_sha256=sha(proof_path),
                           not_bitwise_uninterrupted_rollout_rng_claim=True),stream,indent=2)
        return result

    def init_envs_and_agents(self, batch):
        if getattr(self,'_branpo150_first_batch',None) is not None:
            # Repeat is interleaved by the unchanged rollout.n=4 before this call.
            ids = [str(x['index']) for x in batch.non_tensor_batch['extra_info']]
            expected = [value for value in self._branpo150_first_batch for _ in range(4)]
            assert self.global_steps == 241 and ids == expected, 'Resumed first question batch differs'
            p = Path(os.environ['BRANPO_AUDIT_DIR'])/'cold-first-batch.json'
            with p.open('x') as stream:
                json.dump(dict(job_id=os.environ['SLURM_JOB_ID'],label=241,
                               unique_questions=len(self._branpo150_first_batch),ids_exact=True,
                               proof_sha256=sha(os.environ['BRANPO150_NATIVE_PROOF'])),stream,indent=2)
            self._branpo150_first_batch = None
        return super().init_envs_and_agents(batch)
