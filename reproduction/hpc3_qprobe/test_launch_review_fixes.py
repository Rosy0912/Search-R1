"""CPU regressions for real native4 launch/recovery review findings.

The optional archived model fixture is an actual completed initial smoke. Its
copied phase names below are CPU verification fixtures, never GPU admissions.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.common8 import behavior_runtime,configuration,production_safety_admission as admission
from reproduction.common8.run_pipeline import verify_initial_probe_binding


class SourceAndRestoreTests(unittest.TestCase):
    def test_bound_repo_link_accepts_bytes_but_rejects_internal_link_and_wrong_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'immutable';source.mkdir();base=root/'run';base.mkdir()
            (source/'a.py').write_text('original');(base/'repo').symlink_to(source)
            run=dict(source_sha256={'a.py':hashlib.sha256(b'original').hexdigest()})
            with patch.object(configuration,'REPO',source):
                behavior_runtime.verify_source_snapshot(base,run)
                (source/'a.py').write_text('changed')
                with self.assertRaises(AssertionError):behavior_runtime.verify_source_snapshot(base,run)
                (source/'a.py').unlink();(root/'foreign.py').write_text('original');(source/'a.py').symlink_to(root/'foreign.py')
                with self.assertRaises(AssertionError):behavior_runtime.verify_source_snapshot(base,run)
                (source/'a.py').unlink();(source/'a.py').write_text('original')
                (base/'repo').unlink();other=root/'other';other.mkdir();(other/'a.py').write_text('original');(base/'repo').symlink_to(other)
                with self.assertRaises(AssertionError):behavior_runtime.verify_source_snapshot(base,run)

    def test_fresh_and_paired_reward_probe_binding_are_distinct(self):
        verify_initial_probe_binding(dict(initial_probe_version=None))
        with self.assertRaises(AssertionError):verify_initial_probe_binding(dict(initial_probe_version='unexpected'))
        run=dict(initial_probe_version='reward-r5',full_state_restore=dict(kind='question_probe_pair',reward_probe_version='reward-r5'))
        verify_initial_probe_binding(run)
        for version in (None,'other',''):
            changed=dict(run,initial_probe_version=version)
            with self.assertRaises(AssertionError):verify_initial_probe_binding(changed)


@unittest.skipUnless(os.environ.get('SEARCH_G1_NATIVE4_SMOKE_FIXTURE'),'Actual archived native4 model smoke fixture required')
class Native4ParentAdmissionTests(unittest.TestCase):
    def fixture(self,base,mutation=None):
        original=Path(os.environ['SEARCH_G1_NATIVE4_SMOKE_FIXTURE'])
        proof=json.loads(original.read_text());rows=copy.deepcopy(proof['ranks']);assert len(rows)==4
        run=dict(profile='common8',run_id=proof['run_id'],precision_profile='native4_bf16_varlen_fp32_master',
            behavior_probability_policy='sampling_journal_no_extra_hf_forward',training_world_size=4,
            execution_backend='slurm',execution_site='hpc3',actor_micro_batch_per_rank=4,vllm_gpu_memory_utilization=.1)
        if mutation:mutation(rows)
        probability=dict(slurm_job_id=proof['slurm_job_id'],question_ids=[q for r in rows for q in r['question_ids']],
            actual_cuda_runtime_uuids=[r['cuda_uuid'] for r in rows],resolved_config_sha256=rows[0]['resolved_config_sha256'])
        parent=base/'logs/policy-safety-admission-receipts/parent';parent.mkdir(parents=True);receipts={}
        for phase in ('initial','coexistence'):
            phase_rows=[];hashes={}
            for rank,row in enumerate(rows):
                value=dict(row,phase=phase);p=parent/('model-smoke-'+phase+'-rank-'+str(rank)+'.json');p.write_text(json.dumps(value))
                relative=p.relative_to(base).as_posix();receipts[relative]=admission.digest(p)
                phase_rows.append(value);hashes[str(rank)]=receipts[relative]
            p=parent/('model-smoke-'+phase+'-complete.json')
            p.write_text(json.dumps(dict(accepted=True,run_id=run['run_id'],phase=phase,ranks=phase_rows,actual_grpo_updates=0,rank_file_sha256=hashes)))
            probability[phase+'_proof']=p.relative_to(base).as_posix();probability[phase+'_proof_sha256']=admission.digest(p)
        return run,probability,receipts

    def test_real_four_rank_bf16_receipt_is_not_rejected_as_legacy_eight_fp32(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);run,p,r=self.fixture(b);admission._verify_parent_probability(b,run,p,r)

    def test_four_rank_mode_still_rejects_optimizer_writes_nonfinite_and_internal_drift(self):
        mutations=[lambda rows:rows[0].update(actual_optimizer_writes=1),
            lambda rows:rows[0].update(internal_separate_logprob_max_abs=1.),
            lambda rows:rows[0]['parity'].update(actor_vllm_max_abs=float('nan')),
            lambda rows:rows[0]['parity'].update(cross_backend_drift_is_diagnostic=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as d:
                b=Path(d);run,p,r=self.fixture(b,mutate)
                with self.assertRaises((ValueError,AssertionError)):admission._verify_parent_probability(b,run,p,r)


if __name__=='__main__':unittest.main()
