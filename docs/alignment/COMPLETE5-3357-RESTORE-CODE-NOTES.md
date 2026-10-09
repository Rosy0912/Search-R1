# Actual3357 COMPLETE5 continuation: implementation scope

CODE_CPU_ONLY. This candidate restores the actual3357/e82ef0231a23eedfd68ebba1644966a5759a7575 COMPLETE fifth-outer checkpoint. It does not restart the experiment or replay any Adam write. No remote action, distributed native restore, CPUHF extraction or release was executed for this implementation.

New aligned_complete5_resume.py validates all eight rankv3 states at outer5/attempt5, COMPLETE tags, Adam100 per parameter state, scheduler5 and prescribed6/57 LR, finite model/Adam tensors, exact generation RNG and explicitly absent pending transaction. Model/Adam/scheduler/CPU-CUDA RNG/generation RNG/latest original rank event load together; actual reserialized state_digest must equal the original report digest. Eight-rank collective checks surround loading and witness publication. No optimizer, scheduler or stage operation occurs.

New aligned_complete5_driver.py restores the exact old COMPLETE driver (global5,next6,actual5,attempt5,cursor2560,original accepted accounting), sampler/RNG/BAPO and original latest rank ledger. It constructs the next data iterator before restoring driver RNG, then checks sampler/RNG/BAPO digests. Native checkpoint metadata previous_complete_outer=5 differs from driver previous_completed_outer=4 by the existing producer schema and is checked accordingly. The recovered current process exports the same fifth weights, proves its full fifth native/driver roundtrip and pauses for actual independent acceptance before setting global6. It does not call stage.observe, after_actor, commit_aligned_outer, generate or update_actor. Original3357/3356 records retain their original identities through the separately reviewed gate lineage adapter.

Trainer integration uses exclusive BAPO_ALIGNED_COMPLETE5_RESUME mode, rejects mixing partial/legacy restore, dispatches the new eight-rank RPC, and includes complete5-resume files plus the exact original closure at the fifth gate. It does not require invented newouter1-5 EM/stage artifacts or fresh initial proofs. Launcher/startup claim adaptation and deployment are parent responsibilities. Old3357 source/RUN/evidence stay unchanged.

Actual local command: PYTHONPATH=. ../fresh200-local-test-env/bin/python -m pytest -q tests/baselines/test_aligned_complete5_resume.py tests/baselines/test_aligned_partial_resume.py tests/baselines/test_aligned_outer_commit.py tests/baselines/test_baseline_aligned_first5_gate.py tests/baselines/test_fresh200_gate_skip.py . Result85 passed,2 warnings in5.20s. New focused tests14 passed in1.57s include real tiny CPU Adam100 state and driver RNG restoration/pause-before6 with no training methods available, plus wrong counters/pending/nonfinite/partial-state negatives. Test interpreter is the workspace Python3.9.6/OmegaConf2.3.0 environment. Tests do not establish target Linux/CUDA execution or real full-state restoration; actual target evidence must independently establish that.

search_r1/baselines/aligned_complete5_resume.py SHA256 8bd4ac38bcb4fbb41c202e8bfa5b6360eeea21327145623eaa67bca4109c511f

search_r1/baselines/aligned_complete5_driver.py SHA256 d6858841f70f671b621e9496a7ab22240d2e788b9665d22ae00775539e60a63b

tests/baselines/test_aligned_complete5_resume.py SHA256 c513c8ee9302ad15cf0cbf3b4d38ec11693d04e4386d190991bbba4249076d13

verl/workers/fsdp_workers.py SHA256 c449d53c684c9ad3d01ccc7beb2207501abcbb69923e112adf97419ad128df17

verl/trainer/ppo/ray_trainer.py SHA256 46d12bfa4568ff23fe543ad1e1ecc40ddb03274d0f00cf1027fa87daa984b427

