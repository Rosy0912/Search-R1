# CPU stage validator helper review

No remaining static blocker. The helper selects onlytwo/five/formal, verifies completed own-stage run naming and exit0, clears CUDA_VISIBLE_DEVICES, uses own CPUenv and8-thread limits/offline assets, fixes the3B immutable base and always requires both Tree audit completions. It emits unique stage-validation-JOB JSON/log/HFreload for independent review, never stage acceptance. The submission must retain intended CPU8/64G/noGPU allocation; this helper alone does not allocate Slurm resources.

The discovered same-job log truncation is resolved by set -o noclobber. A real bash redirection fixture creates a fresh log then rejects the identical-job retry without altering the original bytes. Existing validator separately refuses JSON/HFreload overwrite. Syntax passes.14 existing actual Controller rollout/policy audit fixture cases rechecked against the unchanged validator: validtwo-step pair passes, missing/FAILED/extra/changed-binding/trajectory/mass/groups/branch/step/source/pending/five-from-two/missing-directory fail. The probe writes only new reviewer evidence; it leaves production and receipt-frozen files untouched. No actual checkpoint/model/GPU/Slurm was used in these checks.

Real two-update acceptance remains pending: exact finite console actual_outer_updates2/positive optimizer evidence, actor-state consistency, four audit completion files and hashes, CPU HF reload/nonzero parameter delta and full source/cost records.14 CPU fixtures are not evidence that production training passed.

Evidence: cpu-stage-validator-helper-review.json, stage-audit-validator-recheck.json and stage-audit-validator-recheck.py.
