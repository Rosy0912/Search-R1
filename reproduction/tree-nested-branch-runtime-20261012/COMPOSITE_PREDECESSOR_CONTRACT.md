# Truthful composite predecessor contract

Proposed successor API only; the actual composite/full-HF/independent receipts do not exist while replacement evaluation is active. No evidence is fabricated by this file or gate.

The fixed remote inputs are `RUN=/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-branpo-fresh200-3360`, `RUN/output/eval-handoff-hf-acceptance/identity-bound-handoff-hf.json`, and a separate `RUN/output/eval-handoff-hf-acceptance/independent-composite-acceptance.json`. Schema is `tree_train200_eval_handoff_composite_v1`. Composite status is `JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED`. The acceptance worker's `verify_handoff_hf.py` emits the composite with independent_final_acceptance=false after unchanged frozen CPU full-HF verifier + tokenizer semantics checks; this is never itself independent review.

The independent reviewer must create the separate JSON only after reviewing actual completed composite/full-HF/tokenizer/live-terminal outputs and the exact wrapper bytes. Required fields:

```json
{
  "schema_version": "tree_train200_eval_handoff_composite_v1",
  "status": "INDEPENDENT_JOB3360_TRAIN200_EVAL_HANDOFF_ACCEPTED",
  "fork_turns": "none",
  "blocking_findings": [],
  "composite_receipt_sha256": "ACTUAL_RECEIPT_SHA256",
  "training_source_commit": "8b6f99d1b3353145200eb3fa7c16a579a79a987c",
  "evaluation_source_commit": "0e00998fdc645281986250cd8632bb70c6e2fa9b",
  "reviewed_wrapper_files": {
    "verify_handoff_hf.py": {"path": "ACTUAL_ABSOLUTE_REMOTE_WRAPPER", "sha256": "ACTUAL_REVIEWED_WRAPPER_SHA256"}
  },
  "training200_accepted": true,
  "final51713_accepted": true,
  "full_hf_accepted": true,
  "tokenizer_accepted": true,
  "native_preservation_accepted": true,
  "authorized_handoff_accepted": true,
  "cleanup_accepted": true,
  "cost_accepted": true,
  "cost_summary": {"path": "RUN/output/eval-handoff-hf-acceptance/cost-summary-v2.json", "sha256": "ACTUAL_SCOPE_COST_SUMMARY_SHA256"}
}
```

The placeholders above are schema documentation and cannot serve as acceptance. reviewed_wrapper_files must exactly equal composite acceptance_wrapper_files and its bound wrapper input descriptor. The gate additionally rehashes all descriptor files, recomputes original persisted direct200 rank accounting/native200 manifest, verifies native/PT hashes without deserialization and HF inventory, binds every frozen old source/current evaluator source/dataset/provenance/committed batch+done, checks final-metrics51713 and7 exact counts, reads hashed actual full-HF/tokenizer reports, and checks preserved historical cancellation snapshots plus actual replacement terminal cleanup. Missing a descriptor, hash/member/count/source/config/split mismatch or missing independent review refuses launch.

The original old run-result and train200-terminal are absent and stay absent. The honest scheduler observation source is scontrol-preserved, state CANCELLED/ExitCode0:15; training-body exit is null/unknown, original normal-exit claimed=false and original final evaluation completed=false. The accepted outcome is original actual200 training plus authorized handoff/evaluator final51713, not an original normal terminal. No old runtime modification, restart, optimizer restore or foreign process action is permitted.

Package/deploy this successor under a NEW committed/pushed/reviewed source identity. Retain frozen nested0a123e4a unstarted. Resource mode is separately reviewed: a separately implemented explicit own shared-systemd adapter is selected by the successor manifest; legacy Slurm checks stay gated separately. See SHARED_LAUNCH_ADAPTER_PLAN.md.

The independently accepted separate cost summary uses schema_version `tree_train200_eval_handoff_cost_v1` and status `JOB3360_COMPOSITE_ACTUAL_SCOPE_COST_COMPLETE`. It binds training_job_id3360, exact old/new source commits, composite_receipt_sha256 and input_sha256 `{absolute_path: sha256}`. Its scope_flags must accept original_train200, original_dev40, original_partial_evaluation, successful_replacement_evaluation, failed_own_evaluation_startups and actual_cpu_hf_acceptance. The map must cover pinned f39 completed200 report plus every underlying evidence_sha256/input_sha256, successful replacement terminal/allocation/resource/ledger/batch+done (cost.json if emitted), original partial-eval files, actual cold-HF/full-tokenizer/live-terminal outputs, and actual CPU service cost/terminal/elapsed descriptors in cold_hf_service_inputs. failed_own_startup_inputs maps all5 exact failed evaluator unit names to at least their actual journal and terminal/cost observation descriptors; admission failures do not need nonexistent run-result files. Every nested descriptor must also be in input_sha256. Missing systemd CPUUsageNSec stays null/unknown; no unavailable CPU timing is normalized to zero. The existing CPU wrapper remains immutable; this summary and independent review are separate artifacts.
