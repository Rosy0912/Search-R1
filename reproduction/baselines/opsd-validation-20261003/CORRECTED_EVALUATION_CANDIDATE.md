# Corrected evaluation local candidate — NOT DEPLOYABLE

This candidate implements opt-in native scoring-view evidence collection and exact CPU replay. It is not complete GPU evaluation acceptance, and does not claim actual500/HF500 completion. Training job3007/source8e and all immutable old preview directories remain unchanged. Default disabled behavior does not change generation, actor, distillation or teacher mathematics.

## Scope and tests

The writer runs after native DataProto union/unpad and original RewardManager evaluation. It saves integer responses/prompts/attention/info masks, fixed input identities, decoded model-only spans, online EM/F1/format and original rewards. Replay recomputes those values and the original aggregate using actual imported scoring modules. This preserves the native scoring view after the existing response cap; it is not a complete pre-cap turn trace.

Ordered original parquet inputs/targets/questions/source/file-row identities must be manifested and externally pinned before generation. The final entry requires exactly51713 questions and seven sources, batch64, no subsampling/filtering, native do_search, val_only and val_before_train. The expected batches are 808×64 then1. Actual actor/critic update dispatch methods are replaced by fail-closed counter guards; any attempted dispatch or replaced guard rejects completion. Equal model hashes alone do not prove zero updates.

Thirteen source files are required and byte-pinned. Ten CPU-safe modules additionally verify actual imported __file__/SHA, including main, ray_trainer, native generation and dataset. Actor/distillation/vLLM worker files are byte-pinned here; future GPU deployment must independently bind their actual worker import paths/SHA and same-node retrieval/lease/cleanup evidence. The resolved generation/retrieval config is compared against the external stage, including token/turn/batch/resource budgets and the native validation greedy override. No claim of GPU imports is made by CPU tests.

66 CPU tests passed with actual verl.protocol.DataProto and native agent padding/unpad; no AST DataProto fallback. A synthetic CPU tokenizer and synthetic model content files are fixtures, not an HF435-tensor proof. Raw-view unit tests mock only the external training/HF prerequisite boundary; separate tests directly invoke that production checker to prove missing verifier/stage pins reject. They do not prove future500 acceptance.

## Required external stage (currently absent / UNFILLED)

The stage JSON must be pinned through OPSD_CORRECTED_EVAL_STAGE_SHA256, explicitly propagated into the Ray main_task environment, and contain training_commit8e, corrected eval_commit and source_files SHA map, input_manifest_sha256, model_role(base or hf500), model_directory and exact model_files hashes, generation_budget and its canonical JSON SHA, training_hf_receipt_path/SHA, and strict_training_hf_verifier{path,sha256}. Runtime adds reviewed_stage_path/SHA; replay revalidates this external chain and actual imports. The working evaluation checkout must be clean against its actual commit.

The separate strict_training_hf_verifier is **not implemented or supplied by this candidate**. Its absence blocks production load_stage and replay. Future implementation must run the existing actual3007 terminal + all500×8 budgets + periodic0/50/.../500 each64 + final51713 checks and actual independent HF500435-tensor/three-shard/config/tokenizer/finite/loading/generation/changed-weights/cleanup checks. It must validate model-role/path correspondence against those receipts, including the fixed base model and exported HF500. An opaque receipt file/hash alone is never accepted.

The pinned verifier receives --receipt/--receipt-sha256/--model-role/--model-directory and must fail nonzero for any missing or false actual evidence. Only its exact typed JSON result is accepted: kind=strict_actual_training3007_hf500_prerequisites, training_commit8e, formal_job3007, actual_outer_updates500, final_questions51713, receipt_sha256, model_role, model_directory. A returned flag is not a substitute for implementing/reviewing that verifier. GPU job IDs, future HF completion, paired effect receipt and deployed stage remain UNFILLED.

Before final acceptance, root must perform native independent CPU checks and independent source review, implement/review the blocked receipt bridge and runtime supplement, then evaluate base and actual HF500 under the same fixed inputs/budget with corrected source, replay both and assemble the paired evidence. No submit/upload/commit/push is performed by this subagent.

## Reproduction

From the workspace root:

```sh
SEARCH_R1_REQUIRE_REAL_DATAPROTO=1 PYTHONPATH="$PWD/work/opsd" /Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python -m pytest -q work/opsd/tests/reproduction/test_corrected_evaluation.py work/opsd/tests/reproduction/test_reward_and_validation.py work/opsd/tests/reproduction/test_terminal_answer_protocol.py
```

Raw output: corrected-eval-candidate-cpu-tests.txt. Fixed candidate file hashes: corrected-eval-candidate-files.json.
