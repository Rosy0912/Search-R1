# Handoff HF strict source review

Status: **ACCEPTED — source only**.
Wrapper SHA: `8cc378a52009a8a41740831aa2259f380004da94760b3e7e98c4c8243a9c79ff`.
README SHA: `8f6750ec3e6cd86193efd85dc107e8dad3b3405bb52e145575bdf86cdf8a4b23`.

All three previous source blockers are closed. This review used local source inspection and small isolated JSON/function checks; it performed no remote/GPU/model/PT operations.

- **HFH-1 resolved:** check_rank_numerics consumes200x8 actual audits; enforces finite positive norm types/counters, finite sample count/cardinality/layout/coordinates/extrema/nonzero/SHA. Outer1 zero LR distinguishes unchanged sample; outers2..200 require sampled movement. Report states fixed-coordinate scope; no full-shard finiteness claim from samples.
- **HFH-2 resolved:** Supplied handoff equals the journal embedded in pinned preserved scheduler observation; exact own3360 RUNNING historical identity and finite request/stop/observation epoch ordering are enforced.
- **HFH-3 resolved:** exact_attempt_inventory requires full attempt directory to equal consumed sequence ending at200; additional201/fault/pending/nested artifacts are rejected and full filename/content hash inventory is rechecked after CPU verification.

The original remains CANCELLED / 0:15, with original normal run-result and train200-terminal absent. Full coverage is exactly 51,713, with preserved30,880 + replacement20,833 and exact seven counts. The complete ledger independently recomputes question identity/order and metrics. The unchanged frozen full-HF verifier retains complete model loading/keys/shapes/finiteness/finalbase delta and CPU forward checks; tokenizer checks retain backend, AddedToken, special roles, configuration, vocabulary and actual encode/decode/chat behavior.

**Actual artifact acceptance is pending.** No source approval permits new nested fresh200: first require real replacement COMPLETE51713/workerexit0/owncleanup and systemd terminal, actual numerical/ledger acceptance, actual CPU cold HF/tokenizer reports, then independent receipt acceptance.

Positive and negative isolated checks pass: zero-LR and moving samples, malformed/nonfinite/inconsistent numerical proofs, exact200 inventory, extra201 and pending rejection, and altered content hash detection. Actual numerical evidence must still pass during the authorized CPU wrapper run.

Cost evidence remains separately bound; absent fields are not replaced by zeros and wasted slow-evaluator compute is not credited as preserved scores.

## Exact reviewed file SHAs

- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/verify_handoff_hf.py`: `8cc378a52009a8a41740831aa2259f380004da94760b3e7e98c4c8243a9c79ff`
- `work/tree-branpo-final-hf-20261011/verify_final_hf_3360.py`: `ad263ce31eb938542dd5f63d3f70b178605876576356b6fc9a3a21cb404e4ee7`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-branpo-final-hf-20261011/verify_final_hf_3360.py`: `ad263ce31eb938542dd5f63d3f70b178605876576356b6fc9a3a21cb404e4ee7`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py`: `b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-branpo-runtime-20261011/production_contract.py`: `df82d7641d66f3ac6d21ae8b4df6976d9fa8c204bd1909ad2648e3cee2e80b8f`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-branpo-runtime-20261011/attempt_accounting.py`: `8426d39486e92773022dd0d40ef5e4cb0c742ee6317550d20244ed4696913c29`
- `work/tree-speed-repair-20261008/branpo-source/search_r1/baselines/tree_rank_audit.py`: `4ef35ac5e0ab0089a669fd459a342678cd9dd46f00e4cd51c917f191e56c183b`
- `work/tree-speed-repair-20261008/branpo-source/search_r1/baselines/runtime_audit.py`: `d4b2642ece02c8b6ccfde0abf13081c271e8b6abdfb25a7e9fbcc348a17584c6`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/eval_merge.py`: `9159f0b2021a97cb4aa24bcfe8e67c08c6cafc44f7d567976aae8a851e1b205a`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/shared_supervisor.py`: `7c40dc60bdcf6659d54587ba32d04e05f86fc75d9674db8e2243606f777a45c4`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/bind_prefix.py`: `d6dfb36c5fc2584108afb47f94c63171483cbc378de7d71e7f3f4bf24ff76bc1`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/eval_worker.py`: `906095da3e22b4f667fdb36cc635aa52bd8b0d968d116aeb3dcc59f4fda3fcea`
- `outputs/tree-eval-resume-20261013/ORIGINAL_TERMINAL_BINDING.json`: `22e7d9ed71015cf4b99802493205a0e8c1dfa7e97715df191005754499895bb0`
- `outputs/tree-eval-resume-20261013/POLL-SECOND.json`: `af0762ee0af4a94a03138480d03d9e7ec5c2eb8fe7a795a537e72cf8b7cdcef0`
- `outputs/tree-eval-resume-20261013/STATUS.json`: `60e3a72fb3d3017af3223a2f35b3759266a07c0bb3312eb4b7336ee1f5077a41`
- `work/tree-speed-repair-20261008/branpo-source/reproduction/tree-eval-resume-20261013/README.md`: `8f6750ec3e6cd86193efd85dc107e8dad3b3405bb52e145575bdf86cdf8a4b23`
