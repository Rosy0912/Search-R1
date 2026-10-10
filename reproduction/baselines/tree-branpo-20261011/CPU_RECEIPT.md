# CPU verification receipt

2026-10-11 isolated branch codex/tree-branpo-tail-20261011 from ebef786.

`python3 -m unittest test_tree_branpo_tail_cpu test_tree_fresh200_cpu`: 18 tests pass. `git diff --check`: pass. No GPU training or remote launch occurred in implementation verification. Tests cover exact five leaf quota, unique trajectories, three bounded continuations, retrieval-token and selected-logprob ledger inheritance, same-reward one-state backoff, earliest-state clamp, no correct-only stop, actual training reward callback with permuted source metadata, all-zero/all-one degeneracy, population moments, sample-normalization constants, unchanged Tree estimator, and row permutation.

Broader legacy raw-semantics/mu-config suites have existing fixture errors: `capture_native_emissions` now requires `native_stopping`, but their fixture calls omit it. Two representative failures reproduced identically in unchanged ebef786 historical checkout. Native capture safety was preserved; no production protocol change made to pass stale fixtures.

Raw reward statistics use pre-relocation training reward; actual/intra normalized advantages use the unchanged training dtype and relocated estimator reward. Moments are then accumulated in float64. Intra means Var(Z_root), total-within means Var(Z_question+Z_root). Exact-reward-one/zero counts represent strict scalar reward values, not guessed answer categories.
