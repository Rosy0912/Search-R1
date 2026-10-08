# Tree-GRPO terminal-answer reward fix integration

Original fixes: eaa74b8b445e4991f6b9864a5f64408145cf494f (terminal answer/EM), 3137d1fe989697725db2cfb97fe0e582a1e879d9 (F1/action legality), 0fca4cfa3de9fe967c8d0ae7d23e9fa4c1ea1b4e (optional format provenance).

Applied existing remote equivalents fd65112/daae84b/4f23105 with cherry-pick, producing 11d20ff/2645308/8b536a5; then merged origin/reproduce/tree-grpo at 1796f54 to preserve upstream history and allow a normal push. Uncommitted reproduction evidence was protected with stash including untracked files and restored after integration; the recovery stash is retained.

38 targeted tests passed, including terminal protocol, reward validation and Tree batching/audit; complete baseline/reproduction suite: 192 passed, 1 skipped, 4 warnings. Initial targeted collection lacked the legacy test-helper import path; that failed log is retained and rerun with PYTHONPATH=.:tests/baselines passed. Independent review verified reward counterexamples and actual optional RewardManager provenance handling, with Tree math/actor/audit bytes unchanged.

Running job2971 remains frozen4d133ef. No remote source deployment, training restart, old-trajectory rescoring, full state resume or fixed-protocol GPU acceptance is claimed. Existing metrics are pre-fix protocol history and cannot establish corrected-reward effects. Fresh fixed-checkout training and evaluation remain outstanding. Evidence contains no downloaded weights or datasets.
