# OPSD reward protocol fix integration

The three original fixes were already cherry-picked on origin/reproduce/opsd. Local branch fast-forwarded from production8e60976d364ab7d7ae51714415813c1359dafe9a to5f516208a0b0daed88c27c4800c7bcfcf908f043. Stable patch-ID verification confirms exact equivalence: original eaa74b8→a59dbe4, 3137d1f→63abd74, 0fca4cf→5f51620. No duplicate empty cherry-pick, force push, or replacement of local work was performed. Existing OPSD algorithm changes were already committed/pushed in8e; local uncommitted production-code changes were absent. The two older modified review documents and all untracked evidence were preserved; this integration commit contains only selected review/validation records.

## Validation

Local protocol/reward tests:12pass (dependency fallback explicitly retained as a limitation). Native Linux CPU test on an isolated exported8e tree plus the exact fix patch:233pass in24.36seconds, including terminal protocol, reward/validation and all tests/baselines. SEARCH_R1_REQUIRE_REAL_DATAPROTO=1, CUDA_VISIBLE_DEVICES empty, OMP/MKL1. All577 tracked source file contents match the fixed5f archive, aggregate tree SHA fce1e64e88251c7dcabeebc043b7febf5bc5633b313302659fbc500671c5b7b1. The existing environment was reused without installation or upgrade; original running repo remained8e.

Additionally five provenance tests passed through the real verl.trainer.main_ppo_format.RewardManager and real verl.protocol.DataProto. A separate character tokenizer fixture was made compatible with HuggingFace decode accepting both list and tensor inputs; the original tensor-only fixture first failed on lists, and that diagnostic is retained locally. No production code was changed for this fixture adaptation.

## Active training and scientific scope

Actual Slurm3007 continues with immutable training source8e; this integration does not restart, cancel, hot-edit or alter its receipt/tools. Independent review confirms the controller zeroes OPSD advantages/returns and distillation ignores EM/F1 rewards. Five training/generation source files are byte-identical across8e and5f. Therefore pureOPSD does not require retraining for these scoring fixes.

Old8e EM/F1/format metrics remain historical values under a known vulnerable scoring protocol. They are not corrected effect evidence, and this review does not claim exploitation occurred in actual3007. Default validation did not save full per-question model spans, so aggregate logs cannot be exactly rescored. Final efficacy requires separately fixed-source eval-only of the actualHF500 and comparison base with identical data/tool budget, retained per-question raw provenance and new independent actual evidence. Training source and evaluation source must be bound separately. The old blocked HF500 preview remains unchanged and cannot be filled with guessed future IDs or silently relabelled as fixed evaluation.

See REWARD_PROTOCOL_FIX_REVIEW.md for independent reasoning and evidence/reward_fix_* for the actual test and equivalence records. Formal500/final corrected evaluation/HF500 acceptance remain incomplete.
