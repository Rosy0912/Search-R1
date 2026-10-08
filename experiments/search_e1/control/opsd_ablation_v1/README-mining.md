# Frozen ten-cohort selection and seven-variant mining

These three CPU modules are shared by sampling, resume validation and the trainer. They do not start a process, change a reward, infer a usable-pair count, or authorize a resource/storage transition.

`cohort_v1.freeze_from_files(train_identity, validation_identity)` reads the bound original 169103 TRAIN rows and 512 heldout rows, using stable full-file SHA256. It freezes the first 5120 entries of the original isolated `random.Random(42)` shuffled full TRAIN permutation into ten cohorts of 512. Selected ID or exact question-text overlap with validation rejects the preparation; it never replaces or drops selected questions. `verify_from_files` recomputes that same selection. The frozen JSON contains private question/gold text and must remain outside Git.

`miner_v1.verify_rollouts(rows, questions, variant, outer_index, core)` requires the complete frozen 512-question cohort and all 2560 unique sibling slots. The caller supplies the SHA-bound original scientific `core`; its authored-final strict reward is recomputed directly. `mine_pairs` chooses the native successful B by minimum executed searches and original token count, retaining source-order first ties. C0/C1/M/P choose the native failed-preferred farthest complete-response D, with actual rapidfuzz normalized Levenshtein and all-correct success fallback. The native pair projection is compared exactly with original `core.mine_pairs`.

R/W/J choose D uniformly from canonical sibling slots 0–4, including B. The independent SHA256 stream binds version, seed42, outer0–9, question ID and rejection counter; its first unsigned big-endian 64-bit word is accepted only below `2**64-(2**64%5)`, then reduced modulo five. Recorded rejection evidence makes this unbiased draw reproducible without consuming training RNG. Sibling self-pairs and equal response strings are separate fields. All-wrong questions are skipped and all-correct questions are kept.

`weights_v1.correctness_weights` uses the five native binary rewards, sample standard deviation (correction1), epsilon1e-6 and temperature1. It returns five strictly positive `5*softmax(A)` weights with mean1. Only W/J apply the selected D's weight, after component-clipped trajectory token averaging; the global denominator remains the actual number of pairs, never the sum of weights. `verify_pairs` recomputes the exact pair identities and metadata. P counts eligible pairs; context/mask skips and usable K belong to the downstream frozen encoder.

Versions are `search-e1-opsd-ablation-miner-v1`, `search-e1-opsd-uniform-five-sha256-u64-rejection-v1`, `search-e1-opsd-correctness-weights-v1`, and `search-e1-opsd-ten-train-cohorts-seed42-v1`. Each outer manifest must retain these versions, common cohort reference, variant and original outer-start adapter. No random-variant pair should be compared against unmodified native `core.mine_pairs` as its production selector.

CPU fixture entry, from the publisher root:

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_v1 -p 'test_mining_cohort_v1.py' -v
```

The tests compile the actual SHA-bound frozen core/QA source. On a separately deployed helper tree, `OPSD_ONLINE_TEST_RUNTIME` supplies that original runtime path. Twelve meaningful fixtures cover native ties/fallback, unbiased rejection, self-pair identity, all five-reward patterns, strict scoring, pair mutations, complete production cohort binding, original shuffle and heldout exclusion, frozen provenance mutations, and partial JSONL/SHA rejection. This is CPU/source verification, not live seven-group numerical or effect acceptance.
