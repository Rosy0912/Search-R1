# Independent OPSD selection audit (M2)

This standalone CPU tool implements the approved selection/protocol part of M2. It prepares no GPU workload, sends no HTTP request, does not alter pairs or training, and changes neither E5 CPU threads nor affinity. Root must back up, commit, push and verify these sources before deploying or running the full audit. Current local fixtures are small synthetic examples, not full-data acceptance.

## Identity and bounded inspection

Approved input identities are pinned in `audit_selection.py` and also required from the existing debug cases manifest:

| Input | Original absolute path | Complete rows | SHA256 |
|---|---|---:|---|
| rollouts | `/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runs/preparation-3-rollouts.jsonl` | 845515 | `fb4e94fb1efa600757a604db576807d9fd3dbd65a7c75bf36f11b0d2ba89d124` |
| pairs | `/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runs/preparation-3-pairs.jsonl` | 100974 | `a81a3ae61bb402f6b99dd35ae552c4bb76df8da391f3620c17fbd742ba2b1f40` |

2026-10-08 read-only inspection sampled only the first120 records and final10. Rollout head was24 groups of siblings0–4; tail was2 such groups. Pair head had120 unique IDs. The sample does not prove the full merged file is continuous; the implementation never assumes contiguity. No question, answer or response body was printed. Compact inspection evidence is `bounded-input-inspection-20261008.json`.

Production imports the original frozen runtime, not the different publisher/archive core. It hashes the small scientific source files before import and after audit. It requires core SHA `0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc`, strict QA SHA `9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd`, and the original305-file manifest SHA `3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc`. It records the actual rapidfuzz package and C++ implementation path/SHA and requires **3.13.0**, confirmed from the existing K1 environment. No shared environment installation or upgrade is needed.

## One-pass inputs and resource limits

The original rollout file is read physically **once**, while its complete SHA and every line's offset/byte SHA are computed. This pass validates field types, span provenance, strict reward and source metadata, and writes private SQLite rows keyed by `(question_id,sibling)`. Source file row order is preserved. A question appearing in several separated runs is detected and counted; disk grouping correctly joins its five siblings instead of resetting or assuming a new question. Duplicate siblings, missing groups, mixed question/dataset identity and malformed rows fail acceptance.

The private spool is subsequently traversed by first source appearance. Only one five-sibling group is materialized at a time. The original pair file is then physically read **once**, hashes and offsets computed in the same pass, and each preserved pair matched to its reconstructed original native pair. SQLite reads are private spool operations, not repeated original-file passes. Bad JSON, oversize rows and partial final lines are still included in the single complete physical hash and explicitly rejected. Input stat fields include device/inode/size/mtime/ctime before opening, after scanning and again at final acceptance.

Production is one Python process, with64MiB SQLite cache, file-backed temporary storage, SQLite mmap disabled, maximum line8MiB and at most five valid sibling records per group. Linux `RLIMIT_AS` is capped at32GiB before imports; existing lower limits stay lower. No CPU worker pool is launched. This is sufficient parallel independence from M1 while remaining modest relative to the E5 service. A run needs at least32GiB free private scratch; spool and any rollback journal contain QA text and must stay private. `progress.json` updates at most once per60seconds with phase, actual row/question counters, errors and peak RSS. No wall-time promise is made before actual full-input execution.

## Exact selection and analytical counterfactual

For each valid group the helper calls the hashed native `core.mine_pairs` and independently reconstructs its result:

1. Reference: among successes, minimum `(search_calls, token_count)`; if token_count is absent, exactly the native fallback `len(response)`. Python `min` first-in-original-file-order resolves equal keys.
2. Student candidate set: failures other than reference first; if none, other successes.
3. Student: maximum `rapidfuzz.distance.Levenshtein.normalized_distance(reference.response,candidate.response)`, with original default arguments and Python stable first-in-order ties. No epsilon or alternative distance implementation is used.

For the analytical contrast, reference and candidate set stay identical. Each text is the concatenation of original authored `policy_spans`, subtracting the frozen `information_spans` intervals, including delimiters and trailing unclosed information. No separator is invented. This is a character-distance definition, not a modified tokenizer mask, score or training dataset. It may join policy text across observations solely for this metric; strict QA never joins authored spans.

Private per-question records contain IDs, source line offsets/SHA, original order, selected siblings, every distance, reference ties, both exact maximum tie sets, whether selected text changes, whether the original full-text choice is excluded from policy maxima, and whether maximum sets are disjoint. Summary changes are **counts divided by eligible questions**, separately by dataset and CC/CW/all5_wrong. Subtracting two distances does not measure an observation contribution percentage.

Stored pairs have no sibling IDs. If several source siblings share identical response/reward/spans, pair provenance remains ambiguous; all matching original source rows are preserved. The original deterministic predicted selection and its tie bounds must not be mislabeled as proof of one unique stored-pair origin.

All original rows are strict-reward checked. Failures use mutually exclusive `final_invalid` or `valid_gold_mismatch`, also reported for all5_wrong. Literal reserved tag stack balance across the transcript and nonfinal policy spans is separate structural audit: it never changes frozen strict reward. A success with an earlier unmatched think can remain strict success under the original final-span protocol. These counts describe observed protocol failures and do not identify intrinsic hard questions.

## Acceptance and private outputs

`summary.json` is accepted only with complete845515 physical rollout rows,169103 unique valid five-sibling questions,100974 pairs reconstructed and matched, approved complete hashes, stable input/source identities, and no errors. Explicit required-check fields expose bad JSON/partial/oversize/schema/mask counts, duplicates, missing pairs, strict-reward disagreement and selection mismatch. A nonzero error yields `FAILED_AUDIT`, exit2 and retained evidence. A fatal error also records its type and reason. Outputs are never silently replaced: a preexisting run directory causes failure.

Output permissions are private. `source.private.sqlite` contains original QA text; `cases.private.jsonl`, `pairs.private.jsonl`, mismatch and issue files contain private IDs/provenance. Only source/entry/config and reviewed compact aggregate summary/manifest can enter Git. No private spool, full question rows, model weights, environment or credentials may enter Git. The summary does not contain question/response/answers and does not claim tokenizer loss audit, new teacher inference or training causal effect.

## Production command — Root executes after publication

Copy this whole source directory byte-for-byte to a **new** K1 source directory and verify its publication manifest first. The suggested private output directory must not already exist. Preserve original files and the current M1/resident/oracle/K4 processes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python \
/data/shaoshitong/search-e1-opsd-selection-audit-20261008/source/opsd_selection_audit/audit_selection.py \
  --runtime /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd-preparation3-xformers \
  --runtime-manifest /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/receipts/preparation-3-runtime.sha256.json \
  --cases-manifest /data/shaoshitong/search-e1-opsd-debug-20261008/cases/manifest.json \
  --output /data/shaoshitong/search-e1-opsd-selection-audit-20261008/runs/full-selection-v1
```

Do not pipe into a launcher that discards the exit code. A fresh private tmux controller may run this exact command with a separate log, while Root watches `progress.json`; there is no autonomous training restart logic in this helper.

## Small CPU fixtures

Set `E1_AUDIT_TEST_RUNTIME` to the saved frozen305-file runtime, and use an interpreter with original rapidfuzz3.13.0. Local testing used a new isolated dependency folder, not a shared environment update. No live input files are read by the fixtures.

```bash
PYTHONDONTWRITEBYTECODE=1 \
E1_AUDIT_TEST_RUNTIME=/path/to/persisted/opsd-preparation3-xformers \
python test_audit_selection.py -v
```

Thirteen fixtures currently pass: noncontiguous five-sibling grouping with exact native match and an actual full/policy selection change; source-order reference/student ties; min search/token and fail-priority/fallback; Unicode and authored-information removal; duplicate/missing groups/pairs; selection mismatch; malformed schema/mask and stored strict reward; whole-structure defect versus valid final reward; bad JSON/partial/oversize complete hashing; input stat/hash change; wrong frozen source; pair type/extra-field rejection. Fixture success is engineering validation only. Full-data acceptance is still pending Root's subsequent authorized run.
