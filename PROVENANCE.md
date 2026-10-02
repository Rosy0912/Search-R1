# Search-G1 baseline archive

This branch preserves the user-provided Search-G1 code archive as a comparison baseline. It is **Search-G1**, not Search-E1.

- Paper: [Search-G1: Grounded Search Agents via Representation-Based Intrinsic Rewards, v1](https://arxiv.org/pdf/2608.07531v1).
- Original repository declared by the paper and README: `Rosy0912/Search-G1`.
- Input archive: `Search-G1-main.zip`, supplied locally on 2026-10-02.
- Archive SHA256: `f759939622b69b8a4e3640adfa75495360ba2829040bd47e3815df7670239fca`.
- Source import commit: `1a55c92`; all 227 extracted files were imported with their original contents. The archive contained no Git history, so an upstream commit identifier cannot be recovered from it.
- The public GitHub URL returned HTTP 404 during inspection. Its accessibility and a specific upstream commit have not been independently established.
- Preserve `LICENSE` and `Notice.txt`. Importing this code does not assert authorship or successful reproduction.

The `search-g1` branch is a separate source tree in `shaoshitong/Search-R1`; use a separate checkout from the `search-r1` branch. These branches have independent source histories. Downloaded data, checkpoints, environments and credentials are excluded.

The primary Search-G1 paper experiments use Qwen2.5-3B/7B-Instruct and a representation-shaped reward. Our currently running Search-R1 reproduction uses Qwen2.5-3B base and binary outcome EM. Their raw training reward values are not comparable; compare held-out EM only with matched model/data/evaluation conditions. No Search-G1 training run has been started by this import.

See `review/baseline-review.md` for independent findings. The original source is retained for audit; the findings should be resolved and tested before describing this baseline as a verified reproduction. No code is silently substituted into the running Search-R1 process.
